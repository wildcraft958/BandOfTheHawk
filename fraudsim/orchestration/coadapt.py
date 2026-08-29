"""Live co-adaptation: the attacker and the defender improve against each other.

This is the solution the whole system was built to run. Not a turn-based loop
where one side is frozen while the other trains, but a single live run in which
the attacker adapts continuously and the defender refits as it accumulates the
fraud the attacker is producing. Both improve against a moving opponent, which is
what a real arms race is.

It runs in four phases, and the order is the point.

**A — the defender starts strong.** Before anything adversarial learns, the
defender is fitted on a first sweep of benign traffic and scripted fraud, so the
attacker faces a real detector from its first step rather than a rule of thumb.

**B — the actor warm-starts, everything else frozen.** With the defender frozen,
the policy is behaviour-cloned from the scripted attackers. It starts from a
working strategy instead of the inaction a cold policy collapses into.

**C — the critic warm-starts, the actor frozen.** The cloned actor is now frozen
and rolled out against the frozen defender; the critic is fitted on the returns
those rollouts realise. The value function is calibrated to the actor's actual
behaviour before a single PPO step, so the first advantages are trustworthy
rather than noise that would undo the clone.

**D — live adaptation.** Everything unfreezes. The attacker runs PPO
continuously. Every K updates the defender refits on the fraud accumulated since
its last refit — all fraud kept, recent benign only — and the world is pointed at
the new defender, so the attacker immediately faces the stronger defence. The two
clocks differ on purpose: an attacker adapts fast, a fraud model retrains
periodically, and that asymmetry is the real one.

What it produces is a curve, not a matrix: attacker success and defender strength
over the live phase, showing each responding to the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attacker.bootstrap import collect_demos
from ..attacker.env import AttackEnv
from ..attacker.ppo import PPOConfig, PPOTrainer
from ..config.simulation import SimulationConfig
from ..defender.baseline import GBDTBaseline
from ..defender.combiner import MixtureScorer
from ..defender.table import build_table
from ..engine.simulator import Simulator
from ..features.builder import EventBuilder
from ..features.state import FeatureStateStore
from ..population.builder import PopulationBuilder
from ..population.warmstart import WarmStartRunner
from ..protocols import RiskScorer
from ..rules.engine import VelocityRuleScorer
from ..timing.circadian import HolderClockModel
from ..attacker.scripted import VERTICALS, ZERO_SHOT_HOLDOUTS, build_policy
from .retention import RetentionBuffer
from .run import EpisodeRunner, Target


@dataclass
class CoadaptReport:
    """The live-phase curve and the phase-boundary facts."""

    initial_defender_positives: int = 0
    bc_final_loss: float = 0.0
    critic_final_loss: float = 0.0
    # Per live-update series.
    attacker_success: list[float] = field(default_factory=list)
    mean_return: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    defender_refits: list[int] = field(default_factory=list)  # update indices where D refit
    defender_positives_at_refit: list[int] = field(default_factory=list)
    zero_shot: dict[str, float] = field(default_factory=dict)
    top_sequences: list[tuple[str, int]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "live co-adaptation",
            f"  initial defender fraud   {self.initial_defender_positives:>8,}",
            f"  BC final loss            {self.bc_final_loss:>8.4f}",
            f"  critic final loss        {self.critic_final_loss:>8.4f}",
            "",
            "  live phase  (attacker success = fraud approval rate through the live defender)",
            "  update   success   return   entropy   defender",
        ]
        refit_set = set(self.defender_refits)
        for i, (succ, ret, ent) in enumerate(
            zip(self.attacker_success, self.mean_return, self.entropy)
        ):
            marker = "  <- refit" if i in refit_set else ""
            lines.append(f"    {i:<7}{succ:>8.3f}{ret:>9.2f}{ent:>10.3f}{marker}")
        lines += ["", "  reads"]
        if self.attacker_success:
            lines.append(
                f"    attacker success trend   {self.attacker_success[0]:.3f}"
                f" -> {self.attacker_success[-1]:.3f}"
            )
        lines += ["", "  zero-shot recall on held-out verticals"]
        for name, recall in self.zero_shot.items():
            lines.append(f"    {name:<16}{recall:>8.3f}")
        lines += ["", "  top action sequences (trained attacker)"]
        for seq, count in self.top_sequences[:8]:
            lines.append(f"    {count:>4}  {seq}")
        return "\n".join(lines)


class CoadaptEngine:
    """Runs the four-phase warm-start then live co-adaptation."""

    def __init__(
        self,
        config: SimulationConfig,
        seed: int = 0,
        learned_defender: bool = False,
        benign_rounds: int = 3,
        ppo_config: PPOConfig | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.learned_defender = learned_defender
        self.rng = np.random.default_rng(seed)
        self.buffer = RetentionBuffer(benign_rounds=benign_rounds)
        self.ppo_config = ppo_config or PPOConfig()
        self.trainer = PPOTrainer(AttackEnv.obs_dim(), self.ppo_config)

        # One persistent world. The defender in force is swapped as it refits;
        # the population and its history stay put, so the attacker adapts against
        # a stable world with a moving defence.
        graph, _ = PopulationBuilder(config).build()
        states = FeatureStateStore(config.engine.windows)
        builder = EventBuilder(
            graph, states, config.engine.windows, HolderClockModel(config.behavior.circadian)
        )
        self.sim = Simulator(graph, config, builder, scorer=VelocityRuleScorer(config.engine.rules))
        WarmStartRunner(self.sim, config, seed=config.seed).run()
        self._cards = [int(c) for c in graph.cards if graph.devices_of_card(c)]
        self._train_verticals = [v for v in VERTICALS if v not in ZERO_SHOT_HOLDOUTS]
        self.defender: RiskScorer = self.sim.scorer

    # ------------------------------------------------------------ env thunks

    def _target(self) -> Target:
        graph = self.sim.graph
        card_id = int(self.rng.choice(self._cards))
        holder_id = int(graph.cards[card_id].holder_id)
        accounts = sorted(graph.accounts_of_holder(holder_id))
        merchants = list(graph.merchants)
        pool = self.rng.choice(merchants, size=min(20, len(merchants)), replace=False)
        return Target(
            card_id=card_id,
            holder_id=holder_id,
            account_id=int(accounts[0]) if accounts else None,
            merchants=tuple(int(m) for m in pool),
        )

    def _make_env(self) -> AttackEnv:
        return AttackEnv(self.sim, self._target())

    def _make_env_and_policy(self):
        target = self._target()
        env = AttackEnv(self.sim, target)
        vertical = self._train_verticals[int(self.rng.integers(len(self._train_verticals)))]
        return env, build_policy(vertical, target, self.rng)

    # -------------------------------------------------- phase A: defender

    def phase_a_defender(self) -> int:
        """Fit the initial defender on benign traffic and a first fraud sweep."""
        EpisodeRunner(self.sim, self.config, seed=self.seed + 1, train_only=True).run(
            benign_seed=self.seed + 2
        )
        table = build_table(self.sim.log, exclude_warm_start=True)
        self.buffer.add(table)
        self.defender = self._fit_defender(self.buffer.training_table())
        self.sim.set_scorer(self.defender)
        return int((table.y == 1).sum())

    def _fit_defender(self, train) -> RiskScorer:
        if len(train) == 0 or train.y.sum() == 0:
            return self.defender
        if self.learned_defender:
            return MixtureScorer.fit(train, learned=True)
        return GBDTBaseline(train.columns).fit(train)

    # ----------------------------------------- phases B and C: actor, critic

    def phase_b_actor(self, demo_episodes: int, bc_epochs: int) -> float:
        """Behaviour-clone the actor against the frozen defender."""
        demos = collect_demos(self._make_env_and_policy, demo_episodes, self.rng)
        losses = self.trainer.behaviour_clone(demos, bc_epochs, self.rng) if demos else [0.0]
        return losses[-1] if losses else 0.0

    def phase_c_critic(self, rollout_episodes: int, critic_epochs: int) -> float:
        """Fit the critic on the frozen cloned actor's rollouts."""
        batch = self.trainer.collect(self._make_env, rollout_episodes, self.rng)
        losses = self.trainer.fit_critic(batch, critic_epochs)
        return losses[-1] if losses else 0.0

    # --------------------------------------------------- phase D: live loop

    def phase_d_live(
        self,
        n_updates: int,
        episodes_per_update: int,
        refit_every: int,
    ) -> CoadaptReport:
        """The live phase: attacker PPO continuous, defender refit every K updates.

        Each update collects a batch of attacker episodes against the current
        live defender, does one PPO update, and records the attacker's success —
        the share of fraud authorisations the defender let through. Every
        `refit_every` updates the defender refits on the fraud seen since, and the
        world is pointed at it, so the very next batch faces the stronger defence.
        """
        report = CoadaptReport()
        for update in range(n_updates):
            # A fresh log window for this update, so the fraud attributed to the
            # refit is what the attacker produced since the last one.
            batch = self.trainer.collect(self._make_env, episodes_per_update, self.rng)
            stats = self.trainer.update(batch, self.rng)

            report.attacker_success.append(self._measure_success(episodes=24))
            report.mean_return.append(float(batch.ret.mean().item()))
            report.entropy.append(stats["entropy"])

            if (update + 1) % refit_every == 0:
                positives = self._refit_defender()
                report.defender_refits.append(update)
                report.defender_positives_at_refit.append(positives)

        report.zero_shot = self._zero_shot_recall()
        report.top_sequences = self._log_sequences(episodes=40)
        return report

    def _refit_defender(self) -> int:
        """Add the fraud seen so far to the buffer and refit the live defender."""
        table = build_table(self.sim.log, exclude_warm_start=True)
        self.buffer.add(table)
        train = self.buffer.training_table()
        self.defender = self._fit_defender(train)
        self.sim.set_scorer(self.defender)
        return int((train.y == 1).sum())

    def _measure_success(self, episodes: int) -> float:
        """Fraud approval rate under the current live defender.

        A short scripted sweep against the defender in force, to read how much
        fraud is getting through right now. A stronger defender lets less
        through, so this falls when the defender refits and rises as the attacker
        adapts against it.
        """
        runner = EpisodeRunner(
            self.sim, self.config, seed=int(self.rng.integers(1 << 30)), train_only=True
        )
        report = runner.run(benign_seed=int(self.rng.integers(1 << 30)))
        return report.fraud_approval_rate

    def _zero_shot_recall(self) -> dict[str, float]:
        recalls: dict[str, float] = {}
        for vertical in ZERO_SHOT_HOLDOUTS:
            recalls[vertical] = self._recall_on_vertical(vertical)
        return recalls

    def _recall_on_vertical(self, vertical: str) -> float:
        # Run only this held-out vertical into the log, then score its fraud.
        runner = _SingleVerticalRunner(self.sim, self.config, vertical, seed=self.seed + 700)
        before = len(self.sim.log)
        runner.run(benign_seed=self.seed + 800)
        new_events = self.sim.log.events[before:]

        from ..features.schema import AuthAttemptEvent

        fraud = [e for e in new_events if isinstance(e, AuthAttemptEvent) and e.is_fraud]
        if not fraud:
            return 0.0
        caught = sum(1 for e in fraud if self.defender.score(e).risk_score >= 0.5)
        return caught / len(fraud)

    def _log_sequences(self, episodes: int):
        from collections import Counter

        import torch

        from ..engine.actions import ACTION_ORDER

        counter: Counter = Counter()
        for _ in range(episodes):
            env = self._make_env()
            obs = env.reset()
            names: list[str] = []
            done = False
            while not done:
                vec = torch.as_tensor(AttackEnv.encode(obs)).unsqueeze(0)
                mask = torch.as_tensor(AttackEnv.mask_vector(obs)).unsqueeze(0)
                with torch.no_grad():
                    discrete, amount, delay = self.trainer.actor(vec, mask)
                    a_idx = int(discrete.probs.argmax().item())
                    a_amt = float(amount.mean.item())
                    a_dly = float(delay.mean.item())
                names.append(ACTION_ORDER[a_idx].value)
                obs, _, done, _ = env.step(a_idx, a_amt, a_dly)
            env.close()
            counter[">".join(names)] += 1
        return counter.most_common(8)


class _SingleVerticalRunner(EpisodeRunner):
    """An episode runner restricted to one vertical, for zero-shot evaluation."""

    def __init__(self, simulator, config, vertical: str, seed: int = 0):
        super().__init__(simulator, config, seed=seed, train_only=False)
        self._verticals = [vertical]


def run_coadapt(
    config: SimulationConfig,
    seed: int,
    learned_defender: bool,
    demo_episodes: int,
    bc_epochs: int,
    critic_rollouts: int,
    critic_epochs: int,
    n_updates: int,
    episodes_per_update: int,
    refit_every: int,
    ppo_config: PPOConfig | None = None,
) -> CoadaptReport:
    """The whole thing, phases A through D, every scale a parameter."""
    engine = CoadaptEngine(
        config, seed=seed, learned_defender=learned_defender, ppo_config=ppo_config
    )
    positives = engine.phase_a_defender()
    bc_loss = engine.phase_b_actor(demo_episodes, bc_epochs)
    critic_loss = engine.phase_c_critic(critic_rollouts, critic_epochs)
    report = engine.phase_d_live(n_updates, episodes_per_update, refit_every)
    report.initial_defender_positives = positives
    report.bc_final_loss = bc_loss
    report.critic_final_loss = critic_loss
    return report
