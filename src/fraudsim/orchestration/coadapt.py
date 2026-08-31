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

import time

import numpy as np

from ..logs import get_logger
from ..attacker.bootstrap import collect_demos
from ..attacker.env import AttackEnv
from ..attacker.ppo import PPOConfig, PPOTrainer
from ..attacker.selection import ThompsonSelector, card_context
from ..settings.simulation import SimulationConfig
from ..defender.baseline import GBDTBaseline
from ..engine.bands import CostModel, grid_search_bands
from ..defender.combiner import MixtureScorer
from ..defender.table import build_table
from ..engine.simulator import Simulator
from ..features.builder import EventBuilder
from ..features.state import FeatureStateStore
from ..population.builder import PopulationBuilder
from ..population.warmstart import WarmStartRunner
from ..protocols import RiskScorer, Target
from ..rules.engine import VelocityRuleScorer
from ..timing.circadian import HolderClockModel
from ..attacker.scripted import VERTICALS, ZERO_SHOT_HOLDOUTS, build_policy
from .coadapt_eval import (
    log_sequences,
    mask_table,
    measure_success,
    refusal_rate,
    zero_shot_recall,
)
from .coadapt_report import CoadaptReport, _Progress
from .retention import RetentionBuffer
from .run import EpisodeRunner

_log = get_logger(__name__)


class CoadaptEngine:
    """Runs the four-phase warm-start then live co-adaptation."""

    def __init__(
        self,
        config: SimulationConfig,
        seed: int = 0,
        learned_defender: bool = False,
        benign_rounds: int = 3,
        ppo_config: PPOConfig | None = None,
        pool_path=None,
        cfpb_path=None,
        candidates: int = 5,
        selection_warmup: int = 10,
        label_latency_minutes: int = 0,
        fraud_rounds: int | None = None,
        dump_size: int = 3,
        stealth_frozen: bool = False,
        target_prevalence: float | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.learned_defender = learned_defender
        self.rng = np.random.default_rng(seed)
        self.buffer = RetentionBuffer(
            benign_rounds=benign_rounds,
            fraud_rounds=fraud_rounds,
            target_prevalence=target_prevalence,
            seed=seed,
        )
        self.label_latency_minutes = label_latency_minutes
        # How much of the log has already been folded into the retention buffer.
        # Without this every refit re-reads the whole log and the same fraud is
        # counted once per refit — the buffer grew to two hundred thousand rows
        # of mostly duplicates, and the defender was effectively trained on the
        # same attack a dozen times over.
        self._log_consumed = 0
        # Counts the rounds of live benign traffic, so each draws a fresh seed.
        self._benign_rounds = 0
        self._last_fp_rate = 0.0
        # The freeze belongs to the trainer, so every rollout path honours it
        # rather than each call site remembering to. Copied rather than mutated
        # in place: the caller's config is theirs, and an ablation that silently
        # edits it would carry the freeze into whatever ran next.
        import dataclasses as _dc

        base = ppo_config or PPOConfig()
        self.ppo_config = _dc.replace(base, stealth_frozen=stealth_frozen)
        self.trainer = PPOTrainer(AttackEnv.obs_dim(), self.ppo_config)
        # Victim choice is a contextual bandit, not part of the sequential
        # policy: a card is picked, a return is observed, and nothing carries to
        # the next episode. It selects on card-dump features only.
        self.selector = ThompsonSelector(
            warmup_updates=selection_warmup, seed=seed
        )
        self.candidates = candidates
        # How many cards an episode's dump holds. One reproduces the behaviour
        # before the stealth head existed.
        self.dump_size = max(1, dump_size)
        # The ablation switch. With the head frozen to the loud posture the
        # policy is exactly the one that collapsed at the first refit, which is
        # the only control that shows whether stealth did anything — a nicer
        # curve after a dozen edits proves nothing on its own.
        self.stealth_frozen = stealth_frozen
        self._pending_context = None

        # One persistent world. The defender in force is swapped as it refits;
        # the population and its history stay put, so the attacker adapts against
        # a stable world with a moving defence.
        setup = _Progress("setup: world")
        setup.say(f"building the population graph ({config.population.n_holders:,} holders)")
        graph, _ = PopulationBuilder(config).build()
        states = FeatureStateStore(config.engine.windows)
        builder = EventBuilder(
            graph, states, config.engine.windows, HolderClockModel(config.behavior.circadian)
        )
        # The text pool, if one is built, so dispute/ticket/refund actions carry
        # real generated text and its embedding to the text expert. Absent, the
        # simulator falls back to empty artifacts and the text expert sees no
        # text columns — the pipeline still runs, just without that signal.
        from ..generative.pool import load_artifact_source

        setup.say(f"graph holds {len(graph.cards):,} cards, {len(graph.devices):,} devices")
        setup.say("loading the text pool" if pool_path else "no text pool; running without text")
        artifacts = load_artifact_source(pool_path, cfpb_path, seed=seed) if pool_path else None
        self.sim = Simulator(
            graph, config, builder,
            scorer=VelocityRuleScorer(config.engine.rules),
            artifacts=artifacts,
        )
        setup.say("warm start: backdating history so the first events are not feature-poor")
        WarmStartRunner(self.sim, config, seed=config.seed).run()
        setup.done(f"{len(self.sim.log):,} warm-start events")
        self._cards = [int(c) for c in graph.cards if graph.devices_of_card(c)]
        self._train_verticals = [v for v in VERTICALS if v not in ZERO_SHOT_HOLDOUTS]
        self.defender: RiskScorer = self.sim.scorer

    # ----------------------------------------------------------- checkpoints

    def save(self, directory) -> dict:
        """Write the attacker and the defender, returning where each landed."""
        from pathlib import Path

        from ..defender.persist import save_defender

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        attacker_path = directory / "attacker.pt"
        self.trainer.save(attacker_path)

        defender_path = directory / "defender.joblib"
        save_defender(self.defender, defender_path)

        return {"attacker": str(attacker_path), "defender": str(defender_path)}

    # ------------------------------------------------------------ env thunks

    def _card_context(self, card_id: int) -> np.ndarray:
        """The dump-knowable features of a card: BIN tier and age band."""
        card = self.sim.graph.cards[card_id]
        age_days = max(0.0, (self.sim.clock.now - card.issued_ts) / 1440.0)
        return card_context(card.bin_tier, age_days)

    def _target(self) -> Target:
        """Draw candidates, let the bandit choose among them, build the target.

        A handful of candidates rather than the whole population: an attacker
        buying a dump chooses among what the dump holds, not among every card
        that exists, and a small candidate set keeps this a selection problem
        rather than a ranking over thousands.
        """
        graph = self.sim.graph
        k = min(self.candidates, len(self._cards))
        candidate_ids = [
            int(c) for c in self.rng.choice(self._cards, size=k, replace=False)
        ]
        contexts = [self._card_context(c) for c in candidate_ids]
        chosen = self.selector.select(contexts)
        card_id = candidate_ids[chosen]
        # Held so the realised return can be credited to the features that led
        # to this choice once the episode closes.
        self._pending_context = contexts[chosen]

        # The rest of the dump: the candidates the bandit passed over. Taking
        # them from the same draw rather than from a fresh one keeps the bandit's
        # context honest — it still describes the card the episode leads with,
        # and the spares are what a buyer of that same batch would also hold.
        # Drawing spares independently would let an episode reach cards the
        # selector never saw, and the posterior would be crediting returns to a
        # choice it did not make.
        spares = tuple(c for c in candidate_ids if c != card_id)[: self.dump_size - 1]

        holder_id = int(graph.cards[card_id].holder_id)
        accounts = sorted(graph.accounts_of_holder(holder_id))
        merchants = list(graph.merchants)
        pool = self.rng.choice(merchants, size=min(20, len(merchants)), replace=False)
        return Target(
            card_id=card_id,
            holder_id=holder_id,
            account_id=int(accounts[0]) if accounts else None,
            merchants=tuple(int(m) for m in pool),
            card_ids=(card_id,) + spares,
        )

    def _make_env(self) -> AttackEnv:
        env = AttackEnv(self.sim, self._target())
        # The context that chose this victim rides on the env, so the return can
        # be credited back to it when the episode closes.
        env.selection_context = self._pending_context
        return env

    def _credit_selection(self, env, total_reward: float) -> None:
        """Fold one episode's return into the selector's posterior."""
        context = getattr(env, "selection_context", None)
        if context is not None:
            self.selector.record(context, total_reward)

    def _make_env_and_policy(self):
        target = self._target()
        env = AttackEnv(self.sim, target)
        vertical = self._train_verticals[int(self.rng.integers(len(self._train_verticals)))]
        return env, build_policy(vertical, target, self.rng)

    # -------------------------------------------------- phase A: defender

    def phase_a_defender(self) -> int:
        """Fit the initial defender on benign traffic and a first fraud sweep.

        Every step announces itself. The live phase prints per update precisely
        so that a slow run is distinguishable from a hung one, and the same
        reasoning applies here with more force: at twelve thousand holders these
        warm-start phases are twenty minutes of complete silence, and there is
        nothing in the log to say whether the process is working or wedged.
        """
        step = _Progress("phase A: initial defender")
        step.say("collecting benign traffic and a first scripted fraud sweep")
        EpisodeRunner(self.sim, self.config, seed=self.seed + 1, train_only=True).run(
            benign_seed=self.seed + 2
        )
        step.say(f"log holds {len(self.sim.log):,} events; extracting the table")
        table = build_table(self.sim.log, exclude_warm_start=True)
        self.buffer.add(table)
        train = self.buffer.training_table()
        step.say(
            f"fitting the defender on {len(train):,} rows "
            f"({int((train.y == 1).sum()):,} fraud)"
        )
        self.defender = self._fit_defender(train)
        self.sim.set_scorer(self.defender)
        step.done()
        return int((table.y == 1).sum())

    def _fit_defender(self, train) -> RiskScorer:
        if len(train) == 0 or train.y.sum() == 0:
            return self.defender
        if self.learned_defender:
            model = MixtureScorer.fit(train, learned=True)
        else:
            model = GBDTBaseline(train.columns).fit(train)
        model.bands = self._search_bands(model, train)
        return model

    def _search_bands(self, model, train):
        """Place the decision thresholds on the business cost curve.

        Without this the live defender ran on the round-number defaults and had
        exactly one loss condition: missing fraud. Declining a genuine customer
        cost it nothing, so its cheapest winning move was to become stricter
        without limit — and the reason it had not already done so was luck.

        A detector that must stay usable is the constraint that leaves an
        attacker any room at all, which is what makes the contest a contest.
        The search trades a missed fraud against the friction of stepping up a
        real customer and the analyst time of holding one, and lands the bands
        where those meet rather than where the defaults happened to sit.

        Searched on the training set, which slightly favours the defender: the
        thresholds see the same rows the model was fitted on. Stated rather than
        hidden — a held-out split at this prevalence would leave too few
        positives to place a threshold on.
        """
        # The two defenders take different arguments here: the flat tree scores a
        # matrix, the mixture scores a table because its experts each read their
        # own columns. Dispatching on the type rather than catching an exception:
        # a bare except around this silently fell back to the default bands, so
        # the search never ran and nothing said so.
        scores = (
            model.predict_scores(train)
            if isinstance(model, MixtureScorer)
            else model.predict_scores(train.X)
        )
        return grid_search_bands(train.y, np.asarray(scores), CostModel())

    # ----------------------------------------- phases B and C: actor, critic

    def phase_b_actor(self, demo_episodes: int, bc_epochs: int) -> float:
        """Behaviour-clone the actor against the frozen defender."""
        step = _Progress("phase B: actor behaviour cloning")
        step.say(f"running {demo_episodes:,} scripted demonstration episodes")
        demos = collect_demos(self._make_env_and_policy, demo_episodes, self.rng)
        step.say(f"cloning on {len(demos):,} transitions, {bc_epochs} epochs "
                 f"[{self.trainer.device}]")
        losses = self.trainer.behaviour_clone(demos, bc_epochs, self.rng) if demos else [0.0]
        step.done(f"final loss {losses[-1]:.4f}" if losses else "")
        return losses[-1] if losses else 0.0

    def phase_c_critic(self, rollout_episodes: int, critic_epochs: int) -> float:
        """Fit the critic on the frozen cloned actor's rollouts."""
        step = _Progress("phase C: critic")
        step.say(f"collecting {rollout_episodes:,} rollouts from the cloned actor")
        batch = self.trainer.collect(self._make_env, rollout_episodes, self.rng)
        step.say(f"fitting the critic on {len(batch):,} transitions, {critic_epochs} epochs")
        losses = self.trainer.fit_critic(batch, critic_epochs)
        step.done(f"final loss {losses[-1]:.1f}" if losses else "")
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
        # Logged as they happen, not accumulated and rendered at the end. A live
        # phase of this length is otherwise completely silent, and a run that is
        # merely slow is indistinguishable from one that has hung.
        _log.info("live phase: %d updates, refit every %d", n_updates, refit_every)
        _log.info("update  extracted   return   entropy  critic      elapsed")

        started = time.perf_counter()
        for update in range(n_updates):
            # A fresh log window for this update, so the fraud attributed to the
            # refit is what the attacker produced since the last one.
            batch = self.trainer.collect(
                self._make_env, episodes_per_update, self.rng,
                on_episode_end=self._credit_selection,
            )
            stats = self.trainer.update(batch, self.rng)
            # One update done: the selector counts these to decide when its
            # warm-up is over and it may start choosing rather than sampling.
            self.selector.end_update()

            extracted = self._measure_success(episodes=self.EVAL_EPISODES)
            report.attacker_success.append(extracted)
            report.mean_return.append(float(batch.ret.mean().item()))
            report.entropy.append(stats["entropy"])

            # The critic's error relative to the spread of the returns, which is
            # comparable across runs; the absolute loss is not, since it scales
            # with the square of the reward.
            crit_rel = self.trainer.critic_relative_error(batch)
            report.critic_relative.append(crit_rel)

            elapsed = time.perf_counter() - started
            eta = (elapsed / (update + 1)) * (n_updates - update - 1)
            _log.info(
                "  %-7d%9.1f%9.2f%10.3f%8.2f   %5.1fm  eta %5.1fm",
                update, extracted, report.mean_return[-1], stats["entropy"],
                crit_rel, elapsed / 60, eta / 60,
            )

            if (update + 1) % refit_every == 0:
                # Record what the attacker was doing just before the defender
                # retrains, so the strategy either side of a refit is visible.
                report.strategy_history.append(
                    {
                        "update": update,
                        "when": "before_refit",
                        "sequences": [
                            {"sequence": seq, "count": n}
                            for seq, n in self._log_sequences(episodes=12)
                        ],
                    }
                )
                positives = self._refit_defender()
                report.defender_refits.append(update)
                report.defender_positives_at_refit.append(positives)
                report.false_positive_rate.append(self._last_fp_rate)
                _log.info(
                    "    -> defender refit on %s fraud rows"
                    "   (declined %.2f%% of genuine traffic)",
                    f"{positives:,}", self._last_fp_rate * 100,
                )

        report.selection = {
            "describe": self.selector.describe(),
            "weights": self.selector.weights().tolist(),
            "active": self.selector.active,
        }
        report.zero_shot = self._zero_shot_recall()
        report.top_sequences = self._log_sequences(episodes=40)
        return report

    def _refit_defender(self) -> int:
        """Refit on the fraud that has had time to be labelled.

        Labels arrive late in reality: a chargeback takes days to weeks, and a
        detector is therefore always fitted to a picture of the attack that is
        already stale. Modelling that lag is what leaves an adapting attacker
        anything to exploit — with instant labels the detector sees each new
        tactic the moment it appears and the contest is over before it starts.

        The lag is expressed in simulated minutes and applied by withholding the
        most recent events from the refit.
        """
        # The world goes on living between refits. Without this the live phase
        # generated no benign traffic at all: the warm start ran once before
        # training and nothing after, so every event in a refit window came from
        # the attacker. The detector was being fitted at 42% fraud against a
        # design that specifies 0.5%, which is a different and far easier problem
        # — at that balance almost any split separates the classes, and the
        # defender's apparent invincibility was mostly the prevalence.
        #
        # Generated before the window is read, so the benign traffic lands in the
        # same window as the fraud it has to be told apart from.
        self._generate_benign_traffic()

        # Only the events since the last refit. The log is cumulative, so
        # rebuilding from all of it would re-add everything already retained.
        from ..features.schema import EventLog

        window = EventLog()
        for event in self.sim.log.events[self._log_consumed :]:
            window.append(event)
        self._log_consumed = len(self.sim.log)

        table = build_table(window, exclude_warm_start=True)
        if self.label_latency_minutes > 0:
            cutoff = self.sim.clock.now - self.label_latency_minutes
            mature = np.array(
                [getattr(e, "ts", 0) <= cutoff for e in table.events], dtype=bool
            )
            table = mask_table(table, mature)
        self.buffer.add(table)
        train = self.buffer.training_table()
        self.defender = self._fit_defender(train)
        self.sim.set_scorer(self.defender)
        return int((train.y == 1).sum())

    def _generate_benign_traffic(self) -> int:
        """Let the population transact, forward from the current clock.

        `live=True` places the traffic at the present instant and leaves it
        unflagged, which is what makes it the negative class rather than
        burn-in history. The seed advances per round so successive rounds are
        different traffic rather than the same day repeated.
        """
        from ..features.schema import AuthAttemptEvent

        before = len(self.sim.log)
        # Sized to a target volume rather than to the population. The negative
        # class needs enough traffic to hold the prevalence and to represent the
        # world, not one event from every card: a full sweep of twelve thousand
        # holders, repeated at every refit, cost more than the training it was
        # supporting. A sample of the same cards behaves the same way.
        eligible = max(1, sum(1 for c in self.sim.graph.cards
                              if self.sim.graph.devices_of_card(c)))
        per_card = max(1, self.config.warm_start.events_per_entity)
        fraction = min(1.0, self.BENIGN_TARGET_EVENTS / (eligible * per_card))
        WarmStartRunner(
            self.sim, self.config, seed=self.seed + 5000 + self._benign_rounds
        ).run(live=True, card_fraction=fraction)
        self._benign_rounds += 1

        # What the defender did to genuine customers while it was in force.
        #
        # Scored here rather than read off the events: whether an authorisation
        # was approved is an argument to `commit_auth`, not a field the event
        # carries, so the obvious `getattr(event, "approved")` finds nothing and
        # silently reports a perfect zero. It did, for several runs.
        #
        # Scoring with `self.defender` — the one that was in force while this
        # traffic ran, since the refit happens after this returns — keeps the
        # rate a description of decisions actually taken.
        fresh = [
            e for e in self.sim.log.events[before:] if isinstance(e, AuthAttemptEvent)
        ]
        if fresh:
            self._last_fp_rate = self._refusal_rate(fresh)

        # The backdrop is not inside any episode, so it is still unlabelled.
        # Stamping it benign is what turns it into training negatives.
        self.sim.log.stamp_unlabelled_benign()
        return len(self.sim.log) - before

    # How many benign authorisations the false-positive rate is estimated from.
    # A sample rather than the whole sweep: the per-event scoring path rebuilds
    # and realigns a one-row table per call, and five thousand of those took
    # longer than the training they were measuring. Several hundred is ample for
    # a rate, and the sample is the first N of a sweep that is already in
    # arbitrary order.
    # Rollouts used to measure the curve's y-value each update. These are pure
    # measurement -- nothing trains on them -- so they are a straight overhead on
    # top of the episodes that do. Twenty-four of them against eighty training
    # episodes is thirty percent of the run spent watching rather than learning,
    # and the mean of twelve is not meaningfully noisier than the mean of
    # twenty-four at these magnitudes.
    EVAL_EPISODES = 12

    FP_SAMPLE = 400

    # Roughly how many benign events each live sweep should produce. Enough to
    # hold the prevalence and to give the detector a representative negative
    # class, and independent of population size so a larger world costs no more
    # here than a small one.
    BENIGN_TARGET_EVENTS = 6000

    def _refusal_rate(self, events) -> float:
        return refusal_rate(self.defender, events, self.FP_SAMPLE)

    def _measure_success(self, episodes: int) -> float:
        return measure_success(
            self.sim, self.trainer, self._make_env, self.stealth_frozen, episodes,
        )

    def _zero_shot_recall(self) -> dict[str, float]:
        return zero_shot_recall(
            self.sim, self.config, self.defender, self.seed, ZERO_SHOT_HOLDOUTS,
        )

    def _log_sequences(self, episodes: int):
        return log_sequences(
            self.sim, self.trainer, self._make_env, self.stealth_frozen, episodes,
        )


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
    pool_path=None,
    cfpb_path=None,
    checkpoint_dir=None,
    candidates: int = 5,
    selection_warmup: int = 10,
    label_latency_minutes: int = 0,
    fraud_rounds: int | None = None,
    dump_size: int = 3,
    stealth_frozen: bool = False,
    target_prevalence: float | None = None,
) -> CoadaptReport:
    """The whole thing, phases A through D, every scale a parameter.

    Both trained sides are written to `checkpoint_dir` when one is given: the
    attacker's actor and critic, and the final refitted defender. A run of this
    length produces models worth keeping — for re-scoring, for comparing across
    runs, and for resuming — and without saving them they vanish with the
    process.
    """
    engine = CoadaptEngine(
        config, seed=seed, learned_defender=learned_defender, ppo_config=ppo_config,
        pool_path=pool_path, cfpb_path=cfpb_path,
        candidates=candidates, selection_warmup=selection_warmup,
        label_latency_minutes=label_latency_minutes, fraud_rounds=fraud_rounds,
        dump_size=dump_size, stealth_frozen=stealth_frozen,
        target_prevalence=target_prevalence,
    )
    positives = engine.phase_a_defender()
    bc_loss = engine.phase_b_actor(demo_episodes, bc_epochs)
    critic_loss = engine.phase_c_critic(critic_rollouts, critic_epochs)
    report = engine.phase_d_live(n_updates, episodes_per_update, refit_every)
    report.initial_defender_positives = positives
    report.bc_final_loss = bc_loss
    report.critic_final_loss = critic_loss

    if checkpoint_dir is not None:
        report.checkpoints = engine.save(checkpoint_dir)
    return report
