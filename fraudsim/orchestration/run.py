"""Driving episodes through the world.

The warm start fills a benign world with history. This runner adds the other
half: adversarial episodes, interleaved with benign traffic, at a realistic
prevalence. It is the data-collection half on its own — no training, just the
data collection every later phase reads from.

Three things it gets right on purpose.

**Prevalence is measured, not assumed.** Fraud is a share of authorisations, not
of episodes, because an episode can carry one auth or twenty. The runner counts
auth events and stops adding attackers when their auths reach the configured
base rate of the total. Never fifty-fifty; the design is emphatic about it.

**A policy is handed an observation, never the actor.** The runner builds an
`ActorObservation` — stage, legal mask, a small feature mapping — and nothing
else crosses to the policy. A scripted policy that reached past it would be one
the learned policy could never imitate, so the boundary is enforced here where
both kinds of policy pass through.

**Sequences are logged and read.** The top action sequences each round are
recorded. For a scripted attacker they confirm the scripts do what they claim;
for a learned one an absurd sequence means a simulator bug, not a clever
attacker, and the log is where it surfaces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..config.simulation import SimulationConfig
from ..engine.outcome import OutcomeCode
from ..engine.simulator import Actor, ActorKind, Simulator
from ..engine.stages import Stage, StageGate
from ..features.schema import EventType
from ..ids import ActorId, CardId
from ..protocols import ActorObservation, RiskScorer, Target
from ..attacker.scripted import VERTICALS, ZERO_SHOT_HOLDOUTS, build_policy
from ..population.warmstart import WarmStartRunner

_STAGE_INDEX = {stage: i for i, stage in enumerate(Stage)}


@dataclass
class RunReport:
    """What one run produced."""

    benign_auths: int = 0
    fraud_auths: int = 0
    approved_fraud_auths: int = 0
    episodes: int = 0
    reached_monetized: int = 0
    per_vertical: dict[str, int] = field(default_factory=dict)
    top_sequences: list[tuple[str, int]] = field(default_factory=list)
    # Set when the run stopped on its episode budget rather than on reaching the
    # prevalence target. It means the world could no longer produce fraud
    # authorisations — every binding blocklisted, every card frozen — which is a
    # real outcome worth reporting rather than a silent shortfall.
    exhausted: bool = False

    @property
    def fraud_auth_share(self) -> float:
        total = self.benign_auths + self.fraud_auths
        return self.fraud_auths / total if total else 0.0

    @property
    def fraud_approval_rate(self) -> float:
        """Share of fraud authorisations the defender let through.

        The attacker's success against a defender: money that got through. A
        stronger defender declines more, so this falls across the columns of the
        arms-race matrix.
        """
        return self.approved_fraud_auths / self.fraud_auths if self.fraud_auths else 0.0

    def render(self) -> str:
        lines = [
            "episode run",
            f"  benign auths        {self.benign_auths:>10,}",
            f"  fraud auths         {self.fraud_auths:>10,}",
            f"  fraud auth share    {self.fraud_auth_share:>10.4f}",
            f"  episodes            {self.episodes:>10,}",
            f"  reached monetized   {self.reached_monetized:>10,}",
        ]
        if self.exhausted:
            lines.append(
                "  NOTE: stopped on the episode budget, not on the prevalence "
                "target -- the world could no longer produce fraud auths"
            )
        lines += ["", "  episodes per vertical"]
        for name in sorted(self.per_vertical):
            lines.append(f"    {name:<20}{self.per_vertical[name]:>8,}")
        lines += ["", "  top action sequences"]
        for seq, count in self.top_sequences[:10]:
            lines.append(f"    {count:>4}  {seq}")
        return "\n".join(lines)


class EpisodeRunner:
    """Runs adversarial episodes against a world, to a prevalence target."""

    def __init__(
        self,
        simulator: Simulator,
        config: SimulationConfig,
        seed: int = 0,
        train_only: bool = False,
    ) -> None:
        self.simulator = simulator
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.gate = StageGate()
        # Excluding the zero-shot holdouts is how the generalisation test is
        # kept honest: an attack the defender never trained on, evaluated at the
        # end. `train_only` selects the training verticals.
        self._verticals = [
            v for v in VERTICALS if not (train_only and v in ZERO_SHOT_HOLDOUTS)
        ]
        self._sequences: Counter[str] = Counter()
        self._next_actor = 1_000_000

    # ------------------------------------------------------------------ drive

    def run(self, scorer: RiskScorer | None = None, benign_seed: int = 0) -> RunReport:
        """Generate live benign traffic, then fraud to the configured share.

        The world is assumed already warm-started (history), which is excluded
        from training. This adds the live observation window: benign traffic
        that continues forward and is labelled as the negative class, then
        adversarial episodes until fraud reaches the base rate of live auths.
        Prevalence is measured over the live window, since that is what a
        detector trains and is evaluated on.
        """
        # The negative class: ordinary traffic in the observation window, run
        # through the same benign machinery as the history but left unflagged.
        WarmStartRunner(self.simulator, self.config, seed=benign_seed).run(live=True)
        benign_auths = sum(
            1
            for e in self.simulator.log.events
            if not getattr(e, "is_warm_start", False)
            and getattr(e, "event_type", None) == EventType.AUTH_ATTEMPT
        )
        # Ordinary customers also dispute wrong charges, open real support
        # tickets, and ask for genuine refunds. Without these the text expert
        # sees only fraudulent disputes and has no honest ones to contrast — the
        # text signal cannot be learned. This injects them, labelled benign,
        # carrying real text from the pool.
        self._benign_text_sweep(n=self._benign_text_count(benign_auths))

        report = RunReport(benign_auths=benign_auths)
        target_share = self.config.engine.fraud_base_rate

        # Solve for the fraud auths that hit the share: f / (b + f) = r.
        wanted = int(round(target_share * benign_auths / max(1e-9, 1 - target_share)))
        cards = self._targetable_cards()
        bound_cards = self._bound_cards()
        if not cards:
            return report

        # A budget on episodes, not only on auths. The loop's exit condition is a
        # count of auth *events*, and an episode can produce none: an
        # authorisation whose card has no usable binding fails before an event is
        # built, and a vertical that monetises through a refund or a transfer may
        # never attempt one at all. Both were reachable — a long co-adaptation run
        # ends with mitigations having blocklisted devices and frozen cards, and
        # the zero-shot holdout then runs a single vertical against that world.
        # The result was a process that finished training and hung indefinitely in
        # the evaluation afterwards, with the whole run lost.
        #
        # The budget is generous enough that it never binds on a healthy world;
        # it exists so that an unproductive one ends the loop instead of spinning
        # in it, and says so.
        budget = max(200, wanted * 20)
        attempts = 0

        while report.fraud_auths < wanted:
            if attempts >= budget:
                report.exhausted = True
                break
            attempts += 1
            vertical = self._verticals[int(self.rng.integers(len(self._verticals)))]
            pool = bound_cards if vertical in self.TAKEOVER_VERTICALS else cards
            if not pool:
                pool = cards
            target = self._draw_target(pool)
            auths, approved, monetized, seq = self._episode(vertical, target, scorer)
            report.fraud_auths += auths
            report.approved_fraud_auths += approved
            report.episodes += 1
            report.reached_monetized += int(monetized)
            report.per_vertical[vertical] = report.per_vertical.get(vertical, 0) + 1
            self._sequences[seq] += 1
            # An episode emits its auths in a burst, so the crossing overshoots
            # by up to one episode's worth. Once at least the wanted count is
            # reached, stopping here keeps the share close to target rather than
            # letting a run of long episodes drift it upward.
            if report.fraud_auths >= wanted:
                break

        # The benign backdrop was never wrapped in an episode, so it is still
        # unlabelled. Now that collection is done and no fraud episode can claim
        # those events, stamp them benign — the negative class the defender needs.
        self.simulator.log.stamp_unlabelled_benign()

        report.top_sequences = self._sequences.most_common(10)
        return report

    # ----------------------------------------------------- benign text sweep

    # Which benign text action each text vertical is expressed as. These are the
    # legitimate counterparts of the fraud text verticals.
    _BENIGN_TEXT_ACTIONS = (
        ("friendly_fraud", "file_dispute"),
        ("support_se", "open_ticket"),
        ("refund_abuse", "request_refund"),
    )

    def _benign_text_count(self, benign_auths: int) -> int:
        """How many benign text events to inject.

        Scaled to the benign traffic so the text expert has a comparable number
        of honest disputes to the fraudulent ones it will see, rather than a
        published-rate trickle that leaves the class empty in a short window.
        """
        return max(20, benign_auths // 40)

    def _benign_text_sweep(self, n: int) -> None:
        """Have ordinary holders perform legitimate text actions.

        Each draws a holder with a real transaction to dispute or refund, files
        it through the same action an attacker would, and is labelled benign
        because it belongs to no adversarial episode. The action requests text
        from the pool, so the event carries a real embedding.
        """
        from ..engine.actions import Action, ActionName

        graph = self.simulator.graph
        action_names = {
            "file_dispute": ActionName.FILE_DISPUTE,
            "open_ticket": ActionName.OPEN_TICKET,
            "request_refund": ActionName.REQUEST_REFUND,
        }
        cards_with_txn = [
            int(c) for c in graph.cards if graph.merchants_of_card(c)
        ] or [int(c) for c in graph.cards]
        if not cards_with_txn:
            return

        for _ in range(n):
            card_id = int(self.rng.choice(cards_with_txn))
            holder_id = int(graph.cards[card_id].holder_id)
            _, action_key = self._BENIGN_TEXT_ACTIONS[
                int(self.rng.integers(len(self._BENIGN_TEXT_ACTIONS)))
            ]
            # FILE_DISPUTE is legal only at MONETIZED; the other two at BOUND.
            stage = Stage.MONETIZED if action_key == "file_dispute" else Stage.BOUND
            actor_id = self._benign_text_actor(holder_id, card_id, stage)
            self.simulator.step(
                actor_id,
                Action(name=action_names[action_key], target_id=card_id),
            )
            # No episode is opened, so these stay unlabelled until the runner
            # stamps the benign backdrop at the end — which is exactly right,
            # since a legitimate dispute is benign ground truth.

    def _benign_text_actor(self, holder_id: int, card_id: int, stage: Stage):
        """A benign actor at the stage where the intended text action is legal."""
        actor_id = ActorId(self._next_actor)
        self._next_actor += 1
        self.simulator.register_actor(
            Actor(
                actor_id=actor_id,
                kind=ActorKind.BENIGN,
                holder_id=holder_id,
                cards=[CardId(card_id)],
                stage=stage,
            )
        )
        return actor_id

    # --------------------------------------------------------------- episode

    def _episode(self, vertical: str, target: Target, scorer):
        sim = self.simulator
        actor_id = ActorId(self._next_actor)
        self._next_actor += 1

        actor = sim.register_actor(
            Actor(
                actor_id=actor_id,
                kind=ActorKind.ADVERSARIAL,
                holder_id=target.holder_id,
                cards=[target.card_id],
                stage=Stage.NONE,
            )
        )
        sim.open_episode(actor_id)
        policy = build_policy(vertical, target, self.rng)

        auths = 0
        approved = 0
        names: list[str] = []
        episode_cfg = self.config.engine.episode
        # The wall clock at the episode's start, for the duration cap. Enforced
        # here as well as in the learned environment: a control that only one of
        # the two attackers obeys is not a control, and the scripted episodes are
        # what the behaviour clone is fitted to.
        started_at = sim.clock.now
        for _ in range(episode_cfg.max_actions):
            if (sim.clock.now - started_at) / 60.0 >= episode_cfg.max_hours:
                break
            obs = self._observe(actor)
            action = policy.act(obs)
            if action is None:
                break
            before = len(sim.log)
            outcome = sim.step(actor_id, action)
            policy.observe(outcome)
            names.append(action.name.value)
            # Count auths by the event they produced, so the prevalence figure
            # matches what a detector would tally.
            if (
                len(sim.log) > before
                and getattr(sim.log.events[-1], "event_type", None) == EventType.AUTH_ATTEMPT
            ):
                auths += 1
                # An approved fraud auth is money through; a stronger defender
                # declines more of them. This is what the arms-race matrix reads,
                # since it reflects mitigation biting rather than merely the stage
                # an episode reached.
                if outcome.code is OutcomeCode.APPROVED:
                    approved += 1

        monetized = actor.stage in (Stage.MONETIZED, Stage.TERMINAL)
        sim.close_episode(actor_id)
        return auths, approved, monetized, ">".join(names)

    # ----------------------------------------------------------- observation

    def _observe(self, actor: Actor) -> ActorObservation:
        """Everything a policy may see, and nothing more.

        The feature mapping is deliberately thin and graph-free: the clock, the
        actor's own progress counters. It is what the learned policy will read,
        so it may not carry anything a real attacker could not know.
        """
        mask = self.gate.legal_mask(actor.stage)
        features = {
            "now_minutes": float(self.simulator.clock.now),
            "actions_taken": float(actor.actions_taken),
            "value_extracted": float(actor.value_extracted),
            "stage": float(_STAGE_INDEX[actor.stage]),
        }
        return ActorObservation(
            actor_id=actor.actor_id,
            stage=_STAGE_INDEX[actor.stage],
            legal_action_mask=mask.tolist(),
            features=features,
        )

    # --------------------------------------------------------------- targets

    # Verticals that reach a usable state by taking over the victim's account
    # spend through an existing binding, so they must target a card that already
    # has one. The others provision their own device and can target any card.
    TAKEOVER_VERTICALS = frozenset({"phishing_ato"})

    def _targetable_cards(self) -> list[int]:
        """Every card, for verticals that provision their own device."""
        graph = self.simulator.graph
        return [int(c) for c in graph.cards]

    def _bound_cards(self) -> list[int]:
        """Cards that already carry a binding, for takeover verticals."""
        graph = self.simulator.graph
        return [int(c) for c in graph.cards if graph.devices_of_card(c)]

    def _draw_target(self, pool: list[int]) -> Target:
        graph = self.simulator.graph
        card_id = int(self.rng.choice(pool))
        holder_id = int(graph.cards[card_id].holder_id)
        accounts = sorted(graph.accounts_of_holder(holder_id))
        account_id = int(accounts[0]) if accounts else None
        # A modest merchant pool, drawn once per episode. High-liquidity
        # merchants where available, since that is where value is extracted.
        merchants = list(graph.merchants)
        pool = self.rng.choice(merchants, size=min(20, len(merchants)), replace=False)
        return Target(
            card_id=card_id,
            holder_id=holder_id,
            account_id=account_id,
            merchants=tuple(int(m) for m in pool),
        )
