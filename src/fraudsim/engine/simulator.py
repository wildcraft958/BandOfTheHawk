"""The referee.

One step, in a fixed order: check the action is legal, obtain any artifact it
needs, resolve it against the world, build the event, score it, apply the
result, and report back.

The order is load-bearing in two places. The event is built before it is
scored, so the scorer sees exactly what a real detector would see and nothing
more. And the event is committed to the rolling state after it is built, so its
own features never include itself.

Ordinary holders and attackers pass through the same method. Separate paths
would leave a difference in the events that a detector could find instead of
finding behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..clock import SimClock
from ..settings.simulation import SimulationConfig
from ..features.builder import EventBuilder
from ..features.schema import AuthAttemptEvent, EventLog, EventType
from ..ids import ActorId, CardId, DeviceId, MerchantId
from ..protocols import (
    AlwaysApproveScorer,
    ArtifactRequest,
    ArtifactSource,
    NullArtifactSource,
    RiskAction,
    RiskScorer,
)
from ..rng import RngHub
from ..world.graph import EntityGraph
from .actions import Action, ActionName, action_cost
from .outcome import RISK_TO_OUTCOME, Outcome, OutcomeCode
from .resolution import EVENT_FOR_ACTION, ActionResolver
from .mitigation import apply_all
from .stages import Stage, StageGate


class ActorKind(str):
    BENIGN = "benign"
    ADVERSARIAL = "adversarial"


@dataclass(slots=True)
class Actor:
    """Who is acting, and what they can currently do.

    The kind is here and never on an event. It decides what an actor is
    permitted to attempt and how the episode is labelled afterwards, and an
    event carrying it would be handing a detector the answer.
    """

    actor_id: ActorId
    kind: str
    stage: Stage = Stage.NONE
    holder_id: int | None = None
    cards: list[CardId] = field(default_factory=list)
    devices: list[DeviceId] = field(default_factory=list)
    episode_id: int | None = None
    actions_taken: int = 0
    value_extracted: float = 0.0
    cost_incurred: float = 0.0

    # What the actor has obtained so far.
    #
    # These are capabilities rather than graph edges: credentials are held, not
    # connected to anything, and a voice sample is a property of the actor. An
    # action that needs one checks here and fails without it, which is what
    # makes the earlier action that obtains it worth taking.
    credentials: list[float] = field(default_factory=list)
    identities: int = 0
    voice_quality: float = 0.0
    face_quality: float = 0.0
    kyc_passed: bool = False
    controls_number: bool = False
    controls_account: bool = False
    passed_step_up: bool = False
    support_contacts: int = 0
    payees: list[int] = field(default_factory=list)
    laundered: float = 0.0
    launder_hops: int = 0
    # Value already taken at each merchant this episode, for the per-merchant
    # cap. A real card stops working at a merchant long before an attacker has
    # drained it there, and without this the cheapest strategy is to point every
    # authorisation at one merchant and repeat until the action cap.
    value_by_merchant: dict[int, float] = field(default_factory=dict)
    # This episode's offset on the defender's decision thresholds. Drawn once
    # when the episode opens; see EpisodeConfig.threshold_jitter.
    threshold_offset: float = 0.0
    disputes: int = 0
    refunds: int = 0

    @property
    def is_adversarial(self) -> bool:
        return self.kind == ActorKind.ADVERSARIAL


class Simulator:
    """Runs actions against the world and reports what happened."""

    __slots__ = (
        "graph", "clock", "config", "builder", "log", "gate", "resolver",
        "_artifacts", "_scorer", "_hub", "_actors", "_next_episode",
    )

    def __init__(
        self,
        graph: EntityGraph,
        config: SimulationConfig,
        builder: EventBuilder,
        clock: SimClock | None = None,
        artifacts: ArtifactSource | None = None,
        scorer: RiskScorer | None = None,
        hub: RngHub | None = None,
        log: EventLog | None = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.builder = builder
        self.clock = clock or SimClock()
        self.log = log or EventLog()
        self.gate = StageGate()
        # Injected with working defaults, so the world runs before either a
        # content layer or a trained detector exists.
        self._artifacts = artifacts or NullArtifactSource()
        self._scorer = scorer or AlwaysApproveScorer()
        self._hub = hub or RngHub(config.seed)
        self._actors: dict[ActorId, Actor] = {}
        self._next_episode = 0
        self.resolver = ActionResolver(
            graph, self.clock, config, self._hub.stream("resolve")
        )

    # -------------------------------------------------------------- actors

    def register_actor(self, actor: Actor) -> Actor:
        self._actors[actor.actor_id] = actor
        return actor

    def actor(self, actor_id: ActorId) -> Actor:
        return self._actors[actor_id]

    def set_scorer(self, scorer: RiskScorer) -> None:
        """Swap the defender in force.

        Live co-adaptation refits the defender periodically and points the world
        at the new one, so subsequent authorisations are scored and mitigated by
        the model that has seen the most recent fraud. The attacker then faces a
        moving defence within a single run rather than a frozen one.
        """
        self._scorer = scorer

    @property
    def scorer(self) -> RiskScorer:
        return self._scorer

    def open_episode(self, actor_id: ActorId) -> int:
        episode_id = self._next_episode
        self._next_episode += 1
        actor = self._actors[actor_id]
        actor.episode_id = episode_id
        # Fresh caps and a fresh threshold offset for the new episode. The
        # per-merchant tally must not carry over from the last one, and the
        # jitter is drawn once here so it is stable while this actor acts and
        # different the next time anyone tries.
        actor.value_by_merchant = {}
        spread = self.config.engine.episode.threshold_jitter
        actor.threshold_offset = (
            float(self._hub.stream("jitter").uniform(-spread, spread)) if spread else 0.0
        )
        return episode_id

    def close_episode(self, actor_id: ActorId) -> int:
        """Label everything the episode produced.

        Labels are stamped here rather than as events were written, because at
        the moment of scoring nothing knows the answer.
        """
        actor = self._actors[actor_id]
        if actor.episode_id is None:
            return 0
        stamped = self.log.stamp_episode(actor.episode_id, is_fraud=actor.is_adversarial)
        actor.episode_id = None
        return stamped

    # ---------------------------------------------------------------- step

    def step(self, actor_id: ActorId, action: Action) -> Outcome:
        actor = self._actors[actor_id]

        if not self.gate.is_legal(actor.stage, action.name):
            # Nothing happens and nothing is charged. An impossible action is
            # not a failed attempt, and treating it as one would teach a policy
            # that the world pushed back when it never saw the request.
            return Outcome(code=OutcomeCode.ILLEGAL, stage=actor.stage)

        if action.delay_minutes:
            self.clock.advance(action.delay_minutes)

        artifact = None
        if action.needs_artifact:
            # The capability tier the action was taken at. An action may name one
            # in its params; otherwise it is drawn across the range, so the run
            # exercises the whole ladder rather than sitting at tier zero — which
            # would draw every artifact from one corner of the pool and reuse the
            # same handful of texts throughout.
            tier = int(action.params.get("capability_tier", -1))
            if tier < 0:
                tier = int(self._hub.stream("artifact").integers(0, 4))
            artifact = self._artifacts.generate(
                ArtifactRequest(
                    tool_name=action.artifact_tool or "",
                    target_ref=action.target_id,
                    capability_tier=tier,
                )
            )

        cost = action_cost(action.name)
        actor.cost_incurred += cost
        actor.actions_taken += 1

        if action.name is ActionName.ATTEMPT_AUTH:
            outcome = self._resolve_auth(actor, action, cost)
        else:
            outcome = self._resolve_simple(actor, action, cost, artifact)

        actor.stage = outcome.stage
        return outcome

    # ------------------------------------------------------------- resolve

    def _resolve_auth(self, actor: Actor, action: Action, cost: float) -> Outcome:
        card_id = CardId(action.target_id) if action.target_id is not None else None
        merchant_id = (
            MerchantId(action.secondary_id) if action.secondary_id is not None else None
        )
        if card_id is None or merchant_id is None:
            return Outcome(code=OutcomeCode.FAILED, stage=actor.stage, cost=cost)

        devices = self.graph.devices_of_card(card_id)
        # A blocklisted device is refused whatever card it carries, so it drops
        # out of the set an authorisation may run through. This is what makes the
        # blocklist mitigation bite: the capability is not merely flagged, the
        # device can no longer be used at all.
        devices = frozenset(
            d for d in devices if not self.graph.devices[d].blocklisted
        )
        if not devices:
            # No usable binding, so there is nothing to authorise through —
            # either the card never had one, or mitigation removed the last one.
            return Outcome(code=OutcomeCode.FAILED, stage=actor.stage, cost=cost)

        rng = self._hub.stream("resolve")

        # An action may name the device it went through. Where it does not, one
        # of the card's bindings is chosen, weighted towards the newest.
        #
        # Always taking the same member of the set was wrong in a way that hid
        # itself: a third of cards are bound to three or more devices, yet every
        # transaction went through one of them, so the count of devices a card
        # had been seen on never rose above two and the rule keyed on it could
        # not fire. Holders move between a phone, a laptop, and a tablet, and
        # the rule exists to notice when that pattern is unusual.
        if action.device_id is not None and DeviceId(action.device_id) in devices:
            device_id = DeviceId(action.device_id)
        else:
            device_id = self._pick_device(devices, rng)
        amount = action.amount if action.amount is not None else 50.0

        # The per-merchant value cap, one of the anti-reward-hacking controls the
        # design specifies. It was declared in the configuration and enforced
        # nowhere, and the consequence was exactly what the control exists to
        # prevent: with no ceiling on what one merchant would absorb, the policy
        # learned to rotate cards and then hammer authorisations, taking eight to
        # twenty-eight thousand an episode against a stated cap of two thousand.
        # That is a hole in the simulator, not a strategy, and a curve produced
        # against it measures the hole.
        #
        # An attempt over the remaining headroom is refused rather than trimmed
        # to fit. Trimming would silently reward the overreach with whatever was
        # left, which teaches the policy to always ask for more than it can have.
        cap = self.config.engine.episode.max_value_per_merchant
        taken = actor.value_by_merchant.get(int(merchant_id), 0.0)
        if taken + amount > cap:
            return Outcome(code=OutcomeCode.FAILED, stage=actor.stage, cost=cost)

        card = self.graph.cards[card_id]
        holder = self.graph.holders[card.holder_id]
        merchant = self.graph.merchants[merchant_id]
        geo = self._geo_distance(holder, merchant, rng)

        event = self.builder.build_auth(
            ts=self.clock.now,
            card_id=card_id,
            merchant_id=merchant_id,
            device_id=device_id,
            amount=amount,
            entry_mode=action.entry_mode,
            geo_distance_km=geo,
        )
        event.episode_id = actor.episode_id

        assessment = self._scorer.score(event)
        # The episode's threshold offset, applied by shifting the score rather
        # than by rebuilding the bands: the scorer owns its own bands and may not
        # be a banded scorer at all, and moving the score is equivalent to moving
        # every threshold by the same amount in the opposite direction.
        if actor.threshold_offset:
            from .bands import shift_assessment
            assessment = shift_assessment(assessment, actor.threshold_offset, event, self._scorer)
        code = RISK_TO_OUTCOME[assessment.action]

        if code is OutcomeCode.APPROVED and not card.is_usable(self.clock.now):
            code = OutcomeCode.DECLINED
        if (
            code is OutcomeCode.APPROVED
            and rng.random() < self.config.engine.channel.base_decline_rate
        ):
            # Ordinary declines happen for reasons unrelated to risk.
            code = OutcomeCode.DECLINED

        approved = code is OutcomeCode.APPROVED
        self.builder.commit_auth(event, approved=approved)
        self.log.append(event)

        extracted = amount if approved else 0.0
        actor.value_extracted += extracted
        if extracted:
            actor.value_by_merchant[int(merchant_id)] = taken + extracted

        # A score changes nothing on its own; its mitigations are what mutate the
        # world. Applied here, after the event is logged and before the stage is
        # decided, so a deleted binding or a frozen card takes effect for every
        # action the actor attempts next.
        if assessment.mitigations:
            apply_all(assessment.mitigations, self.graph, self.clock.now)

        stage = self.gate.advance(actor.stage, action.name, approved)
        if assessment.action is RiskAction.BLOCK:
            stage = Stage.TERMINAL

        return Outcome(
            code=code,
            stage=stage,
            reward=extracted - cost,
            value_extracted=extracted,
            cost=cost,
            event_id=event.event_id,
        )

    def _resolve_simple(self, actor: Actor, action: Action, cost: float, artifact=None) -> Outcome:
        """Everything that is not an authorisation.

        Delegates to the resolver registered for this action, which performs
        the mutation the action claims or fails. A single method standing in
        for all of them was how nineteen actions came to report success while
        leaving the world untouched.

        The event is emitted only where the action succeeded. Emitting one
        regardless is what let the log say a device had been bound when no such
        edge existed.
        """
        outcome = self.resolver.resolve(actor, action, cost)
        event_id = None

        if outcome.succeeded and actor.holder_id is not None:
            event_type = EVENT_FOR_ACTION.get(action.name)
            if event_type is not None:
                event = self.builder.build_binding(
                    ts=self.clock.now,
                    event_type=event_type,
                    actor_id=int(actor.actor_id),
                    target_id=action.target_id or 0,
                    holder_id=actor.holder_id,
                    device_id=actor.devices[-1] if actor.devices else None,
                )
                event.episode_id = actor.episode_id
                # Where the action presented rendered text to a control, its
                # embedding and scores ride on the event, so the text expert sees
                # what the control saw rather than only that a ticket was opened.
                if artifact is not None and artifact.embedding:
                    event.text_embedding = tuple(artifact.embedding)
                    if artifact.scores:
                        event.text_score_names = tuple(artifact.scores.keys())
                        event.text_scores = tuple(artifact.scores.values())
                self.builder.commit_binding(event)
                self.log.append(event)
                event_id = event.event_id

        actor.value_extracted += outcome.value_extracted
        stage = self.gate.advance(actor.stage, action.name, outcome.succeeded)
        return Outcome(
            code=outcome.code,
            stage=stage,
            reward=outcome.reward,
            value_extracted=outcome.value_extracted,
            cost=cost,
            event_id=event_id,
        )

    def _pick_device(self, devices, rng: np.random.Generator) -> DeviceId:
        """Choose among a card's bindings, favouring the most recent.

        A replaced handset should carry most of the traffic while the older
        devices keep appearing occasionally, which is what a household with a
        phone and a laptop actually looks like.
        """
        ordered = sorted(
            devices, key=lambda d: self.graph.devices[d].first_seen_ts, reverse=True
        )
        if len(ordered) == 1:
            return ordered[0]
        weights = np.array([0.6**index for index in range(len(ordered))], dtype=float)
        weights /= weights.sum()
        return ordered[int(rng.choice(len(ordered), p=weights))]

    def _geo_distance(self, holder, merchant, rng: np.random.Generator) -> float:
        geo = self.config.population.geo
        if rng.random() < geo.travel_share:
            return float(rng.exponential(geo.travel_distance_km))
        return float(rng.exponential(geo.home_radius_km))

    # -------------------------------------------------------------- status

    def summary(self) -> dict[str, int]:
        return {
            "actors": len(self._actors),
            "events": len(self.log),
            "labelled": len(self.log.labelled()),
            "clock": self.clock.now,
        }

