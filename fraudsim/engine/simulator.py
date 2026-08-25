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
from ..config.simulation import SimulationConfig
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

    @property
    def is_adversarial(self) -> bool:
        return self.kind == ActorKind.ADVERSARIAL


class Simulator:
    """Runs actions against the world and reports what happened."""

    __slots__ = (
        "graph", "clock", "config", "builder", "log", "gate",
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

    # -------------------------------------------------------------- actors

    def register_actor(self, actor: Actor) -> Actor:
        self._actors[actor.actor_id] = actor
        return actor

    def actor(self, actor_id: ActorId) -> Actor:
        return self._actors[actor_id]

    def open_episode(self, actor_id: ActorId) -> int:
        episode_id = self._next_episode
        self._next_episode += 1
        self._actors[actor_id].episode_id = episode_id
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
            artifact = self._artifacts.generate(
                ArtifactRequest(
                    tool_name=action.artifact_tool or "",
                    target_ref=action.target_id,
                )
            )

        cost = action_cost(action.name)
        actor.cost_incurred += cost
        actor.actions_taken += 1

        if action.name is ActionName.ATTEMPT_AUTH:
            outcome = self._resolve_auth(actor, action, cost)
        else:
            outcome = self._resolve_simple(actor, action, cost)

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
        if not devices:
            # No binding, so there is nothing to authorise through. This is the
            # structural constraint the stage machine exists to express.
            return Outcome(code=OutcomeCode.FAILED, stage=actor.stage, cost=cost)
        device_id = next(iter(devices))

        rng = self._hub.stream("resolve")
        amount = action.amount if action.amount is not None else 50.0

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

    def _resolve_simple(self, actor: Actor, action: Action, cost: float) -> Outcome:
        """Actions whose effect is on capability rather than money."""
        succeeded = True
        event_id = None

        if action.name in _BINDING_EVENTS and actor.holder_id is not None:
            event = self.builder.build_binding(
                ts=self.clock.now,
                event_type=_BINDING_EVENTS[action.name],
                actor_id=int(actor.actor_id),
                target_id=action.target_id or 0,
                holder_id=actor.holder_id,
            )
            event.episode_id = actor.episode_id
            self.builder.commit_binding(event)
            self.log.append(event)
            event_id = event.event_id

        stage = self.gate.advance(actor.stage, action.name, succeeded)
        return Outcome(
            code=OutcomeCode.APPROVED if succeeded else OutcomeCode.FAILED,
            stage=stage,
            reward=-cost,
            cost=cost,
            event_id=event_id,
        )

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


_BINDING_EVENTS: dict[ActionName, EventType] = {
    ActionName.ADD_DEVICE_SELFSERVE: EventType.DEVICE_BIND,
    ActionName.RESET_PASSWORD: EventType.AUTH_RESET,
    ActionName.OPEN_TICKET: EventType.SUPPORT_TICKET,
    ActionName.ADD_PAYEE: EventType.PAYEE_ADD,
    ActionName.CALL_IVR_PROVISION: EventType.IVR_CALL,
    ActionName.SUBMIT_KYC: EventType.KYC_SUBMIT,
    ActionName.SIM_SWAP: EventType.SIM_CHANGE,
    ActionName.ESCALATE_LIMIT: EventType.LIMIT_CHANGE,
}
