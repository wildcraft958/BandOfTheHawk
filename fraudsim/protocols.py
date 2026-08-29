"""Plug-in seams.

Everything the simulator delegates to is declared here as a Protocol, with a
null implementation so the system runs end-to-end before the real components
exist. The simulator imports these types only; it never imports a concrete
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .ids import ActorId


class RiskAction(Enum):
    APPROVE = "approve"
    STEP_UP = "step_up"
    HOLD = "hold"
    DECLINE = "decline"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """A scorer's judgement on a single event."""

    risk_score: float
    action: RiskAction = RiskAction.APPROVE
    expert_weights: tuple[float, ...] = ()
    mitigations: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactRequest:
    """A request for rendered content backing an action."""

    tool_name: str
    target_ref: int | None = None
    capability_tier: int = 0
    persona_hint: str = ""


@dataclass(frozen=True, slots=True)
class Artifact:
    """Rendered content plus the scores a control will check it against."""

    scores: Mapping[str, float] = field(default_factory=dict)
    content: str | None = None
    embedding: tuple[float, ...] = ()

    def score(self, name: str, default: float = 0.0) -> float:
        return self.scores.get(name, default)


@dataclass(frozen=True, slots=True)
class ActorObservation:
    """What an acting policy is allowed to see. Never the graph itself."""

    actor_id: ActorId
    stage: int
    legal_action_mask: Sequence[bool]
    features: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class ActorPolicy(Protocol):
    """Chooses the next action for an actor."""

    def act(self, obs: ActorObservation) -> Any: ...

    def observe(self, outcome: Any) -> None: ...


@runtime_checkable
class ArtifactSource(Protocol):
    """Renders artifacts on demand. Called after an action is chosen."""

    def generate(self, request: ArtifactRequest) -> Artifact: ...


@runtime_checkable
class RiskScorer(Protocol):
    """Scores an event with no access to ground truth."""

    def score(self, event: Any) -> RiskAssessment: ...


@runtime_checkable
class ArrivalProcess(Protocol):
    """Draws the gap in minutes until an entity's next event."""

    def next_gap(self, history: Sequence[int], now: int) -> int: ...

    def reset(self) -> None: ...


class NullArtifactSource:
    """Returns an empty artifact, so artifact-gated actions resolve on graph facts."""

    __slots__ = ()

    def generate(self, request: ArtifactRequest) -> Artifact:
        return Artifact()


class AlwaysApproveScorer:
    """Baseline scorer used during warm start and before a real one exists."""

    __slots__ = ()

    def score(self, event: Any) -> RiskAssessment:
        return RiskAssessment(risk_score=0.0, action=RiskAction.APPROVE)
