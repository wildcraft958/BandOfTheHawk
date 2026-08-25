"""What an action produced."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..protocols import RiskAction
from .stages import Stage


class OutcomeCode(Enum):
    APPROVED = "approved"
    DECLINED = "declined"
    STEPPED_UP = "stepped_up"
    HELD = "held"
    BLOCKED = "blocked"
    ILLEGAL = "illegal"
    FAILED = "failed"

    @property
    def succeeded(self) -> bool:
        return self is OutcomeCode.APPROVED


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of one step.

    Carries what the actor is entitled to know: whether it worked, what it
    cost, and where that leaves it. Not the risk score, and not whether the
    world considers it fraudulent.
    """

    code: OutcomeCode
    stage: Stage
    reward: float = 0.0
    value_extracted: float = 0.0
    cost: float = 0.0
    event_id: int | None = None

    @property
    def succeeded(self) -> bool:
        return self.code.succeeded


RISK_TO_OUTCOME: dict[RiskAction, OutcomeCode] = {
    RiskAction.APPROVE: OutcomeCode.APPROVED,
    RiskAction.STEP_UP: OutcomeCode.STEPPED_UP,
    RiskAction.HOLD: OutcomeCode.HELD,
    RiskAction.DECLINE: OutcomeCode.DECLINED,
    RiskAction.BLOCK: OutcomeCode.BLOCKED,
}
