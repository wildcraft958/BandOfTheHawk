"""Risk bands: turning a score into an action and a graph mutation.

A classifier emits a number; deciding what to do with it is a separate, and
cheaper, problem. The band boundaries are free parameters, but leaving them at
round numbers wastes a real result — grid-searched against a business cost curve
(fraud loss plus friction plus review cost) they become a defended operating
point rather than an arbitrary one, which turns a FREE parameter into a CITED
one and lets the claim be "we optimise the cost curve, not just AUC".

Each band maps to a mitigation whose severity matches the confidence: a step-up
where the score is merely elevated, a frozen card higher, a deleted binding or a
blocklisted device where it is near-certain. The mutation is chosen from the
event, so the mitigation acts on the capability the event used.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..features.schema import AuthAttemptEvent
from ..protocols import RiskAction, RiskAssessment
from .mitigation import BlocklistDevice, FreezeCard, UnbindDevice


@dataclass(frozen=True, slots=True)
class RiskBands:
    """Score thresholds and the action each band takes.

    The defaults are round numbers, a starting point the cost-curve search
    refines. `decide` returns the action and any mitigations, reading the event
    to target the mutation.
    """

    step_up_at: float = 0.30
    hold_at: float = 0.60
    decline_at: float = 0.80
    block_at: float = 0.95

    def action_for(self, score: float) -> RiskAction:
        if score >= self.block_at:
            return RiskAction.BLOCK
        if score >= self.decline_at:
            return RiskAction.DECLINE
        if score >= self.hold_at:
            return RiskAction.HOLD
        if score >= self.step_up_at:
            return RiskAction.STEP_UP
        return RiskAction.APPROVE

    def decide(self, score: float, event) -> tuple[RiskAction, tuple]:
        """The action and the mitigations it carries.

        Mitigations escalate with the band. A hold freezes the card; a block
        deletes the binding the event ran through and blocklists the device, so
        the capability is gone rather than paused. Only authorisation events
        carry a device to act on; others take the card-level mitigation.
        """
        action = self.action_for(score)
        mitigations: list = []

        card_id = getattr(event, "card_id", None)
        device_id = getattr(event, "device_id", None)

        if action is RiskAction.HOLD and card_id is not None:
            mitigations.append(FreezeCard(card_id=int(card_id), hours=24))
        elif action is RiskAction.DECLINE and card_id is not None:
            mitigations.append(FreezeCard(card_id=int(card_id), hours=72))
        elif action is RiskAction.BLOCK:
            if card_id is not None and device_id is not None:
                mitigations.append(
                    UnbindDevice(card_id=int(card_id), device_id=int(device_id))
                )
            if device_id is not None:
                mitigations.append(BlocklistDevice(device_id=int(device_id)))
        return action, tuple(mitigations)


@dataclass(frozen=True, slots=True)
class CostModel:
    """The business cost the bands are searched against.

    A missed fraud costs its value. A friction event — a step-up or a hold on a
    genuine customer — costs a fixed amount, the annoyance and abandonment it
    causes. A review costs the analyst's time. The search minimises the sum, so
    the bands land where the three trade off rather than at round numbers.
    """

    friction_cost: float = 5.0
    review_cost: float = 8.0

    def evaluate(self, y_true, scores, bands: RiskBands) -> float:
        """Total cost of operating these bands on this scored set.

        Fraud above the decline threshold is stopped; below it, its value is
        lost. Genuine traffic above the step-up threshold pays friction; above
        the hold threshold it also pays a review. Amounts are unit where the
        event value is not to hand, so this is a shape to minimise, not a dollar
        figure — stated as such.
        """
        import numpy as np

        y = np.asarray(y_true)
        s = np.asarray(scores)
        stopped = s >= bands.decline_at
        fraud_loss = float(((y == 1) & ~stopped).sum())
        friction = float(((y == 0) & (s >= bands.step_up_at)).sum()) * self.friction_cost / 100
        review = float(((y == 0) & (s >= bands.hold_at)).sum()) * self.review_cost / 100
        return fraud_loss + friction + review


def shift_assessment(assessment: RiskAssessment, offset: float, event, scorer) -> RiskAssessment:
    """Re-decide an assessment with the episode's threshold offset applied.

    The offset moves the score rather than the thresholds. The two are
    equivalent, and moving the score works for any scorer, including one
    carrying no bands at all.

    The decision is rebuilt from scratch at the shifted score, mitigations
    included. The reported risk_score stays the model's own, unshifted.
    """
    bands = getattr(scorer, "bands", None) or RiskBands()
    action, mitigations = bands.decide(assessment.risk_score - offset, event)
    return RiskAssessment(
        risk_score=assessment.risk_score,
        action=action,
        mitigations=tuple(mitigations),
    )


def grid_search_bands(y_true, scores, cost: CostModel | None = None, steps: int = 9) -> RiskBands:
    """Search band boundaries against the cost curve.

    A coarse grid over ordered thresholds, keeping the cheapest. Coarse on
    purpose: the point is a defended operating point, not a hand-tuned one, and a
    fine grid would invite overfitting the evaluation set.
    """
    import numpy as np

    cost = cost or CostModel()
    grid = np.linspace(0.1, 0.95, steps)
    best = RiskBands()
    best_cost = cost.evaluate(y_true, scores, best)
    for step_up in grid:
        for hold in grid[grid > step_up]:
            for decline in grid[grid > hold]:
                bands = RiskBands(
                    step_up_at=float(step_up),
                    hold_at=float(hold),
                    decline_at=float(decline),
                    block_at=float(min(0.95, decline + 0.1)),
                )
                c = cost.evaluate(y_true, scores, bands)
                if c < best_cost:
                    best_cost, best = c, bands
    return best
