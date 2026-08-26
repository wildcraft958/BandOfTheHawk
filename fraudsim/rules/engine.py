"""Rule evaluation.

Two things live here. The engine reports which rules an event trips, which is
how the share of ordinary traffic that looks suspicious gets measured. And a
thin adapter presents the same engine as a scorer, so the simulator has a
working defender from the start rather than waiting for a trained one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.engine import VelocityRuleConfig
from ..features.schema import AuthAttemptEvent
from ..protocols import RiskAction, RiskAssessment
from .definitions import VelocityRule, build_rules


@dataclass(frozen=True, slots=True)
class RuleReport:
    """Which rules fired on one event."""

    triggered: tuple[str, ...]

    @property
    def any_triggered(self) -> bool:
        return bool(self.triggered)

    @property
    def n_triggered(self) -> int:
        return len(self.triggered)

    def __contains__(self, rule_id: str) -> bool:
        return rule_id in self.triggered


@dataclass
class TriggerRates:
    """How often each rule fires across a set of events."""

    n_events: int
    per_rule: dict[str, float] = field(default_factory=dict)
    any_rule: float = 0.0

    def render(self, target: float | None = None, tolerance: float = 0.02) -> str:
        lines = [f"rule trigger rates over {self.n_events:,} events", ""]
        for rule_id in sorted(self.per_rule):
            lines.append(f"  {rule_id:<6}{self.per_rule[rule_id]:>9.4f}")
        lines.append(f"  {'any':<6}{self.any_rule:>9.4f}")
        if target is not None:
            verdict = "within target" if abs(self.any_rule - target) < 0.02 else "off target"
            lines.append(f"\n  target {target:.3f}, {verdict}")
        return "\n".join(lines)


class VelocityRuleEngine:
    """Evaluates the rule set against events."""

    __slots__ = ("_rules",)

    def __init__(self, config: VelocityRuleConfig | None = None) -> None:
        self._rules = build_rules(config or VelocityRuleConfig())

    @property
    def rules(self) -> tuple[VelocityRule, ...]:
        return self._rules

    def evaluate(self, event: AuthAttemptEvent) -> RuleReport:
        return RuleReport(
            triggered=tuple(rule.rule_id for rule in self._rules if rule.triggers(event))
        )

    def trigger_rates(self, events: list[AuthAttemptEvent]) -> TriggerRates:
        """Share of events tripping each rule, and any rule at all.

        The combined figure is the one that matters: it is the share of
        ordinary traffic a naive engine would flag, which is what makes a
        false-positive rate mean something.
        """
        counts = {rule.rule_id: 0 for rule in self._rules}
        any_count = 0
        for event in events:
            report = self.evaluate(event)
            for rule_id in report.triggered:
                counts[rule_id] += 1
            if report.any_triggered:
                any_count += 1

        total = max(len(events), 1)
        return TriggerRates(
            n_events=len(events),
            per_rule={rule_id: count / total for rule_id, count in counts.items()},
            any_rule=any_count / total,
        )

    def describe(self) -> str:
        return "\n".join(f"  {r.rule_id}  {r.description}" for r in self._rules)


class VelocityRuleScorer:
    """The rule engine presented as a scorer.

    Gives the simulator a defender from the first run, and doubles as the
    baseline a learned model has to improve on. Risk rises with how many rules
    fired rather than which, since a naive engine has no basis for weighting
    one above another.
    """

    __slots__ = ("_engine", "_step_up_at", "_decline_at")

    def __init__(
        self,
        config: VelocityRuleConfig | None = None,
        step_up_at: int = 1,
        decline_at: int = 3,
    ) -> None:
        self._engine = VelocityRuleEngine(config)
        self._step_up_at = step_up_at
        self._decline_at = decline_at

    def score(self, event: AuthAttemptEvent) -> RiskAssessment:
        report = self._engine.evaluate(event)
        fired = report.n_triggered
        risk = min(1.0, fired / max(len(self._engine.rules), 1))

        if fired >= self._decline_at:
            action = RiskAction.DECLINE
        elif fired >= self._step_up_at:
            action = RiskAction.STEP_UP
        else:
            action = RiskAction.APPROVE

        return RiskAssessment(risk_score=risk, action=action)
