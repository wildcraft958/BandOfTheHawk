"""Canonical velocity rules.

Eight rules drawn from published fraud-detection practice. They serve two
purposes here: a naive baseline any learned detector has to beat, and the
instrument that measures how much ordinary traffic looks suspicious.

That second use is why they exist at all. A false-positive rate means nothing
unless some legitimate behaviour actually trips something, and asserting that a
few per cent of ordinary traffic looks odd is weaker than measuring it.

The rule identifiers are the ones used in the literature and are kept as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..settings.engine import VelocityRuleConfig
from ..features.schema import AuthAttemptEvent


@dataclass(frozen=True, slots=True)
class VelocityRule:
    """One rule: a field, a comparison, and a threshold."""

    rule_id: str
    description: str
    predicate: Callable[[AuthAttemptEvent], bool]

    def triggers(self, event: AuthAttemptEvent) -> bool:
        return bool(self.predicate(event))


def build_rules(config: VelocityRuleConfig) -> tuple[VelocityRule, ...]:
    """The eight rules, bound to their configured thresholds.

    All eight are computable here. The published benchmark this set comes from
    could only evaluate six, because its feature subset lacked the columns for
    failed transactions and distinct addresses; the event schema carries both.
    """
    return (
        VelocityRule(
            rule_id="R1",
            description=f"more than {config.txn_count_1h} transactions in an hour",
            predicate=lambda e: e.auths_last_1h > config.txn_count_1h,
        ),
        VelocityRule(
            rule_id="R2",
            description=f"more than {config.distinct_merchants_24h} merchants in a day",
            predicate=lambda e: e.distinct_merchants_24h > config.distinct_merchants_24h,
        ),
        VelocityRule(
            rule_id="R3",
            description=f"more than {config.amount_sum_24h:.0f} spent in a day",
            predicate=lambda e: e.amount_sum_24h > config.amount_sum_24h,
        ),
        VelocityRule(
            rule_id="R4",
            description=(
                f"more than {config.new_account_txn_count} transactions on an account "
                f"younger than {config.new_account_days} days"
            ),
            predicate=lambda e: (
                e.account_age_days < config.new_account_days
                and e.auths_last_24h > config.new_account_txn_count
            ),
        ),
        VelocityRule(
            rule_id="R5",
            description=(
                f"card seen on more than {config.distinct_payment_methods_7d} devices "
                "in a week"
            ),
            # Devices per card, not cards per device. The latter is the shared
            # fingerprint fan-out, which is heavy tailed among ordinary holders
            # by design, so a rule keyed on it fires on half of all legitimate
            # traffic and measures the generator rather than the behaviour.
            predicate=lambda e: e.card_n_devices > config.distinct_payment_methods_7d,
        ),
        VelocityRule(
            rule_id="R6",
            description=f"amount more than {config.amount_ratio_30d:.1f}x the usual",
            predicate=lambda e: (
                e.amount_vs_median is not None
                and e.amount_vs_median > config.amount_ratio_30d
            ),
        ),
        VelocityRule(
            rule_id="R7",
            description=f"more than {config.declines_1h} declines in an hour",
            predicate=lambda e: e.declines_last_1h > config.declines_1h,
        ),
        VelocityRule(
            rule_id="R8",
            description=f"more than {config.distinct_ips_24h} networks in a day",
            predicate=lambda e: e.distinct_ips_24h > config.distinct_ips_24h,
        ),
    )
