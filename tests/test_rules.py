"""The rule set is both a baseline and an instrument.

As a baseline it is what a learned detector has to beat. As an instrument it
measures how much ordinary traffic looks suspicious, which is what makes a
false-positive rate mean anything.
"""

from __future__ import annotations

import pytest

from fraudsim.config.engine import VelocityRuleConfig
from fraudsim.features.schema import AuthAttemptEvent
from fraudsim.protocols import RiskAction, RiskScorer
from fraudsim.rules.engine import VelocityRuleEngine, VelocityRuleScorer


def event(**overrides) -> AuthAttemptEvent:
    """A quiet authorisation that trips nothing."""
    base = dict(
        event_id=0, ts=0, card_id=1, merchant_id=1, device_id=1,
        amount=50.0, category_cluster=0, entry_mode=0, merchant_risk_tier=0,
        is_high_liquidity=False,
        device_age_days=200, device_new_to_card=False, device_n_cards=2,
        card_n_devices=1, ip_asn=10, geo_distance_km=5.0,
        auths_last_60s=0, auths_last_1h=0, auths_last_24h=1,
        distinct_categories_1h=0, distinct_merchants_24h=1, distinct_ips_24h=1,
        amount_sum_24h=50.0, declines_last_1h=0, seconds_since_last_auth=3600,
        is_first_txn_this_merchant=False, hour_of_day=14, is_weekend=False,
        within_usual_hours=True, amount_vs_median=1.0,
        account_age_days=900, holder_tenure_days=900,
    )
    base.update(overrides)
    return AuthAttemptEvent(**base)


@pytest.fixture
def engine() -> VelocityRuleEngine:
    return VelocityRuleEngine(VelocityRuleConfig())


def test_a_quiet_event_trips_nothing(engine) -> None:
    assert engine.evaluate(event()).triggered == ()


def test_all_eight_rules_are_present(engine) -> None:
    """The benchmark this set comes from could evaluate only six, for want of
    the columns behind the other two. The event schema carries both."""
    assert tuple(rule.rule_id for rule in engine.rules) == (
        "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"
    )


def _just_over(threshold):
    """A value that clears a configured threshold.

    Derived rather than restated, since the thresholds are tuned against
    generated traffic and a hardcoded copy here would break silently whenever
    one moved.
    """
    return threshold + (1 if isinstance(threshold, int) else 1.0)


@pytest.mark.parametrize(
    "rule_id,field,threshold_name,extra",
    [
        ("R1", "auths_last_1h", "txn_count_1h", {}),
        ("R2", "distinct_merchants_24h", "distinct_merchants_24h", {}),
        ("R3", "amount_sum_24h", "amount_sum_24h", {}),
        ("R4", "auths_last_24h", "new_account_txn_count", {"account_age_days": 2}),
        ("R5", "card_n_devices", "distinct_payment_methods_7d", {}),
        ("R6", "amount_vs_median", "amount_ratio_30d", {}),
        ("R7", "declines_last_1h", "declines_1h", {}),
        ("R8", "distinct_ips_24h", "distinct_ips_24h", {}),
    ],
)
def test_each_rule_fires_on_its_own_condition(
    engine, rule_id, field, threshold_name, extra
) -> None:
    threshold = getattr(VelocityRuleConfig(), threshold_name)
    overrides = {field: _just_over(threshold), **extra}
    assert rule_id in engine.evaluate(event(**overrides))


def test_r5_counts_devices_per_card_not_cards_per_device(engine) -> None:
    """Cards per device is the shared fingerprint fan-out, heavy tailed among
    ordinary holders by design. A rule keyed on it fires on half of legitimate
    traffic and measures the generator rather than the behaviour."""
    threshold = VelocityRuleConfig().distinct_payment_methods_7d
    assert "R5" not in engine.evaluate(event(device_n_cards=400, card_n_devices=1))
    assert "R5" in engine.evaluate(
        event(device_n_cards=1, card_n_devices=threshold + 5)
    )


def test_r4_needs_both_a_new_account_and_activity(engine) -> None:
    threshold = VelocityRuleConfig().new_account_txn_count
    assert "R4" not in engine.evaluate(event(account_age_days=2, auths_last_24h=1))
    assert "R4" not in engine.evaluate(
        event(account_age_days=900, auths_last_24h=threshold + 6)
    )
    assert "R4" in engine.evaluate(
        event(account_age_days=2, auths_last_24h=threshold + 6)
    )


def test_r6_stays_quiet_without_history(engine) -> None:
    """A card with no median has nothing to be a multiple of, and treating
    absence as a ratio of zero would fire on every first transaction."""
    assert "R6" not in engine.evaluate(event(amount_vs_median=None))


def test_thresholds_come_from_configuration(engine) -> None:
    strict = VelocityRuleEngine(VelocityRuleConfig(txn_count_1h=1))
    sample = event(auths_last_1h=2)
    assert "R1" in strict.evaluate(sample)
    assert "R1" not in engine.evaluate(sample)


def test_trigger_rates_report_each_rule_and_the_whole_set() -> None:
    engine = VelocityRuleEngine(VelocityRuleConfig())
    events = [event() for _ in range(90)] + [event(auths_last_1h=99) for _ in range(10)]
    rates = engine.trigger_rates(events)
    assert rates.n_events == 100
    assert rates.per_rule["R1"] == pytest.approx(0.10)
    assert rates.any_rule == pytest.approx(0.10)


def test_trigger_rates_count_an_event_once_however_many_rules_fire() -> None:
    engine = VelocityRuleEngine(VelocityRuleConfig())
    noisy = event(auths_last_1h=99, distinct_merchants_24h=99, amount_sum_24h=9000.0)
    rates = engine.trigger_rates([noisy] * 10)
    assert rates.any_rule == pytest.approx(1.0)
    assert rates.per_rule["R1"] == pytest.approx(1.0)


def test_scorer_satisfies_the_scorer_protocol() -> None:
    assert isinstance(VelocityRuleScorer(), RiskScorer)


def test_scorer_escalates_with_the_number_of_rules() -> None:
    scorer = VelocityRuleScorer(VelocityRuleConfig())
    assert scorer.score(event()).action is RiskAction.APPROVE
    assert scorer.score(event(auths_last_1h=99)).action is RiskAction.STEP_UP

    many = event(auths_last_1h=99, distinct_merchants_24h=99, amount_sum_24h=9000.0)
    assessment = scorer.score(many)
    assert assessment.action is RiskAction.DECLINE
    assert assessment.risk_score > 0.3


def test_scorer_risk_stays_within_range() -> None:
    scorer = VelocityRuleScorer(VelocityRuleConfig())
    everything = event(
        auths_last_1h=99, distinct_merchants_24h=99, amount_sum_24h=99_999.0,
        account_age_days=1, auths_last_24h=99, card_n_devices=99,
        amount_vs_median=99.0, declines_last_1h=99, distinct_ips_24h=99,
    )
    assert 0.0 <= scorer.score(everything).risk_score <= 1.0
    assert scorer.score(event()).risk_score == pytest.approx(0.0)
