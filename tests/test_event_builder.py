"""An event must describe the state before itself, and must not say who acted."""

from __future__ import annotations

import pytest

from fraudsim.features.builder import EventBuilder, haversine_km
from fraudsim.features.schema import EventLog, EventType
from fraudsim.features.state import FeatureStateStore
from fraudsim.population.builder import PopulationBuilder
from fraudsim.settings.engine import WindowConfig
from fraudsim.settings.simulation import SimulationConfig
from fraudsim.timing.circadian import HolderClock, HolderClockModel

MINUTE = 1
HOUR_MINUTES = 60
DAY_MINUTES = 1440


@pytest.fixture
def builder():
    """A fresh world per test.

    Committing an event writes to the graph as well as to the rolling state, so
    a shared graph would carry transactions between tests and quietly change
    what a later one observes.
    """
    config = SimulationConfig.model_validate({"population": {"n_holders": 400}})
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    clocks = HolderClockModel(config.behavior.circadian)
    return EventBuilder(graph, states, config.engine.windows, clocks), graph


def first_binding(graph):
    card_id, device_id = next(iter(graph.provisioned))
    merchant_id = next(iter(graph.merchants))
    return card_id, device_id, merchant_id


def build(builder_pair, ts, amount=50.0):
    builder, graph = builder_pair
    card_id, device_id, merchant_id = first_binding(graph)
    return builder.build_auth(
        ts=ts, card_id=card_id, merchant_id=merchant_id, device_id=device_id,
        amount=amount, entry_mode=0, geo_distance_km=5.0,
    )


def test_an_event_does_not_count_itself(builder) -> None:
    """The gate. Collapsing build and commit leaves every count off by one
    while the numbers stay entirely plausible."""
    event = build(builder, ts=0)
    assert event.auths_last_1h == 0
    builder[0].commit_auth(event)

    second = build(builder, ts=10)
    assert second.auths_last_1h == 1
    builder[0].commit_auth(second)

    third = build(builder, ts=20)
    assert third.auths_last_1h == 2


def test_amount_sum_excludes_the_current_amount(builder) -> None:
    first = build(builder, ts=0, amount=100.0)
    builder[0].commit_auth(first)
    second = build(builder, ts=60, amount=250.0)
    assert second.amount_sum_24h == pytest.approx(100.0)


def test_distinct_counts_exclude_the_current_event(builder) -> None:
    event = build(builder, ts=0)
    assert event.distinct_merchants_24h == 0
    builder[0].commit_auth(event)
    assert build(builder, ts=30).distinct_merchants_24h == 1


def test_first_transaction_flag_flips_after_commit(builder) -> None:
    event = build(builder, ts=0)
    assert event.is_first_txn_this_merchant
    builder[0].commit_auth(event)
    assert not build(builder, ts=60).is_first_txn_this_merchant


def test_median_ratio_is_absent_until_history_exists(builder) -> None:
    """Reporting zero would be a claim; absence is the honest answer."""
    event = build(builder, ts=0)
    assert event.amount_vs_median is None
    for index in range(8):
        builder[0].commit_auth(build(builder, ts=index * 10, amount=50.0))
    assert build(builder, ts=200, amount=150.0).amount_vs_median == pytest.approx(3.0)


def test_time_since_last_auth_is_absent_at_first(builder) -> None:
    event = build(builder, ts=0)
    assert event.seconds_since_last_auth is None
    builder[0].commit_auth(event)
    assert build(builder, ts=5).seconds_since_last_auth == 300


def test_declines_are_counted_only_when_they_happen(builder) -> None:
    approved = build(builder, ts=0)
    builder[0].commit_auth(approved, approved=True)
    declined = build(builder, ts=10)
    builder[0].commit_auth(declined, approved=False)
    assert build(builder, ts=20).declines_last_1h == 1


def test_scoring_fields_exclude_the_label(builder) -> None:
    """Excluded structurally rather than by convention, so a scorer cannot read
    the answer even by accident."""
    event = build(builder, ts=0)
    event.is_fraud = True
    event.episode_id = 7
    fields = event.scoring_fields()
    assert "is_fraud" not in fields
    assert "episode_id" not in fields
    assert "amount" in fields


def test_compound_features_are_flattened_by_name(builder) -> None:
    event = build(builder, ts=0)
    fields = event.scoring_fields()
    assert len(event.compound_features) == WindowConfig().n_compound_features
    assert "category_cluster_count_3600s" in fields


def test_events_carry_no_actor_identity(builder) -> None:
    """The same builder serves an ordinary holder and an attacker, so no field
    may say which produced the row."""
    fields = build(builder, ts=0).scoring_fields()
    forbidden = {"actor", "actor_id", "is_attacker", "intent", "vertical"}
    assert not (forbidden & set(fields))


def test_warm_start_events_are_flagged(builder) -> None:
    """These are feature-poorer by construction, since the history they would
    read is what they are creating."""
    builder[0].set_warm_start(True)
    assert build(builder, ts=0).is_warm_start
    builder[0].set_warm_start(False)
    assert not build(builder, ts=10).is_warm_start


def test_within_usual_hours_reads_this_holder_not_the_population(builder) -> None:
    """The feature has to be about the acting holder.

    A population-level interval is the same test for everyone, so it says
    nothing about whether this holder is behaving unusually. Two holders with
    opposite habits must disagree about the same hour.
    """
    event_builder, graph = builder
    card_id, _, _ = first_binding(graph)
    holder_id = int(graph.cards[card_id].holder_id)

    # A holder who shops late in the evening.
    event_builder.clocks._clocks[holder_id] = HolderClock(preferred_hour=23.0, kappa=4.0)
    assert build(builder, ts=23 * HOUR_MINUTES).within_usual_hours is True
    assert build(builder, ts=11 * HOUR_MINUTES).within_usual_hours is False

    # The same hours, for a holder who shops in the morning.
    event_builder.clocks._clocks[holder_id] = HolderClock(preferred_hour=11.0, kappa=4.0)
    assert build(builder, ts=23 * HOUR_MINUTES).within_usual_hours is False
    assert build(builder, ts=11 * HOUR_MINUTES).within_usual_hours is True


def test_within_usual_hours_is_absent_for_an_unregistered_holder(builder) -> None:
    """No habit on record is an absence, not a verdict."""
    assert build(builder, ts=3 * HOUR_MINUTES).within_usual_hours is None


def test_hour_and_weekend_derive_from_the_clock(builder) -> None:
    event = build(builder, ts=5 * DAY_MINUTES + 14 * HOUR_MINUTES)
    assert event.hour_of_day == 14
    assert event.is_weekend


def test_device_fanout_is_reported(builder) -> None:
    _, graph = builder
    event = build(builder, ts=0)
    assert event.device_n_cards == graph.device_card_count(event.device_id)


def test_binding_events_report_the_recovery_chain(builder) -> None:
    """Each step is unremarkable alone; the sequence is what carries signal."""
    event_builder, graph = builder
    holder_id = next(iter(graph.holders))
    now = 10 * HOUR_MINUTES

    reset = event_builder.build_binding(
        ts=now, event_type=EventType.AUTH_RESET, actor_id=1, target_id=1,
        holder_id=holder_id,
    )
    assert not reset.recovery_chain_within_1h
    event_builder.commit_binding(reset)

    call = event_builder.build_binding(
        ts=now + 5, event_type=EventType.SUPPORT_TICKET, actor_id=1, target_id=1,
        holder_id=holder_id,
    )
    event_builder.commit_binding(call)

    bind = event_builder.build_binding(
        ts=now + 10, event_type=EventType.DEVICE_BIND, actor_id=1, target_id=1,
        holder_id=holder_id,
    )
    event_builder.commit_binding(bind)

    after = event_builder.build_binding(
        ts=now + 15, event_type=EventType.DEVICE_BIND, actor_id=1, target_id=2,
        holder_id=holder_id,
    )
    assert after.recovery_chain_within_1h


def test_event_log_stamps_labels_after_the_fact(builder) -> None:
    """Nothing knows the answer at scoring time, so labels arrive later."""
    log = EventLog()
    for index in range(3):
        event = build(builder, ts=index * 10)
        event.episode_id = 42
        builder[0].commit_auth(event)
        log.append(event)

    assert log.labelled() == []
    assert log.stamp_episode(42, is_fraud=True) == 3
    assert len(log.labelled()) == 3


def test_event_log_separates_warm_start_rows(builder) -> None:
    log = EventLog()
    builder[0].set_warm_start(True)
    log.append(build(builder, ts=0))
    builder[0].set_warm_start(False)
    log.append(build(builder, ts=10))
    assert len(log) == 2
    assert len(log.scoreable()) == 1


def test_haversine_is_symmetric_and_zero_at_a_point() -> None:
    assert haversine_km(40.0, -74.0, 40.0, -74.0) == pytest.approx(0.0)
    there = haversine_km(40.7, -74.0, 34.0, -118.2)
    back = haversine_km(34.0, -118.2, 40.7, -74.0)
    assert there == pytest.approx(back)
    assert 3900 < there < 4000


def test_no_leak_holds_across_many_events(builder) -> None:
    """The gate at scale, against a history kept independently of the windows.

    Timestamps advance monotonically here because the simulator drives a single
    clock; the structures rely on that, and refuse events that break it.
    """
    import numpy as np

    event_builder, graph = builder
    rng = np.random.default_rng(0)

    seen, pairs = set(), []
    for card_id, device_id in graph.provisioned:
        if card_id not in seen:
            seen.add(card_id)
            pairs.append((card_id, device_id))
        if len(pairs) >= 60:
            break
    merchants = list(graph.merchants)[:40]

    history: dict[int, list[tuple[int, float]]] = {}
    ts = 0
    for _ in range(4000):
        ts += int(rng.integers(1, 8))
        card_id, device_id = pairs[int(rng.integers(0, len(pairs)))]
        merchant_id = merchants[int(rng.integers(0, len(merchants)))]
        amount = float(rng.lognormal(4.0, 0.8))

        event = event_builder.build_auth(
            ts=ts, card_id=card_id, merchant_id=merchant_id, device_id=device_id,
            amount=amount, entry_mode=0, geo_distance_km=3.0,
        )

        past = history.setdefault(card_id, [])
        seconds = ts * 60
        assert event.auths_last_60s == sum(1 for t, _ in past if t * 60 > seconds - 60)
        assert event.auths_last_1h == sum(1 for t, _ in past if t * 60 > seconds - 3600)
        assert event.auths_last_24h == sum(1 for t, _ in past if t * 60 > seconds - 86_400)
        assert event.amount_sum_24h == pytest.approx(
            sum(a for t, a in past if t * 60 > seconds - 86_400), rel=1e-9
        )

        event_builder.commit_auth(event)
        past.append((ts, amount))
