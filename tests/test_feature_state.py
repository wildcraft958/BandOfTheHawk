"""Feature state has to describe history before the current event, not after."""

from __future__ import annotations

import pytest

from fraudsim.features.state import CardFeatureState, FeatureStateStore
from fraudsim.features.windows import CompoundKey
from fraudsim.settings.engine import WindowConfig

HOUR = 3600
DAY = 86_400


def key() -> CompoundKey:
    return CompoundKey(category_cluster=1, entry_mode=0, merchant_risk_tier=0)


def commit(state: CardFeatureState, ts: int, amount: float = 50.0, **kwargs) -> None:
    state.commit_auth(
        ts=ts, amount=amount, merchant_id=kwargs.get("merchant_id", 1),
        category=kwargs.get("category", 1), ip_asn=kwargs.get("ip_asn", 10),
        device_id=kwargs.get("device_id", 100), key=kwargs.get("key", key()),
        approved=kwargs.get("approved", True),
    )


def test_reading_before_committing_excludes_the_current_event() -> None:
    """The error this ordering prevents is invisible: every count stays
    plausible while being off by exactly one."""
    state = CardFeatureState.create(WindowConfig())
    commit(state, ts=0)
    commit(state, ts=100)

    before = state.auth_times.count_within(200, HOUR)
    commit(state, ts=200)
    after = state.auth_times.count_within(200, HOUR)

    assert before == 2
    assert after == 3


def test_counts_track_committed_events() -> None:
    state = CardFeatureState.create(WindowConfig())
    for ts in range(0, 5 * HOUR, HOUR):
        commit(state, ts=ts)
    now = 5 * HOUR
    assert state.auth_times.count_within(now, HOUR) == 0
    assert state.auth_times.count_within(now, DAY) == 5
    assert state.n_auths == 5


def test_amount_sum_covers_its_window() -> None:
    state = CardFeatureState.create(WindowConfig())
    for index in range(4):
        commit(state, ts=index * HOUR, amount=100.0)
    assert state.amounts.sum_within(4 * HOUR, DAY) == pytest.approx(400.0)


def test_declines_are_recorded_separately() -> None:
    state = CardFeatureState.create(WindowConfig())
    commit(state, ts=0, approved=False)
    commit(state, ts=60, approved=False)
    commit(state, ts=120, approved=True)
    assert state.declines.count_within(180, HOUR) == 2
    assert state.auth_times.count_within(180, HOUR) == 3


def test_distinct_counts_deduplicate() -> None:
    state = CardFeatureState.create(WindowConfig())
    for merchant in (1, 1, 2, 3, 3):
        commit(state, ts=0, merchant_id=merchant)
    assert state.merchants.count_within(1, DAY) == 3


def test_median_withholds_a_value_until_it_has_history() -> None:
    state = CardFeatureState.create(WindowConfig())
    commit(state, ts=0, amount=10.0)
    assert state.median_amount.value() is None
    for index in range(1, 6):
        commit(state, ts=index * 60, amount=10.0 * (index + 1))
    assert state.median_amount.value() is not None


def test_time_since_last_auth() -> None:
    state = CardFeatureState.create(WindowConfig())
    assert state.seconds_since_last_auth(100) is None
    commit(state, ts=100)
    assert state.seconds_since_last_auth(400) == 300


def test_recovery_chain_needs_every_step() -> None:
    """Each step is unremarkable alone; the sequence is the signal."""
    store = FeatureStateStore(WindowConfig())
    holder = store.holder(1)
    now = 10 * HOUR

    assert not holder.recovery_chain_within(now)
    holder.record_password_reset(now - 600)
    assert not holder.recovery_chain_within(now)
    holder.record_support_call(now - 400)
    assert not holder.recovery_chain_within(now)
    holder.record_device_bind(now - 200)
    assert holder.recovery_chain_within(now)


def test_recovery_chain_respects_its_window() -> None:
    store = FeatureStateStore(WindowConfig())
    holder = store.holder(1)
    now = 10 * HOUR
    holder.record_password_reset(now - 5 * HOUR)
    holder.record_support_call(now - 400)
    holder.record_device_bind(now - 200)
    assert not holder.recovery_chain_within(now, hours=1.0)
    assert holder.recovery_chain_within(now, hours=6.0)


def test_store_creates_state_on_first_use() -> None:
    store = FeatureStateStore(WindowConfig())
    assert store.n_cards() == 0
    store.card(1)
    store.card(1)
    store.card(2)
    assert store.n_cards() == 2


def test_eviction_releases_across_the_store() -> None:
    store = FeatureStateStore(WindowConfig())
    for card_id in range(20):
        commit(store.card(card_id), ts=0)
    store.evict_all(60 * DAY)
    assert all(len(store.card(c).auth_times) == 0 for c in range(20))
