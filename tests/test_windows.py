"""Window structures are checked against naive implementations.

The naive version keeps every event and filters on read. It is obviously
correct and far too slow, which makes it exactly the right thing to compare
against: any disagreement is a bug in the incremental structure, since the
reference has no state to get wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.features.windows import (
    CompoundKey,
    CompoundWindowIndex,
    DistinctWindow,
    RingWindow,
    RollingSum,
    RunningMedian,
)

HOUR = 3600
DAY = 86_400
WEEK = 604_800


class NaiveWindow:
    """Keeps everything, filters on read."""

    def __init__(self) -> None:
        self.events: list[tuple[int, float, int]] = []

    def push(self, ts: int, amount: float = 0.0, value: int = 0) -> None:
        self.events.append((ts, amount, value))

    def count_within(self, now: int, seconds: int) -> int:
        return sum(1 for ts, _, _ in self.events if ts > now - seconds)

    def sum_within(self, now: int, seconds: int) -> float:
        return float(sum(a for ts, a, _ in self.events if ts > now - seconds))

    def distinct_within(self, now: int, seconds: int) -> int:
        return len({v for ts, _, v in self.events if ts > now - seconds})


def random_stream(n: int, seed: int) -> list[tuple[int, float, int]]:
    rng = np.random.default_rng(seed)
    ts = 0
    out = []
    for _ in range(n):
        ts += int(rng.exponential(900))
        out.append((ts, float(rng.lognormal(4.0, 0.9)), int(rng.integers(0, 12))))
    return out


def test_ring_window_matches_the_reference() -> None:
    window = RingWindow(WEEK, capacity=4096)
    naive = NaiveWindow()
    for ts, amount, value in random_stream(3000, seed=1):
        window.push(ts)
        naive.push(ts, amount, value)
        window.evict(ts)
        for span in (HOUR, DAY, WEEK):
            assert window.count_within(ts, span) == naive.count_within(ts, span)


def test_rolling_sum_matches_the_reference() -> None:
    window = RollingSum(WEEK, capacity=4096)
    naive = NaiveWindow()
    for ts, amount, value in random_stream(3000, seed=2):
        window.push(ts, amount)
        naive.push(ts, amount, value)
        window.evict(ts)
        for span in (HOUR, DAY, WEEK):
            assert window.sum_within(ts, span) == pytest.approx(
                naive.sum_within(ts, span), rel=1e-9
            )


def test_distinct_window_matches_the_reference() -> None:
    window = DistinctWindow(WEEK, capacity=4096)
    naive = NaiveWindow()
    for ts, amount, value in random_stream(3000, seed=3):
        window.push(ts, value)
        naive.push(ts, amount, value)
        window.evict(ts)
        for span in (HOUR, DAY, WEEK):
            assert window.count_within(ts, span) == naive.distinct_within(ts, span)


def test_eviction_actually_releases_events() -> None:
    """A window that never shrinks is a leak rather than a window."""
    window = RingWindow(HOUR)
    for ts in range(0, 10_000, 100):
        window.push(ts)
    window.evict(100_000)
    assert len(window) == 0


def test_window_refuses_a_span_beyond_its_horizon() -> None:
    """Answering over a span longer than the horizon would silently undercount,
    since the events needed were already evicted."""
    window = RingWindow(HOUR)
    with pytest.raises(ValueError, match="horizon"):
        window.count_within(0, DAY)


def test_boundary_is_exclusive() -> None:
    window = RingWindow(HOUR)
    window.push(0)
    assert window.count_within(HOUR, HOUR) == 0
    assert window.count_within(HOUR - 1, HOUR) == 1


def test_compound_index_matches_the_reference() -> None:
    criteria = ("category_cluster", "entry_mode", "merchant_risk_tier")
    windows = (HOUR, DAY, WEEK)
    index = CompoundWindowIndex(windows, criteria, capacity=4096)

    history: list[tuple[int, float, CompoundKey]] = []
    rng = np.random.default_rng(5)
    ts = 0
    for _ in range(1500):
        ts += int(rng.exponential(1200))
        key = CompoundKey(
            category_cluster=int(rng.integers(0, 8)),
            entry_mode=int(rng.integers(0, 4)),
            merchant_risk_tier=int(rng.integers(0, 3)),
        )
        amount = float(rng.lognormal(4.0, 0.8))

        index.push(ts, amount, key)
        history.append((ts, amount, key))
        index.evict(ts)

        observed = index.aggregates(ts, key)
        expected: list[float] = []
        for criterion in criteria:
            wanted = key.value_for(criterion)
            for window in windows:
                cutoff = ts - window
                rows = [
                    (t, a)
                    for t, a, k in history
                    if k.value_for(criterion) == wanted and t > cutoff
                ]
                expected.append(float(len(rows)))
                expected.append(float(sum(a for _, a in rows)))
        assert observed == pytest.approx(tuple(expected), rel=1e-9)


def test_compound_index_reports_a_stable_feature_order() -> None:
    index = CompoundWindowIndex((HOUR, DAY), ("category_cluster", "entry_mode"))
    names = index.feature_names()
    assert len(names) == index.n_features == 8
    assert names[0] == "category_cluster_count_3600s"
    assert names == CompoundWindowIndex(
        (HOUR, DAY), ("category_cluster", "entry_mode")
    ).feature_names()


def test_compound_index_drops_empty_buckets() -> None:
    """A card that shops somewhere once would otherwise hold that bucket for
    the whole run."""
    index = CompoundWindowIndex((HOUR,), ("category_cluster",))
    for cluster in range(50):
        index.push(cluster * 10, 20.0, CompoundKey(cluster, 0, 0))
    index.evict(1_000_000)
    assert len(index) == 0


def test_compound_key_rejects_an_unknown_criterion() -> None:
    with pytest.raises(KeyError, match="unknown criterion"):
        CompoundKey(1, 2, 3).value_for("geo_bucket")


def test_running_median_withholds_a_value_until_it_has_history() -> None:
    """A feature has to tell no history from a history that says zero."""
    median = RunningMedian(minimum=5)
    for value in (10.0, 20.0, 30.0):
        median.push(value)
    assert median.value() is None
    assert median.ratio(100.0) is None
    for value in (40.0, 50.0):
        median.push(value)
    assert median.value() == pytest.approx(30.0)
    assert median.ratio(60.0) == pytest.approx(2.0)


def test_running_median_matches_a_sorted_reference() -> None:
    rng = np.random.default_rng(9)
    values = rng.lognormal(4.0, 1.0, 300)
    median = RunningMedian(capacity=64, minimum=5)
    for value in values:
        median.push(float(value))
    assert median.value() == pytest.approx(float(np.median(values[-64:])), rel=1e-9)


def test_running_median_is_bounded() -> None:
    median = RunningMedian(capacity=32)
    for value in range(1000):
        median.push(float(value))
    assert len(median) == 32


def test_out_of_order_events_are_refused() -> None:
    """Eviction drops from the front assuming it is the oldest, so an event
    arriving late finds part of its own history already discarded. The counts
    are then wrong with nothing looking wrong, which is why this raises."""
    from fraudsim.features.windows import OutOfOrderError

    window = RingWindow(HOUR)
    window.push(1000)
    window.push(2000)
    with pytest.raises(OutOfOrderError, match="non-decreasing"):
        window.push(1500)


def test_equal_timestamps_are_allowed() -> None:
    """Several events can share a clock tick."""
    window = RingWindow(HOUR)
    for _ in range(5):
        window.push(1000)
    assert len(window) == 5
