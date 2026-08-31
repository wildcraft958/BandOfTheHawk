"""Rolling windows over recent activity.

Every structure here answers a question about the last N seconds and does it in
constant amortised time. Recomputing from history instead would be quadratic in
the number of events, which the hot path cannot afford.

They hold redundant state, which is the hazard: a count and the events behind
it can disagree if eviction is missed on any path. Each is checked against a
naive implementation that keeps everything and filters on read, since that
version is obviously correct and obviously too slow.

**Events must arrive in non-decreasing time order.** Eviction drops from the
front on the assumption that the front is the oldest, so an event arriving
after a later one has already been seen will find its own history partly
discarded. The simulator advances a single clock and so satisfies this
naturally; `assert_ordered` makes the requirement checkable where a caller
might not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


class OutOfOrderError(RuntimeError):
    """Raised when an event arrives before one already recorded.

    Eviction assumes the front of the deque is the oldest entry. An event
    arriving out of order finds part of its own history already discarded, and
    the resulting counts are wrong without anything looking wrong.
    """


class RingWindow:
    """Timestamps within a horizon, counted over any shorter span."""

    __slots__ = ("_horizon", "_capacity", "_events", "_last_ts")

    def __init__(self, horizon_seconds: int, capacity: int = 512) -> None:
        self._horizon = horizon_seconds
        self._capacity = capacity
        self._events: deque[int] = deque(maxlen=capacity)
        self._last_ts: int | None = None

    def push(self, ts: int) -> None:
        if self._last_ts is not None and ts < self._last_ts:
            raise OutOfOrderError(
                f"event at {ts} arrives before {self._last_ts}; windows require "
                "non-decreasing time"
            )
        self._last_ts = ts
        self._events.append(ts)

    def evict(self, now: int) -> None:
        cutoff = now - self._horizon
        events = self._events
        while events and events[0] <= cutoff:
            events.popleft()

    def count_within(self, now: int, seconds: int) -> int:
        """Events in the last `seconds`, excluding any at exactly the boundary."""
        if seconds > self._horizon:
            raise ValueError(
                f"{seconds}s exceeds this window's {self._horizon}s horizon"
            )
        cutoff = now - seconds
        return sum(1 for ts in self._events if ts > cutoff)

    def latest(self) -> int | None:
        return self._events[-1] if self._events else None

    def __len__(self) -> int:
        return len(self._events)


class RollingSum:
    """Amounts within a horizon, summed over any shorter span."""

    __slots__ = ("_horizon", "_events")

    def __init__(self, horizon_seconds: int, capacity: int = 512) -> None:
        self._horizon = horizon_seconds
        self._events: deque[tuple[int, float]] = deque(maxlen=capacity)

    def push(self, ts: int, value: float) -> None:
        self._events.append((ts, value))

    def evict(self, now: int) -> None:
        cutoff = now - self._horizon
        events = self._events
        while events and events[0][0] <= cutoff:
            events.popleft()

    def sum_within(self, now: int, seconds: int) -> float:
        if seconds > self._horizon:
            raise ValueError(
                f"{seconds}s exceeds this window's {self._horizon}s horizon"
            )
        cutoff = now - seconds
        return float(sum(value for ts, value in self._events if ts > cutoff))

    def count_within(self, now: int, seconds: int) -> int:
        cutoff = now - seconds
        return sum(1 for ts, _ in self._events if ts > cutoff)

    def __len__(self) -> int:
        return len(self._events)


class DistinctWindow:
    """How many distinct values appeared within a horizon."""

    __slots__ = ("_horizon", "_events")

    def __init__(self, horizon_seconds: int, capacity: int = 512) -> None:
        self._horizon = horizon_seconds
        self._events: deque[tuple[int, int]] = deque(maxlen=capacity)

    def push(self, ts: int, value: int) -> None:
        self._events.append((ts, value))

    def evict(self, now: int) -> None:
        cutoff = now - self._horizon
        events = self._events
        while events and events[0][0] <= cutoff:
            events.popleft()

    def count_within(self, now: int, seconds: int) -> int:
        if seconds > self._horizon:
            raise ValueError(
                f"{seconds}s exceeds this window's {self._horizon}s horizon"
            )
        cutoff = now - seconds
        return len({value for ts, value in self._events if ts > cutoff})

    def contains(self, now: int, seconds: int, value: int) -> bool:
        cutoff = now - seconds
        return any(v == value and ts > cutoff for ts, v in self._events)

    def __len__(self) -> int:
        return len(self._events)


@dataclass(frozen=True, slots=True)
class CompoundKey:
    """The attributes a compound aggregate conditions on.

    Geography is deliberately absent. The only source with merchant locations
    places them in a ring around each customer, so conditioning on a
    geographic bucket would put an uncalibrated value under a third of these
    features. Merchant risk tier stands in its place.
    """

    category_cluster: int
    entry_mode: int
    merchant_risk_tier: int

    def value_for(self, criterion: str) -> int:
        if criterion == "category_cluster":
            return self.category_cluster
        if criterion == "entry_mode":
            return self.entry_mode
        if criterion == "merchant_risk_tier":
            return self.merchant_risk_tier
        raise KeyError(f"unknown criterion {criterion!r}")


@dataclass(slots=True)
class _Bucket:
    """Events sharing one criterion value."""

    events: deque[tuple[int, float]] = field(default_factory=deque)


class CompoundWindowIndex:
    """Counts and sums per window, conditioned on a second attribute.

    A window keyed on the card alone misses that the same count means
    different things depending on what it is made of. Published feature work
    finds this second condition carries most of the lift, and that leaving it
    out is where a plain velocity feature falls short.

    One deque per criterion value rather than one per card. A card that shops
    in three categories holds three small deques, not one large one filtered
    three ways on every read.
    """

    __slots__ = ("_windows", "_criteria", "_horizon", "_capacity", "_buckets")

    def __init__(
        self,
        windows_seconds: tuple[int, ...],
        criteria: tuple[str, ...],
        capacity: int = 512,
    ) -> None:
        self._windows = tuple(windows_seconds)
        self._criteria = tuple(criteria)
        self._horizon = max(windows_seconds)
        self._capacity = capacity
        self._buckets: dict[tuple[str, int], _Bucket] = {}

    @property
    def n_features(self) -> int:
        """Windows times criteria, counted and summed in each cell."""
        return len(self._windows) * len(self._criteria) * 2

    def feature_names(self) -> tuple[str, ...]:
        """Fixed order, so the vector is a contract rather than an accident."""
        names: list[str] = []
        for criterion in self._criteria:
            for window in self._windows:
                names.append(f"{criterion}_count_{window}s")
                names.append(f"{criterion}_sum_{window}s")
        return tuple(names)

    def push(self, ts: int, amount: float, key: CompoundKey) -> None:
        for criterion in self._criteria:
            bucket_key = (criterion, key.value_for(criterion))
            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                bucket = _Bucket(deque(maxlen=self._capacity))
                self._buckets[bucket_key] = bucket
            bucket.events.append((ts, amount))

    def evict(self, now: int) -> None:
        cutoff = now - self._horizon
        empty: list[tuple[str, int]] = []
        for bucket_key, bucket in self._buckets.items():
            events = bucket.events
            while events and events[0][0] <= cutoff:
                events.popleft()
            if not events:
                empty.append(bucket_key)
        # Buckets are keyed by criterion value, so a card that shops somewhere
        # once would otherwise leave that bucket behind for the whole run.
        for bucket_key in empty:
            del self._buckets[bucket_key]

    def aggregates(self, now: int, key: CompoundKey) -> tuple[float, ...]:
        """The fixed-order feature vector for this key.

        A pass per window, which re-reads the recent events once for each. That
        looks wasteful and a single reverse pass filling every window at once
        was tried instead; measured, it ran twice as slow, because the
        bookkeeping needed to know which window an event still belongs to costs
        more per event in Python than simply re-reading a short deque. Eviction
        keeps these deques short, which is what makes the simple version the
        fast one here.
        """
        out: list[float] = []
        for criterion in self._criteria:
            bucket = self._buckets.get((criterion, key.value_for(criterion)))
            events = bucket.events if bucket else ()
            for window in self._windows:
                cutoff = now - window
                count = 0
                total = 0.0
                for ts, amount in events:
                    if ts > cutoff:
                        count += 1
                        total += amount
                out.append(float(count))
                out.append(total)
        return tuple(out)

    def __len__(self) -> int:
        return sum(len(bucket.events) for bucket in self._buckets.values())


class RunningMedian:
    """Approximate median over recent values.

    A card's median transaction is a derived field: it summarises realised
    behaviour rather than being sampled alongside it. Sampling it independently
    would leave a card whose median disagrees with its own transactions, which
    is an inconsistency a detector could learn.

    Returns None until enough values have arrived, so a feature built on it can
    tell "no history yet" from "history says zero".
    """

    __slots__ = ("_values", "_minimum")

    def __init__(self, capacity: int = 256, minimum: int = 5) -> None:
        self._values: deque[float] = deque(maxlen=capacity)
        self._minimum = minimum

    def push(self, value: float) -> None:
        self._values.append(value)

    @property
    def is_ready(self) -> bool:
        return len(self._values) >= self._minimum

    def value(self) -> float | None:
        if not self.is_ready:
            return None
        ordered = sorted(self._values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return float((ordered[middle - 1] + ordered[middle]) / 2.0)

    def ratio(self, amount: float) -> float | None:
        """How this amount compares to the recent median."""
        median = self.value()
        if median is None or median <= 0:
            return None
        return float(amount / median)

    def __len__(self) -> int:
        return len(self._values)
