"""Per-entity feature state.

Everything an event needs to know about recent history lives here, maintained
incrementally as events arrive.

The important rule is that reading and writing are separate operations. An
event's features have to describe the state before that event, so building and
committing are two calls rather than one. A single method doing both would make
every count include itself, and that error is invisible: the numbers stay
plausible and every feature is off by exactly one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.engine import WindowConfig
from ..ids import CardId
from ..clock import SECONDS_PER_DAY as DAY, SECONDS_PER_HOUR as HOUR, SECONDS_PER_WEEK as WEEK
from .windows import (
    CompoundKey,
    CompoundWindowIndex,
    DistinctWindow,
    RingWindow,
    RollingSum,
    RunningMedian,
)


@dataclass(slots=True)
class CardFeatureState:
    """Rolling history for one card."""

    auth_times: RingWindow
    declines: RingWindow
    amounts: RollingSum
    merchants: DistinctWindow
    categories: DistinctWindow
    ip_asns: DistinctWindow
    devices: DistinctWindow
    compound: CompoundWindowIndex
    median_amount: RunningMedian
    category_counts: dict[int, int] = field(default_factory=dict)
    last_auth_ts: int | None = None
    first_auth_ts: int | None = None
    n_auths: int = 0

    @classmethod
    def create(cls, config: WindowConfig) -> "CardFeatureState":
        horizon = max(config.windows_seconds)
        capacity = config.max_events_per_window
        return cls(
            auth_times=RingWindow(horizon, capacity),
            declines=RingWindow(HOUR, capacity),
            amounts=RollingSum(horizon, capacity),
            merchants=DistinctWindow(horizon, capacity),
            categories=DistinctWindow(horizon, capacity),
            ip_asns=DistinctWindow(horizon, capacity),
            devices=DistinctWindow(WEEK, capacity),
            compound=CompoundWindowIndex(
                config.windows_seconds, config.compound_criteria, capacity
            ),
            median_amount=RunningMedian(),
        )

    def evict(self, now: int) -> None:
        self.auth_times.evict(now)
        self.declines.evict(now)
        self.amounts.evict(now)
        self.merchants.evict(now)
        self.categories.evict(now)
        self.ip_asns.evict(now)
        self.devices.evict(now)
        self.compound.evict(now)

    def commit_auth(
        self,
        ts: int,
        amount: float,
        merchant_id: int,
        category: int,
        ip_asn: int,
        device_id: int,
        key: CompoundKey,
        approved: bool = True,
    ) -> None:
        """Fold an authorisation into the history.

        Called after the event is built, never before. The ordering is the
        whole reason these are separate methods.
        """
        self.auth_times.push(ts)
        self.amounts.push(ts, amount)
        self.merchants.push(ts, merchant_id)
        self.categories.push(ts, category)
        self.ip_asns.push(ts, ip_asn)
        self.devices.push(ts, device_id)
        self.compound.push(ts, amount, key)
        self.median_amount.push(amount)
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

        if not approved:
            self.declines.push(ts)

        self.last_auth_ts = ts
        if self.first_auth_ts is None:
            self.first_auth_ts = ts
        self.n_auths += 1

    def seconds_since_last_auth(self, now: int) -> int | None:
        if self.last_auth_ts is None:
            return None
        return max(0, now - self.last_auth_ts)

    def has_seen_merchant(self, now: int, merchant_id: int, seconds: int = WEEK) -> bool:
        return self.merchants.contains(now, seconds, merchant_id)

    def has_seen_device(self, now: int, device_id: int, seconds: int = WEEK) -> bool:
        return self.devices.contains(now, seconds, device_id)


@dataclass(slots=True)
class HolderFeatureState:
    """History that belongs to a person rather than a card.

    Account recovery is the reason this exists. A password reset, a support
    call, and a device binding are each unremarkable alone, and the sequence is
    what a binding detector keys on. Nothing published measures how often a
    legitimate holder does all three within an hour, so the rate is swept, and
    the timestamps have to be tracked regardless of which card was involved.
    """

    last_password_reset_ts: int | None = None
    last_support_call_ts: int | None = None
    last_device_bind_ts: int | None = None
    n_binds_30d: RingWindow = field(default_factory=lambda: RingWindow(30 * DAY, 64))

    def record_password_reset(self, ts: int) -> None:
        self.last_password_reset_ts = ts

    def record_support_call(self, ts: int) -> None:
        self.last_support_call_ts = ts

    def record_device_bind(self, ts: int) -> None:
        self.last_device_bind_ts = ts
        self.n_binds_30d.push(ts)

    def hours_since_password_reset(self, now: int) -> float | None:
        if self.last_password_reset_ts is None:
            return None
        return (now - self.last_password_reset_ts) / 3600.0

    def hours_since_support_call(self, now: int) -> float | None:
        if self.last_support_call_ts is None:
            return None
        return (now - self.last_support_call_ts) / 3600.0

    def recovery_chain_within(self, now: int, hours: float = 1.0) -> bool:
        """Whether reset, call, and binding all happened inside one window."""
        reset = self.hours_since_password_reset(now)
        call = self.hours_since_support_call(now)
        if reset is None or call is None or self.last_device_bind_ts is None:
            return False
        bind = (now - self.last_device_bind_ts) / 3600.0
        return max(reset, call, bind) <= hours


class FeatureStateStore:
    """Feature state for every entity, created on first use."""

    __slots__ = ("_config", "_cards", "_holders")

    def __init__(self, config: WindowConfig) -> None:
        self._config = config
        self._cards: dict[CardId, CardFeatureState] = {}
        self._holders: dict[int, HolderFeatureState] = {}

    def card(self, card_id: CardId) -> CardFeatureState:
        state = self._cards.get(card_id)
        if state is None:
            state = CardFeatureState.create(self._config)
            self._cards[card_id] = state
        return state

    def holder(self, holder_id: int) -> HolderFeatureState:
        state = self._holders.get(holder_id)
        if state is None:
            state = HolderFeatureState()
            self._holders[holder_id] = state
        return state

    def evict_all(self, now: int) -> None:
        for state in self._cards.values():
            state.evict(now)

    def n_cards(self) -> int:
        return len(self._cards)

    def n_holders(self) -> int:
        return len(self._holders)
