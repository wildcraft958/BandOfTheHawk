"""Event construction.

Two calls, never one. `build` reads the state as it stands and returns the
event; `commit` folds that event into the state afterwards. Collapsing them
would make every count include the event describing it, which is an error that
leaves the numbers looking entirely reasonable while every feature is off by
one.

The builder is blind to intent. It receives entity references and reads the
graph, and nothing in its signature says who is acting. That is what keeps an
ordinary holder and an attacker producing the same shape of row.
"""

from __future__ import annotations

import math

from ..config.engine import WindowConfig
from ..ids import CardId, DeviceId, MerchantId
from ..timing.circadian import CircadianClock
from ..world.entities import CategoryCluster, RiskTier
from ..world.graph import EntityGraph
from .schema import AuthAttemptEvent, BindingEvent, EventType
from .state import FeatureStateStore
from .windows import CompoundKey

MINUTES_PER_DAY = 1440
SECONDS_PER_MINUTE = 60
HOUR = 3600
DAY = 86_400
WEEK = 604_800
EARTH_RADIUS_KM = 6371.0

_CLUSTER_INDEX = {cluster: index for index, cluster in enumerate(CategoryCluster)}
_RISK_INDEX = {tier: index for index, tier in enumerate(RiskTier)}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class EventBuilder:
    """Builds events from graph facts and rolling state."""

    __slots__ = ("_graph", "_states", "_config", "_clock", "_next_id", "_warm_start")

    def __init__(
        self,
        graph: EntityGraph,
        states: FeatureStateStore,
        config: WindowConfig,
        clock: CircadianClock | None = None,
    ) -> None:
        self._graph = graph
        self._states = states
        self._config = config
        self._clock = clock
        self._next_id = 0
        self._warm_start = False

    @property
    def warm_start(self) -> bool:
        return self._warm_start

    def set_warm_start(self, active: bool) -> None:
        """Mark events as backdated history.

        Warm-start rows are feature-poorer by construction, since the history
        they would read is what they are in the middle of creating. Flagging
        them lets training exclude them rather than learn the difference.
        """
        self._warm_start = active

    def _mint(self) -> int:
        event_id = self._next_id
        self._next_id += 1
        return event_id

    # ------------------------------------------------------------- reading

    def build_auth(
        self,
        ts: int,
        card_id: CardId,
        merchant_id: MerchantId,
        device_id: DeviceId,
        amount: float,
        entry_mode: int,
        geo_distance_km: float,
    ) -> AuthAttemptEvent:
        """Read the state as it stands. Nothing here mutates."""
        graph = self._graph
        card = graph.cards[card_id]
        merchant = graph.merchants[merchant_id]
        device = graph.devices[device_id]
        holder = graph.holders[card.holder_id]
        state = self._states.card(card_id)

        seconds = ts * SECONDS_PER_MINUTE
        state.evict(seconds)

        key = CompoundKey(
            category_cluster=_CLUSTER_INDEX[merchant.category],
            entry_mode=entry_mode,
            merchant_risk_tier=_RISK_INDEX[merchant.risk_tier],
        )

        accounts = graph.accounts_of_holder(holder.holder_id)
        account_age = 0
        if accounts:
            opened = min(graph.accounts[a].opened_ts for a in accounts)
            account_age = max(0, (ts - opened) // MINUTES_PER_DAY)

        minute_of_day = ts % MINUTES_PER_DAY
        within_usual = (
            self._clock.contains_timestamp(ts) if self._clock is not None else None
        )

        return AuthAttemptEvent(
            event_id=self._mint(),
            ts=ts,
            card_id=int(card_id),
            merchant_id=int(merchant_id),
            device_id=int(device_id),
            amount=amount,
            category_cluster=key.category_cluster,
            entry_mode=entry_mode,
            merchant_risk_tier=key.merchant_risk_tier,
            is_high_liquidity=merchant.is_high_liquidity,
            device_age_days=device.age_days(ts),
            device_new_to_card=not state.has_seen_device(seconds, int(device_id)),
            device_n_cards=graph.device_card_count(device_id),
            card_n_devices=state.devices.count_within(seconds, WEEK),
            ip_asn=device.ip_asn,
            geo_distance_km=geo_distance_km,
            auths_last_60s=state.auth_times.count_within(seconds, 60),
            auths_last_1h=state.auth_times.count_within(seconds, HOUR),
            auths_last_24h=state.auth_times.count_within(seconds, DAY),
            distinct_categories_1h=state.categories.count_within(seconds, HOUR),
            distinct_merchants_24h=state.merchants.count_within(seconds, DAY),
            distinct_ips_24h=state.ip_asns.count_within(seconds, DAY),
            amount_sum_24h=state.amounts.sum_within(seconds, DAY),
            declines_last_1h=state.declines.count_within(seconds, HOUR),
            seconds_since_last_auth=state.seconds_since_last_auth(seconds),
            is_first_txn_this_merchant=not graph.has_transacted(card_id, merchant_id),
            hour_of_day=minute_of_day // 60,
            is_weekend=(ts // MINUTES_PER_DAY) % 7 >= 5,
            within_usual_hours=within_usual,
            amount_vs_median=state.median_amount.ratio(amount),
            account_age_days=account_age,
            holder_tenure_days=holder.tenure_days,
            compound_features=state.compound.aggregates(seconds, key),
            compound_feature_names=state.compound.feature_names(),
            is_warm_start=self._warm_start,
        )

    def build_binding(
        self,
        ts: int,
        event_type: EventType,
        actor_id: int,
        target_id: int,
        holder_id: int,
        method: int = 0,
        channel: int = 0,
        device_id: DeviceId | None = None,
    ) -> BindingEvent:
        graph = self._graph
        holder_state = self._states.holder(holder_id)
        seconds = ts * SECONDS_PER_MINUTE

        accounts = graph.accounts_of_holder(holder_id)
        account_age = 0
        if accounts:
            opened = min(graph.accounts[a].opened_ts for a in accounts)
            account_age = max(0, (ts - opened) // MINUTES_PER_DAY)

        since_last_bind = None
        if holder_state.last_device_bind_ts is not None:
            since_last_bind = max(
                0, (seconds - holder_state.last_device_bind_ts) // DAY
            )

        device_age = None
        device_cards = None
        if device_id is not None and device_id in graph.devices:
            device_age = graph.devices[device_id].age_days(ts)
            device_cards = graph.device_card_count(device_id)

        return BindingEvent(
            event_id=self._mint(),
            ts=ts,
            event_type_value=event_type.value,
            actor_id=actor_id,
            target_id=target_id,
            holder_id=holder_id,
            method=method,
            channel=channel,
            time_since_account_open_days=account_age,
            time_since_last_bind_days=since_last_bind,
            n_binds_30d=len(holder_state.n_binds_30d),
            hour_of_day=(ts % MINUTES_PER_DAY) // 60,
            hours_since_password_reset=holder_state.hours_since_password_reset(seconds),
            hours_since_support_call=holder_state.hours_since_support_call(seconds),
            recovery_chain_within_1h=holder_state.recovery_chain_within(seconds, hours=1.0),
            device_age_days=device_age,
            device_n_cards=device_cards,
            is_warm_start=self._warm_start,
        )

    # ------------------------------------------------------------- writing

    def commit_auth(self, event: AuthAttemptEvent, approved: bool = True) -> None:
        """Fold a built event into the state it was read from."""
        state = self._states.card(CardId(event.card_id))
        state.commit_auth(
            ts=event.ts * SECONDS_PER_MINUTE,
            amount=event.amount,
            merchant_id=event.merchant_id,
            category=event.category_cluster,
            ip_asn=event.ip_asn,
            device_id=event.device_id,
            key=CompoundKey(
                category_cluster=event.category_cluster,
                entry_mode=event.entry_mode,
                merchant_risk_tier=event.merchant_risk_tier,
            ),
            approved=approved,
        )
        if approved:
            self._graph.record_transaction(
                CardId(event.card_id),
                MerchantId(event.merchant_id),
                event.amount,
                event.ts,
            )
            card = self._graph.cards[CardId(event.card_id)]
            card.median_amount = state.median_amount.value()
            counts = card.category_counts
            cluster = list(CategoryCluster)[event.category_cluster]
            counts[cluster] = counts.get(cluster, 0) + 1

    def commit_binding(self, event: BindingEvent) -> None:
        holder_state = self._states.holder(event.holder_id)
        seconds = event.ts * SECONDS_PER_MINUTE
        kind = event.event_type
        if kind is EventType.AUTH_RESET:
            holder_state.record_password_reset(seconds)
        elif kind is EventType.SUPPORT_TICKET:
            holder_state.record_support_call(seconds)
        elif kind is EventType.DEVICE_BIND:
            holder_state.record_device_bind(seconds)
