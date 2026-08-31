"""Legitimate behaviour that looks suspicious.

Without these a false-positive rate has nothing to measure. Every rule keyed on
travel, on bursts, on a device nobody has seen before, or on a recovery
sequence stays silent forever against traffic that never does any of it, and
the resulting rate says only that ordinary spending is ordinary.

Each injector produces a plan rather than a transaction, because most of these
are not about the amount. Travel changes where. A shopping session changes how
many and how close together. A new device changes what the transaction goes
through. A recovery is not a transaction at all.

On provenance: only the dispute rate comes from a published figure. Device
replacement is a reasonable proxy from upgrade cycles, travel and session rates
are tuned towards a target, and the recovery chain is unmeasurable. No bank,
regulator, or paper publishes how often a legitimate holder resets a password,
calls support, and rebinds a device inside an hour, so it is swept across three
orders of magnitude and no claim may rest on a point estimate of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..behavior.amount import AmountModel
from ..behavior.loyalty import LoyaltyModel
from ..features.schema import EventType
from ..ids import DeviceId, MerchantId
from ..settings.behavior import AmountConfig, HardNegativeConfig
from ..settings.world import GeoConfig


class NegativeKind(Enum):
    ORDINARY = "ordinary"
    LARGE_PURCHASE = "large_purchase"
    GIFT_CARD = "gift_card"
    TRAVEL = "travel"
    SESSION = "session"
    NEW_DEVICE = "new_device"
    DISPUTE = "dispute"
    RECOVERY = "recovery"


@dataclass(slots=True)
class PlannedAuth:
    """One authorisation within a plan."""

    merchant_id: MerchantId
    amount: float
    geo_distance_km: float
    offset_minutes: int = 0
    device_id: DeviceId | None = None


@dataclass(slots=True)
class Plan:
    """What one scheduled slot turns into.

    Usually a single authorisation. Sometimes a burst of them, a binding
    followed by spending, or a sequence of non-payment events with no
    authorisation at all.
    """

    kind: NegativeKind
    auths: list[PlannedAuth] = field(default_factory=list)
    bindings: list[tuple[EventType, int]] = field(default_factory=list)
    bind_device: bool = False


class NegativeInjector:
    """Decides what a scheduled slot becomes."""

    def __init__(
        self,
        negatives: HardNegativeConfig,
        amount: AmountConfig,
        geo: GeoConfig,
        rng: np.random.Generator,
        amounts: AmountModel | None = None,
        loyalty: LoyaltyModel | None = None,
    ) -> None:
        self.negatives = negatives
        self.amount = amount
        self.geo = geo
        self.rng = rng
        # Amounts come from the acting card's own level where one exists. A
        # shared curve makes every card alike and halves the spread of per-card
        # means, which a pooled comparison cannot see.
        self.amounts = amounts
        # Merchants come from the acting card's own regulars where it has
        # them. A uniform draw over the whole roster makes almost every
        # transaction the card's first at that merchant, so the feature that
        # asks carries no information at all.
        self.loyalty = loyalty
        self._card_id: int | None = None
        self.counts: dict[str, int] = {kind.value: 0 for kind in NegativeKind}
        self._thresholds = self._build_thresholds(negatives)

    @staticmethod
    def _build_thresholds(
        negatives: HardNegativeConfig,
    ) -> list[tuple[float, NegativeKind]]:
        """Cumulative shares, checked in order.

        Yearly rates become a per-event share by assuming a holder transacts on
        the order of a few hundred times a year. That conversion is an
        assumption in its own right and is why these end up tuned against the
        target rather than derived from it.
        """
        per_event = 1.0 / 250.0
        shares = [
            (negatives.large_purchase_share, NegativeKind.LARGE_PURCHASE),
            (negatives.gift_card_share, NegativeKind.GIFT_CARD),
            (negatives.travel_rate_yearly * per_event, NegativeKind.TRAVEL),
            (negatives.shopping_session_share, NegativeKind.SESSION),
            (negatives.new_device_rate_yearly * per_event, NegativeKind.NEW_DEVICE),
            (negatives.dispute_rate, NegativeKind.DISPUTE),
            (negatives.password_reset_rate_yearly * per_event, NegativeKind.RECOVERY),
        ]
        thresholds: list[tuple[float, NegativeKind]] = []
        cumulative = 0.0
        for share, kind in shares:
            cumulative += share
            thresholds.append((cumulative, kind))
        return thresholds

    # ------------------------------------------------------------- sampling

    def for_card(self, card_id: int) -> None:
        """Name the card the next plan is for, so its level is used."""
        self._card_id = card_id

    def _base_amount(self) -> float:
        if self.amounts is not None and self._card_id is not None:
            profile = self.amounts.profile(self._card_id)
            if profile is not None:
                return profile.sample(self.amount, self.rng)
        return float(
            np.clip(
                self.rng.lognormal(self.amount.lognormal_mu, self.amount.lognormal_sigma),
                1.0,
                self.amount.upper_bound,
            )
        )

    def _local_distance(self) -> float:
        return float(self.rng.exponential(self.geo.home_radius_km))

    def _pick(self, options: list) -> int:
        return options[int(self.rng.integers(0, len(options)))]

    def _pick_merchant(self, options: list) -> int:
        """This card's usual merchant, or a uniform draw if it has none.

        Constrained pools are passed through untouched by the callers that
        need them: a card's habits should not override "this has to be a
        travel merchant".
        """
        if self.loyalty is not None and self._card_id is not None:
            picked = self.loyalty.pick_merchant(
                self._card_id, self.rng, np.asarray(options, dtype=int)
            )
            if picked is not None:
                return picked
        return self._pick(options)

    # ------------------------------------------------------------ planning

    def plan(
        self,
        merchants: list[MerchantId],
        liquid: list[MerchantId],
        travel_merchants: list[MerchantId],
    ) -> Plan:
        roll = float(self.rng.random())
        kind = NegativeKind.ORDINARY
        for threshold, candidate in self._thresholds:
            if roll < threshold:
                kind = candidate
                break

        plan = getattr(self, f"_plan_{kind.value}")(merchants, liquid, travel_merchants)
        self.counts[kind.value] += 1
        return plan

    def _plan_ordinary(self, merchants, liquid, travel_merchants) -> Plan:
        return Plan(
            kind=NegativeKind.ORDINARY,
            auths=[
                PlannedAuth(
                    merchant_id=self._pick_merchant(merchants),
                    amount=self._base_amount(),
                    geo_distance_km=self._local_distance(),
                )
            ],
        )

    def _plan_large_purchase(self, merchants, liquid, travel_merchants) -> Plan:
        """A legitimate purchase far above what this card usually sees.

        Trips the amount-ratio rule, which is the point: an unusual amount is
        not by itself evidence of anything.
        """
        return Plan(
            kind=NegativeKind.LARGE_PURCHASE,
            auths=[
                PlannedAuth(
                    merchant_id=self._pick_merchant(merchants),
                    amount=self._base_amount() * float(self.rng.uniform(8.0, 20.0)),
                    geo_distance_km=self._local_distance(),
                )
            ],
        )

    def _plan_gift_card(self, merchants, liquid, travel_merchants) -> Plan:
        """Buying something easily resold. Ordinary, and indistinguishable at
        the moment of purchase from buying it to launder."""
        return Plan(
            kind=NegativeKind.GIFT_CARD,
            auths=[
                PlannedAuth(
                    merchant_id=self._pick(liquid),
                    amount=self._base_amount(),
                    geo_distance_km=self._local_distance(),
                )
            ],
        )

    def _plan_travel(self, merchants, liquid, travel_merchants) -> Plan:
        """A trip: several transactions far from home over a few days.

        Trips distance-based checks and, because a trip visits several places,
        the distinct-merchant rule as well.

        This raises the travel share of generated traffic above the configured
        category mix, by roughly three points: it fires on under one percent of
        slots but emits several authorisations each, all of them travel. That
        is the behaviour, not a defect - people on a trip do spend on travel -
        and the mix describes ordinary spending rather than the total. Do not
        try to correct it by constraining this pool, which would remove the
        hard negative instead of the discrepancy.
        """
        distance = float(self.rng.exponential(self.geo.travel_distance_km))
        pool = travel_merchants or merchants
        count = int(self.rng.integers(2, 6))
        auths = []
        offset = 0
        for _ in range(count):
            offset += int(self.rng.exponential(6 * 60))
            auths.append(
                PlannedAuth(
                    merchant_id=self._pick(pool),
                    amount=self._base_amount() * float(self.rng.uniform(1.0, 2.5)),
                    geo_distance_km=distance * float(self.rng.uniform(0.8, 1.2)),
                    offset_minutes=offset,
                )
            )
        return Plan(kind=NegativeKind.TRAVEL, auths=auths)

    def _plan_session(self, merchants, liquid, travel_merchants) -> Plan:
        """An afternoon of shopping: several purchases minutes apart.

        This is what the transactions-per-hour rule exists to catch, and it is
        also what a perfectly ordinary Saturday looks like.
        """
        count = int(self.rng.integers(3, 8))
        auths = []
        offset = 0
        for _ in range(count):
            offset += int(self.rng.exponential(11))
            auths.append(
                PlannedAuth(
                    merchant_id=self._pick_merchant(merchants),
                    amount=self._base_amount(),
                    geo_distance_km=self._local_distance(),
                    offset_minutes=offset,
                )
            )
        return Plan(kind=NegativeKind.SESSION, auths=auths)

    def _plan_new_device(self, merchants, liquid, travel_merchants) -> Plan:
        """A replaced phone, bound and then used.

        Trips the payment-method rule and anything keyed on device age. The
        binding is emitted as its own event, so a detector sees the sequence
        rather than only its consequence.
        """
        count = int(self.rng.integers(1, 4))
        auths = []
        offset = 0
        for _ in range(count):
            offset += int(self.rng.exponential(90))
            auths.append(
                PlannedAuth(
                    merchant_id=self._pick_merchant(merchants),
                    amount=self._base_amount(),
                    geo_distance_km=self._local_distance(),
                    offset_minutes=offset,
                )
            )
        return Plan(
            kind=NegativeKind.NEW_DEVICE,
            auths=auths,
            bindings=[(EventType.DEVICE_BIND, 0)],
            bind_device=True,
        )

    def _plan_dispute(self, merchants, liquid, travel_merchants) -> Plan:
        """A genuine complaint about a real charge.

        The only injector whose rate comes from a published figure.
        """
        return Plan(
            kind=NegativeKind.DISPUTE,
            bindings=[(EventType.SUPPORT_TICKET, 0), (EventType.DISPUTE_FILED, 30)],
        )

    def _plan_recovery(self, merchants, liquid, travel_merchants) -> Plan:
        """Locked out, called support, signed in on the new phone.

        Reset, call, and binding are each unremarkable alone, and this sequence
        inside an hour is exactly what an account-takeover detector keys on.
        How often a legitimate holder does it is unmeasured, so the rate is
        swept and no claim may rest on a point estimate.
        """
        chained = float(self.rng.random()) < self.negatives.recovery_chain_probability * 1e4
        spacing = 12 if chained else 6 * 60
        return Plan(
            kind=NegativeKind.RECOVERY,
            bindings=[
                (EventType.AUTH_RESET, 0),
                (EventType.SUPPORT_TICKET, spacing),
                (EventType.DEVICE_BIND, spacing * 2),
            ],
            bind_device=chained,
        )
