"""Entity nodes.

Slotted dataclasses rather than a graph library: the simulator touches these
millions of times, and attribute access on a slotted class is a fixed offset
where a library's attribute dict is a hash lookup.

Fields marked derived are computed from realised behaviour after warm start,
never sampled. Sampling them independently of the transactions they summarise
creates an inconsistency a detector can exploit as a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..ids import AccountId, BucketId, CardId, DeviceId, HolderId, MerchantId, PayeeId


class Archetype(Enum):
    COMMUTER = "commuter"
    HOMEBODY = "homebody"
    ONLINE_HEAVY = "online_heavy"
    TRAVELLER = "traveller"
    SENIOR = "senior"
    BUSINESS = "business"


class ActivityTier(Enum):
    """Most holders transact rarely. Real data: median 2 events, 39.5% singletons."""

    DORMANT = "dormant"
    OCCASIONAL = "occasional"
    REGULAR = "regular"
    HEAVY = "heavy"


class CardStatus(Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    REISSUED = "reissued"


class KycLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    FULL = "full"


class RiskTier(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CategoryCluster(Enum):
    """Eight clusters folded from the fourteen source categories."""

    GROCERY = "grocery"
    FUEL_TRANSIT = "fuel_transit"
    DINING = "dining"
    RETAIL = "retail"
    ONLINE = "online"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    TRAVEL = "travel"


class EntryMode(Enum):
    CHIP = "chip"
    CONTACTLESS = "contactless"
    CARD_NOT_PRESENT = "cnp"
    TOKEN = "token"


@dataclass(slots=True)
class Cardholder:
    holder_id: HolderId
    home_lat: float
    home_lon: float
    city_pop: int
    age_years: int
    job_code: int
    tenure_days: int
    archetype: Archetype
    activity_tier: ActivityTier
    household_id: int
    voice_reference: int | None = None
    face_reference: int | None = None


@dataclass(slots=True)
class Card:
    card_id: CardId
    holder_id: HolderId
    issued_ts: int
    credit_line: float
    bin_tier: int
    status: CardStatus = CardStatus.ACTIVE
    frozen_until: int | None = None
    # derived from realised transactions, None until warm start fills them
    median_amount: float | None = None
    category_counts: dict[CategoryCluster, int] = field(default_factory=dict)

    def is_usable(self, now: int) -> bool:
        if self.status is CardStatus.ACTIVE:
            return True
        if self.status is CardStatus.FROZEN and self.frozen_until is not None:
            return now >= self.frozen_until
        return False


@dataclass(slots=True)
class Device:
    """A physical device. Sharing is household-scale, and this is what a
    mitigation may blocklist."""

    device_id: DeviceId
    bucket_id: BucketId
    first_seen_ts: int
    household_id: int
    os_code: int
    browser_code: int
    app_version: int
    ip_asn: int
    is_emulator: bool = False
    reputation: float = 1.0
    blocklisted: bool = False

    def age_days(self, now: int) -> int:
        return max(0, (now - self.first_seen_ts) // (24 * 60))


@dataclass(slots=True)
class FingerprintBucket:
    """A configuration signature shared by many unrelated devices.

    Observed device-to-card fan-out is heavy tailed only because a fingerprint
    collapses everyone running the same OS, browser, and screen size into one
    key. That crowd is represented here rather than as a device, so a
    blocklist mitigation can never take out hundreds of unrelated cardholders.
    """

    bucket_id: BucketId
    os_code: int
    browser_code: int
    screen_code: int
    is_common_configuration: bool = False


@dataclass(slots=True)
class Account:
    account_id: AccountId
    holder_id: HolderId
    opened_ts: int
    balance: float
    kyc_level: KycLevel = KycLevel.FULL
    kyc_via: str = "branch"

    def age_days(self, now: int) -> int:
        return max(0, (now - self.opened_ts) // (24 * 60))


@dataclass(slots=True)
class Merchant:
    merchant_id: MerchantId
    category: CategoryCluster
    avg_ticket: float
    chargeback_rate: float
    risk_tier: RiskTier
    is_high_liquidity: bool
    is_card_not_present: bool
    popularity_rank: int


@dataclass(slots=True)
class Payee:
    payee_id: PayeeId
    target_account_id: AccountId
    first_added_ts: int
    is_mule: bool = False
