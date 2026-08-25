"""Entity edges.

Fraud is the creation of an edge that should not exist; mitigation deletes one
or raises its cost. Edge payloads are kept separate from the adjacency indices
so a traversal never pays for attributes it does not read.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..ids import AccountId, CardId, DeviceId, MerchantId, PayeeId


class BindMethod(Enum):
    SELF_SERVICE = "self_service"
    IVR = "ivr"
    BRANCH = "branch"
    RECOVERY = "recovery"


class AddMethod(Enum):
    APP = "app"
    WEB = "web"
    BRANCH = "branch"
    PHONE = "phone"


@dataclass(slots=True)
class ProvisionedEdge:
    """A card usable on a device."""

    card_id: CardId
    device_id: DeviceId
    bind_ts: int
    bind_method: BindMethod
    bind_trust: float = 1.0
    challenge_required: bool = False
    step_up_until: int | None = None

    def requires_step_up(self, now: int) -> bool:
        if self.step_up_until is None:
            return self.challenge_required
        return now < self.step_up_until


@dataclass(slots=True)
class AddedEdge:
    """A payee registered against an account, possibly still cooling off."""

    account_id: AccountId
    payee_id: PayeeId
    add_ts: int
    add_method: AddMethod
    cooling_off_until: int | None = None

    def is_cooling_off(self, now: int) -> bool:
        return self.cooling_off_until is not None and now < self.cooling_off_until


@dataclass(slots=True)
class TransactsEdge:
    """Running history between a card and a merchant."""

    card_id: CardId
    merchant_id: MerchantId
    first_ts: int
    last_ts: int
    count: int = 0
    total_amount: float = 0.0

    def observe(self, ts: int, amount: float) -> None:
        self.last_ts = ts
        self.count += 1
        self.total_amount += amount


@dataclass(slots=True)
class UsedByEdge:
    """A device seen operating an account."""

    device_id: DeviceId
    account_id: AccountId
    first_ts: int
    last_ts: int
    count: int = 0

    def observe(self, ts: int) -> None:
        self.last_ts = ts
        self.count += 1
