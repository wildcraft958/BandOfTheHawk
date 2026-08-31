"""Typed entity identifiers and the minter that issues them.

Identifiers are plain integers so they index adjacency structures directly.
The NewType wrappers cost nothing at runtime but stop a card id being passed
where a device id is expected.
"""

from __future__ import annotations

from enum import Enum
from typing import NewType

HolderId = NewType("HolderId", int)
CardId = NewType("CardId", int)
DeviceId = NewType("DeviceId", int)
BucketId = NewType("BucketId", int)
AccountId = NewType("AccountId", int)
MerchantId = NewType("MerchantId", int)
PayeeId = NewType("PayeeId", int)
ActorId = NewType("ActorId", int)
EventId = NewType("EventId", int)
EpisodeId = NewType("EpisodeId", int)


class EntityKind(Enum):
    HOLDER = "holder"
    CARD = "card"
    DEVICE = "device"
    FINGERPRINT_BUCKET = "fingerprint_bucket"
    ACCOUNT = "account"
    MERCHANT = "merchant"
    PAYEE = "payee"
    ACTOR = "actor"
    EVENT = "event"
    EPISODE = "episode"


class IdMinter:
    """Issues monotonic ids, counted separately per entity kind."""

    __slots__ = ("_counters",)

    def __init__(self) -> None:
        self._counters: dict[EntityKind, int] = dict.fromkeys(EntityKind, 0)

    def mint(self, kind: EntityKind) -> int:
        nxt = self._counters[kind]
        self._counters[kind] = nxt + 1
        return nxt

    def mint_many(self, kind: EntityKind, count: int) -> range:
        start = self._counters[kind]
        self._counters[kind] = start + count
        return range(start, start + count)

    def issued(self, kind: EntityKind) -> int:
        return self._counters[kind]
