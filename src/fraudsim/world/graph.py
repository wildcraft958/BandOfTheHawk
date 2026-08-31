"""The entity graph.

Node stores are id-keyed dicts; edges are payload dicts keyed by an endpoint
tuple, plus adjacency indices in both directions so every traversal the event
builder needs is a set lookup rather than a scan.

The indices are redundant state, which is the one real hazard here. Every
mutation therefore goes through a method on this class, never by touching an
index directly, and `check_invariants` re-derives the indices from the payload
stores to prove they agree.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from ..ids import AccountId, BucketId, CardId, DeviceId, HolderId, MerchantId, PayeeId
from .edges import AddedEdge, ProvisionedEdge, TransactsEdge, UsedByEdge
from .entities import Account, Card, Cardholder, Device, FingerprintBucket, Merchant, Payee

_EMPTY_CARDS: frozenset[CardId] = frozenset()
_EMPTY_DEVICES: frozenset[DeviceId] = frozenset()


# The adjacency indices are keyed by NewType ids, whose Mapping key type is
# invariant, so the comparison helper is generic rather than pinned to int.
_K = TypeVar("_K")
_V = TypeVar("_V", bound=int)


class GraphInvariantError(RuntimeError):
    """Raised when an adjacency index disagrees with the edge payloads."""


class EntityGraph:
    """Mutable world state. Only the simulator is permitted to hold one."""

    __slots__ = (
        "_accounts_of_device",
        "_accounts_of_holder",
        "_cards_of_device",
        "_cards_of_holder",
        "_devices_of_account",
        "_devices_of_bucket",
        "_devices_of_card",
        "_merchants_of_card",
        "_payees_of_account",
        "accounts",
        "added",
        "buckets",
        "cards",
        "devices",
        "holders",
        "merchants",
        "payees",
        "provisioned",
        "transacts",
        "used_by",
    )

    def __init__(self) -> None:
        self.holders: dict[HolderId, Cardholder] = {}
        self.cards: dict[CardId, Card] = {}
        self.devices: dict[DeviceId, Device] = {}
        self.buckets: dict[BucketId, FingerprintBucket] = {}
        self.accounts: dict[AccountId, Account] = {}
        self.merchants: dict[MerchantId, Merchant] = {}
        self.payees: dict[PayeeId, Payee] = {}

        self.provisioned: dict[tuple[CardId, DeviceId], ProvisionedEdge] = {}
        self.added: dict[tuple[AccountId, PayeeId], AddedEdge] = {}
        self.transacts: dict[tuple[CardId, MerchantId], TransactsEdge] = {}
        self.used_by: dict[tuple[DeviceId, AccountId], UsedByEdge] = {}

        self._cards_of_holder: dict[HolderId, set[CardId]] = {}
        self._accounts_of_holder: dict[HolderId, set[AccountId]] = {}
        self._devices_of_card: dict[CardId, set[DeviceId]] = {}
        self._cards_of_device: dict[DeviceId, set[CardId]] = {}
        self._devices_of_bucket: dict[BucketId, set[DeviceId]] = {}
        self._payees_of_account: dict[AccountId, set[PayeeId]] = {}
        self._merchants_of_card: dict[CardId, set[MerchantId]] = {}
        self._accounts_of_device: dict[DeviceId, set[AccountId]] = {}
        self._devices_of_account: dict[AccountId, set[DeviceId]] = {}

    # ---------------------------------------------------------------- nodes

    def add_holder(self, holder: Cardholder) -> None:
        self.holders[holder.holder_id] = holder
        self._cards_of_holder.setdefault(holder.holder_id, set())
        self._accounts_of_holder.setdefault(holder.holder_id, set())

    def add_card(self, card: Card) -> None:
        if card.holder_id not in self.holders:
            raise KeyError(f"card {card.card_id} references unknown holder {card.holder_id}")
        self.cards[card.card_id] = card
        self._cards_of_holder[card.holder_id].add(card.card_id)
        self._devices_of_card.setdefault(card.card_id, set())
        self._merchants_of_card.setdefault(card.card_id, set())

    def add_bucket(self, bucket: FingerprintBucket) -> None:
        self.buckets[bucket.bucket_id] = bucket
        self._devices_of_bucket.setdefault(bucket.bucket_id, set())

    def add_device(self, device: Device) -> None:
        if device.bucket_id not in self.buckets:
            raise KeyError(f"device {device.device_id} references unknown bucket")
        self.devices[device.device_id] = device
        self._devices_of_bucket[device.bucket_id].add(device.device_id)
        self._cards_of_device.setdefault(device.device_id, set())
        self._accounts_of_device.setdefault(device.device_id, set())

    def add_account(self, account: Account) -> None:
        if account.holder_id not in self.holders:
            raise KeyError(f"account {account.account_id} references unknown holder")
        self.accounts[account.account_id] = account
        self._accounts_of_holder[account.holder_id].add(account.account_id)
        self._payees_of_account.setdefault(account.account_id, set())
        self._devices_of_account.setdefault(account.account_id, set())

    def add_merchant(self, merchant: Merchant) -> None:
        self.merchants[merchant.merchant_id] = merchant

    def add_payee(self, payee: Payee) -> None:
        self.payees[payee.payee_id] = payee

    # ---------------------------------------------------------------- edges

    def bind_device(self, edge: ProvisionedEdge) -> bool:
        """Make a card usable on a device. False if the binding already exists."""
        key = (edge.card_id, edge.device_id)
        if key in self.provisioned:
            return False
        if edge.card_id not in self.cards:
            raise KeyError(f"unknown card {edge.card_id}")
        if edge.device_id not in self.devices:
            raise KeyError(f"unknown device {edge.device_id}")
        self.provisioned[key] = edge
        self._devices_of_card[edge.card_id].add(edge.device_id)
        self._cards_of_device[edge.device_id].add(edge.card_id)
        return True

    def unbind_device(self, card_id: CardId, device_id: DeviceId) -> bool:
        """Remove a binding. This is the mitigation that shrinks capability."""
        if self.provisioned.pop((card_id, device_id), None) is None:
            return False
        self._devices_of_card[card_id].discard(device_id)
        self._cards_of_device[device_id].discard(card_id)
        return True

    def attach_payee(self, edge: AddedEdge) -> bool:
        key = (edge.account_id, edge.payee_id)
        if key in self.added:
            return False
        if edge.account_id not in self.accounts:
            raise KeyError(f"unknown account {edge.account_id}")
        if edge.payee_id not in self.payees:
            raise KeyError(f"unknown payee {edge.payee_id}")
        self.added[key] = edge
        self._payees_of_account[edge.account_id].add(edge.payee_id)
        return True

    def detach_payee(self, account_id: AccountId, payee_id: PayeeId) -> bool:
        if self.added.pop((account_id, payee_id), None) is None:
            return False
        self._payees_of_account[account_id].discard(payee_id)
        return True

    def record_transaction(
        self, card_id: CardId, merchant_id: MerchantId, amount: float, ts: int
    ) -> TransactsEdge:
        key = (card_id, merchant_id)
        edge = self.transacts.get(key)
        if edge is None:
            if card_id not in self.cards:
                raise KeyError(f"unknown card {card_id}")
            if merchant_id not in self.merchants:
                raise KeyError(f"unknown merchant {merchant_id}")
            edge = TransactsEdge(card_id=card_id, merchant_id=merchant_id, first_ts=ts, last_ts=ts)
            self.transacts[key] = edge
            self._merchants_of_card[card_id].add(merchant_id)
        edge.observe(ts, amount)
        return edge

    def record_device_usage(
        self, device_id: DeviceId, account_id: AccountId, ts: int
    ) -> UsedByEdge:
        key = (device_id, account_id)
        edge = self.used_by.get(key)
        if edge is None:
            if device_id not in self.devices:
                raise KeyError(f"unknown device {device_id}")
            if account_id not in self.accounts:
                raise KeyError(f"unknown account {account_id}")
            edge = UsedByEdge(device_id=device_id, account_id=account_id, first_ts=ts, last_ts=ts)
            self.used_by[key] = edge
            self._accounts_of_device[device_id].add(account_id)
            self._devices_of_account[account_id].add(device_id)
        edge.observe(ts)
        return edge

    # ------------------------------------------------------------- queries

    def has_binding(self, card_id: CardId, device_id: DeviceId) -> bool:
        return (card_id, device_id) in self.provisioned

    def binding(self, card_id: CardId, device_id: DeviceId) -> ProvisionedEdge | None:
        return self.provisioned.get((card_id, device_id))

    def cards_of_device(self, device_id: DeviceId) -> frozenset[CardId]:
        return frozenset(self._cards_of_device.get(device_id, _EMPTY_CARDS))

    def device_card_count(self, device_id: DeviceId) -> int:
        """Device fan-out. The benign distribution here is heavy tailed."""
        return len(self._cards_of_device.get(device_id, _EMPTY_CARDS))

    def devices_of_card(self, card_id: CardId) -> frozenset[DeviceId]:
        return frozenset(self._devices_of_card.get(card_id, _EMPTY_DEVICES))

    def devices_of_bucket(self, bucket_id: BucketId) -> frozenset[DeviceId]:
        return frozenset(self._devices_of_bucket.get(bucket_id, _EMPTY_DEVICES))

    def bucket_card_count(self, bucket_id: BucketId) -> int:
        """Cards reachable through any device sharing a fingerprint.

        This is the quantity a naive reading of the data mistakes for device
        fan-out. Reported for comparison, never used as a mitigation target.
        """
        cards: set[CardId] = set()
        for device_id in self._devices_of_bucket.get(bucket_id, _EMPTY_DEVICES):
            cards |= self._cards_of_device.get(device_id, _EMPTY_CARDS)
        return len(cards)

    def cards_of_holder(self, holder_id: HolderId) -> frozenset[CardId]:
        return frozenset(self._cards_of_holder.get(holder_id, _EMPTY_CARDS))

    def accounts_of_holder(self, holder_id: HolderId) -> frozenset[AccountId]:
        return frozenset(self._accounts_of_holder.get(holder_id, set()))

    def merchants_of_card(self, card_id: CardId) -> frozenset[MerchantId]:
        return frozenset(self._merchants_of_card.get(card_id, set()))

    def payees_of_account(self, account_id: AccountId) -> frozenset[PayeeId]:
        return frozenset(self._payees_of_account.get(account_id, set()))

    def accounts_of_device(self, device_id: DeviceId) -> frozenset[AccountId]:
        return frozenset(self._accounts_of_device.get(device_id, set()))

    def has_transacted(self, card_id: CardId, merchant_id: MerchantId) -> bool:
        return (card_id, merchant_id) in self.transacts

    def fanout_distribution(self) -> list[int]:
        """Cards per device, for comparison against the measured anchor."""
        return [len(cards) for cards in self._cards_of_device.values()]

    def bucket_fanout_distribution(self) -> list[int]:
        return [self.bucket_card_count(bucket_id) for bucket_id in self.buckets]

    # ------------------------------------------------------------ integrity

    def check_invariants(self) -> None:
        """Re-derive every index from the payload stores and compare.

        O(E). Called by tests and after warm start, never on the hot path.
        """
        devices_of_card: dict[CardId, set[DeviceId]] = {c: set() for c in self.cards}
        cards_of_device: dict[DeviceId, set[CardId]] = {d: set() for d in self.devices}
        for card_id, device_id in self.provisioned:
            devices_of_card[card_id].add(device_id)
            cards_of_device[device_id].add(card_id)
        self._compare("devices_of_card", self._devices_of_card, devices_of_card)
        self._compare("cards_of_device", self._cards_of_device, cards_of_device)

        payees: dict[AccountId, set[PayeeId]] = {a: set() for a in self.accounts}
        for account_id, payee_id in self.added:
            payees[account_id].add(payee_id)
        self._compare("payees_of_account", self._payees_of_account, payees)

        merchants: dict[CardId, set[MerchantId]] = {c: set() for c in self.cards}
        for card_id, merchant_id in self.transacts:
            merchants[card_id].add(merchant_id)
        self._compare("merchants_of_card", self._merchants_of_card, merchants)

        accounts_of_device: dict[DeviceId, set[AccountId]] = {d: set() for d in self.devices}
        devices_of_account: dict[AccountId, set[DeviceId]] = {a: set() for a in self.accounts}
        for device_id, account_id in self.used_by:
            accounts_of_device[device_id].add(account_id)
            devices_of_account[account_id].add(device_id)
        self._compare("accounts_of_device", self._accounts_of_device, accounts_of_device)
        self._compare("devices_of_account", self._devices_of_account, devices_of_account)

        for device in self.devices.values():
            if device.device_id not in self._devices_of_bucket.get(device.bucket_id, set()):
                raise GraphInvariantError(
                    f"device {device.device_id} missing from bucket {device.bucket_id}"
                )

    @staticmethod
    def _compare(
        label: str,
        actual: Mapping[_K, set[_V]],
        expected: Mapping[_K, set[_V]],
    ) -> None:
        for key, expected_set in expected.items():
            actual_set = actual.get(key, set())
            if actual_set != expected_set:
                raise GraphInvariantError(
                    f"{label}[{key}]: index holds {sorted(actual_set)}, "
                    f"edges imply {sorted(expected_set)}"
                )

    def summary(self) -> dict[str, int]:
        return {
            "holders": len(self.holders),
            "cards": len(self.cards),
            "devices": len(self.devices),
            "buckets": len(self.buckets),
            "accounts": len(self.accounts),
            "merchants": len(self.merchants),
            "payees": len(self.payees),
            "provisioned": len(self.provisioned),
            "added": len(self.added),
            "transacts": len(self.transacts),
            "used_by": len(self.used_by),
        }
