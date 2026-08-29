"""Detection is a score; mitigation is a graph mutation.

A risk score on its own changes nothing — the attacker's next action still
resolves against the same world. Mitigation is what closes the loop: a scored
event triggers a mutation that deletes an edge or freezes a card, so the
capability the attacker was using no longer exists, and the next action fails at
the level of the world rather than because a rule fired.

This is the mitigation half of the edge symmetry the design puts at its core.
Fraud creates an edge that should not exist; mitigation deletes one or raises
its cost. The four mitigations here each map onto a graph primitive the world
already exposes:

    FreezeCard      -> Card.status = FROZEN, until a horizon
    UnbindDevice    -> graph.unbind_device, the capability itself removed
    DetachPayee     -> graph.detach_payee, the cash-out route removed
    BlocklistDevice -> Device.blocklisted, every card on it refused

They are typed rather than a bag of strings, so the simulator applies exactly
what was decided and an unknown mitigation is a programming error caught at the
boundary rather than a silent no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..world.entities import CardStatus

MINUTES_PER_HOUR = 60


@dataclass(frozen=True, slots=True)
class Mitigation:
    """Base of the typed mitigations. Subclasses carry their own targets."""

    def apply(self, graph, now: int) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FreezeCard(Mitigation):
    """Freeze a card until a horizon, refusing authorisations meanwhile.

    The gentlest mitigation: reversible, and the card's bindings survive. Used
    where the score is high but not certain, so a false positive costs the
    holder a temporary freeze rather than a deleted device.
    """

    card_id: int
    hours: int = 24

    def apply(self, graph, now: int) -> bool:
        card = graph.cards.get(self.card_id)
        if card is None:
            return False
        card.status = CardStatus.FROZEN
        card.frozen_until = now + self.hours * MINUTES_PER_HOUR
        return True


@dataclass(frozen=True, slots=True)
class UnbindDevice(Mitigation):
    """Delete a binding, removing the capability an authorisation ran through.

    This is the sharp one. After it, an authorisation through that device fails
    because the edge is gone — not throttled, not flagged, absent. The attacker
    has to spend actions and cost rebuilding it, which is exactly the loop the
    design is after.
    """

    card_id: int
    device_id: int

    def apply(self, graph, now: int) -> bool:
        return graph.unbind_device(self.card_id, self.device_id)


@dataclass(frozen=True, slots=True)
class DetachPayee(Mitigation):
    """Remove a transfer destination, cutting a cash-out route."""

    account_id: int
    payee_id: int

    def apply(self, graph, now: int) -> bool:
        return graph.detach_payee(self.account_id, self.payee_id)


@dataclass(frozen=True, slots=True)
class BlocklistDevice(Mitigation):
    """Refuse a device outright, across every card it touches.

    The heaviest, and deliberately keyed on a physical device rather than a
    fingerprint bucket. A bucket over-merges unrelated hardware, so blocklisting
    one would take out hundreds of ordinary holders sharing a configuration; a
    device is one household's, so this stays proportionate.
    """

    device_id: int

    def apply(self, graph, now: int) -> bool:
        device = graph.devices.get(self.device_id)
        if device is None:
            return False
        device.blocklisted = True
        return True


def apply_all(mitigations, graph, now: int) -> int:
    """Apply a sequence, returning how many actually changed the world.

    A mitigation whose target is already gone returns False and is not counted,
    so the number reported is real mutations, not attempts.
    """
    return sum(1 for m in mitigations if isinstance(m, Mitigation) and m.apply(graph, now))
