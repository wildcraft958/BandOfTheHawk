"""What each action actually does to the world.

One resolver per action, registered by name. The alternative, a single method
standing in for every action that is not an authorisation, was what allowed
nineteen actions to report success while changing nothing: the stage advanced,
the actor believed it had gained a capability, and the graph disagreed.

The worst case was a device binding, which emitted an event saying a device had
been bound without creating the edge. The event log and the world then
contradicted each other, and the next authorisation failed for a reason nothing
in the log explained.

Every resolver here either performs its mutation or fails. An action that
cannot do what it claims returns FAILED rather than APPROVED, so a stage never
advances on a capability that was not obtained.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from ..features.schema import EventType
from ..ids import AccountId, CardId, DeviceId, MerchantId, PayeeId
from ..world.edges import AddedEdge, AddMethod, BindMethod, ProvisionedEdge
from ..world.entities import Device, Payee
from .actions import Action, ActionName
from .outcome import Outcome, OutcomeCode

MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440

# Which event each action emits, where it emits one.
EVENT_FOR_ACTION: Mapping[ActionName, EventType] = {
    ActionName.ADD_DEVICE_SELFSERVE: EventType.DEVICE_BIND,
    ActionName.CALL_IVR_PROVISION: EventType.IVR_CALL,
    ActionName.SUBMIT_KYC: EventType.KYC_SUBMIT,
    ActionName.SIM_SWAP: EventType.SIM_CHANGE,
    ActionName.RESET_PASSWORD: EventType.AUTH_RESET,
    ActionName.ADD_PAYEE: EventType.PAYEE_ADD,
    ActionName.OPEN_TICKET: EventType.SUPPORT_TICKET,
    ActionName.ESCALATE_LIMIT: EventType.LIMIT_CHANGE,
    ActionName.TRANSFER_P2P: EventType.TRANSFER,
    ActionName.CASH_OUT: EventType.CASHOUT,
    ActionName.LAUNDER_CHAIN: EventType.TRANSFER,
    ActionName.FILE_DISPUTE: EventType.DISPUTE_FILED,
    ActionName.REQUEST_REFUND: EventType.REFUND_REQUEST,
    ActionName.COMPLETE_3DS: EventType.THREEDS_RESULT,
    ActionName.PHISH_HOLDER: EventType.SUPPORT_TICKET,
}

_REGISTRY: dict[ActionName, Callable] = {}


def resolves(name: ActionName):
    """Attach a resolver to an action."""

    def register(function: Callable) -> Callable:
        _REGISTRY[name] = function
        return function

    return register


def resolver_for(name: ActionName) -> Callable | None:
    return _REGISTRY.get(name)


def registered_actions() -> frozenset[ActionName]:
    return frozenset(_REGISTRY)


class ActionResolver:
    """Applies an action to the world.

    Holds no state of its own. The simulator passes what a resolver needs, so
    the same instance serves every actor.
    """

    def __init__(self, graph, clock, config, rng: np.random.Generator) -> None:
        self.graph = graph
        self.clock = clock
        self.config = config
        self.rng = rng
        self._next_device = 50_000_000
        self._next_payee = 50_000_000

    # ------------------------------------------------------------ helpers

    def _fail(self, actor, cost: float) -> Outcome:
        return Outcome(code=OutcomeCode.FAILED, stage=actor.stage, reward=-cost, cost=cost)

    def _ok(self, actor, cost: float, value: float = 0.0) -> Outcome:
        return Outcome(
            code=OutcomeCode.APPROVED,
            stage=actor.stage,
            reward=value - cost,
            value_extracted=value,
            cost=cost,
        )

    def _holder_accounts(self, actor) -> list[AccountId]:
        if actor.holder_id is None:
            return []
        return sorted(self.graph.accounts_of_holder(actor.holder_id))

    def _primary_account(self, actor) -> AccountId | None:
        accounts = self._holder_accounts(actor)
        return accounts[0] if accounts else None

    def mint_device(self, actor, ts: int) -> DeviceId:
        """A device the actor controls."""
        device_id = DeviceId(self._next_device)
        self._next_device += 1
        bucket_id = int(self.rng.choice(list(self.graph.buckets)))
        household = 0
        if actor.holder_id is not None and actor.holder_id in self.graph.holders:
            household = self.graph.holders[actor.holder_id].household_id
        self.graph.add_device(
            Device(
                device_id=device_id,
                bucket_id=bucket_id,
                first_seen_ts=ts,
                household_id=household,
                os_code=int(self.rng.integers(0, 12)),
                browser_code=int(self.rng.integers(0, 6)),
                app_version=int(self.rng.integers(1, 40)),
                ip_asn=int(self.rng.integers(0, 5000)),
            )
        )
        return device_id

    # --------------------------------------------------- acquiring a means

    @resolves(ActionName.BUY_CREDS)
    def buy_creds(self, actor, action: Action, cost: float) -> Outcome:
        """Obtain card details.

        Credentials are a capability rather than a graph edge, so this records
        what the actor now holds. Without it an actor could reach the binding
        stage having acquired nothing.
        """
        count = int(action.params.get("count", 1))
        quality = float(action.params.get("quality", 0.6))
        actor.credentials.extend([quality] * max(1, count))
        return self._ok(actor, cost)

    @resolves(ActionName.MAKE_SYNTH_ID)
    def make_synth_id(self, actor, action: Action, cost: float) -> Outcome:
        actor.identities += 1
        return self._ok(actor, cost)

    @resolves(ActionName.PHISH_HOLDER)
    def phish_holder(self, actor, action: Action, cost: float) -> Outcome:
        """Contact a holder under a pretext.

        Succeeds sometimes. A contact that always works would let an actor
        treat the first stage as free, and the rate at which people respond is
        the reason phishing is a numbers game.
        """
        if self.rng.random() > float(action.params.get("success_rate", 0.25)):
            return self._fail(actor, cost)
        actor.credentials.append(0.5)
        return self._ok(actor, cost)

    @resolves(ActionName.HARVEST_VOICE)
    def harvest_voice(self, actor, action: Action, cost: float) -> Outcome:
        actor.voice_quality = max(actor.voice_quality, float(action.params.get("quality", 0.0)))
        return self._ok(actor, cost)

    @resolves(ActionName.HARVEST_FACE)
    def harvest_face(self, actor, action: Action, cost: float) -> Outcome:
        actor.face_quality = max(actor.face_quality, float(action.params.get("quality", 0.0)))
        return self._ok(actor, cost)

    # ------------------------------------------------------------- binding

    @resolves(ActionName.ADD_DEVICE_SELFSERVE)
    def add_device(self, actor, action: Action, cost: float) -> Outcome:
        """Bind a device to a card.

        Creates the edge. Emitting the event without it was the sharpest
        inconsistency in the system: the log said a device had been bound while
        the graph had no such edge, so the next authorisation failed for a
        reason nothing in the log explained.
        """
        if action.target_id is None:
            return self._fail(actor, cost)
        card_id = CardId(action.target_id)
        if card_id not in self.graph.cards:
            return self._fail(actor, cost)
        if not actor.credentials:
            # Nothing to bind with.
            return self._fail(actor, cost)

        device_id = self.mint_device(actor, self.clock.now)
        bound = self.graph.bind_device(
            ProvisionedEdge(
                card_id=card_id,
                device_id=device_id,
                bind_ts=self.clock.now,
                bind_method=BindMethod.SELF_SERVICE,
                bind_trust=0.4,
            )
        )
        if not bound:
            return self._fail(actor, cost)
        actor.devices.append(device_id)
        return self._ok(actor, cost)

    @resolves(ActionName.CALL_IVR_PROVISION)
    def call_ivr(self, actor, action: Action, cost: float) -> Outcome:
        """Provision a card by phone.

        The voice sample is checked against the control's threshold. An actor
        that never harvested one has nothing to present and fails, which is the
        whole point of the earlier step.
        """
        if action.target_id is None or not actor.credentials:
            return self._fail(actor, cost)
        threshold = self.config.engine.channel.voice_similarity_threshold
        if actor.voice_quality < threshold:
            return self._fail(actor, cost)

        card_id = CardId(action.target_id)
        if card_id not in self.graph.cards:
            return self._fail(actor, cost)

        device_id = self.mint_device(actor, self.clock.now)
        if not self.graph.bind_device(
            ProvisionedEdge(
                card_id=card_id,
                device_id=device_id,
                bind_ts=self.clock.now,
                bind_method=BindMethod.IVR,
                bind_trust=0.3,
            )
        ):
            return self._fail(actor, cost)
        actor.devices.append(device_id)
        return self._ok(actor, cost)

    @resolves(ActionName.SUBMIT_KYC)
    def submit_kyc(self, actor, action: Action, cost: float) -> Outcome:
        if actor.identities < 1:
            return self._fail(actor, cost)
        if actor.face_quality < self.config.engine.channel.liveness_threshold:
            return self._fail(actor, cost)
        actor.kyc_passed = True
        return self._ok(actor, cost)

    @resolves(ActionName.SIM_SWAP)
    def sim_swap(self, actor, action: Action, cost: float) -> Outcome:
        if not actor.credentials:
            return self._fail(actor, cost)
        actor.controls_number = True
        return self._ok(actor, cost)

    @resolves(ActionName.RESET_PASSWORD)
    def reset_password(self, actor, action: Action, cost: float) -> Outcome:
        if not actor.credentials and not actor.controls_number:
            return self._fail(actor, cost)
        actor.controls_account = True
        return self._ok(actor, cost)

    @resolves(ActionName.ADD_PAYEE)
    def add_payee(self, actor, action: Action, cost: float) -> Outcome:
        """Register a transfer destination.

        The cooling-off period is the point. A payee added now cannot receive
        money immediately, so an actor has to either wait or find another
        route, and a transfer that ignored it would make the control
        decorative.
        """
        account_id = self._primary_account(actor)
        if account_id is None:
            return self._fail(actor, cost)

        payee_id = PayeeId(self._next_payee)
        self._next_payee += 1
        self.graph.add_payee(
            Payee(
                payee_id=payee_id,
                target_account_id=account_id,
                first_added_ts=self.clock.now,
                is_mule=True,
            )
        )
        cooling = self.config.engine.channel.payee_cooling_off_hours * MINUTES_PER_HOUR
        if not self.graph.attach_payee(
            AddedEdge(
                account_id=account_id,
                payee_id=payee_id,
                add_ts=self.clock.now,
                add_method=AddMethod.APP,
                cooling_off_until=self.clock.now + cooling,
            )
        ):
            return self._fail(actor, cost)
        actor.payees.append(payee_id)
        return self._ok(actor, cost)

    @resolves(ActionName.OPEN_TICKET)
    def open_ticket(self, actor, action: Action, cost: float) -> Outcome:
        actor.support_contacts += 1
        return self._ok(actor, cost)

    @resolves(ActionName.ESCALATE_LIMIT)
    def escalate_limit(self, actor, action: Action, cost: float) -> Outcome:
        """Raise a card's limit.

        Changes the card, so a later authorisation can exceed what it could
        before. Without the mutation the action was a no-op an actor could
        repeat for free.
        """
        if action.target_id is None:
            return self._fail(actor, cost)
        card_id = CardId(action.target_id)
        card = self.graph.cards.get(card_id)
        if card is None:
            return self._fail(actor, cost)
        if actor.support_contacts < 1 and not actor.controls_account:
            return self._fail(actor, cost)

        factor = float(action.params.get("factor", 1.5))
        card.credit_line = min(card.credit_line * factor, 250_000.0)
        return self._ok(actor, cost)

    @resolves(ActionName.COMPLETE_3DS)
    def complete_3ds(self, actor, action: Action, cost: float) -> Outcome:
        """Answer a step-up challenge.

        Passing needs the one-time code, which means controlling the number.
        An actor without it fails, which is what makes intercepting the number
        worth the earlier action.
        """
        if not actor.controls_number and not actor.controls_account:
            return self._fail(actor, cost)
        actor.passed_step_up = True
        return self._ok(actor, cost)

    # ---------------------------------------------------------- extracting

    @resolves(ActionName.TRANSFER_P2P)
    def transfer_p2p(self, actor, action: Action, cost: float) -> Outcome:
        """Send money to a payee.

        Moves balance, and refuses where the balance is not there or the payee
        is still cooling off. A transfer that skipped both was extracting value
        from nothing.
        """
        account_id = self._primary_account(actor)
        if account_id is None or not actor.payees:
            return self._fail(actor, cost)

        payee_id = (
            PayeeId(action.secondary_id) if action.secondary_id is not None
            else actor.payees[-1]
        )
        edge = self.graph.added.get((account_id, payee_id))
        if edge is None or edge.is_cooling_off(self.clock.now):
            return self._fail(actor, cost)

        account = self.graph.accounts[account_id]
        amount = float(action.amount or 0.0)
        if amount <= 0 or amount > account.balance:
            return self._fail(actor, cost)

        account.balance -= amount
        actor.laundered += amount
        # Moved, not realised. Counting a transfer as extracted value and then
        # counting the cash-out as well credits the same money twice, which
        # would let an actor inflate its take by adding hops.
        return self._ok(actor, cost)

    @resolves(ActionName.CASH_OUT)
    def cash_out(self, actor, action: Action, cost: float) -> Outcome:
        """Convert to an untraceable form.

        Only what was already moved can be taken out, and a haircut applies,
        because converting stolen funds is never at par. Without either the
        action minted value from nothing.
        """
        amount = float(action.amount or 0.0)
        available = actor.laundered
        if amount <= 0 or available <= 0:
            return self._fail(actor, cost)

        taken = min(amount, available)
        haircut = float(action.params.get("haircut", 0.35))
        actor.laundered -= taken
        return self._ok(actor, cost, value=taken * (1.0 - haircut))

    @resolves(ActionName.LAUNDER_CHAIN)
    def launder_chain(self, actor, action: Action, cost: float) -> Outcome:
        """Move funds through intermediaries.

        Each hop loses a share. The chain buys distance from the source, which
        is what it is for, and pays for it.
        """
        if actor.laundered <= 0:
            return self._fail(actor, cost)
        hops = max(1, int(action.params.get("hops", 2)))
        loss = float(action.params.get("loss_per_hop", 0.05))
        actor.laundered *= (1.0 - loss) ** hops
        actor.launder_hops += hops
        return self._ok(actor, cost)

    @resolves(ActionName.FILE_DISPUTE)
    def file_dispute(self, actor, action: Action, cost: float) -> Outcome:
        """Dispute a settled transaction.

        Needs a transaction to dispute. Filing against nothing was possible
        before and is the kind of thing a policy would learn to repeat.
        """
        if action.target_id is None:
            return self._fail(actor, cost)
        card_id = CardId(action.target_id)
        merchants = self.graph.merchants_of_card(card_id)
        if not merchants:
            return self._fail(actor, cost)

        merchant_id = MerchantId(next(iter(merchants)))
        edge = self.graph.transacts.get((card_id, merchant_id))
        if edge is None or edge.count == 0:
            return self._fail(actor, cost)

        recovered = edge.total_amount / edge.count
        actor.disputes += 1
        return self._ok(actor, cost, value=recovered)

    @resolves(ActionName.REQUEST_REFUND)
    def request_refund(self, actor, action: Action, cost: float) -> Outcome:
        if action.target_id is None:
            return self._fail(actor, cost)
        card_id = CardId(action.target_id)
        merchants = self.graph.merchants_of_card(card_id)
        if not merchants:
            return self._fail(actor, cost)
        edge = self.graph.transacts.get((card_id, MerchantId(next(iter(merchants)))))
        if edge is None or edge.count == 0:
            return self._fail(actor, cost)
        actor.refunds += 1
        return self._ok(actor, cost, value=edge.total_amount / edge.count)

    # ------------------------------------------------------------ dispatch

    def resolve(self, actor, action: Action, cost: float) -> Outcome:
        handler = _REGISTRY.get(action.name)
        if handler is None:
            raise KeyError(f"no resolver registered for {action.name.value}")
        return handler(self, actor, action, cost)
