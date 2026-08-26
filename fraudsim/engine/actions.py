"""The action space.

Twenty actions, fixed. The count is not incidental: it sets the width of the
legality mask and the shape of any policy head that later chooses among them,
so changing it after those exist means rebuilding both and discarding anything
trained.

One vertical was cut rather than added to. Merchant collusion needs a
settlement and clawback model that the money layer does not implement, and no
available source carries merchant onboarding or settlement data, so simulating
it would have meant inventing every parameter it depends on. It is described
rather than run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ActionName(Enum):
    # Acquiring the means
    PHISH_HOLDER = "phish_holder"
    BUY_CREDS = "buy_creds"
    MAKE_SYNTH_ID = "make_synth_id"
    HARVEST_VOICE = "harvest_voice"
    HARVEST_FACE = "harvest_face"

    # Binding them to something usable
    CALL_IVR_PROVISION = "call_ivr_provision"
    SUBMIT_KYC = "submit_kyc"
    ADD_DEVICE_SELFSERVE = "add_device_selfserve"
    SIM_SWAP = "sim_swap"
    RESET_PASSWORD = "reset_password"
    ADD_PAYEE = "add_payee"
    OPEN_TICKET = "open_ticket"
    ESCALATE_LIMIT = "escalate_limit"

    # Spending
    ATTEMPT_AUTH = "attempt_auth"
    COMPLETE_3DS = "complete_3ds"
    TRANSFER_P2P = "transfer_p2p"
    REQUEST_REFUND = "request_refund"

    # Extracting
    FILE_DISPUTE = "file_dispute"
    CASH_OUT = "cash_out"
    LAUNDER_CHAIN = "launder_chain"


ACTION_ORDER: tuple[ActionName, ...] = tuple(ActionName)
ACTION_INDEX: Mapping[ActionName, int] = {
    name: index for index, name in enumerate(ACTION_ORDER)
}
N_ACTIONS = len(ACTION_ORDER)


# Actions needing rendered content. Most do not: they are decisions about
# amounts and timing, and the content layer is called only where an artifact is
# actually presented to a control.
NEEDS_ARTIFACT: Mapping[ActionName, str] = {
    ActionName.PHISH_HOLDER: "write_phish",
    ActionName.HARVEST_VOICE: "clone_voice",
    ActionName.HARVEST_FACE: "deepfake_selfie",
    ActionName.SIM_SWAP: "pretext",
    ActionName.OPEN_TICKET: "write_ticket",
    ActionName.REQUEST_REFUND: "write_refund_claim",
    ActionName.FILE_DISPUTE: "write_dispute",
}


@dataclass(frozen=True, slots=True)
class Action:
    """A chosen action and its parameters.

    Amount and delay are continuous because they are where the interesting
    decisions live. Which action to take is often forced by the stage; how much
    and how soon are not.
    """

    name: ActionName
    target_id: int | None = None
    secondary_id: int | None = None
    device_id: int | None = None
    amount: float | None = None
    delay_minutes: int = 0
    category_cluster: int | None = None
    entry_mode: int = 0
    params: Mapping[str, float] = field(default_factory=dict)

    @property
    def needs_artifact(self) -> bool:
        return self.name in NEEDS_ARTIFACT

    @property
    def artifact_tool(self) -> str | None:
        return NEEDS_ARTIFACT.get(self.name)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """What an action costs and what it emits."""

    name: ActionName
    cost: float
    emits_event: bool
    description: str


ACTION_SPECS: Mapping[ActionName, ActionSpec] = {
    ActionName.PHISH_HOLDER: ActionSpec(
        ActionName.PHISH_HOLDER, 1.0, True, "contact a holder under a pretext"
    ),
    ActionName.BUY_CREDS: ActionSpec(
        ActionName.BUY_CREDS, 2.0, False, "obtain card details"
    ),
    ActionName.MAKE_SYNTH_ID: ActionSpec(
        ActionName.MAKE_SYNTH_ID, 3.0, False, "assemble an identity"
    ),
    ActionName.HARVEST_VOICE: ActionSpec(
        ActionName.HARVEST_VOICE, 1.5, False, "obtain a voice sample"
    ),
    ActionName.HARVEST_FACE: ActionSpec(
        ActionName.HARVEST_FACE, 1.5, False, "obtain a face sample"
    ),
    ActionName.CALL_IVR_PROVISION: ActionSpec(
        ActionName.CALL_IVR_PROVISION, 3.0, True, "provision a card by phone"
    ),
    ActionName.SUBMIT_KYC: ActionSpec(
        ActionName.SUBMIT_KYC, 3.0, True, "submit identity documents"
    ),
    ActionName.ADD_DEVICE_SELFSERVE: ActionSpec(
        ActionName.ADD_DEVICE_SELFSERVE, 1.0, True, "bind a device in the app"
    ),
    ActionName.SIM_SWAP: ActionSpec(
        ActionName.SIM_SWAP, 5.0, True, "move a number to a new carrier"
    ),
    ActionName.RESET_PASSWORD: ActionSpec(
        ActionName.RESET_PASSWORD, 1.0, True, "reset account credentials"
    ),
    ActionName.ADD_PAYEE: ActionSpec(
        ActionName.ADD_PAYEE, 1.0, True, "register a transfer destination"
    ),
    ActionName.OPEN_TICKET: ActionSpec(
        ActionName.OPEN_TICKET, 2.0, True, "raise a support request"
    ),
    ActionName.ESCALATE_LIMIT: ActionSpec(
        ActionName.ESCALATE_LIMIT, 2.0, True, "request a higher limit"
    ),
    ActionName.ATTEMPT_AUTH: ActionSpec(
        ActionName.ATTEMPT_AUTH, 0.5, True, "authorise a purchase"
    ),
    ActionName.COMPLETE_3DS: ActionSpec(
        ActionName.COMPLETE_3DS, 1.0, True, "answer a step-up challenge"
    ),
    ActionName.TRANSFER_P2P: ActionSpec(
        ActionName.TRANSFER_P2P, 0.5, True, "send money to a payee"
    ),
    ActionName.REQUEST_REFUND: ActionSpec(
        ActionName.REQUEST_REFUND, 2.0, True, "ask a merchant for a refund"
    ),
    ActionName.FILE_DISPUTE: ActionSpec(
        ActionName.FILE_DISPUTE, 2.0, True, "dispute a settled transaction"
    ),
    ActionName.CASH_OUT: ActionSpec(
        ActionName.CASH_OUT, 1.0, True, "convert to an untraceable form"
    ),
    ActionName.LAUNDER_CHAIN: ActionSpec(
        ActionName.LAUNDER_CHAIN, 2.0, True, "move funds through intermediaries"
    ),
}


def action_cost(name: ActionName) -> float:
    """Every action costs something.

    A free action invites a policy to spam it, and the resulting behaviour says
    more about the reward than about fraud.
    """
    return ACTION_SPECS[name].cost
