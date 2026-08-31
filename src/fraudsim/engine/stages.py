"""The stage machine.

An actor moves through stages as it acquires capability, and each stage permits
only the actions that make sense from it. Spending a card nobody has bound is
not a strategy that fails, it is a thing that cannot happen, and encoding that
structurally keeps a learning policy from spending its capacity rediscovering
it.

Stages advance only on success. A failed provisioning attempt leaves the actor
exactly where it was, holding credentials it still cannot use.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from .actions import ACTION_INDEX, ACTION_ORDER, N_ACTIONS, ActionName


class Stage(Enum):
    NONE = "none"
    ACQUIRED = "acquired"
    BOUND = "bound"
    MONETIZED = "monetized"
    TERMINAL = "terminal"


LEGAL_ACTIONS: dict[Stage, frozenset[ActionName]] = {
    Stage.NONE: frozenset(
        {
            ActionName.PHISH_HOLDER,
            ActionName.BUY_CREDS,
            ActionName.MAKE_SYNTH_ID,
            ActionName.HARVEST_VOICE,
            ActionName.HARVEST_FACE,
        }
    ),
    Stage.ACQUIRED: frozenset(
        {
            ActionName.CALL_IVR_PROVISION,
            ActionName.SUBMIT_KYC,
            ActionName.ADD_DEVICE_SELFSERVE,
            ActionName.ADD_PAYEE,
            ActionName.SIM_SWAP,
            ActionName.RESET_PASSWORD,
            ActionName.HARVEST_VOICE,
            ActionName.HARVEST_FACE,
        }
    ),
    Stage.BOUND: frozenset(
        {
            ActionName.ATTEMPT_AUTH,
            ActionName.COMPLETE_3DS,
            ActionName.TRANSFER_P2P,
            ActionName.REQUEST_REFUND,
            ActionName.ADD_PAYEE,
            ActionName.ESCALATE_LIMIT,
            ActionName.OPEN_TICKET,
        }
    ),
    Stage.MONETIZED: frozenset(
        {
            ActionName.CASH_OUT,
            ActionName.LAUNDER_CHAIN,
            ActionName.FILE_DISPUTE,
            ActionName.ATTEMPT_AUTH,
        }
    ),
    Stage.TERMINAL: frozenset(),
}


# Which successful actions carry an actor forward. Anything absent leaves the
# stage unchanged even when it succeeds.
ADVANCES: dict[tuple[Stage, ActionName], Stage] = {
    (Stage.NONE, ActionName.PHISH_HOLDER): Stage.ACQUIRED,
    (Stage.NONE, ActionName.BUY_CREDS): Stage.ACQUIRED,
    (Stage.NONE, ActionName.MAKE_SYNTH_ID): Stage.ACQUIRED,
    (Stage.ACQUIRED, ActionName.CALL_IVR_PROVISION): Stage.BOUND,
    (Stage.ACQUIRED, ActionName.SUBMIT_KYC): Stage.BOUND,
    (Stage.ACQUIRED, ActionName.ADD_DEVICE_SELFSERVE): Stage.BOUND,
    (Stage.ACQUIRED, ActionName.SIM_SWAP): Stage.BOUND,
    # Account takeover reaches a usable state without minting a device: the
    # attacker resets the password and spends through the victim's own,
    # already-aged binding. Without this transition every attacker had to
    # add a fresh device, which handed the detector a device-age-zero tell
    # that swamped every subtler signal.
    (Stage.ACQUIRED, ActionName.RESET_PASSWORD): Stage.BOUND,
    (Stage.BOUND, ActionName.ATTEMPT_AUTH): Stage.MONETIZED,
    (Stage.BOUND, ActionName.TRANSFER_P2P): Stage.MONETIZED,
}


class StageGate:
    """Decides what an actor may attempt, and where success takes it."""

    __slots__ = ("_masks",)

    def __init__(self) -> None:
        # Precomputed, since a mask is read on every decision.
        self._masks: dict[Stage, np.ndarray] = {}
        for stage, allowed in LEGAL_ACTIONS.items():
            mask = np.zeros(N_ACTIONS, dtype=bool)
            for name in allowed:
                mask[ACTION_INDEX[name]] = True
            self._masks[stage] = mask

    def is_legal(self, stage: Stage, name: ActionName) -> bool:
        return name in LEGAL_ACTIONS[stage]

    def legal_mask(self, stage: Stage) -> np.ndarray:
        """A boolean over the whole action space.

        Built now although nothing consumes it yet. A policy choosing among
        actions needs exactly this, and retrofitting it later would mean
        threading it back through the gate after the gate is settled.
        """
        return self._masks[stage].copy()

    def legal_actions(self, stage: Stage) -> tuple[ActionName, ...]:
        return tuple(name for name in ACTION_ORDER if name in LEGAL_ACTIONS[stage])

    def advance(self, stage: Stage, name: ActionName, succeeded: bool) -> Stage:
        """Where the actor stands after attempting this.

        Failure never advances. An actor that could not provision a card is
        still holding credentials it cannot spend, which is the state the
        attempt was trying to leave.
        """
        if not succeeded:
            return stage
        return ADVANCES.get((stage, name), stage)

    def is_terminal(self, stage: Stage) -> bool:
        return stage is Stage.TERMINAL


def describe_stages() -> str:
    lines = []
    gate = StageGate()
    for stage in Stage:
        actions = gate.legal_actions(stage)
        listed = ", ".join(a.value for a in actions) if actions else "nothing"
        lines.append(f"  {stage.value:<12}{listed}")
    return "\n".join(lines)
