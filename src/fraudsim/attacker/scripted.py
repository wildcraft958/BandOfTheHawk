"""One scripted policy per vertical.

These are the first thing to drive the action layer, and they earn their place
three times over. They produce the first labelled fraud, so a detector has
positives to learn from. They are the demonstrations the reinforcement-learning
policy is cloned from, so it starts from a working strategy rather than random
exploration. And they are a complete, submittable red team on their own — if the
learned policy never converges, versioned scripts still give the arms-race chart.

Each policy is a small stage machine. It reads only what a policy is allowed to
see — the stage, the legal-action mask, and a feature mapping — and never the
graph or the actor's internal capability flags. It tracks its own progress
through `observe`, exactly as the learned policy will have to, so a script that
cannot be expressed against that boundary is a script the learned policy could
never imitate.

Every choice that could be a constant is a draw instead. A deterministic script
clones into a deterministic policy with no entropy for PPO to explore from, so
amounts, delays and target merchants are sampled. The randomness is the point,
not decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.actions import ACTION_INDEX, Action, ActionName
from ..engine.outcome import Outcome
from ..protocols import ActorObservation

# Quality floors the channel controls enforce, cleared with a margin so a
# well-resourced attacker gets through and a poorly resourced one does not. The
# controls themselves live in config; these are what a script chooses to buy.
VOICE_QUALITY = 0.9  # clears voice_similarity_threshold 0.85
FACE_QUALITY = 0.95  # clears liveness_threshold 0.90


@dataclass(slots=True)
class ScriptState:
    """What a script remembers about its own run.

    The policy cannot read the actor, so anything it needs to condition on it
    records here from the outcomes it is handed. This mirrors the recurrent
    state the learned policy carries.
    """

    steps: int = 0
    acquired: bool = False
    bound: bool = False
    voiced: bool = False
    faced: bool = False
    synth: bool = False
    payee_added_at: int | None = None
    number_controlled: bool = False
    account_controlled: bool = False
    reset_done: bool = False
    support_called: bool = False
    transferred: float = 0.0
    now_minutes: int = 0
    auths: int = 0
    done: bool = False
    auth_budget: int = -1


class ScriptedPolicy:
    """A stage machine over the action space, one per vertical.

    Subclasses fill in the three stage handlers. The base class enforces the
    contract: only legal actions are returned, progress is tracked from
    outcomes, and a terminal or exhausted script yields None so the runner can
    close the episode.
    """

    vertical: str = "base"

    def __init__(self, target, rng: np.random.Generator) -> None:
        # `target` is a small immutable record the runner hands over: which card,
        # merchant pool and account the episode acts against. It is not the
        # graph; it is the equivalent of an attacker knowing a card number.
        self.target = target
        self.rng = rng
        self.state = ScriptState()

    # -------------------------------------------------------- policy contract

    def act(self, obs: ActorObservation) -> Action | None:
        self.state.steps += 1
        self.state.now_minutes = int(obs.features.get("now_minutes", self.state.now_minutes))
        stage = obs.stage
        if stage == 0:  # NONE
            action = self._acquire(obs)
        elif stage == 1:  # ACQUIRED
            action = self._bind(obs)
        elif stage in (2, 3):  # BOUND, MONETIZED
            action = self._monetize(obs)
        else:  # TERMINAL
            action = None
        return self._legal(action, obs)

    def observe(self, outcome: Outcome) -> None:
        """Update memory from what the world reported.

        Stage is authoritative for the coarse transitions; the finer capability
        flags are inferred from which action just succeeded, since the policy
        set them up itself.
        """
        if outcome is None:
            return
        # Stage is 0..4; map the coarse transitions the runner will also see.
        self.state.acquired = self.state.acquired or outcome.stage.value != "none"
        self.state.bound = self.state.bound or outcome.stage.value in ("bound", "monetized")

    # ------------------------------------------------------------- overridden

    def _acquire(self, obs: ActorObservation) -> Action | None:
        raise NotImplementedError

    def _bind(self, obs: ActorObservation) -> Action | None:
        raise NotImplementedError

    def _monetize(self, obs: ActorObservation) -> Action | None:
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers

    def _legal(self, action: Action | None, obs: ActorObservation) -> Action | None:
        """Never return an action the mask forbids.

        A script that proposes an illegal action would resolve to a no-op the
        learned policy would then imitate as a wasted step. Where the intended
        action is masked, fall through to None and let the runner end the run.
        """
        if action is None:
            return None
        idx = ACTION_INDEX[action.name]
        if idx < len(obs.legal_action_mask) and obs.legal_action_mask[idx]:
            return action
        return None

    def _amount(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def _delay(self, lo_min: int, hi_min: int) -> int:
        return int(self.rng.integers(lo_min, hi_min + 1))

    def _merchant(self) -> int:
        return int(self.rng.choice(self.target.merchants))

    def _budget(self, lo: int, hi: int) -> int:
        """How many authorisations this run will make, drawn once and cached.

        A fresh draw each step would let the cap wander, leaving the run neither
        reliably short nor long; drawing once fixes the length up front.
        """
        if self.state.auth_budget < 0:
            self.state.auth_budget = int(self.rng.integers(lo, hi))
        return self.state.auth_budget


# ------------------------------------------------------------------ verticals


class CardTesting(ScriptedPolicy):
    """Buy a batch of stolen cards, bind one, probe with small authorisations.

    The defining behaviour is many low-value attempts in a burst, which is what
    the velocity rules key on and the most common vertical in reality.
    """

    vertical = "card_testing"

    def _acquire(self, obs):
        return Action(name=ActionName.BUY_CREDS, params={"count": 5, "quality": 0.6})

    def _bind(self, obs):
        return Action(name=ActionName.ADD_DEVICE_SELFSERVE, target_id=self.target.card_id)

    def _monetize(self, obs):
        # A run of small probes, then stop. Small amounts on purpose: a tester
        # is checking which cards are live, not extracting value yet.
        if self.state.auths >= self._budget(4, 9):
            return None
        self.state.auths += 1
        return Action(
            name=ActionName.ATTEMPT_AUTH,
            target_id=self.target.card_id,
            secondary_id=self._merchant(),
            amount=self._amount(1.0, 15.0),
            delay_minutes=self._delay(1, 20),
            entry_mode=2,  # card-not-present
        )


class VoiceCloneProvisioning(ScriptedPolicy):
    """Harvest a voice sample, provision the card by phone, then spend.

    The voice quality has to clear the IVR control, so the harvest is not
    optional — an attacker who skips it fails at provisioning.
    """

    vertical = "voice_clone"

    def _acquire(self, obs):
        if not self.state.voiced:
            self.state.voiced = True
            return Action(name=ActionName.HARVEST_VOICE, params={"quality": VOICE_QUALITY})
        return Action(name=ActionName.BUY_CREDS, params={"count": 1, "quality": 0.7})

    def _bind(self, obs):
        if not self.state.voiced:
            self.state.voiced = True
            return Action(name=ActionName.HARVEST_VOICE, params={"quality": VOICE_QUALITY})
        return Action(name=ActionName.CALL_IVR_PROVISION, target_id=self.target.card_id)

    def _monetize(self, obs):
        if self.state.auths >= self._budget(3, 6):
            return None
        self.state.auths += 1
        return Action(
            name=ActionName.ATTEMPT_AUTH,
            target_id=self.target.card_id,
            secondary_id=self._merchant(),
            amount=self._amount(200.0, 900.0),
            delay_minutes=self._delay(30, 240),
        )


class DeepfakeOnboarding(ScriptedPolicy):
    """Assemble a synthetic identity, clear liveness with a deepfake, pass KYC.

    KYC opens an account rather than taking over one, so monetisation here is a
    fresh line the identity itself unlocked.
    """

    vertical = "deepfake_onboarding"

    def _acquire(self, obs):
        if not self.state.synth:
            self.state.synth = True
            return Action(name=ActionName.MAKE_SYNTH_ID)
        self.state.faced = True
        return Action(name=ActionName.HARVEST_FACE, params={"quality": FACE_QUALITY})

    def _bind(self, obs):
        if not self.state.faced:
            self.state.faced = True
            return Action(name=ActionName.HARVEST_FACE, params={"quality": FACE_QUALITY})
        return Action(name=ActionName.SUBMIT_KYC)

    def _monetize(self, obs):
        # A synthetic account has no bound card here, so the value is taken by
        # escalating a limit and transferring. Kept short.
        if self.state.auths >= self._budget(2, 5):
            return None
        self.state.auths += 1
        return Action(
            name=ActionName.ATTEMPT_AUTH,
            target_id=self.target.card_id,
            secondary_id=self._merchant(),
            amount=self._amount(100.0, 500.0),
            delay_minutes=self._delay(60, 720),
        )


class PhishingATO(ScriptedPolicy):
    """Phish a holder, reset the password, rebind a device, take over.

    The recovery chain — reset then rebind in short order — is what the binding
    detector keys on, so the timing between them is the interesting knob.
    """

    vertical = "phishing_ato"

    def _acquire(self, obs):
        return Action(name=ActionName.PHISH_HOLDER, params={"success_rate": 0.35})

    def _bind(self, obs):
        # Reset the password and take over, rather than add a device. The victim
        # already has an aged binding; spending through it leaves no fresh-device
        # signal, which is what makes an account takeover hard to see. The delay
        # decides whether the reset and the first spend fall inside the recovery
        # chain window the binding detector keys on.
        self.state.reset_done = True
        return Action(name=ActionName.RESET_PASSWORD, delay_minutes=self._delay(5, 120))

    def _monetize(self, obs):
        if self.state.auths >= self._budget(3, 7):
            return None
        self.state.auths += 1
        # No device_id given, so the resolver spends through one of the card's
        # existing bindings — the victim's own device.
        return Action(
            name=ActionName.ATTEMPT_AUTH,
            target_id=self.target.card_id,
            secondary_id=self._merchant(),
            amount=self._amount(150.0, 800.0),
            delay_minutes=self._delay(10, 180),
        )


class SimSwapOTP(ScriptedPolicy):
    """Move the number to a new SIM, intercept the OTP, clear the step-up.

    Held out of training as a zero-shot vertical: the defender should catch it
    without ever having trained on it.
    """

    vertical = "sim_swap"

    def _acquire(self, obs):
        return Action(name=ActionName.BUY_CREDS, params={"count": 1, "quality": 0.7})

    def _bind(self, obs):
        if not self.state.number_controlled:
            self.state.number_controlled = True
            return Action(name=ActionName.SIM_SWAP)
        return Action(name=ActionName.ADD_DEVICE_SELFSERVE, target_id=self.target.card_id)

    def _monetize(self, obs):
        if self.state.auths >= self._budget(3, 6):
            return None
        self.state.auths += 1
        return Action(
            name=ActionName.ATTEMPT_AUTH,
            target_id=self.target.card_id,
            secondary_id=self._merchant(),
            amount=self._amount(300.0, 1200.0),
            delay_minutes=self._delay(5, 90),
        )


class MuleLayering(ScriptedPolicy):
    """Add a payee, wait out the cooling-off, transfer, then launder and cash out.

    The cooling-off is the whole shape of this vertical: the payee cannot
    receive money immediately, so the script has to wait, which spreads the
    episode across real time.
    """

    vertical = "mule_layering"

    def _acquire(self, obs):
        return Action(name=ActionName.BUY_CREDS, params={"count": 1, "quality": 0.7})

    def _bind(self, obs):
        return Action(name=ActionName.ADD_DEVICE_SELFSERVE, target_id=self.target.card_id)

    def _monetize(self, obs):
        if self.state.payee_added_at is None:
            self.state.payee_added_at = self.state.now_minutes
            return Action(name=ActionName.ADD_PAYEE)
        waited = self.state.now_minutes - self.state.payee_added_at
        cooling = 24 * 60
        if waited < cooling:
            # Wait past the cooling-off before attempting the transfer.
            return Action(
                name=ActionName.TRANSFER_P2P,
                amount=self._amount(200.0, 1500.0),
                delay_minutes=cooling - waited + self._delay(10, 120),
            )
        if self.state.transferred == 0.0:
            amount = self._amount(200.0, 1500.0)
            self.state.transferred = amount
            return Action(name=ActionName.TRANSFER_P2P, amount=amount)
        if self.state.done:
            return None
        # Move the balance out.
        self.state.done = True
        return Action(
            name=ActionName.CASH_OUT,
            amount=self.state.transferred,
            params={"haircut": 0.35},
        )


class FriendlyFraud(ScriptedPolicy):
    """Spend normally, then dispute the settled charge as unauthorised.

    The dispute carries generated text, so this is one of the verticals the
    generative layer renders for. The behaviour looks ordinary until the
    dispute, which is what makes it hard.
    """

    vertical = "friendly_fraud"

    def _acquire(self, obs):
        return Action(name=ActionName.BUY_CREDS, params={"count": 1, "quality": 0.7})

    def _bind(self, obs):
        return Action(name=ActionName.ADD_DEVICE_SELFSERVE, target_id=self.target.card_id)

    def _monetize(self, obs):
        if self.state.done:
            return None
        if not self.state.transferred:
            self.state.transferred = 1.0  # marker: a purchase was made
            return Action(
                name=ActionName.ATTEMPT_AUTH,
                target_id=self.target.card_id,
                secondary_id=self._merchant(),
                amount=self._amount(80.0, 600.0),
                delay_minutes=self._delay(30, 300),
            )
        self.state.done = True
        return Action(name=ActionName.FILE_DISPUTE, target_id=self.target.card_id)


class SupportSocialEngineering(ScriptedPolicy):
    """Open a support ticket under a pretext, escalate a limit, spend.

    The ticket carries generated text. Escalating the limit is what the ticket
    is for, and it changes the card so a later authorisation can exceed what it
    could before.
    """

    vertical = "support_se"

    def _acquire(self, obs):
        return Action(name=ActionName.BUY_CREDS, params={"count": 1, "quality": 0.7})

    def _bind(self, obs):
        return Action(name=ActionName.ADD_DEVICE_SELFSERVE, target_id=self.target.card_id)

    def _monetize(self, obs):
        if not self.state.support_called:
            self.state.support_called = True
            return Action(name=ActionName.OPEN_TICKET)
        if self.state.auths >= self._budget(3, 6):
            return None
        self.state.auths += 1
        return Action(
            name=ActionName.ATTEMPT_AUTH,
            target_id=self.target.card_id,
            secondary_id=self._merchant(),
            amount=self._amount(400.0, 1500.0),
            delay_minutes=self._delay(20, 180),
        )


class RefundAbuse(ScriptedPolicy):
    """Buy, then claim the item never arrived and request a refund.

    The refund claim carries generated evidence, so this is the third text
    vertical, and it is the second zero-shot holdout — a generalisation test
    across the text modality as well as to an unseen attack.
    """

    vertical = "refund_abuse"

    def _acquire(self, obs):
        return Action(name=ActionName.BUY_CREDS, params={"count": 1, "quality": 0.7})

    def _bind(self, obs):
        return Action(name=ActionName.ADD_DEVICE_SELFSERVE, target_id=self.target.card_id)

    def _monetize(self, obs):
        if self.state.done:
            return None
        if not self.state.transferred:
            self.state.transferred = 1.0
            return Action(
                name=ActionName.ATTEMPT_AUTH,
                target_id=self.target.card_id,
                secondary_id=self._merchant(),
                amount=self._amount(60.0, 400.0),
                delay_minutes=self._delay(30, 300),
            )
        self.state.done = True
        return Action(name=ActionName.REQUEST_REFUND, target_id=self.target.card_id)


# The registry the runner samples from. Zero-shot holdouts are marked so the
# orchestrator can exclude them from training while still evaluating on them.
VERTICALS: dict[str, type[ScriptedPolicy]] = {
    p.vertical: p
    for p in (
        CardTesting,
        VoiceCloneProvisioning,
        DeepfakeOnboarding,
        PhishingATO,
        SimSwapOTP,
        MuleLayering,
        FriendlyFraud,
        SupportSocialEngineering,
        RefundAbuse,
    )
}

ZERO_SHOT_HOLDOUTS: frozenset[str] = frozenset({"sim_swap", "refund_abuse"})


def build_policy(vertical: str, target, rng: np.random.Generator) -> ScriptedPolicy:
    return VERTICALS[vertical](target, rng)
