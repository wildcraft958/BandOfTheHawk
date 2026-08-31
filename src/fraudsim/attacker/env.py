"""The environment the policy acts in.

The simulator is the world; this wraps it into the loop a policy trains against.
An episode places one adversarial actor against a target, and each step turns the
policy's chosen action into an `Action`, resolves it, and returns the reward and
the next observation. Benign traffic and prevalence are the runner's concern, not
this — here the concern is one attacker's trajectory, which is what PPO learns
from.

The observation is exactly what the design permits a policy to see: the stage,
the legal-action mask, and a feature mapping. Never the graph. The encoder turns
that into a fixed vector, and its width is reported so the networks match it.

**Stealth is resolved here, not chosen here.** The policy emits a posture — ride
an aged binding, cool off, rotate cards — and this module turns it into the
concrete `device_id`, `delay` and `target_id` the simulator needs. That
resolution reads the graph; the policy does not. The distinction matters and is
easy to misread as a leak: an attacker who has taken over an account knows which
of its devices is the familiar one without being handed the bank's device table,
and the abstraction is what keeps the policy honest while letting the behaviour
exist at all. Before this, no posture was expressible: every authorisation went
through whatever binding the world preferred, which is the newest, so every
attack announced itself with a device minutes old.

Reward is the attacker's own: value extracted, minus the cost of every action,
minus a penalty when the defender flags the event. A small shaping term rewards
advancing a stage, because without it the reward is flat until the first
cash-out and a policy from a cold start learns nothing before it gives up. The
shaping is a free parameter and never presented as empirical; it is kept small
against realised value so it guides rather than dominates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.actions import ACTION_ORDER, Action
from ..engine.outcome import Outcome, OutcomeCode
from ..engine.simulator import Actor, ActorKind, Simulator
from ..ids import CardId
from ..engine.stages import Stage, StageGate
from ..protocols import ActorObservation
from .nets import (
    STEALTH_AGED,
    STEALTH_AGED_COOL,
    STEALTH_ROTATE,
    squash_amount,
    squash_delay,
)

_STAGE_INDEX = {s: i for i, s in enumerate(Stage)}
N_STAGES = len(Stage)

# The feature keys the encoder reads, in a fixed order so the vector is stable.
#
# The last four were added with the stealth head. A policy that can choose
# between a loud and a quiet route needs to know whether loud is working, and
# without any feedback on its own detection history the choice is unlearnable —
# both postures look identical from inside the observation. Every one of these is
# a fact the attacker itself holds: how often its own actions came back refused,
# whether the last one did, how much of its dump is unspent, and how long it has
# been at this. None of them is a bank fact and none comes from the graph.
_FEATURE_KEYS = (
    "actions_taken",
    "value_extracted",
    "now_minutes",
    "flags_so_far",
    "last_action_flagged",
    "cards_remaining",
    "hours_elapsed",
    "hour_of_day",
)

# How long "cool off" waits, at minimum. Long enough to read as a separate
# session rather than a continuation of the same one, and deliberately not a
# whole number of days: a floor of exactly twenty-four hours lands the action at
# the same hour it would have run at anyway, so the wait would buy separation in
# the velocity windows while leaving the time-of-day tell exactly where it was.
from ..clock import MINUTES_PER_DAY

COOL_OFF_MINUTES = 20 * 60


@dataclass(slots=True)
class RewardWeights:
    """Pure design parameters (SINK 9). Tuned until the simulation behaves.

    Never presented as empirical. The detection penalty and the stage-advance
    shaping are the two that most change behaviour; both are kept modest against
    realised value so the policy optimises extraction, not the shaping.
    """

    # These were mis-balanced in a way that made the optimal policy inaction: a
    # three-hundred-dollar fraud scaled to three points of reward while being
    # caught cost five, so attempting an attack had negative expected value once
    # the defender was any good, and the policy correctly learned to stop trying.
    # Extraction now outweighs a single detection, so pressing an attack is worth
    # it when it works and the penalty shapes *how* rather than *whether*.
    detection_penalty: float = 2.0
    failed_action_penalty: float = 0.25
    stage_bonus: float = 1.0
    terminal_bonus: float = 10.0
    value_scale: float = 0.05  # a $300 fraud is worth 15, against a 2.0 detection cost


class AttackEnv:
    """One attacker episode against the simulator.

    Constructed per episode with a target and a policy-facing view of the world.
    `reset` places the actor; `step` takes the policy's raw outputs and resolves
    them; `encode` turns an observation into the vector the networks read.
    """

    def __init__(
        self,
        simulator: Simulator,
        target,
        weights: RewardWeights | None = None,
        actor_id: int | None = None,
    ) -> None:
        self.sim = simulator
        self.target = target
        self.weights = weights or RewardWeights()
        self.gate = StageGate()
        self.actor_id = actor_id if actor_id is not None else _fresh_actor_id()
        self._actor: Actor | None = None
        self._steps = 0
        self._stuck = 0
        # Detection history, so the policy can tell whether its current posture
        # is working. Attacker-side facts only: these count refusals it saw.
        self._flags = 0
        self._last_flagged = False
        self._started_at = 0
        # Which card of the dump is currently being worked. Rotation moves this
        # along; everything else leaves it alone.
        self._card_index = 0

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> ActorObservation:
        self._actor = self.sim.register_actor(
            Actor(
                actor_id=self.actor_id,
                kind=ActorKind.ADVERSARIAL,
                holder_id=self.target.holder_id,
                # The whole dump, not just the primary. Nothing in the engine
                # reads this — resolvers take the card off the action — but an
                # actor whose record disagrees with what it is actually spending
                # is the kind of quiet inconsistency that costs an afternoon
                # later.
                cards=[CardId(c) for c in self._dump],
                stage=Stage.NONE,
            )
        )
        self.sim.open_episode(self.actor_id)
        self._steps = 0
        self._stuck = 0
        self._flags = 0
        self._last_flagged = False
        self._started_at = self.sim.clock.now
        self._card_index = 0
        return self._observe()

    def close(self) -> int:
        return self.sim.close_episode(self.actor_id)

    # ------------------------------------------------------------------ dump

    @property
    def _dump(self) -> tuple[int, ...]:
        """The cards this episode may work, in order.

        A target carries a small set because a dump does. Older targets carry
        one, and the fallback keeps those working unchanged.
        """
        cards = getattr(self.target, "card_ids", None)
        if cards:
            return tuple(int(c) for c in cards)
        return (int(self.target.card_id),)

    @property
    def active_card(self) -> int:
        cards = self._dump
        return cards[self._card_index % len(cards)]

    # -------------------------------------------------------------------- step

    def step(
        self,
        action_idx: int,
        amount_raw: float,
        delay_raw: float,
        stealth_idx: int = 0,
    ):
        """Resolve one policy decision, return (obs, reward, done, outcome).

        The discrete index picks the action; the continuous outputs are squashed
        into an amount and a delay and attached; the stealth index decides how
        the action is carried. Reward folds in realised value, the action's cost,
        a detection penalty, and the stage-advance shaping.

        `stealth_idx` defaults to the loud posture so a caller written before the
        modifier head existed still behaves exactly as it did.
        """
        actor = self._actor
        before_stage = actor.stage
        name = ACTION_ORDER[action_idx]

        amount = float(squash_amount(_as_tensor(amount_raw)).item())
        delay = int(float(squash_delay(_as_tensor(delay_raw)).item()))

        # Rotation happens before the action, since it changes what the action
        # acts on. Applied here rather than inside the resolver because moving to
        # another card in the dump is the attacker's decision, not the world's.
        if stealth_idx == STEALTH_ROTATE and len(self._dump) > 1:
            self._card_index += 1

        card_id = self.active_card

        if stealth_idx == STEALTH_AGED_COOL:
            delay = max(delay, COOL_OFF_MINUTES)

        action = Action(
            name=name,
            target_id=card_id,
            secondary_id=self._merchant(),
            device_id=self._device_for(stealth_idx, card_id),
            amount=amount,
            delay_minutes=delay,
            entry_mode=self._entry_mode(stealth_idx),
        )

        outcome = self.sim.step(self.actor_id, action)
        self._steps += 1

        # Detection history, for the observation. A refusal is the only signal
        # the attacker gets about whether the current posture is working.
        flagged = outcome.code in (
            OutcomeCode.BLOCKED,
            OutcomeCode.HELD,
            OutcomeCode.DECLINED,
        )
        self._last_flagged = flagged
        self._flags += int(flagged)

        # A step that neither advanced the stage nor extracted value nor was even
        # legal is non-progress. A run of them is the policy stuck on a
        # legal-but-failing action, exploiting the action cap; the episode ends
        # so that pattern cannot farm reward, and the sequence log shows it.
        progressed = (
            _STAGE_INDEX[actor.stage] > _STAGE_INDEX[before_stage]
            or outcome.value_extracted > 0
        )
        self._stuck = 0 if progressed else self._stuck + 1

        reward = self._reward(outcome, before_stage, actor.stage)
        done = self._done(actor.stage)
        return self._observe(), reward, done, outcome

    # ----------------------------------------------------------- stealth

    def _device_for(self, stealth_idx: int, card_id: int) -> int | None:
        """The binding an aged posture routes through, or None for the default.

        Returning None hands the choice back to the simulator, which prefers the
        newest binding — the loud posture, and what every action did before this
        head existed. The aged postures name the card's oldest surviving binding
        instead, which is the point: an account takeover spends through the
        device the victim has always used, and a detector leaning on device
        novelty has nothing to see.

        Blocklisted devices are skipped because the simulator would refuse them
        anyway; naming one would spend the step on a guaranteed failure.
        """
        if stealth_idx not in (STEALTH_AGED, STEALTH_AGED_COOL):
            return None
        graph = self.sim.graph
        try:
            devices = graph.devices_of_card(CardId(card_id))
        except (KeyError, TypeError):
            return None
        usable = [d for d in devices if not graph.devices[d].blocklisted]
        if not usable:
            return None
        return int(min(usable, key=lambda d: graph.devices[d].first_seen_ts))

    @staticmethod
    def _entry_mode(stealth_idx: int) -> int:
        """How the transaction presents itself.

        This was drawn from an RNG keyed on the step counter, which meant the
        policy had no say in one of the fields a detector reads. An aged posture
        presents as a chip transaction, the ordinary in-person mode; the loud one
        presents as card-not-present, which is what a fresh remote binding
        actually looks like.
        """
        return 1 if stealth_idx in (STEALTH_AGED, STEALTH_AGED_COOL) else 0

    def _reward(self, outcome: Outcome, before: Stage, after: Stage) -> float:
        w = self.weights
        reward = outcome.value_extracted * w.value_scale - outcome.cost

        # Detection penalty: any flag heavier than an approval is a cost, since a
        # flagged action is one a real attacker would rather not have taken.
        if outcome.code in (OutcomeCode.BLOCKED, OutcomeCode.HELD, OutcomeCode.DECLINED):
            reward -= w.detection_penalty
        # A small extra cost on an outright failed action, so the policy is
        # nudged off a legal-but-failing choice rather than only stopped when it
        # repeats one to the stuck limit.
        if outcome.code is OutcomeCode.FAILED:
            reward -= w.failed_action_penalty

        # Shaping: advancing a stage is progress toward value, rewarded so the
        # gradient is not flat until the first cash-out.
        if _STAGE_INDEX[after] > _STAGE_INDEX[before]:
            reward += w.stage_bonus

        # The terminal bonus is paid for money, not for a stage label. Two
        # actions reach MONETIZED: an authorisation, which extracts the amount
        # on approval, and a transfer, which moves balance into the laundering
        # pot and extracts nothing until a later cash-out. Paying the bonus on
        # arrival made the transfer worth +10.5 for zero realised value, and the
        # policy duly learned to reach the stage and stop — a strategy that
        # scored well on the reward and extracted nothing the metric could see.
        # Scaling by whether the step actually realised value keeps the shaping
        # pointing at the same quantity the arms race is measured in.
        if (
            after is Stage.MONETIZED
            and before is not Stage.MONETIZED
            and outcome.value_extracted > 0
        ):
            reward += w.terminal_bonus
        return reward

    # How many non-progress steps end an episode. Enough to try a couple of
    # alternatives, not enough to farm a legal-but-failing action to the cap.
    STUCK_LIMIT = 6

    def _done(self, stage: Stage) -> bool:
        if stage is Stage.TERMINAL:
            return True
        if self._stuck >= self.STUCK_LIMIT:
            return True
        episode = self.sim.config.engine.episode
        if self._steps >= episode.max_actions:
            return True
        # The wall-clock cap, which was declared in the configuration and
        # enforced nowhere. Without it the delay head could stretch an episode
        # across months of simulated time — long enough for a brand-new device to
        # age into an ordinary one, and for velocity windows to forget everything
        # that came before. That is not patience, it is stepping outside the
        # window the detector is defined over.
        elapsed_hours = (self.sim.clock.now - self._started_at) / 60.0
        return elapsed_hours >= episode.max_hours

    # ------------------------------------------------------------- observation

    def _observe(self) -> ActorObservation:
        actor = self._actor
        mask = self.gate.legal_mask(actor.stage)
        features = {
            "actions_taken": float(actor.actions_taken),
            "value_extracted": float(actor.value_extracted),
            "now_minutes": float(self.sim.clock.now),
            "flags_so_far": float(self._flags),
            "last_action_flagged": 1.0 if self._last_flagged else 0.0,
            "cards_remaining": float(len(self._dump) - 1 - self._card_index),
            "hours_elapsed": float(self.sim.clock.now - self._started_at) / 60.0,
            # What time it is where the victim is. Without it the delay head is
            # choosing a wait in the dark: benign traffic follows a circadian
            # curve and fraud, having no clock at all, spread itself uniformly
            # across the day — so roughly a third of it landed in hours when
            # genuine volume is near zero. That handed the detector a tell as
            # decisive as device novelty and just as untouchable by any posture,
            # because the policy could not see the clock it was being judged
            # against. A wall clock is not a bank fact; anyone can read one.
            "hour_of_day": float((self.sim.clock.now % MINUTES_PER_DAY) // 60),
        }
        return ActorObservation(
            actor_id=self.actor_id,
            stage=_STAGE_INDEX[actor.stage],
            legal_action_mask=mask.tolist(),
            features=features,
        )

    def _merchant(self) -> int:
        """A merchant from the target's pool, walked as the episode proceeds.

        Offset by the active card so rotating to another card also moves off the
        merchant the previous one was being spent at. Spreading a dump across one
        merchant would be a pattern in its own right.
        """
        pool = self.target.merchants
        return int(pool[(self._steps + self._card_index) % len(pool)])

    # ---------------------------------------------------------------- encoding

    @staticmethod
    def obs_dim() -> int:
        """Width of the encoded observation: stage one-hot + mask + features."""
        from ..engine.actions import N_ACTIONS

        # One more than the key count: the hour is encoded as a sine/cosine
        # pair, so the vector is wider than the mapping it comes from.
        return N_STAGES + N_ACTIONS + len(_FEATURE_KEYS) + 1

    @staticmethod
    def encode(obs: ActorObservation) -> np.ndarray:
        """Observation to a fixed vector.

        Stage one-hot, then the legal-action mask as floats (the policy benefits
        from seeing what it may do, not only having illegal actions masked out),
        then the features in a fixed order with a light normalisation so their
        scales do not swamp the trunk.
        """
        from ..engine.actions import N_ACTIONS

        stage = np.zeros(N_STAGES)
        stage[obs.stage] = 1.0
        mask = np.asarray(obs.legal_action_mask, dtype=float)
        if mask.shape[0] != N_ACTIONS:
            mask = np.zeros(N_ACTIONS)
        feats = np.array(
            [
                obs.features.get("actions_taken", 0.0) / 40.0,
                obs.features.get("value_extracted", 0.0) * 0.001,
                obs.features.get("now_minutes", 0.0) / (1440.0 * 30),
                obs.features.get("flags_so_far", 0.0) / 5.0,
                obs.features.get("last_action_flagged", 0.0),
                obs.features.get("cards_remaining", 0.0) / 4.0,
                obs.features.get("hours_elapsed", 0.0) / 72.0,
                # Cyclic, not linear: hour 23 and hour 0 are adjacent, and a
                # single scaled column puts them at opposite ends so the trunk
                # would have to learn the wrap-around from scratch.
                np.sin(2 * np.pi * obs.features.get("hour_of_day", 0.0) / 24.0),
                np.cos(2 * np.pi * obs.features.get("hour_of_day", 0.0) / 24.0),
            ]
        )
        return np.concatenate([stage, mask, feats]).astype(np.float32)

    @staticmethod
    def mask_vector(obs: ActorObservation) -> np.ndarray:
        from ..engine.actions import N_ACTIONS

        mask = np.asarray(obs.legal_action_mask, dtype=bool)
        if mask.shape[0] != N_ACTIONS:
            return np.zeros(N_ACTIONS, dtype=bool)
        return mask


_ACTOR_COUNTER = [9_000_000]


def _fresh_actor_id() -> int:
    _ACTOR_COUNTER[0] += 1
    return _ACTOR_COUNTER[0]


def _as_tensor(x):
    import torch

    return torch.as_tensor(float(x))
