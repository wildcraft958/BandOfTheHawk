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
from ..engine.stages import Stage, StageGate
from ..protocols import ActorObservation
from .nets import squash_amount, squash_delay

_STAGE_INDEX = {s: i for i, s in enumerate(Stage)}
N_STAGES = len(Stage)

# The feature keys the encoder reads, in a fixed order so the vector is stable.
_FEATURE_KEYS = ("actions_taken", "value_extracted", "now_minutes")


@dataclass(slots=True)
class RewardWeights:
    """Pure design parameters (SINK 9). Tuned until the simulation behaves.

    Never presented as empirical. The detection penalty and the stage-advance
    shaping are the two that most change behaviour; both are kept modest against
    realised value so the policy optimises extraction, not the shaping.
    """

    detection_penalty: float = 5.0
    failed_action_penalty: float = 0.25
    stage_bonus: float = 1.0
    terminal_bonus: float = 10.0
    value_scale: float = 0.01  # realised dollars are large; scale into reward range


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

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> ActorObservation:
        self._actor = self.sim.register_actor(
            Actor(
                actor_id=self.actor_id,
                kind=ActorKind.ADVERSARIAL,
                holder_id=self.target.holder_id,
                cards=[self.target.card_id],
                stage=Stage.NONE,
            )
        )
        self.sim.open_episode(self.actor_id)
        self._steps = 0
        self._stuck = 0
        return self._observe()

    def close(self) -> int:
        return self.sim.close_episode(self.actor_id)

    # -------------------------------------------------------------------- step

    def step(self, action_idx: int, amount_raw: float, delay_raw: float):
        """Resolve one policy decision, return (obs, reward, done, outcome).

        The discrete index picks the action; the continuous outputs are squashed
        into an amount and a delay and attached. Reward folds in realised value,
        the action's cost, a detection penalty, and the stage-advance shaping.
        """
        actor = self._actor
        before_stage = actor.stage
        name = ACTION_ORDER[action_idx]

        amount = float(squash_amount(_as_tensor(amount_raw)).item())
        delay = int(float(squash_delay(_as_tensor(delay_raw)).item()))

        action = Action(
            name=name,
            target_id=self.target.card_id,
            secondary_id=self._merchant(),
            amount=amount,
            delay_minutes=delay,
            entry_mode=int(np.random.default_rng(self._steps).integers(0, 4)),
        )

        outcome = self.sim.step(self.actor_id, action)
        self._steps += 1

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
        if after is Stage.MONETIZED and before is not Stage.MONETIZED:
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
        cap = self.sim.config.engine.episode.max_actions
        return self._steps >= cap

    # ------------------------------------------------------------- observation

    def _observe(self) -> ActorObservation:
        actor = self._actor
        mask = self.gate.legal_mask(actor.stage)
        features = {
            "actions_taken": float(actor.actions_taken),
            "value_extracted": float(actor.value_extracted),
            "now_minutes": float(self.sim.clock.now),
        }
        return ActorObservation(
            actor_id=self.actor_id,
            stage=_STAGE_INDEX[actor.stage],
            legal_action_mask=mask.tolist(),
            features=features,
        )

    def _merchant(self) -> int:
        return int(self.target.merchants[self._steps % len(self.target.merchants)])

    # ---------------------------------------------------------------- encoding

    @staticmethod
    def obs_dim() -> int:
        """Width of the encoded observation: stage one-hot + mask + features."""
        from ..engine.actions import N_ACTIONS

        return N_STAGES + N_ACTIONS + len(_FEATURE_KEYS)

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
