"""Attacker training parameters: PPO, the warm-start, the live loop, rewards.

These lived in four places that disagreed. `PPOConfig` carried fifteen fields of
which only four were reachable from any command line. `orchestration/cli.py` and
`attacker/cli.py` both declared the bootstrap sizes as argparse defaults and gave
different answers: 300 demo episodes against 400, 48 episodes per update against
64. `main.py` held a fourth table of sixty numbers, one row per profile.

Every default here is the value the code used before it moved, so adopting this
module changes no behaviour on its own.

Nothing in this file may import torch. It is plain data on the runtime side of
the import firewall; the attacker tier reads it and builds the network.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .base import PositiveFloat, StrictModel, UnitInterval


class PPOConfig(StrictModel):
    """Optimiser and objective settings for the policy."""

    gamma: UnitInterval = 0.99
    gae_lambda: UnitInterval = 0.95
    clip_eps: PositiveFloat = 0.2
    entropy_coef: Annotated[float, Field(ge=0.0)] = 0.01
    # No value coefficient: the critic has its own optimiser and its loss is
    # backpropagated separately, so there is no combined loss to weight it in.
    max_grad_norm: PositiveFloat = 0.5
    lr_actor: PositiveFloat = 3e-4
    lr_critic: PositiveFloat = 1e-3
    epochs_per_update: Annotated[int, Field(ge=1, le=64)] = 4
    minibatch_size: Annotated[int, Field(ge=8, le=8192)] = 256
    hidden_dim: Annotated[int, Field(ge=16, le=4096)] = 256
    n_layers: Annotated[int, Field(ge=1, le=8)] = 2
    # Behaviour-cloning regularisation, annealed over this many updates.
    bc_kl_coef: Annotated[float, Field(ge=0.0)] = 0.5
    bc_kl_anneal_updates: Annotated[int, Field(ge=1)] = 20
    # "auto" takes the GPU when one is visible and falls back to the CPU when it
    # is not, so CUDA_VISIBLE_DEVICES alone decides where this runs. An explicit
    # "cpu" or "cuda" overrides that.
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    # Seeds torch. Left unset it follows the root seed, which is what makes a
    # run reproducible; torch was previously never seeded at all.
    seed: Annotated[int, Field(ge=0)] | None = None
    # Pins the stealth head to the loud posture, reproducing the policy from
    # before the head existed. This is the control arm of the stealth ablation.
    stealth_frozen: bool = False


class ActionSpaceConfig(StrictModel):
    """The ranges the policy's continuous heads squash into.

    These are the network's output ranges, independent of the engine's own caps
    on what an episode may extract, which is how the network came to be able to
    propose an amount the engine would never allow.
    """

    amount_min: PositiveFloat = 1.0
    amount_max: PositiveFloat = 5000.0
    delay_min_hours: Annotated[float, Field(ge=0.0)] = 0.0
    delay_max_hours: PositiveFloat = 72.0
    n_stealth_postures: Annotated[int, Field(ge=1, le=16)] = 4
    cool_off_hours: Annotated[float, Field(ge=0.0)] = 20.0
    # Consecutive illegal or failed actions before an episode is abandoned.
    stuck_limit: Annotated[int, Field(ge=1)] = 6


class RewardConfig(StrictModel):
    """Pure design parameters (SINK 9). Tuned until the simulation behaves.

    Never presented as empirical. The detection penalty and the stage-advance
    shaping are the two that most change behaviour; both are kept modest against
    realised value so the policy optimises extraction, not the shaping.
    """

    detection_penalty: Annotated[float, Field(ge=0.0)] = 2.0
    failed_action_penalty: Annotated[float, Field(ge=0.0)] = 0.25
    stage_bonus: Annotated[float, Field(ge=0.0)] = 1.0
    terminal_bonus: Annotated[float, Field(ge=0.0)] = 10.0
    value_scale: PositiveFloat = 0.05  # a $300 fraud is worth 15, against 2.0 detection


class BootstrapConfig(StrictModel):
    """Sizes for the three warm-start phases, before anything runs live."""

    demo_episodes: Annotated[int, Field(ge=1)] = 300
    bc_epochs: Annotated[int, Field(ge=1)] = 10
    critic_rollouts: Annotated[int, Field(ge=1)] = 48
    critic_epochs: Annotated[int, Field(ge=1)] = 20


class LoopConfig(StrictModel):
    """The live co-adaptation loop: how long it runs and how often each side moves."""

    updates: Annotated[int, Field(ge=1)] = 60
    episodes_per_update: Annotated[int, Field(ge=1)] = 48
    # The two clocks differ on purpose: an attacker adapts every update, a fraud
    # model retrains periodically, and that asymmetry is the real one.
    refit_every: Annotated[int, Field(ge=1)] = 10
    eval_episodes: Annotated[int, Field(ge=1)] = 12
    # Events sampled when measuring what share of genuine traffic is refused.
    false_positive_sample: Annotated[int, Field(ge=1)] = 400
    benign_target_events: Annotated[int, Field(ge=1)] = 6000
    benign_rounds: Annotated[int, Field(ge=1)] = 3
    # Cards offered to the victim-selection bandit each episode.
    candidates: Annotated[int, Field(ge=1)] = 5
    selection_warmup: Annotated[int, Field(ge=0)] = 10
    # Cards an episode's dump holds. One reproduces the single-card behaviour.
    dump_size: Annotated[int, Field(ge=1)] = 3
    label_latency_minutes: Annotated[int, Field(ge=0)] = 0


class TrainingConfig(StrictModel):
    """Everything the attacker's training reads."""

    ppo: PPOConfig = Field(default_factory=PPOConfig)
    action_space: ActionSpaceConfig = Field(default_factory=ActionSpaceConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)


def seeded(ppo: PPOConfig, root_seed: int) -> PPOConfig:
    """PPO settings that follow the run's root seed unless given their own.

    `training.ppo.seed` is null by default so that one --seed reseeds the whole
    run, torch included. Setting it explicitly pins the network independently of
    the world, which is what an ablation on initialisation would want.
    """
    return ppo if ppo.seed is not None else ppo.model_copy(update={"seed": root_seed})
