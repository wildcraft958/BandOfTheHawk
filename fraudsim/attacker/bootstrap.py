"""Bootstrapping the learned attacker from the scripted ones.

The order is the whole point. A policy from random weights does not reach a
cash-out often enough to learn anything before it collapses to inaction, so it is
first cloned from the scripted policies, then refined with PPO. This module
collects the demonstrations, runs the clone, fits the critic on the
demonstrations' returns, and then runs PPO updates, logging the top action
sequences each round so an absurd one surfaces as the simulator bug it is rather
than a clever attacker.

The demonstrations are collected through the same environment the learned policy
acts in, so they are recorded as encoded observations and action indices — the
exact form BC needs. A script that read the actor directly could not be cloned;
collecting through the env is what guarantees it can.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..engine.actions import ACTION_INDEX
from .env import AttackEnv, RewardWeights
from .nets import AMOUNT_MAX, AMOUNT_MIN
from .ppo import PPOConfig, PPOTrainer


@dataclass(slots=True)
class Demo:
    """One demonstration step, in the form BC consumes."""

    obs: np.ndarray
    mask: np.ndarray
    action_idx: int
    amount_raw: float
    delay_raw: float


def _inverse_amount_raw(amount: float) -> float:
    """The raw pre-squash value that produces an amount, for cloning targets.

    The env squashes a Gaussian sample through a sigmoid and a log-scale; this
    inverts it so a scripted amount becomes a target the continuous head can be
    regressed onto.
    """
    import math

    amount = min(max(amount, AMOUNT_MIN + 1e-3), AMOUNT_MAX - 1e-3)
    lo, hi = math.log(AMOUNT_MIN), math.log(AMOUNT_MAX)
    unit = (math.log(amount) - lo) / (hi - lo)
    unit = min(max(unit, 1e-4), 1 - 1e-4)
    return math.log(unit / (1 - unit))  # logit


def collect_demos(make_env_and_policy, n_episodes: int, rng: np.random.Generator) -> list[Demo]:
    """Run scripted policies through the env, recording encoded transitions.

    `make_env_and_policy` returns a fresh `(AttackEnv, ScriptedPolicy)` pair. The
    scripted policy chooses actions from the same observation the learned policy
    will see, and each choice is recorded as the vector and index BC trains on.
    """
    demos: list[Demo] = []
    for _ in range(n_episodes):
        env, policy = make_env_and_policy()
        obs = env.reset()
        done = False
        while not done:
            action = policy.act(obs)
            if action is None:
                break
            vec = AttackEnv.encode(obs)
            mask = AttackEnv.mask_vector(obs)
            amount_raw = _inverse_amount_raw(action.amount if action.amount else AMOUNT_MIN + 1)
            # Delay is stored as a raw logit around the sigmoid midpoint; the
            # scripts' delays are coarse, so the continuous head only needs a
            # gentle nudge rather than a precise target.
            delay_raw = 0.0
            demos.append(
                Demo(
                    obs=vec,
                    mask=mask,
                    action_idx=ACTION_INDEX[action.name],
                    amount_raw=amount_raw,
                    delay_raw=delay_raw,
                )
            )
            nxt, _, done, _ = env.step(ACTION_INDEX[action.name], amount_raw, delay_raw)
            policy.observe(env_last_outcome(env))
            obs = nxt
        env.close()
    return demos


def env_last_outcome(env: AttackEnv):
    """The most recent outcome, for feeding a scripted policy's `observe`.

    The scripted policies update their memory from outcomes; during collection
    the env holds the last one implicitly, so this returns a minimal stand-in
    carrying the stage, which is what the scripts read.
    """
    from ..engine.outcome import Outcome, OutcomeCode

    actor = env.sim.actor(env.actor_id)
    return Outcome(code=OutcomeCode.APPROVED, stage=actor.stage)


@dataclass
class TrainReport:
    """What a bootstrap run produced, round by round."""

    bc_losses: list[float]
    critic_losses: list[float]
    update_stats: list[dict]
    mean_return: list[float]
    top_sequences: list[tuple[str, int]]

    def render(self) -> str:
        lines = ["attacker bootstrap"]
        if self.bc_losses:
            lines.append(f"  BC final loss       {self.bc_losses[-1]:>10.4f}")
        if self.critic_losses:
            lines.append(f"  critic final loss   {self.critic_losses[-1]:>10.4f}")
        lines.append("")
        lines.append("  update   policy_loss   entropy   bc_coef   mean_return")
        for i, (st, ret) in enumerate(zip(self.update_stats, self.mean_return)):
            lines.append(
                f"    {i:<7}{st['policy_loss']:>10.3f}"
                f"{st['entropy']:>10.3f}{st['bc_coef']:>10.3f}{ret:>13.3f}"
            )
        lines += ["", "  top action sequences"]
        for seq, count in self.top_sequences[:10]:
            lines.append(f"    {count:>4}  {seq}")
        return "\n".join(lines)


def bootstrap_and_train(
    trainer: PPOTrainer,
    make_env_and_policy,
    make_env,
    demo_episodes: int,
    bc_epochs: int,
    critic_epochs: int,
    n_updates: int,
    episodes_per_update: int,
    seed: int = 0,
) -> TrainReport:
    """The full attacker pipeline: demos, clone, critic fit, PPO.

    Every count is a parameter, so the same function runs a smoke test and a real
    training run. The defaults live at the call sites, not here.
    """
    rng = np.random.default_rng(seed)

    demos = collect_demos(make_env_and_policy, demo_episodes, rng)
    bc_losses = trainer.behaviour_clone(demos, bc_epochs, rng) if demos else []

    warm = trainer.collect(make_env, max(1, episodes_per_update), rng)
    critic_losses = trainer.fit_critic(warm, critic_epochs)

    update_stats: list[dict] = []
    mean_return: list[float] = []
    sequences: Counter[str] = Counter()

    for _ in range(n_updates):
        batch = trainer.collect(make_env, episodes_per_update, rng)
        stats = trainer.update(batch, rng)
        update_stats.append(stats)
        mean_return.append(float(batch.ret.mean().item()))

    # A final greedy rollout pass to log the sequences the trained policy runs.
    sequences = _log_sequences(trainer, make_env, n_episodes=min(50, episodes_per_update), rng=rng)

    return TrainReport(
        bc_losses=bc_losses,
        critic_losses=critic_losses,
        update_stats=update_stats,
        mean_return=mean_return,
        top_sequences=sequences.most_common(10),
    )


def _log_sequences(trainer, make_env, n_episodes: int, rng) -> Counter:
    """Roll out the trained policy and record its action sequences.

    Read, not just logged: an absurd sequence means a simulator exploit the
    policy found, which is a hole a real attacker would find too and worth
    fixing rather than a sign of cleverness.
    """
    import torch

    from ..engine.actions import ACTION_ORDER

    counter: Counter = Counter()
    for _ in range(n_episodes):
        env = make_env()
        obs = env.reset()
        names: list[str] = []
        done = False
        while not done:
            vec = torch.as_tensor(AttackEnv.encode(obs)).unsqueeze(0)
            mask = torch.as_tensor(AttackEnv.mask_vector(obs)).unsqueeze(0)
            with torch.no_grad():
                discrete, amount, delay = trainer.actor(vec, mask)
                a_idx = int(discrete.probs.argmax().item())
                a_amt = float(amount.mean.item())
                a_dly = float(delay.mean.item())
            names.append(ACTION_ORDER[a_idx].value)
            obs, _, done, _ = env.step(a_idx, a_amt, a_dly)
        env.close()
        counter[">".join(names)] += 1
    return counter
