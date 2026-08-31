"""Attacker training entry point.

    python -m fraudsim.attacker.cli train --holders 2000 --updates 50

Warms a world, then bootstraps the learned attacker: scripted demonstrations,
behaviour cloning, a critic fit, and PPO against the current (frozen, here the
default-approve) defender. Every scale is a flag, so the same command runs a
smoke test and a real training run; the defaults are set for a real one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..cli import add_scale_flags, base_parser, load_config
from ..population.factory import build_warm_world
from ..protocols import AlwaysApproveScorer, Target
from ..attacker.scripted import VERTICALS, ZERO_SHOT_HOLDOUTS, build_policy
from .bootstrap import bootstrap_and_train
from .env import AttackEnv
from .ppo import PPOConfig, PPOTrainer


class WorldFactory:
    """Hands out env and scripted-policy thunks against one warm world.

    The world is built and warmed once; each episode registers a fresh actor and
    draws a target, so training runs many episodes against a stable population
    rather than rebuilding it each time.
    """

    def __init__(self, config, scorer, train_only: bool = True, seed: int = 0):
        self.config = config
        self.scorer = scorer
        self.rng = np.random.default_rng(seed)
        world = build_warm_world(config, scorer=scorer)
        self.sim = world.simulator
        self._cards = [int(c) for c in world.graph.cards if world.graph.devices_of_card(c)]
        self._verticals = [
            v for v in VERTICALS if not (train_only and v in ZERO_SHOT_HOLDOUTS)
        ]

    def _target(self) -> Target:
        graph = self.sim.graph
        card_id = int(self.rng.choice(self._cards))
        holder_id = int(graph.cards[card_id].holder_id)
        accounts = sorted(graph.accounts_of_holder(holder_id))
        merchants = list(graph.merchants)
        pool = self.rng.choice(merchants, size=min(20, len(merchants)), replace=False)
        return Target(
            card_id=card_id,
            holder_id=holder_id,
            account_id=int(accounts[0]) if accounts else None,
            merchants=tuple(int(m) for m in pool),
        )

    def make_env(self) -> AttackEnv:
        return AttackEnv(self.sim, self._target())

    def make_env_and_policy(self):
        target = self._target()
        env = AttackEnv(self.sim, target)
        vertical = self._verticals[int(self.rng.integers(len(self._verticals)))]
        policy = build_policy(vertical, target, self.rng)
        return env, policy


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args)

    factory = WorldFactory(config, AlwaysApproveScorer(), train_only=True, seed=config.seed)

    ppo_cfg = PPOConfig(
        hidden_dim=args.hidden,
        n_layers=args.layers,
        minibatch_size=args.minibatch,
        bc_kl_anneal_updates=max(1, args.updates // 3),
    )
    trainer = PPOTrainer(AttackEnv.obs_dim(), ppo_cfg)

    report = bootstrap_and_train(
        trainer,
        factory.make_env_and_policy,
        factory.make_env,
        demo_episodes=args.demo_episodes,
        bc_epochs=args.bc_epochs,
        critic_epochs=args.critic_epochs,
        n_updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        seed=config.seed,
    )
    print(report.render())

    if args.out:
        trainer.save(args.out)
        print(f"\n  checkpoint written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fraudsim.attacker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="bootstrap and train the learned attacker")
    add_scale_flags(train)
    # Real-run defaults; a smoke test passes smaller.
    train.add_argument("--demo-episodes", type=int, default=400)
    train.add_argument("--bc-epochs", type=int, default=10)
    train.add_argument("--critic-epochs", type=int, default=20)
    train.add_argument("--updates", type=int, default=60)
    train.add_argument("--episodes-per-update", type=int, default=64)
    train.add_argument("--hidden", type=int, default=256)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--minibatch", type=int, default=256)
    train.add_argument("--out", type=Path, default=None)
    train.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
