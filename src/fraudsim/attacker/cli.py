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

from ..logs import emit
from ..cli import add_scale_flags, base_parser, load_config, overlay
from ..settings.training import seeded
from ..population.factory import build_warm_world
from ..protocols import AlwaysApproveScorer, Target
from ..attacker.scripted import VERTICALS, ZERO_SHOT_HOLDOUTS, build_policy
from .bootstrap import bootstrap_and_train
from .env import AttackEnv
from .ppo import PPOTrainer


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
        policy = build_policy(vertical, target, self.rng, self.config)
        return env, policy


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args)

    factory = WorldFactory(config, AlwaysApproveScorer(), train_only=True, seed=config.seed)

    boot = overlay(
        config.training.bootstrap,
        demo_episodes=args.demo_episodes, bc_epochs=args.bc_epochs,
        critic_epochs=args.critic_epochs,
    )
    loop = overlay(
        config.training.loop,
        updates=args.updates, episodes_per_update=args.episodes_per_update,
    )
    ppo_cfg = overlay(
        config.training.ppo,
        hidden_dim=args.hidden, n_layers=args.layers, minibatch_size=args.minibatch,
    ).model_copy(update={"bc_kl_anneal_updates": max(1, loop.updates // 3)})
    ppo_cfg = seeded(ppo_cfg, config.seed)
    trainer = PPOTrainer(AttackEnv.obs_dim(), ppo_cfg)

    report = bootstrap_and_train(
        trainer,
        factory.make_env_and_policy,
        factory.make_env,
        demo_episodes=boot.demo_episodes,
        bc_epochs=boot.bc_epochs,
        critic_epochs=boot.critic_epochs,
        n_updates=loop.updates,
        episodes_per_update=loop.episodes_per_update,
        seed=config.seed,
    )
    emit(report.render())

    if args.out:
        trainer.save(args.out)
        emit(f"\n  checkpoint written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fraudsim.attacker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="bootstrap and train the learned attacker")
    add_scale_flags(train)
    # Unset means the configured value; see configs/simulation.yaml training.
    train.add_argument("--demo-episodes", type=int, default=None)
    train.add_argument("--bc-epochs", type=int, default=None)
    train.add_argument("--critic-epochs", type=int, default=None)
    train.add_argument("--updates", type=int, default=None)
    train.add_argument("--episodes-per-update", type=int, default=None)
    train.add_argument("--hidden", type=int, default=None)
    train.add_argument("--layers", type=int, default=None)
    train.add_argument("--minibatch", type=int, default=None)
    train.add_argument("--out", type=Path, default=None)
    train.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
