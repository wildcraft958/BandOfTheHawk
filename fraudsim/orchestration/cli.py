"""Episode-runner entry point.

    python -m fraudsim.orchestration.cli run --holders 2000

Builds a benign world, warms it, then runs adversarial episodes to the
configured prevalence and prints what they produced — including the top action
sequences, which are there to be read.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..calibration.artifact import FittedParams
from ..config.simulation import resolve
from ..engine.simulator import Simulator
from ..features.builder import EventBuilder
from ..features.state import FeatureStateStore
from ..population.builder import PopulationBuilder
from ..population.warmstart import WarmStartRunner
from ..protocols import AlwaysApproveScorer
from ..timing.circadian import HolderClockModel
from .coadapt import run_coadapt
from .run import EpisodeRunner

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "simulation.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "fitted_params.json"


def cmd_run(args: argparse.Namespace) -> int:
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    overrides = {"population": {"n_holders": args.holders}} if args.holders else None
    config = resolve(args.config, artifact=artifact, overrides=overrides).config

    started = time.perf_counter()
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(
        graph, states, config.engine.windows, HolderClockModel(config.behavior.circadian)
    )
    simulator = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())

    warm = WarmStartRunner(simulator, config, seed=config.seed).run()

    runner = EpisodeRunner(simulator, config, seed=config.seed + 1, train_only=args.train_only)
    report = runner.run(benign_seed=config.seed + 2)
    elapsed = time.perf_counter() - started

    print(warm.render())
    print()
    print(report.render())
    simulator.graph.check_invariants()
    print(f"\n  graph invariants   {'hold':>10}")
    print(f"  built and run      {elapsed:>10.1f}s")
    return 0


def cmd_coadapt(args: argparse.Namespace) -> int:
    """Warm-start the defender, actor and critic, then live co-adaptation."""
    from ..attacker.ppo import PPOConfig

    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    overrides: dict = {}
    if args.holders:
        overrides["population"] = {"n_holders": args.holders}
    if args.fraud_rate:
        overrides["engine"] = {"fraud_base_rate": args.fraud_rate}
    config = resolve(args.config, artifact=artifact, overrides=overrides or None).config

    ppo_cfg = PPOConfig(
        hidden_dim=args.hidden,
        minibatch_size=args.minibatch,
        bc_kl_anneal_updates=max(1, args.updates // 3),
    )
    report = run_coadapt(
        config,
        seed=config.seed,
        learned_defender=args.learned,
        demo_episodes=args.demo_episodes,
        bc_epochs=args.bc_epochs,
        critic_rollouts=args.critic_rollouts,
        critic_epochs=args.critic_epochs,
        n_updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        refit_every=args.refit_every,
        ppo_config=ppo_cfg,
        pool_path=args.pool,
        cfpb_path=args.cfpb,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(report.render())

    # The same numbers as data, so the curve can be plotted and the attacker
    # strategies inspected without re-running or scraping the log.
    if args.metrics:
        import json

        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\n  metrics written to {args.metrics}")
    for name, where in report.checkpoints.items():
        print(f"  {name} saved to {where}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.orchestration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="warm start, then adversarial episodes to prevalence")
    run.add_argument("--holders", type=int, default=None)
    run.add_argument(
        "--train-only",
        action="store_true",
        help="exclude the zero-shot holdout verticals",
    )
    run.set_defaults(func=cmd_run)

    co = subparsers.add_parser("coadapt", help="warm-start then live attacker/defender co-adaptation")
    co.add_argument("--holders", type=int, default=None)
    co.add_argument("--fraud-rate", type=float, default=None)
    co.add_argument("--learned", action="store_true", help="use the mixture defender")
    co.add_argument("--demo-episodes", type=int, default=300)
    co.add_argument("--bc-epochs", type=int, default=10)
    co.add_argument("--critic-rollouts", type=int, default=48)
    co.add_argument("--critic-epochs", type=int, default=20)
    co.add_argument("--updates", type=int, default=60)
    co.add_argument("--episodes-per-update", type=int, default=48)
    co.add_argument("--refit-every", type=int, default=10)
    co.add_argument("--hidden", type=int, default=256)
    co.add_argument("--minibatch", type=int, default=256)
    co.add_argument("--pool", type=Path, default=ROOT / "artifacts" / "text_pool.json",
                    help="text pool to feed dispute/ticket/refund text and embeddings")
    co.add_argument("--cfpb", type=Path,
                    default=ROOT / "Dataset" / "complaints" / "cfpb_payments_all.parquet")
    co.add_argument("--metrics", type=Path, default=ROOT / "artifacts" / "coadapt_metrics.json",
                    help="write the live curve and attacker sequences as JSON, for plotting")
    co.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts" / "checkpoints",
                    help="where the trained attacker and final defender are written")
    co.set_defaults(func=cmd_coadapt)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
