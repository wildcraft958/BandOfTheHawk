"""Simulator entry point.

    python -m fraudsim.engine.cli demo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..calibration.artifact import FittedParams
from ..config.simulation import resolve
from ..features.builder import EventBuilder
from ..features.state import FeatureStateStore
from ..population.builder import PopulationBuilder
from ..population.warmstart import WarmStartRunner
from ..protocols import AlwaysApproveScorer
from ..rules.engine import VelocityRuleEngine
from ..timing.circadian import HolderClockModel
from .simulator import Simulator
from .stages import describe_stages

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "simulation.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "fitted_params.json"


def cmd_demo(args: argparse.Namespace) -> int:
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    overrides = {"population": {"n_holders": args.holders}} if args.holders else None
    config = resolve(args.config, artifact=artifact, overrides=overrides).config

    started = time.perf_counter()
    graph, population = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(
        graph, states, config.engine.windows, HolderClockModel(config.behavior.circadian)
    )
    # History is not shaped by a defender that does not exist yet.
    simulator = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
    report = WarmStartRunner(simulator, config, seed=config.seed).run()
    elapsed = time.perf_counter() - started

    print("stage machine")
    print(describe_stages())
    print()
    print(report.render())

    events = [event for event in simulator.log.events if hasattr(event, "amount")]
    if events:
        stamps = [event.ts for event in events]
        span = (max(stamps) - min(stamps)) / 1440
        print(f"\n  history spans      {span:>10.0f} days")

        rates = VelocityRuleEngine(config.engine.rules).trigger_rates(events)
        print()
        negatives = config.behavior.hard_negatives
        print(
            rates.render(
                target=negatives.naive_rule_trip_target,
                tolerance=negatives.naive_rule_trip_tolerance,
            )
        )

    graph.check_invariants()
    print(f"\n  graph invariants   {'hold':>10}")
    print(f"  fan-out mean       {population.fanout['mean']:>10.2f}"
          f"  (target {population.fanout['target_mean']:.2f})")
    print(f"  built and warmed   {elapsed:>10.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.engine")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="build a world and fill it with history")
    demo.add_argument("--holders", type=int, default=None)
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
