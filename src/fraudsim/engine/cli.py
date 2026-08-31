"""Simulator entry point.

    python -m fraudsim.engine.cli demo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..paths import DEFAULT_ARTIFACT, DEFAULT_CONFIG
from ..calibration.artifact import FittedParams
from ..settings.simulation import resolve
from ..population.factory import build_warm_world
from ..rules.engine import VelocityRuleEngine
from .stages import describe_stages


def cmd_demo(args: argparse.Namespace) -> int:
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    overrides = {"population": {"n_holders": args.holders}} if args.holders else None
    config = resolve(args.config, artifact=artifact, overrides=overrides).config

    started = time.perf_counter()
    world = build_warm_world(config)
    elapsed = time.perf_counter() - started

    print("stage machine")
    print(describe_stages())
    print()

    events = [event for event in world.simulator.log.events if hasattr(event, "amount")]
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

    world.graph.check_invariants()
    print(f"\n  graph invariants   {'hold':>10}")
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
