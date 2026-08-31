"""Simulator entry point.

    python -m fraudsim.engine.cli demo
"""

from __future__ import annotations

import argparse
import time

from ..cli import add_scale_flags, base_parser, load_config
from ..logs import emit
from ..population.factory import build_warm_world
from ..rules.engine import VelocityRuleEngine
from .stages import describe_stages


def cmd_demo(args: argparse.Namespace) -> int:
    config = load_config(args)

    started = time.perf_counter()
    world = build_warm_world(config)
    elapsed = time.perf_counter() - started

    emit("stage machine")
    emit(describe_stages())
    emit()

    events = [event for event in world.simulator.log.events if hasattr(event, "amount")]
    if events:
        stamps = [event.ts for event in events]
        span = (max(stamps) - min(stamps)) / 1440
        emit(f"\n  history spans      {span:>10.0f} days")

        rates = VelocityRuleEngine(config.engine.rules).trigger_rates(events)
        emit()
        negatives = config.behavior.hard_negatives
        emit(
            rates.render(
                target=negatives.naive_rule_trip_target,
                tolerance=negatives.naive_rule_trip_tolerance,
            )
        )

    world.graph.check_invariants()
    emit(f"\n  graph invariants   {'hold':>10}")
    emit(f"  built and warmed   {elapsed:>10.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fraudsim.engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="build a world and fill it with history")
    add_scale_flags(demo)
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
