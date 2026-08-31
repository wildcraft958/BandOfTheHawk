"""Population entry point.

    python -m fraudsim.population.cli preview
"""

from __future__ import annotations

import argparse
import time

from ..cli import add_scale_flags, base_parser, load_artifact, load_config
from .builder import PopulationBuilder


def cmd_preview(args: argparse.Namespace) -> int:
    if load_artifact(args) is None:
        print(f"no artifact at {args.artifact}, using configured values only")
        print("run: python -m fraudsim.calibration.cli fit\n")

    config = load_config(args)

    started = time.perf_counter()
    graph, report = PopulationBuilder(config).build()
    elapsed = time.perf_counter() - started

    print(report.render())
    print(f"\n  signatures derived   {config.population.resolved_fingerprint_count():>10,}")
    print(f"  built in             {elapsed:>10.1f}s")

    fanout = report.fanout
    ratio = fanout["mean"] / max(fanout["target_mean"], 1e-9)
    print(
        f"\n  fan-out mean is {ratio:.2f}x its target; dispersion "
        f"{fanout['variance_to_mean']:.0f} against a bound of 1 for independent assignment"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fraudsim.population")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="build a population and report it")
    add_scale_flags(preview)
    preview.set_defaults(func=cmd_preview)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
