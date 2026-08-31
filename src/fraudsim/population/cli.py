"""Population entry point.

    python -m fraudsim.population.cli preview
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..calibration.artifact import FittedParams
from ..settings.simulation import resolve
from .builder import PopulationBuilder

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "simulation.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "fitted_params.json"


def cmd_preview(args: argparse.Namespace) -> int:
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    if artifact is None:
        print(f"no artifact at {args.artifact}, using configured values only")
        print("run: python -m fraudsim.calibration.cli fit\n")

    overrides = {"population": {"n_holders": args.holders}} if args.holders else None
    config = resolve(args.config, artifact=artifact, overrides=overrides).config

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
    parser = argparse.ArgumentParser(prog="fraudsim.population")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="build a population and report it")
    preview.add_argument("--holders", type=int, default=None)
    preview.set_defaults(func=cmd_preview)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
