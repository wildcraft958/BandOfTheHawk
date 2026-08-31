"""Configuration entry points.

    python -m fraudsim.settings.cli show        resolved values with their origin
    python -m fraudsim.settings.cli provenance  the origin table alone
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..calibration.artifact import FittedParams
from .simulation import resolve

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "simulation.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "fitted_params.json"


def _load(args: argparse.Namespace):
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    if artifact is None:
        print(f"no artifact at {args.artifact}; showing configured values only")
        print("run: python -m fraudsim.calibration.cli fit\n")
    return resolve(args.config, artifact=artifact)


def cmd_show(args: argparse.Namespace) -> int:
    resolved = _load(args)
    config = resolved.config
    print(resolved.render())
    print()
    print("population")
    print(f"  holders               {config.population.n_holders:,}")
    print(f"  merchants             {config.population.merchants.count:,}")
    print(f"  fingerprints          {config.population.resolved_fingerprint_count():,}")
    print(f"  device household mean {config.population.devices.household_mean}")
    print(f"  fan-out exponent      {config.population.fanout.exponent}")
    print()
    print("behaviour")
    print(f"  amount median target  {config.behavior.amount.tail_threshold:,.0f} splice")
    print(f"  tail index            {config.behavior.amount.tail_index:.3f}")
    print(f"  arrival model         {config.behavior.arrival.model}")
    print(f"  circadian components  {len(config.behavior.circadian.means)}")
    print()
    print("engine")
    print(f"  windows               {config.engine.windows.windows_seconds}")
    print(f"  compound criteria     {config.engine.windows.compound_criteria}")
    print(f"  compound features     {config.engine.windows.n_compound_features}")
    print(f"  fraud base rate       {config.engine.fraud_base_rate}")
    return 0


def cmd_provenance(args: argparse.Namespace) -> int:
    resolved = _load(args)
    print(resolved.ledger.table())
    grouped = resolved.ledger.by_origin()
    for origin, paths in grouped.items():
        if not paths:
            continue
        print(f"\n{origin.value}")
        for path in paths:
            print(f"  {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.settings")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="resolved values with their origin")
    show.set_defaults(func=cmd_show)

    provenance = subparsers.add_parser("provenance", help="the origin table alone")
    provenance.set_defaults(func=cmd_provenance)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
