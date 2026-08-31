"""Configuration entry points.

    python -m fraudsim.settings.cli show        resolved values with their origin
    python -m fraudsim.settings.cli provenance  the origin table alone
"""

from __future__ import annotations

import argparse

from ..logs import emit
from ..cli import base_parser, load_artifact, load_resolved


def _load(args: argparse.Namespace):
    if load_artifact(args) is None:
        emit(f"no artifact at {args.artifact}; showing configured values only")
        emit("run: python -m fraudsim.calibration.cli fit\n")
    return load_resolved(args)


def cmd_show(args: argparse.Namespace) -> int:
    resolved = _load(args)
    config = resolved.config
    emit(resolved.render())
    emit()
    emit("population")
    emit(f"  holders               {config.population.n_holders:,}")
    emit(f"  merchants             {config.population.merchants.count:,}")
    emit(f"  fingerprints          {config.population.resolved_fingerprint_count():,}")
    emit(f"  device household mean {config.population.devices.household_mean}")
    emit(f"  fan-out exponent      {config.population.fanout.exponent}")
    emit()
    emit("behaviour")
    emit(f"  amount median target  {config.behavior.amount.tail_threshold:,.0f} splice")
    emit(f"  tail index            {config.behavior.amount.tail_index:.3f}")
    emit(f"  arrival model         {config.behavior.arrival.model}")
    emit(f"  circadian components  {len(config.behavior.circadian.means)}")
    emit()
    emit("engine")
    emit(f"  windows               {config.engine.windows.windows_seconds}")
    emit(f"  compound criteria     {config.engine.windows.compound_criteria}")
    emit(f"  compound features     {config.engine.windows.n_compound_features}")
    emit(f"  fraud base rate       {config.engine.fraud_base_rate}")
    return 0


def cmd_provenance(args: argparse.Namespace) -> int:
    resolved = _load(args)
    emit(resolved.ledger.table())
    grouped = resolved.ledger.by_origin()
    for origin, paths in grouped.items():
        if not paths:
            continue
        emit(f"\n{origin.value}")
        for path in paths:
            emit(f"  {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fraudsim.settings")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="resolved values with their origin")
    show.set_defaults(func=cmd_show)

    provenance = subparsers.add_parser("provenance", help="the origin table alone")
    provenance.set_defaults(func=cmd_provenance)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
