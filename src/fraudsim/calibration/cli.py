"""Calibration entry points.

    python -m fraudsim.calibration.cli split         entity-disjoint halves
    python -m fraudsim.calibration.cli noise-floor   degradation denominators
    python -m fraudsim.calibration.cli fanout        shared-attribute degrees
    python -m fraudsim.calibration.cli taxonomy      categories and demographics
    python -m fraudsim.calibration.cli fit           every fit, writes the artifact
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..logs import emit
from ..paths import ARTIFACT_DIR
from .behavioral import fanout_stats, fraud_rate_by_fanout
from .loaders import IeeeCisLoader, SparkovLoader
from .noise_floor import NoiseFloorBuilder
from .pipeline import run_calibration
from .splits import entity_level_split, row_level_split


def cmd_split(args: argparse.Namespace) -> int:
    frame = IeeeCisLoader().transactions().benign()
    split = entity_level_split(frame, "entity", seed=args.seed)
    emit("entity-level split")
    for key, value in split.summary().items():
        emit(f"  {key:<16} {value}")

    if args.compare_row_split:
        leaky = row_level_split(frame, "entity", seed=args.seed)
        shared = set(leaky.left["entity"].unique()) & set(leaky.right["entity"].unique())
        emit("\nrow-level split, shown only as the counterexample")
        emit(f"  disjoint         {leaky.is_disjoint()}")
        emit(f"  entities in both {len(shared):,}")
        emit("  a shared entity puts its own history on both sides, which drives")
        emit("  the floor towards zero and inflates every ratio measured against it")

    if not split.is_disjoint():
        emit("\nFAIL: split leaked entities")
        return 1
    return 0


def cmd_noise_floor(args: argparse.Namespace) -> int:
    frame = IeeeCisLoader().transactions().benign()
    builder = NoiseFloorBuilder(
        frame, "entity", "TransactionDT", "TransactionAmt", seed=args.seed
    )
    floors = builder.build()
    emit(floors.render())

    out = ARTIFACT_DIR / "noise_floors.json"
    floors.save(out)
    emit(f"\nwrote {out}")

    signal = floors.targets.get("autocorrelation_mean", float("nan"))
    noise = floors.floors.get("autocorrelation_gap", float("nan"))
    if noise > 0:
        emit(
            f"\nautocorrelation target {signal:.4f} sits {signal / noise:.1f}x above its floor "
            f"{noise:.4f}, so a timing model that samples independently is detectable"
        )
    return 0


def cmd_fanout(args: argparse.Namespace) -> int:
    joined = IeeeCisLoader().fingerprint_to_entity()
    benign = joined[joined["isFraud"] == 0]
    stats = fanout_stats(benign, "fingerprint", "entity")

    emit("fingerprint fan-out, benign rows only")
    for key, value in stats.summary().items():
        emit(f"  {key:<20} {value:>12.3f}")
    emit(f"  {'hill_index':<20} {stats.hill_index():>12.3f}")
    emit("\n  a variance-to-mean far above one is out of reach of independent assignment")
    emit("  a hill index near or below one marks the tail as a measurement artefact,")
    emit("  which is why the crowd behind a fingerprint is modelled apart from a device")

    emit("\nfraud rate by fan-out band")
    emit(fraud_rate_by_fanout(joined, "fingerprint", "entity", "isFraud").to_string())
    emit("\n  a flat or falling profile means sharing is ordinary behaviour;")
    emit("  a profile that climbs with degree would mean sharing was stamped in as fraud")

    if args.save:
        out = ARTIFACT_DIR / "noise_floors.json"
        if out.exists():
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["targets"].update(
                {f"fanout_{k}": float(v) for k, v in stats.summary().items()}
            )
            payload["targets"]["fanout_hill_index"] = float(stats.hill_index())
            out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            emit(f"\nappended fan-out targets to {out}")
    return 0


def cmd_taxonomy(args: argparse.Namespace) -> int:
    loader = SparkovLoader()
    mix = loader.cluster_mix()
    emit("category clusters, share of benign transactions")
    for cluster, share in mix.items():
        emit(f"  {cluster:<16} {share:.4f}")

    demographics = loader.demographics()
    ages = demographics["age_years"]
    emit(f"\nholders {len(demographics):,}")
    emit(f"  age      p10 {ages.quantile(0.1):.0f}  p50 {ages.quantile(0.5):.0f}  "
          f"p90 {ages.quantile(0.9):.0f}")
    emit(f"  jobs     {demographics['job'].nunique()} distinct")
    emit("\n  taxonomy and demographics only: this source's geo is an annulus around")
    emit("  each customer and its per-category amounts are inverted, so neither is fitted")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    params = run_calibration(seed=args.seed, include_rejected_hawkes=not args.skip_hawkes)
    emit()
    emit(params.render())
    emit(f"\nwrote {params.save(args.out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split", help="entity-level split of the judge dataset")
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--compare-row-split", action="store_true")
    split.set_defaults(func=cmd_split)

    floors = subparsers.add_parser("noise-floor", help="compute every degradation denominator")
    floors.add_argument("--seed", type=int, default=0)
    floors.set_defaults(func=cmd_noise_floor)

    fanout = subparsers.add_parser("fanout", help="measure shared-attribute fan-out")
    fanout.add_argument("--save", action="store_true")
    fanout.set_defaults(func=cmd_fanout)

    taxonomy = subparsers.add_parser("taxonomy", help="category clusters and demographics")
    taxonomy.set_defaults(func=cmd_taxonomy)

    fit = subparsers.add_parser("fit", help="run every fit and write the artifact")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--out", type=Path, default=ARTIFACT_DIR / "fitted_params.json")
    fit.add_argument("--skip-hawkes", action="store_true")
    fit.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
