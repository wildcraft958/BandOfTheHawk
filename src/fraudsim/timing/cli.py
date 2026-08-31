"""Timing entry point.

    python -m fraudsim.timing.cli gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..calibration.artifact import FittedParams
from ..settings.simulation import resolve
from .arrival import DriftingRateProcess, burstiness, lag1_autocorrelation
from .circadian import CircadianClock, resultant_length

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "simulation.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "fitted_params.json"
DEFAULT_FLOORS = ROOT / "artifacts" / "noise_floors.json"


def cmd_gate(args: argparse.Namespace) -> int:
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    config = resolve(args.config, artifact=artifact).config
    arrival = config.behavior.arrival

    process = DriftingRateProcess(arrival)
    rng = np.random.default_rng(args.seed)
    rhos, bursts = [], []
    for _ in range(args.entities):
        state = process.new_state(rng)
        gaps = process.sample_gaps(state, args.events, rng)
        rho, burst = lag1_autocorrelation(gaps), burstiness(gaps)
        if np.isfinite(rho):
            rhos.append(rho)
        if np.isfinite(burst):
            bursts.append(burst)

    observed_rho = float(np.mean(rhos))
    observed_burst = float(np.mean(bursts))

    floors = {}
    if args.floors.exists():
        floors = json.loads(args.floors.read_text(encoding="utf-8")).get("floors", {})

    print(f"timing gate  {args.entities:,} entities, {args.events} events each")
    print(f"  seed {args.seed}, held out from the fit\n")
    print(f"  {'metric':<20}{'target':>10}{'generated':>12}{'ratio':>10}")
    print(f"  {'-' * 20}{'-' * 10}{'-' * 12}{'-' * 10}")

    for name, target, observed, floor_key in (
        ("autocorrelation", arrival.target_autocorrelation, observed_rho, "autocorrelation_gap"),
        ("burstiness", arrival.target_burstiness, observed_burst, "burstiness_gap"),
    ):
        floor = floors.get(floor_key)
        ratio = abs(target - observed) / floor if floor else float("nan")
        print(f"  {name:<20}{target:>+10.4f}{observed:>+12.4f}{ratio:>10.2f}")

    print(f"\n  share of entities with positive correlation  {np.mean(np.array(rhos) > 0):.3f}")
    print("  independent draws cannot exceed zero, so a positive mean is the check")

    clock = CircadianClock(config.behavior.circadian)
    hours = clock.sample_hour(rng, 100_000)
    print(f"\n  circadian resultant {resultant_length(hours):.4f}")

    if observed_rho <= 0:
        print("\nFAIL: generated gaps carry no positive correlation")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.timing")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--floors", type=Path, default=DEFAULT_FLOORS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("gate", help="check generated timing against its targets")
    gate.add_argument("--entities", type=int, default=3000)
    gate.add_argument("--events", type=int, default=30)
    gate.add_argument("--seed", type=int, default=7777)
    gate.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
