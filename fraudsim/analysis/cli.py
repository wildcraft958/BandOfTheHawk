"""Analysis entry points.

    python -m fraudsim.analysis.cli metrics
    python -m fraudsim.analysis.cli compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..calibration.artifact import FittedParams
from ..config.simulation import resolve
from ..engine.simulator import Simulator
from ..features.builder import EventBuilder
from ..features.state import FeatureStateStore
from ..population.builder import PopulationBuilder
from ..population.warmstart import WarmStartRunner
from ..protocols import AlwaysApproveScorer
from .graph_snapshot import GraphSnapshot

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "simulation.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "fitted_params.json"


def _build(args: argparse.Namespace, warm: bool = True):
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    overrides = {"population": {"n_holders": args.holders}} if args.holders else None
    config = resolve(args.config, artifact=artifact, overrides=overrides).config

    graph, _ = PopulationBuilder(config).build()
    if warm:
        states = FeatureStateStore(config.engine.windows)
        builder = EventBuilder(graph, states, config.engine.windows)
        simulator = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
        WarmStartRunner(simulator, config, seed=config.seed).run()
    return graph, config


def cmd_metrics(args: argparse.Namespace) -> int:
    graph, _ = _build(args)
    print(GraphSnapshot(graph).render())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Generated structure beside what was measured on real data."""
    graph, config = _build(args)
    snapshot = GraphSnapshot(graph)
    target = config.population.fanout

    reference = {
        "mean": target.target_mean,
        "share_shared": target.target_share_shared,
        "p99": target.target_p99,
        "variance_to_mean": target.target_variance_to_mean,
    }
    generated = snapshot.fingerprint_card_degrees().as_dict()

    print("shared-signature fan-out")
    print(f"  {'statistic':<20}{'generated':>12}{'measured':>12}{'ratio':>9}")
    print(f"  {'-' * 20}{'-' * 12}{'-' * 12}{'-' * 9}")
    for key, measured in reference.items():
        value = generated[key]
        ratio = value / measured if measured else float("nan")
        print(f"  {key:<20}{value:>12.3f}{measured:>12.3f}{ratio:>9.2f}")

    devices = snapshot.device_card_degrees()
    print("\nphysical devices, which a mitigation may block")
    print(f"  {'mean':<20}{devices.mean:>12.3f}")
    print(f"  {'max':<20}{devices.degrees.max():>12.3f}")
    print(f"  {'variance_to_mean':<20}{devices.variance_to_mean:>12.3f}")
    print("\n  independent assignment caps dispersion at one, whatever it draws")
    print("  from, so the signature figure above rules it out while the device")
    print("  figure stays where a household would put it")

    motifs = snapshot.motifs()
    print("\nentity projection, cards sharing a device")
    for key, value in motifs.as_dict().items():
        print(f"  {key:<20}{value:>12.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.analysis")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics = subparsers.add_parser("metrics", help="degree and motif metrics")
    metrics.add_argument("--holders", type=int, default=None)
    metrics.set_defaults(func=cmd_metrics)

    compare = subparsers.add_parser("compare", help="generated beside measured")
    compare.add_argument("--holders", type=int, default=None)
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
