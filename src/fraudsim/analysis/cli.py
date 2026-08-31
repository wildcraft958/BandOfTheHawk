"""Analysis entry points.

    python -m fraudsim.analysis.cli metrics
    python -m fraudsim.analysis.cli compare
    python -m fraudsim.analysis.cli entity-stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..cli import add_scale_flags, base_parser, load_config
from ..logs import emit
from ..population.builder import PopulationBuilder
from ..population.factory import build_warm_world
from .entity_report import render_entity_report
from .graph_snapshot import GraphSnapshot


def _build(args: argparse.Namespace, warm: bool = True):
    """Build a world, optionally run its warm start, and hand back both.

    The simulator is returned rather than dropped because its event log is the
    generated side of every comparison here. Building the world again to get
    at it would draw a different one.
    """
    config = load_config(args)

    if warm:
        world = build_warm_world(config)
        return world.graph, config, world.simulator
    graph, _ = PopulationBuilder(config).build()
    return graph, config, None


def cmd_metrics(args: argparse.Namespace) -> int:
    graph, _, _ = _build(args)
    emit(GraphSnapshot(graph).render())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Generated structure beside what was measured on real data."""
    graph, config, _ = _build(args)
    snapshot = GraphSnapshot(graph)
    target = config.population.fanout

    reference = {
        "mean": target.target_mean,
        "share_shared": target.target_share_shared,
        "p99": target.target_p99,
        "variance_to_mean": target.target_variance_to_mean,
    }
    generated = snapshot.fingerprint_card_degrees().as_dict()

    emit("shared-signature fan-out")
    emit(f"  {'statistic':<20}{'generated':>12}{'measured':>12}{'ratio':>9}")
    emit(f"  {'-' * 20}{'-' * 12}{'-' * 12}{'-' * 9}")
    for key, measured in reference.items():
        value = generated[key]
        ratio = value / measured if measured else float("nan")
        emit(f"  {key:<20}{value:>12.3f}{measured:>12.3f}{ratio:>9.2f}")

    devices = snapshot.device_card_degrees()
    emit("\nphysical devices, which a mitigation may block")
    emit(f"  {'mean':<20}{devices.mean:>12.3f}")
    emit(f"  {'max':<20}{devices.degrees.max():>12.3f}")
    emit(f"  {'variance_to_mean':<20}{devices.variance_to_mean:>12.3f}")
    emit("\n  independent assignment caps dispersion at one, whatever it draws")
    emit("  from, so the signature figure above rules it out while the device")
    emit("  figure stays where a household would put it")

    motifs = snapshot.motifs()
    emit("\nentity projection, cards sharing a device")
    for key, value in motifs.as_dict().items():
        emit(f"  {key:<20}{value:>12.3f}")
    return 0


def cmd_entity_stats(args: argparse.Namespace) -> int:
    """Per-entity structure, generated beside real.

    The statistics a marginal comparison cannot see. Each is reported against
    the judge dataset rather than against a threshold, because what matters is
    not whether the generated value looks reasonable but whether it sits where
    the real one does.
    """
    _, _, simulator = _build(args)
    emit(render_entity_report(simulator.log, args.judge, min_events=args.min_events))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fraudsim.analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics = subparsers.add_parser("metrics", help="degree and motif metrics")
    add_scale_flags(metrics)
    metrics.set_defaults(func=cmd_metrics)

    compare = subparsers.add_parser("compare", help="generated beside measured")
    add_scale_flags(compare)
    compare.set_defaults(func=cmd_compare)

    entity = subparsers.add_parser(
        "entity-stats", help="per-entity structure, generated beside real"
    )
    add_scale_flags(entity)
    entity.set_defaults(holders=4000)
    entity.add_argument("--min-events", type=int, default=5)
    entity.add_argument(
        "--judge", type=Path, default=None,
        help="judge dataset root; omit to report the generated side alone",
    )
    entity.set_defaults(func=cmd_entity_stats)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
