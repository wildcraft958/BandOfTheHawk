"""Rule entry points.

    python -m fraudsim.rules.cli describe
    python -m fraudsim.rules.cli rate
"""

from __future__ import annotations

import argparse

import numpy as np

from ..cli import base_parser, load_config
from ..features.builder import EventBuilder
from ..features.state import FeatureStateStore
from ..population.builder import PopulationBuilder
from ..timing.arrival import DriftingRateProcess
from ..timing.circadian import HolderClockModel
from .engine import VelocityRuleEngine


def _load(args: argparse.Namespace):
    return load_config(args)


def cmd_describe(args: argparse.Namespace) -> int:
    config = _load(args)
    print("velocity rules")
    print(VelocityRuleEngine(config.engine.rules).describe())
    return 0


def cmd_rate(args: argparse.Namespace) -> int:
    """Measure how much ordinary traffic a naive engine would flag."""
    config = _load(args)
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(
        graph, states, config.engine.windows, HolderClockModel(config.behavior.circadian)
    )
    process = DriftingRateProcess(config.behavior.arrival)
    rng = np.random.default_rng(args.seed)

    devices = {}
    for card_id, device_id in graph.provisioned:
        devices.setdefault(card_id, device_id)
    cards = list(devices)[: args.cards]

    schedule = []
    for card_id in cards:
        state = process.new_state(rng)
        elapsed = 0.0
        for _ in range(args.events):
            elapsed += process.next_gap_seconds(state, rng)
            schedule.append((int(elapsed / 60), card_id))
    schedule.sort()

    amount = config.behavior.amount
    merchants = list(graph.merchants)
    events = []
    for ts, card_id in schedule:
        merchant_id = merchants[int(rng.integers(0, len(merchants)))]
        value = float(
            np.clip(
                rng.lognormal(amount.lognormal_mu, amount.lognormal_sigma),
                1.0,
                amount.upper_bound,
            )
        )
        event = builder.build_auth(
            ts=ts, card_id=card_id, merchant_id=merchant_id, device_id=devices[card_id],
            amount=value, entry_mode=int(rng.integers(0, 4)),
            geo_distance_km=float(rng.exponential(config.population.geo.home_radius_km)),
        )
        events.append(event)
        builder.commit_auth(event)

    rates = VelocityRuleEngine(config.engine.rules).trigger_rates(events)
    target = config.behavior.hard_negatives.naive_rule_trip_target
    tolerance = config.behavior.hard_negatives.naive_rule_trip_tolerance
    print(rates.render(target=target, tolerance=tolerance))
    print("\n  this traffic carries no travel, sessions, or new devices yet,")
    print("  so the rules keyed on those stay quiet until they are injected")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fraudsim.rules")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="list the rules and thresholds")
    describe.set_defaults(func=cmd_describe)

    rate = subparsers.add_parser("rate", help="share of ordinary traffic that trips a rule")
    rate.add_argument("--cards", type=int, default=1500)
    rate.add_argument("--events", type=int, default=40)
    rate.add_argument("--seed", type=int, default=11)
    rate.set_defaults(func=cmd_rate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
