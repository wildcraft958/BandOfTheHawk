"""Warm start is a gate.

A cold world leaves device age, tenure, and every prior count degenerate
through the burn-in, so the earliest events a detector would train on are
describing the generator settling rather than anyone's behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.config.simulation import SimulationConfig
from fraudsim.engine.simulator import Simulator
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.population.builder import PopulationBuilder
from fraudsim.population.warmstart import WarmStartRunner
from fraudsim.protocols import AlwaysApproveScorer
from fraudsim.rules.engine import VelocityRuleEngine

MINUTES_PER_DAY = 1440


@pytest.fixture(scope="module")
def warmed():
    config = SimulationConfig.model_validate({"population": {"n_holders": 1200}})
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    # Nothing is refused during the warm start. History should not be shaped by
    # a defender that does not exist yet.
    simulator = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
    report = WarmStartRunner(simulator, config, seed=3).run()
    return simulator, config, report


def auth_events(simulator):
    return [event for event in simulator.log.events if hasattr(event, "amount")]


def test_history_spans_the_lookback_window(warmed) -> None:
    """The clock has to start behind the observation window.

    Left at the origin it refuses to move backwards, every event lands at the
    same instant, and the traffic reads as one enormous burst.
    """
    simulator, config, _ = warmed
    stamps = np.array([event.ts for event in auth_events(simulator)])
    span_days = (stamps.max() - stamps.min()) / MINUTES_PER_DAY
    assert stamps.min() < 0
    assert span_days > config.warm_start.lookback_days * 0.5


def test_events_do_not_pile_onto_one_instant(warmed) -> None:
    """Timestamps have to spread across the window.

    Some sharing is expected rather than suspect: a shopping session is several
    purchases minutes apart, and a recovery is a sequence at one sitting. What
    would be wrong is everything landing together, which is what happens when
    the clock never leaves its origin.
    """
    simulator, _, _ = warmed
    events = auth_events(simulator)
    distinct = len({event.ts for event in events})
    assert distinct > len(events) * 0.25
    assert distinct > 100


def test_history_arrives_in_time_order(warmed) -> None:
    """The windows evict from the front assuming it is the oldest entry, so
    a card receiving its own history out of order loses part of it."""
    simulator, _, _ = warmed
    stamps = [event.ts for event in auth_events(simulator)]
    assert stamps == sorted(stamps)


def test_activity_is_sparse(warmed) -> None:
    """Most holders transact rarely, and a uniformly active population hands
    every history-dependent feature more than it would ever have."""
    _, _, report = warmed
    assert report.events_per_entity["median"] <= 4
    assert report.dormant_share > 0.3


def test_rolling_state_is_populated(warmed) -> None:
    """The point of the exercise: windows that start full rather than empty."""
    simulator, _, report = warmed
    assert report.n_events > 0
    assert report.cards_with_median > 0.1


def test_derived_fields_are_realised_not_asserted(warmed) -> None:
    """A card's usual amount comes from its own transactions. Sampling it
    separately would leave a card disagreeing with its own history."""
    simulator, _, _ = warmed
    graph = simulator.graph
    with_median = [
        card for card in graph.cards.values() if card.median_amount is not None
    ]
    assert with_median
    assert all(card.median_amount > 0 for card in with_median)
    assert all(card.category_counts for card in with_median)


def test_warm_start_events_are_flagged_and_separable(warmed) -> None:
    """These are feature-poorer by construction, so training on them would
    learn that difference rather than behaviour."""
    simulator, _, _ = warmed
    assert all(event.is_warm_start for event in simulator.log.events)
    assert simulator.log.scoreable() == []


def test_the_builder_stops_flagging_afterwards(warmed) -> None:
    simulator, _, _ = warmed
    assert not simulator.builder.warm_start


def test_hard_negatives_were_injected(warmed) -> None:
    """Without ordinary behaviour that looks suspicious, a false-positive rate
    has nothing to measure."""
    _, _, report = warmed
    assert report.hard_negatives["large_purchase"] > 0
    assert report.hard_negatives["gift_card"] > 0


def test_naive_rules_trip_at_the_intended_rate(warmed) -> None:
    """The gate. Too low and the false-positive metric is vacuous; too high and
    the traffic is not ordinary behaviour.

    Against the configured tolerance rather than a wider range of its own. A
    range the printed verdict disagrees with lets a regression pass the suite
    while the report calls it off target.
    """
    simulator, config, _ = warmed
    negatives = config.behavior.hard_negatives
    rates = VelocityRuleEngine(config.engine.rules).trigger_rates(auth_events(simulator))
    assert abs(rates.any_rule - negatives.naive_rule_trip_target) <= (
        negatives.naive_rule_trip_tolerance
    )


def test_no_single_rule_dominates(warmed) -> None:
    """One rule carrying everything means it is keyed on something the
    generator produces rather than on behaviour."""
    simulator, config, _ = warmed
    rates = VelocityRuleEngine(config.engine.rules).trigger_rates(auth_events(simulator))
    assert max(rates.per_rule.values()) < 0.7 * max(rates.any_rule, 1e-9) + 0.05


def test_graph_invariants_survive(warmed) -> None:
    simulator, _, _ = warmed
    simulator.graph.check_invariants()


def test_warm_start_is_reproducible() -> None:
    def run(seed: int) -> int:
        config = SimulationConfig.model_validate({"population": {"n_holders": 400}})
        graph, _ = PopulationBuilder(config).build()
        states = FeatureStateStore(config.engine.windows)
        builder = EventBuilder(graph, states, config.engine.windows)
        simulator = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
        return WarmStartRunner(simulator, config, seed=seed).run().n_events

    assert run(5) == run(5)


def test_every_injector_fires(warmed) -> None:
    """All seven, not just the two that only needed a different amount.

    Travel changes where, a session changes how many and how close together,
    a new device changes what the transaction goes through, and a recovery is
    not a transaction at all. Each needs its own shape, which is why they were
    missing while the amount-only pair worked.
    """
    _, _, report = warmed
    for kind in (
        "large_purchase", "gift_card", "travel",
        "session", "new_device", "dispute", "recovery",
    ):
        assert report.hard_negatives.get(kind, 0) > 0, f"{kind} never fired"


def test_ordinary_traffic_still_dominates(warmed) -> None:
    """These are meant to be the exceptions. If awkward behaviour is the norm,
    the rate measures the injectors rather than a population."""
    _, _, report = warmed
    total = sum(report.hard_negatives.values())
    assert report.hard_negatives["ordinary"] / total > 0.85


def test_every_rule_can_fire(warmed) -> None:
    """A rule that never fires on any traffic is untested, and its threshold is
    a number nobody has checked."""
    simulator, config, _ = warmed
    rates = VelocityRuleEngine(config.engine.rules).trigger_rates(auth_events(simulator))
    silent = [rule for rule, rate in rates.per_rule.items() if rate == 0.0]
    # Declines are produced by the defender rather than by ordinary behaviour,
    # so the rule keyed on them stays quiet against approve-everything traffic.
    assert set(silent) <= {"R7"}


def test_cards_are_seen_on_more_than_one_device(warmed) -> None:
    """Holders move between a phone, a laptop, and a tablet.

    Routing every transaction through the same binding left the count of
    devices a card had been used on stuck at one, so the rule keyed on it could
    not fire however many devices the card was actually bound to.
    """
    simulator, _, _ = warmed
    counts = [event.card_n_devices for event in auth_events(simulator)]
    assert max(counts) > 2


def test_travel_reaches_beyond_the_home_radius(warmed) -> None:
    simulator, config, _ = warmed
    distances = [event.geo_distance_km for event in auth_events(simulator)]
    assert max(distances) > config.population.geo.home_radius_km * 5


def test_sessions_produce_close_together_transactions(warmed) -> None:
    simulator, _, _ = warmed
    assert max(event.auths_last_1h for event in auth_events(simulator)) >= 3
