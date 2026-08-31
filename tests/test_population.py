"""A built population has to reproduce the structure it was calibrated against,
and stay internally consistent while doing it."""

from __future__ import annotations

import pytest

from fraudsim.population.archetypes import build_profiles
from fraudsim.population.builder import PopulationBuilder
from fraudsim.rng import RngHub
from fraudsim.settings.simulation import SimulationConfig
from fraudsim.settings.world import PopulationConfig
from fraudsim.world.entities import Archetype, CategoryCluster


def small_config(**overrides) -> SimulationConfig:
    payload = {"population": {"n_holders": 4000, **overrides}}
    return SimulationConfig.model_validate(payload)


@pytest.fixture(scope="module")
def built():
    return PopulationBuilder(small_config()).build()


def test_graph_invariants_hold(built) -> None:
    graph, _ = built
    graph.check_invariants()


def test_every_entity_is_reachable(built) -> None:
    graph, report = built
    assert report.counts["holders"] == 4000
    assert report.counts["cards"] > report.counts["holders"]
    assert report.counts["devices"] > 0
    assert report.counts["provisioned"] > 0


def test_fanout_reaches_its_targets(built) -> None:
    """The gate: a population where one device carries one card would let a
    graph detector score perfectly for a reason unrelated to detection."""
    _, report = built
    assert report.fanout["mean"] == pytest.approx(report.fanout["target_mean"], rel=0.35)
    assert report.fanout["p99"] == pytest.approx(report.fanout["target_p99"], rel=0.5)
    assert report.fanout["variance_to_mean"] > 50.0


def test_devices_stay_at_household_scale(built) -> None:
    """Blocking a device is only proportionate if it is not a crowd."""
    graph, report = built
    assert report.device_fanout["max"] <= graph.cards.__len__()
    config = small_config().population
    assert report.device_fanout["max"] <= config.devices.household_max
    assert report.device_fanout["mean"] == pytest.approx(
        config.devices.household_mean, rel=0.2
    )


def test_signature_reach_exceeds_device_reach(built) -> None:
    _, report = built
    assert report.fanout["max"] > report.device_fanout["max"] * 20


def test_mixes_follow_the_configured_shares(built) -> None:
    _, report = built
    config = small_config().population
    for name, share in config.archetype_weights.items():
        assert report.archetype_mix[name] == pytest.approx(share, abs=0.03)
    for name, share in config.activity.tier_weights.items():
        assert report.activity_mix[name] == pytest.approx(share, abs=0.03)


def test_signature_count_is_derived_from_the_target() -> None:
    """Count and reach are the same quantity seen twice, so fixing one fixes
    the other. A hand-set count silently contradicts the target."""
    config = PopulationConfig(n_holders=20_000)
    derived = config.resolved_fingerprint_count()
    expected = (
        config.n_holders * config.devices_per_holder_mean * config.devices.household_mean
    ) / config.fanout.target_mean
    assert derived == pytest.approx(expected, rel=0.05)


def test_explicit_signature_count_is_respected() -> None:
    config = PopulationConfig(n_holders=1000, fingerprint_count=42)
    assert config.resolved_fingerprint_count() == 42


def test_build_is_reproducible() -> None:
    first, _ = PopulationBuilder(small_config(), RngHub(7)).build()
    second, _ = PopulationBuilder(small_config(), RngHub(7)).build()
    assert first.summary() == second.summary()
    assert sorted(first.provisioned) == sorted(second.provisioned)


def test_different_seeds_differ() -> None:
    first, _ = PopulationBuilder(small_config(), RngHub(1)).build()
    second, _ = PopulationBuilder(small_config(), RngHub(2)).build()
    assert sorted(first.provisioned) != sorted(second.provisioned)


def test_cards_belong_to_a_known_holder(built) -> None:
    graph, _ = built
    for card in graph.cards.values():
        assert card.holder_id in graph.holders


def test_derived_card_fields_start_empty(built) -> None:
    """Realised behaviour fills these; sampling them independently would leave
    an inconsistency a detector could exploit."""
    graph, _ = built
    assert all(card.median_amount is None for card in graph.cards.values())


def test_archetype_profiles_prefer_their_categories() -> None:
    profiles = build_profiles(SimulationConfig().behavior.categories.mix)
    order = tuple(CategoryCluster)
    online = profiles[Archetype.ONLINE_HEAVY]
    homebody = profiles[Archetype.HOMEBODY]
    index = order.index(CategoryCluster.ONLINE)
    assert online.category_weights[index] > homebody.category_weights[index]


def test_archetype_weights_form_a_distribution() -> None:
    for profile in build_profiles(SimulationConfig().behavior.categories.mix).values():
        assert profile.category_weights.sum() == pytest.approx(1.0)
