"""Cards shop where they usually shop, and buy what they usually buy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudsim.behavior.loyalty import LoyaltyModel, archetype_weights, clusters_from_graph
from fraudsim.calibration.entity_stats import categorical_entity_concentration
from fraudsim.config.behavior import LoyaltyConfig
from fraudsim.config.simulation import SimulationConfig
from fraudsim.population.archetypes import build_profiles
from fraudsim.population.builder import PopulationBuilder
from fraudsim.world.entities import Archetype, CategoryCluster


def small_world(n_holders=600):
    config = SimulationConfig.model_validate({"population": {"n_holders": n_holders}})
    graph, _ = PopulationBuilder(config).build()
    return graph, config


def draw(config, loyalty_config, n_cards=400, n_events=12, seed=0):
    """Merchants and categories for a population of cards."""
    graph, base = small_world()
    by_cluster, popularity = clusters_from_graph(
        graph, base.population.merchants.popularity_exponent
    )
    model = LoyaltyModel(loyalty_config, by_cluster, popularity)
    profiles = build_profiles(base.behavior.categories.mix, base.population.archetype_weights)
    rng = np.random.default_rng(seed)
    everything = np.asarray(list(graph.merchants), dtype=int)

    rows = []
    archetypes = list(Archetype)
    for card in range(n_cards):
        weights = archetype_weights(profiles, archetypes[card % len(archetypes)])
        model.register(card, rng, weights)
        for _ in range(n_events):
            merchant = model.pick_merchant(card, rng, everything)
            rows.append({"card": card, "merchant": merchant})
    return pd.DataFrame(rows), graph


def test_loyalty_makes_cards_revisit() -> None:
    frame, _ = draw(None, LoyaltyConfig(merchant_loyalty=0.7, merchant_preferred_set_mean=8.0))
    spread = categorical_entity_concentration(
        frame, "card", "merchant", min_events=5, n_shuffles=6
    )
    assert spread.ratio > 2.0
    assert spread.z_against_null > 5.0


def test_zero_loyalty_reproduces_the_uniform_draw() -> None:
    """The sweep has to contain the behaviour it replaces.

    At the low end of the range the merchant draw is uniform again, which is
    what lets a claim be stated as holding across a range that includes the
    defect.
    """
    frame, _ = draw(None, LoyaltyConfig(merchant_loyalty=0.0))
    spread = categorical_entity_concentration(
        frame, "card", "merchant", min_events=5, n_shuffles=6
    )
    assert spread.z_against_null < 4.0


def test_loyalty_is_monotone_in_the_swept_parameter() -> None:
    ratios = []
    for loyalty in (0.0, 0.4, 0.8):
        frame, _ = draw(None, LoyaltyConfig(merchant_loyalty=loyalty,
                                            merchant_preferred_set_mean=8.0))
        ratios.append(
            categorical_entity_concentration(
                frame, "card", "merchant", min_events=5, n_shuffles=4
            ).ratio
        )
    assert ratios[0] < ratios[1] < ratios[2]


def test_a_regular_set_larger_than_the_history_is_no_habit() -> None:
    """Why the roster is sized per card rather than per category.

    A card with a hundred regulars and five transactions never revisits any of
    them, so the habit exists on paper and nowhere in the data.
    """
    tight, _ = draw(None, LoyaltyConfig(merchant_loyalty=0.8, merchant_preferred_set_mean=4.0))
    loose, _ = draw(None, LoyaltyConfig(merchant_loyalty=0.8, merchant_preferred_set_mean=150.0))

    def ratio(frame):
        return categorical_entity_concentration(
            frame, "card", "merchant", min_events=5, n_shuffles=4
        ).ratio

    assert ratio(tight) > ratio(loose)


def test_category_concentration_gives_cards_their_own_mix() -> None:
    graph, base = small_world()
    by_cluster, popularity = clusters_from_graph(
        graph, base.population.merchants.popularity_exponent
    )
    profiles = build_profiles(base.behavior.categories.mix, base.population.archetype_weights)

    def mixes(concentration: float) -> float:
        model = LoyaltyModel(
            LoyaltyConfig(category_concentration=concentration), by_cluster, popularity
        )
        rng = np.random.default_rng(0)
        rows = []
        for card in range(500):
            model.register(card, rng, archetype_weights(profiles, Archetype.COMMUTER))
            profile = model.category(card)
            for _ in range(15):
                rows.append({"card": card, "cluster": profile.pick(rng)})
        return categorical_entity_concentration(
            pd.DataFrame(rows), "card", "cluster", min_events=5, n_shuffles=4
        ).ratio

    # A high concentration is the defect: every card of an archetype shares
    # one mix, so no card has anything of its own.
    assert mixes(200.0) < 1.15
    assert mixes(2.0) > 1.4


def test_archetype_tilts_preserve_the_population_mix() -> None:
    """Tilting each archetype and normalising it alone moves the mix.

    The average of the tilted mixes, weighted by how common each archetype is,
    has to come back to the mix that was fitted - otherwise the archetypes are
    not redistributing the population's spending but inventing more of it.
    """
    config = SimulationConfig()
    profiles = build_profiles(
        config.behavior.categories.mix, config.population.archetype_weights
    )
    order = list(CategoryCluster)
    average = np.zeros(len(order))
    for archetype in Archetype:
        average += (
            config.population.archetype_weights[archetype.value]
            * profiles[archetype].category_weights
        )
    for index, cluster in enumerate(order):
        assert average[index] == pytest.approx(
            config.behavior.categories.mix[cluster.value], rel=0.01
        )


def test_archetypes_still_differ_from_one_another() -> None:
    """Preserving the population mix must not flatten the archetypes."""
    config = SimulationConfig()
    profiles = build_profiles(
        config.behavior.categories.mix, config.population.archetype_weights
    )
    travel = list(CategoryCluster).index(CategoryCluster.TRAVEL)
    assert (
        profiles[Archetype.TRAVELLER].category_weights[travel]
        > 3 * profiles[Archetype.HOMEBODY].category_weights[travel]
    )
