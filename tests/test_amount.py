"""Cards differ from one another, not only from a shared curve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudsim.behavior.amount import AmountModel, level_spread
from fraudsim.calibration.fit_heterogeneity import entity_level_spread, fit_heterogeneity
from fraudsim.settings.behavior import AmountConfig
from fraudsim.world.entities import Archetype


def synthetic_panel(between_sd, within_sd, n_entities=1200, n_events=40, seed=0):
    """Entities with known levels, so the estimator has a target to recover."""
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(n_entities):
        level = rng.normal(4.4, between_sd)
        for _ in range(n_events):
            rows.append(
                {"entity": entity, "amount": float(np.exp(rng.normal(level, within_sd)))}
            )
    return pd.DataFrame(rows)


def test_decomposition_recovers_known_components() -> None:
    fit = fit_heterogeneity(synthetic_panel(0.6, 0.7), "entity", "amount")
    assert fit.between_sd == pytest.approx(0.6, rel=0.15)
    assert fit.within_sd == pytest.approx(0.7, rel=0.15)


def test_sampling_noise_is_subtracted() -> None:
    """The scatter of entity means is the true spread plus sampling noise.

    Left in, the estimate reads high, and a generator built on it produces a
    spread that stays wide however many events a card accumulates where a real
    one narrows.
    """
    sparse = fit_heterogeneity(
        synthetic_panel(0.6, 0.7, n_events=6, seed=1), "entity", "amount"
    )
    dense = fit_heterogeneity(
        synthetic_panel(0.6, 0.7, n_events=80, seed=1), "entity", "amount"
    )
    # Both target the same 0.6, from panels whose raw scatter differs a lot.
    assert sparse.between_sd == pytest.approx(dense.between_sd, rel=0.2)


def test_no_between_spread_is_recognised() -> None:
    """Every entity from one curve has to read as no heterogeneity at all."""
    fit = fit_heterogeneity(synthetic_panel(0.001, 0.7), "entity", "amount")
    assert fit.between_sd < 0.1
    assert fit.between_share < 0.05


def test_pooled_draws_understate_the_spread() -> None:
    """The defect this replaces: one curve leaves only sampling to separate two
    cards, and a marginal comparison cannot see the difference."""
    rng = np.random.default_rng(0)
    config = AmountConfig()
    pooled = {
        card: list(np.exp(rng.normal(config.level_mean, config.within_sd, 20)))
        for card in range(600)
    }
    assert level_spread(pooled) < config.between_sd * 0.5


def test_per_card_levels_reproduce_the_spread() -> None:
    rng = np.random.default_rng(0)
    config = AmountConfig()
    model = AmountModel(config)
    amounts = {}
    for card in range(800):
        model.register(card, rng)
        amounts[card] = [model.sample(card, rng) for _ in range(30)]
    assert level_spread(amounts) == pytest.approx(config.between_sd, rel=0.3)


def test_archetype_tilt_redistributes_rather_than_adds() -> None:
    """The spread was measured across real cardholders, who already are a
    mixture of habits, so a shift layered on top double-counts."""
    rng = np.random.default_rng(0)
    config = AmountConfig()

    def spread(with_archetypes: bool) -> float:
        model = AmountModel(config)
        archetypes = list(Archetype)
        amounts = {}
        for card in range(900):
            model.register(
                card, rng, archetypes[card % len(archetypes)] if with_archetypes else None
            )
            amounts[card] = [model.sample(card, rng) for _ in range(25)]
        return level_spread(amounts)

    plain, tilted = spread(False), spread(True)
    assert tilted == pytest.approx(plain, rel=0.35)


def test_archetypes_land_at_different_levels() -> None:
    """Redistributing still has to separate them, or the tilt does nothing."""
    rng = np.random.default_rng(0)
    model = AmountModel(AmountConfig())
    levels = {}
    for index, archetype in enumerate(Archetype):
        drawn = [
            model.register(index * 1000 + card, rng, archetype).level
            for card in range(400)
        ]
        levels[archetype] = float(np.mean(drawn))
    assert levels[Archetype.BUSINESS] > levels[Archetype.COMMUTER]


def test_amounts_stay_bounded_and_quantised() -> None:
    rng = np.random.default_rng(0)
    config = AmountConfig()
    model = AmountModel(config)
    model.register(1, rng)
    drawn = [model.sample(1, rng) for _ in range(4000)]
    assert max(drawn) <= config.upper_bound
    assert min(drawn) > 0
    whole = float(np.mean([value == round(value) for value in drawn]))
    assert whole == pytest.approx(config.whole_number_share, abs=0.08)


def test_unregistered_card_is_refused() -> None:
    with pytest.raises(KeyError, match="no amount profile"):
        AmountModel(AmountConfig()).sample(99, np.random.default_rng(0))


def test_entity_level_spread_ignores_thin_histories() -> None:
    """A card seen once has a mean equal to its single purchase, which says
    nothing about its habits."""
    frame = pd.DataFrame(
        [{"entity": entity, "amount": 50.0} for entity in range(100)]
        + [{"entity": 999, "amount": float(v)} for v in range(10, 110, 10)]
    )
    assert len(entity_level_spread(frame, "entity", "amount", min_events=5)) == 1
