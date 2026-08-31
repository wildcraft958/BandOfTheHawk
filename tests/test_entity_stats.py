"""Estimators that can see per-entity structure.

Each is checked against a panel with planted parameters, so there is a known
target rather than a plausible-looking number. The corrections are checked the
way the amount decomposition was: by drawing sparse and dense panels from the
same truth and requiring them to agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudsim.calibration.distances import circular_w1, w1
from fraudsim.calibration.entity_stats import (
    categorical_entity_concentration,
    circular_entity_spread,
    matched_by_event_count,
    resultant_to_kappa,
    unbiased_simpson,
    _bessel_ratio,
)

TWO_PI = 2.0 * np.pi


def circular_panel(kappa_within, kappa_between, mode_hour=20.5,
                   n_entities=1200, n_events=30, seed=0):
    """Entities with their own preferred hours, drawn around a population mode."""
    rng = np.random.default_rng(seed)
    mode = mode_hour * TWO_PI / 24.0
    rows = []
    for entity in range(n_entities):
        preferred = rng.vonmises(mode, kappa_between) if kappa_between < 400 else mode
        for _ in range(n_events):
            angle = rng.vonmises(preferred, kappa_within)
            rows.append({"entity": entity, "hour": (angle % TWO_PI) * 24.0 / TWO_PI})
    return pd.DataFrame(rows)


def categorical_panel(concentration, n_values=8, n_entities=900, n_events=30, seed=0):
    """Entities with their own mix, drawn Dirichlet around a shared base.

    Large concentration means every entity shares the base mix, which is the
    defect; small means entities specialise.
    """
    rng = np.random.default_rng(seed)
    base = np.full(n_values, 1.0 / n_values)
    rows = []
    for entity in range(n_entities):
        weights = rng.dirichlet(base * concentration)
        picks = rng.choice(n_values, size=n_events, p=weights)
        rows.extend({"entity": entity, "value": int(v)} for v in picks)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ circular


def test_recovers_planted_concentration() -> None:
    fit = circular_entity_spread(circular_panel(1.1, 3.5), "entity", "hour", min_events=10)
    assert fit.within_r == pytest.approx(_bessel_ratio(1.1), rel=0.10)
    assert fit.between_r == pytest.approx(_bessel_ratio(3.5), rel=0.10)


def test_correction_is_stable_across_event_counts() -> None:
    """The test that earns the correction.

    An entity seen k times has a resultant inflated by roughly 1/k, so the raw
    estimate confounds concentration with history length and drifts with the
    cutoff. Corrected, panels drawn from the same truth agree.
    """
    sparse = circular_entity_spread(
        circular_panel(1.1, 3.5, n_events=6, seed=1), "entity", "hour", min_events=5
    )
    dense = circular_entity_spread(
        circular_panel(1.1, 3.5, n_events=60, seed=1), "entity", "hour", min_events=5
    )
    assert sparse.within_r == pytest.approx(dense.within_r, rel=0.15)
    # And the uncorrected figure would not have agreed, which is the point.
    assert abs(sparse.within_r_raw - dense.within_r_raw) > 0.15 * dense.within_r_raw


def test_resultants_multiply() -> None:
    """The identity the hour design rests on.

    A von Mises within a von Mises has a marginal resultant equal to the
    product of the two. If this ever fails, the hierarchical fit is wrong.
    """
    fit = circular_entity_spread(
        circular_panel(1.1, 3.5, n_entities=2000), "entity", "hour", min_events=10
    )
    assert fit.marginal_r == pytest.approx(fit.within_r * fit.between_r, rel=0.08)


def test_identical_preferences_read_as_no_between_spread() -> None:
    """Every entity peaking at the same hour has to read as full agreement."""
    fit = circular_entity_spread(
        circular_panel(1.1, 500.0), "entity", "hour", min_events=10
    )
    # Not 1.0: each entity's preferred angle is itself estimated from thirty
    # events, so the agreement between them carries that estimation noise.
    assert fit.between_r > 0.95


def test_scattered_preferences_are_distinguished_from_the_marginal() -> None:
    """The trap a marginal cannot see.

    Entities that each concentrate tightly but disagree about where produce a
    flat marginal. A generator matching only the marginal would look correct
    while every entity was wrong.
    """
    scattered = circular_entity_spread(
        circular_panel(4.0, 0.0), "entity", "hour", min_events=10
    )
    assert scattered.within_r > 0.7
    assert scattered.between_r < 0.2
    assert scattered.marginal_r < 0.2


def test_kappa_inversion_round_trips() -> None:
    for kappa in (0.3, 1.1, 3.5, 12.0):
        assert resultant_to_kappa(_bessel_ratio(kappa)) == pytest.approx(kappa, rel=1e-3)


# --------------------------------------------------------------- categorical


def test_unbiased_simpson_is_flat_in_event_count() -> None:
    """The plug-in form reads high on short histories and the unbiased one
    does not, which is the same correction in a different coordinate."""
    sparse = categorical_entity_concentration(
        categorical_panel(8.0, n_events=8, seed=2), "entity", "value",
        min_events=5, n_shuffles=3,
    )
    dense = categorical_entity_concentration(
        categorical_panel(8.0, n_events=80, seed=2), "entity", "value",
        min_events=5, n_shuffles=3,
    )
    assert sparse.within_simpson == pytest.approx(dense.within_simpson, rel=0.15)
    assert sparse.within_simpson_raw > dense.within_simpson_raw


def test_shared_curve_reads_as_no_concentration() -> None:
    """Every entity drawn from one mix has to land on the null.

    This is the defect: a generator that picks each event independently from a
    population curve produces exactly this, and its ratio sits at chance.
    """
    spread = categorical_entity_concentration(
        categorical_panel(10_000.0, seed=3), "entity", "value",
        min_events=10, n_shuffles=8,
    )
    assert spread.ratio == pytest.approx(1.0, abs=0.05)
    assert abs(spread.z_against_null) < 4.0


def test_specialised_entities_are_found() -> None:
    spread = categorical_entity_concentration(
        categorical_panel(0.8, seed=4), "entity", "value", min_events=10, n_shuffles=8
    )
    assert spread.ratio > 1.5
    assert spread.z_against_null > 5.0


def test_unbiased_simpson_needs_two_events() -> None:
    assert np.isnan(unbiased_simpson(np.array([1.0])))
    assert unbiased_simpson(np.array([2.0])) == pytest.approx(1.0)


# ------------------------------------------------------------------ distance


def test_circular_w1_wraps_across_midnight() -> None:
    """The reason the linear form cannot be reused.

    Two samples an hour apart across midnight are an hour apart. The linear
    distance calls them twenty-three and returns it without complaint.
    """
    late = np.full(500, 23.5)
    early = np.full(500, 0.5)
    assert circular_w1(late, early) == pytest.approx(1.0, abs=0.05)
    assert w1(late, early) == pytest.approx(23.0, abs=0.05)


def test_circular_w1_is_zero_for_the_same_sample() -> None:
    rng = np.random.default_rng(0)
    hours = rng.uniform(0, 24, 800)
    assert circular_w1(hours, hours) == pytest.approx(0.0, abs=1e-9)


def test_circular_w1_grows_with_separation() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(12.0, 1.0, 900) % 24
    near = (base + 1.0) % 24
    far = (base + 5.0) % 24
    assert circular_w1(base, near) < circular_w1(base, far)


# ------------------------------------------------------------------ matching


def test_matched_comparison_separates_bands() -> None:
    """Sparse and dense entities are compared against their own kind."""
    # Deliberately mismatched history lengths: real entities are sparse, the
    # generated ones dense. Pooled, that difference alone would move any of
    # these statistics.
    real = pd.concat([
        circular_panel(1.1, 3.5, n_entities=200, n_events=12, seed=5),
        circular_panel(1.1, 3.5, n_entities=200, n_events=60, seed=7).assign(
            entity=lambda d: d.entity + 1000
        ),
    ])
    generated = pd.concat([
        circular_panel(1.1, 3.5, n_entities=200, n_events=12, seed=8),
        circular_panel(1.1, 3.5, n_entities=200, n_events=60, seed=6).assign(
            entity=lambda d: d.entity + 1000
        ),
    ])

    table = matched_by_event_count(
        real, generated, "entity",
        statistic=lambda f: circular_entity_spread(
            f, "entity", "hour", min_events=5
        ).within_r,
        bands=((10, 19), (50, 10**9)),
    )
    assert not table.empty
    # Each band compares like with like, so every ratio sits near one even
    # though the two populations have very different history lengths.
    assert (table["ratio"].dropna() > 0.8).all()
    assert (table["ratio"].dropna() < 1.25).all()


# ------------------------------------------------ small-sample autocorrelation


def arrival_panel(n_events, n_entities=1500, seed=0):
    """Gaps from one process, varying only how many each entity gets."""
    from fraudsim.settings.behavior import ArrivalConfig
    from fraudsim.timing.arrival import DriftingRateProcess

    process = DriftingRateProcess(ArrivalConfig())
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(n_entities):
        state = process.new_state(rng)
        elapsed = 0.0
        for _ in range(n_events):
            elapsed += process.next_gap_seconds(state, rng)
            rows.append({"entity": entity, "t": elapsed})
    return pd.DataFrame(rows)


def test_autocorrelation_sign_depends_on_history_length() -> None:
    """One process, opposite signs, purely from how much history each entity has.

    Lag-1 autocorrelation carries a bias of roughly -1/(n-1): with a handful of
    gaps it is computed against a mean estimated from those same gaps, which
    forces it negative. Real cardholders show it too - entities with five to
    nine events measure -0.13 while entities with fifty or more measure +0.07,
    from the same population.

    Comparing a sparse generated population against a pooled real target
    therefore reports a difference in census as a difference in behaviour, and
    reads as the generator inverting a correlation it reproduces correctly.
    """
    from fraudsim.calibration.behavioral import inter_event_stats

    sparse = inter_event_stats(arrival_panel(6), "entity", "t", min_events=5)
    dense = inter_event_stats(arrival_panel(60), "entity", "t", min_events=5)

    assert sparse.mean_autocorrelation < 0
    assert dense.mean_autocorrelation > 0
    # And the gap between them is the bias, not the process.
    assert dense.mean_autocorrelation - sparse.mean_autocorrelation > 0.15


def test_matched_comparison_survives_the_bias() -> None:
    """Compared within a band, two draws from one process agree.

    This is the check the arrival comparison needs. Pooled, these two
    populations look like different processes; matched, they do not.
    """
    from fraudsim.calibration.behavioral import inter_event_stats

    real = pd.concat([
        arrival_panel(7, n_entities=800, seed=1),
        arrival_panel(40, n_entities=800, seed=2).assign(
            entity=lambda d: d.entity + 10_000
        ),
    ])
    generated = arrival_panel(7, n_entities=800, seed=3)

    table = matched_by_event_count(
        real, generated, "entity",
        statistic=lambda f: inter_event_stats(
            f, "entity", "t", min_events=5
        ).mean_autocorrelation,
        bands=((5, 9),),
    )
    assert not table.empty
    row = table.iloc[0]
    assert abs(row["generated"] - row["real"]) < 0.06
