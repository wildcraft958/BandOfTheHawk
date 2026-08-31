"""Fitters are checked against data whose parameters are known, so a failure
points at the estimator rather than at the source."""

from __future__ import annotations

import numpy as np
import pytest


from fraudsim.calibration.fit_amount import fit_amount, hill_index
from fraudsim.calibration.fit_arrival import fit_arrival, measure_targets, simulate_arrival
from fraudsim.calibration.fit_timing import fit_hawkes, simulate_hawkes


def test_hill_index_recovers_a_known_tail() -> None:
    rng = np.random.default_rng(0)
    alpha = 2.5
    sample = (1.0 - rng.random(200_000)) ** (-1.0 / alpha)
    index, _ = hill_index(sample, tail_fraction=0.05)
    assert index == pytest.approx(alpha, rel=0.15)


def test_amount_fit_reproduces_its_own_summary() -> None:
    rng = np.random.default_rng(1)
    sample = rng.lognormal(4.2, 0.8, 60_000)
    fit = fit_amount(sample)
    drawn = fit.sample(60_000, rng)
    assert np.median(drawn) == pytest.approx(np.median(sample), rel=0.2)


def test_amount_tail_is_bounded() -> None:
    """An unbounded Pareto draws far past anything in the source."""
    rng = np.random.default_rng(2)
    sample = rng.lognormal(4.0, 1.0, 40_000)
    fit = fit_amount(sample)
    assert fit.sample(200_000, rng).max() <= fit.upper_bound


def test_amount_preserves_whole_number_share() -> None:
    rng = np.random.default_rng(3)
    sample = np.round(rng.lognormal(4.0, 0.8, 40_000))
    fit = fit_amount(sample)
    drawn = fit.sample(40_000, rng)
    share = float(np.mean(drawn == np.floor(drawn)))
    assert share == pytest.approx(fit.whole_number_share, abs=0.05)


def test_hawkes_recovers_known_parameters() -> None:
    rng = np.random.default_rng(0)
    mu, alpha, beta = 1 / 3600, 6e-4, 1 / 600
    sequences = [
        simulate_hawkes(mu, alpha, beta, horizon=60 * 86_400, rng=rng, max_events=150)
        for _ in range(120)
    ]
    fit = fit_hawkes([s for s in sequences if len(s) >= 5], max_entities=120)
    assert fit.branching_ratio == pytest.approx(alpha / beta, rel=0.3)
    assert fit.is_stable


def test_hawkes_gate_accepts_data_from_its_own_model() -> None:
    """The gate must pass where the model is right, or it says nothing when it fails."""
    rng = np.random.default_rng(1)
    sequences = [
        simulate_hawkes(1 / 3600, 6e-4, 1 / 600, horizon=60 * 86_400, rng=rng, max_events=150)
        for _ in range(120)
    ]
    fit = fit_hawkes([s for s in sequences if len(s) >= 5], max_entities=120)
    assert fit.ks_pvalue > 0.01


def test_arrival_drift_controls_autocorrelation() -> None:
    """Persistence in the rate is what couples consecutive gaps."""
    rng = np.random.default_rng(0)
    sequences = [
        np.concatenate([[0.0], np.cumsum(rng.gamma(0.8, 1.0, 40))]) for _ in range(400)
    ]
    fit = fit_arrival(sequences, min_events=10)
    flat = fit.__class__(**{**fit.as_dict(), "drift_sigma": 0.0, "drift_persistence": 0.0,
                            "n_entities": int(fit.n_entities), "n_gaps": int(fit.n_gaps)})
    drifting = fit.__class__(**{**fit.as_dict(), "drift_sigma": 1.0, "drift_persistence": 0.95,
                                "n_entities": int(fit.n_entities), "n_gaps": int(fit.n_gaps)})
    still = measure_targets(simulate_arrival(flat, 400, 30, np.random.default_rng(7)))
    moving = measure_targets(simulate_arrival(drifting, 400, 30, np.random.default_rng(7)))
    assert moving["autocorrelation"] > still["autocorrelation"]


def test_arrival_without_drift_has_no_positive_autocorrelation() -> None:
    """A plain renewal draw cannot produce positively correlated gaps.

    The bound is one-sided on purpose. Independence puts the expectation at or
    below zero, and a short simulated sample scatters a little way below it, so
    the property worth asserting is that nothing positive appears.
    """
    rng = np.random.default_rng(5)
    sequences = [
        np.concatenate([[0.0], np.cumsum(rng.gamma(1.0, 1.0, 40))]) for _ in range(400)
    ]
    fit = fit_arrival(sequences, min_events=10, drift_grid=(0.0,), persistence_grid=(0.0,),
                      shape_grid=(1.0,))
    observed = measure_targets(simulate_arrival(fit, 500, 30, np.random.default_rng(9)))
    assert observed["autocorrelation"] < 0.02
