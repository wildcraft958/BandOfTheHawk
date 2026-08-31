"""Timing is a gate.

If generated gaps carry no correlation, every velocity feature the detector
depends on is measuring an artefact of the generator rather than behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.settings.behavior import ArrivalConfig, CircadianConfig
from fraudsim.timing.arrival import (
    ArrivalScheduler,
    DriftingRateProcess,
    burstiness,
    lag1_autocorrelation,
)
from fraudsim.timing.circadian import (
    CircadianClock,
    circular_mean_hour,
    resultant_length,
)
from fraudsim.world.entities import Archetype


def mean_over_entities(config: ArrivalConfig, seed: int, n: int = 1500, events: int = 30):
    process = DriftingRateProcess(config)
    rng = np.random.default_rng(seed)
    rhos, bursts = [], []
    for _ in range(n):
        state = process.new_state(rng)
        gaps = process.sample_gaps(state, events, rng)
        rho, burst = lag1_autocorrelation(gaps), burstiness(gaps)
        if np.isfinite(rho):
            rhos.append(rho)
        if np.isfinite(burst):
            bursts.append(burst)
    return float(np.mean(rhos)), float(np.mean(bursts))


def test_generated_gaps_are_positively_correlated() -> None:
    """Independent draws cannot exceed zero, whatever the distribution."""
    rho, _ = mean_over_entities(ArrivalConfig(), seed=7777)
    assert rho > 0.0


def test_autocorrelation_lands_near_its_target() -> None:
    config = ArrivalConfig()
    rho, _ = mean_over_entities(config, seed=7777)
    assert rho == pytest.approx(config.target_autocorrelation, abs=0.02)


def test_burstiness_lands_near_its_target() -> None:
    config = ArrivalConfig()
    _, burst = mean_over_entities(config, seed=7777)
    assert burst == pytest.approx(config.target_burstiness, abs=0.03)


def test_removing_the_drift_removes_the_correlation() -> None:
    """The drift is the whole mechanism, so disabling it has to undo the effect."""
    flat, _ = mean_over_entities(
        ArrivalConfig(drift_persistence=0.0, drift_sigma=0.0), seed=11
    )
    drifting, _ = mean_over_entities(ArrivalConfig(), seed=11)
    assert flat <= 0.0
    assert drifting > flat


def test_persistence_raises_the_correlation() -> None:
    low, _ = mean_over_entities(ArrivalConfig(drift_persistence=0.1), seed=5)
    high, _ = mean_over_entities(ArrivalConfig(drift_persistence=0.95), seed=5)
    assert high > low


def test_rate_spread_is_per_entity() -> None:
    """A shared rate has nowhere to put the spread across entities and ends up
    absorbing it into the coupling instead."""
    process = DriftingRateProcess(ArrivalConfig())
    rng = np.random.default_rng(3)
    scales = [process.new_state(rng).base_scale_seconds for _ in range(2000)]
    assert np.quantile(scales, 0.9) / np.quantile(scales, 0.1) > 5.0


def test_gaps_never_fall_below_one_tick() -> None:
    process = DriftingRateProcess(ArrivalConfig())
    rng = np.random.default_rng(1)
    state = process.new_state(rng)
    assert all(process.next_gap_minutes(state, rng) >= 1 for _ in range(500))


def test_scheduler_applies_archetype_and_activity_scaling() -> None:
    scheduler = ArrivalScheduler(
        ArrivalConfig(), archetype_rate_scale={Archetype.BUSINESS: 4.0}
    )
    rng = np.random.default_rng(2)
    busy = [
        scheduler.register(i, rng, Archetype.BUSINESS).base_scale_seconds
        for i in range(400)
    ]
    quiet = [
        scheduler.register(1000 + i, rng, Archetype.SENIOR, activity_multiplier=0.2)
        .base_scale_seconds
        for i in range(400)
    ]
    assert np.median(busy) < np.median(quiet)


def test_scheduler_rejects_unregistered_entities() -> None:
    with pytest.raises(KeyError, match="never registered"):
        ArrivalScheduler(ArrivalConfig()).next_gap_minutes(99, np.random.default_rng(0))


def test_circadian_matches_its_concentration() -> None:
    clock = CircadianClock(CircadianConfig())
    hours = clock.sample_hour(np.random.default_rng(3), 100_000)
    assert resultant_length(hours) == pytest.approx(0.4341, abs=0.06)


def test_usual_range_wraps_midnight() -> None:
    """A linear interval around a late mean would exclude the hours either side
    of midnight and include the middle of the day, which is backwards."""
    clock = CircadianClock(CircadianConfig())
    assert clock.contains(23.7)
    assert clock.contains(0.3)
    assert not clock.contains(8.5)


def test_usual_range_covers_roughly_its_confidence() -> None:
    clock = CircadianClock(CircadianConfig(confidence=0.9))
    hours = clock.sample_hour(np.random.default_rng(4), 20_000)
    inside = float(np.mean([clock.contains(float(h)) for h in hours]))
    assert inside == pytest.approx(0.9, abs=0.08)


def test_density_integrates_to_one() -> None:
    clock = CircadianClock(CircadianConfig())
    grid = np.linspace(0, 24, 2000, endpoint=False)
    assert float(np.trapezoid(clock.density(grid), grid)) == pytest.approx(1.0, rel=0.02)


def test_circular_mean_handles_the_wrap() -> None:
    """The average of 23:00 and 01:00 is midnight, not noon."""
    assert circular_mean_hour(np.array([23.0, 1.0])) == pytest.approx(0.0, abs=0.1)
