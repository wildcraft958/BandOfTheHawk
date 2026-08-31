"""Holders keep their own hours, not the population's."""

from __future__ import annotations

import pytest

pd = pytest.importorskip(
    "pandas", reason='install the "calibration" extra'
)

import numpy as np


from fraudsim.calibration.entity_stats import circular_entity_spread
from fraudsim.settings.behavior import CircadianConfig
from fraudsim.timing.circadian import CircadianClock, HolderClock, HolderClockModel

TWO_PI = 2.0 * np.pi


def generated_hours(config: CircadianConfig, n_holders=1500, n_events=25, seed=0):
    """Hours from a population of per-holder clocks."""
    rng = np.random.default_rng(seed)
    model = HolderClockModel(config)
    rows = []
    for holder in range(n_holders):
        clock = model.register(holder, rng)
        for _ in range(n_events):
            rows.append({"card": holder, "hour": clock.sample_hour(rng)})
    return pd.DataFrame(rows)


def test_all_three_statistics_together() -> None:
    """One test, three targets.

    Split across separate tests, a change that fixed the marginal while
    breaking the between-holder term would leave one of them green and look
    like a partial success. They are properties of a single population and
    are asserted as one.
    """
    config = CircadianConfig()
    spread = circular_entity_spread(generated_hours(config), "card", "hour", min_events=10)

    assert spread.marginal_r == pytest.approx(0.4341, rel=0.15)
    assert spread.within_r == pytest.approx(0.4814, rel=0.15)
    assert spread.between_r == pytest.approx(0.8502, rel=0.15)
    assert spread.marginal_mean == pytest.approx(20.5, abs=1.5)


def test_uniform_hours_fail_the_between_statistic() -> None:
    """The defect this replaces.

    Events placed with no time-of-day term at all give holders no hours of
    their own, and the between-holder term collapses even though the marginal
    is the more obvious casualty.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        [
            {"card": holder, "hour": float(rng.uniform(0, 24))}
            for holder in range(1200)
            for _ in range(25)
        ]
    )
    spread = circular_entity_spread(frame, "card", "hour", min_events=10)
    assert spread.between_r < 0.35
    assert spread.marginal_r < 0.15


def test_a_shared_population_clock_overshoots_agreement() -> None:
    """The trap, and it runs the opposite way to the obvious guess.

    Drawing every event from the population mixture reproduces the marginal
    almost exactly - it is the right curve, after all. What it gets wrong is
    the spread between holders, and not by scattering them: every holder's
    preferred hour converges on the same population mean, so they agree with
    one another far more than real holders do.

    Real cardholders sit between the two failures. They largely share an
    evening rhythm, which is why the between term is high at 0.79, but they
    are not copies of one another, which is why it is not 0.94. A generator
    matching only the marginal lands on the wrong side of that and no
    marginal comparison would say so.
    """
    rng = np.random.default_rng(0)
    shared = CircadianClock(CircadianConfig())
    frame = pd.DataFrame(
        [
            {"card": holder, "hour": float(shared.sample_hour(rng, size=1)[0])}
            for holder in range(1200)
            for _ in range(25)
        ]
    )
    shared_spread = circular_entity_spread(frame, "card", "hour", min_events=10)

    # The marginal is fine, which is exactly why this is a trap.
    assert shared_spread.marginal_r == pytest.approx(0.4341, rel=0.20)
    # And the holders are near-copies of each other.
    assert shared_spread.between_r > 0.90

    # Per-holder clocks land near the measured 0.85, on the right side of it.
    per_holder = circular_entity_spread(
        generated_hours(CircadianConfig()), "card", "hour", min_events=10
    )
    assert per_holder.between_r < shared_spread.between_r - 0.05
    assert per_holder.between_r == pytest.approx(0.8502, rel=0.15)


def test_between_concentration_drives_agreement() -> None:
    """Loosening how much holders agree has to show up as scatter."""
    tight = circular_entity_spread(
        generated_hours(CircadianConfig(kappa_between=8.0)), "card", "hour", min_events=10
    )
    loose = circular_entity_spread(
        generated_hours(CircadianConfig(kappa_between=0.5)), "card", "hour", min_events=10
    )
    assert tight.between_r > loose.between_r + 0.3


def test_usual_hours_are_this_holders_hours() -> None:
    late = HolderClock(preferred_hour=23.0, kappa=4.0)
    morning = HolderClock(preferred_hour=9.0, kappa=4.0)
    assert late.contains(23.5) and not late.contains(9.0)
    assert morning.contains(9.5) and not morning.contains(23.0)


def test_usual_hours_wrap_across_midnight() -> None:
    """An arc centred at 23:30 has to include 00:30, which a linear interval
    would exclude while including the middle of the afternoon."""
    clock = HolderClock(preferred_hour=23.5, kappa=6.0)
    assert clock.contains(0.5)
    assert not clock.contains(12.0)


def test_a_holder_with_no_habit_has_no_unusual_hours() -> None:
    """Reporting unusual hours for a holder with no rhythm invents a signal."""
    clock = HolderClock(preferred_hour=12.0, kappa=0.0)
    assert all(clock.contains(hour) for hour in range(24))


def test_clocks_are_drawn_once_and_kept() -> None:
    rng = np.random.default_rng(0)
    model = HolderClockModel(CircadianConfig())
    first = model.register(7, rng)
    assert model.clock(7) is first
    assert model.require(7, rng) is first
    assert model.clock(99) is None
