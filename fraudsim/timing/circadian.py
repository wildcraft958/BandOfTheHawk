"""Time of day at runtime.

Hour is circular, so 23:00 and 01:00 are two hours apart rather than
twenty-two, and any spread or interval computed on the raw number is wrong
across midnight. A von Mises mixture handles it directly.

The same object does two jobs. It draws hours when generating activity, and it
answers whether an hour falls inside a holder's usual range, which is a feature
in its own right. Fitting once and using it for both keeps the generator and
the feature consistent by construction, rather than by remembering to update
two places.
"""

from __future__ import annotations

import numpy as np

from ..config.behavior import CircadianConfig

HOURS_PER_DAY = 24.0
TWO_PI = 2.0 * np.pi
MINUTES_PER_DAY = 1440
_GRID = np.linspace(0.0, HOURS_PER_DAY, 480, endpoint=False)


def _to_angle(hours: np.ndarray | float) -> np.ndarray:
    return np.asarray(hours) * TWO_PI / HOURS_PER_DAY


def _to_hours(angles: np.ndarray | float) -> np.ndarray:
    """Angles back to hours in [0, 24).

    The wrap is applied after scaling as well as before. An angle at exactly
    two pi is zero on the circle, but scaling it first lands on 24.0, which is
    outside the interval this claims to return.
    """
    hours = (np.asarray(angles) % TWO_PI) * HOURS_PER_DAY / TWO_PI
    return hours % HOURS_PER_DAY


def _bessel_i0(kappa: float) -> float:
    """Modified Bessel function of the first kind, order zero.

    Written out rather than imported so this module stays on the runtime tier,
    which carries no scipy dependency.
    """
    x = abs(float(kappa))
    if x < 3.75:
        t = (x / 3.75) ** 2
        return (
            1.0
            + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
            + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
        )
    t = 3.75 / x
    coefficients = (
        0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281,
        -0.02057706, 0.02635537, -0.01647633, 0.00392377,
    )
    series = 0.0
    for power, coefficient in enumerate(coefficients):
        series += coefficient * t**power
    return float(np.exp(x) / np.sqrt(x) * series)


class CircadianClock:
    """A von Mises mixture over hour of day."""

    __slots__ = ("_means", "_kappas", "_weights", "_confidence", "_cutoff")

    def __init__(self, config: CircadianConfig) -> None:
        self._means = np.asarray(config.means, dtype=float)
        self._kappas = np.asarray(config.concentrations, dtype=float)
        self._weights = np.asarray(config.weights, dtype=float)
        self._weights = self._weights / self._weights.sum()
        self._confidence = config.confidence
        self._cutoff = self._density_cutoff(config.confidence)

    @property
    def n_components(self) -> int:
        return len(self._means)

    def sample_hour(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        picks = rng.choice(self.n_components, size=size, p=self._weights)
        angles = np.empty(size)
        for component in range(self.n_components):
            mask = picks == component
            count = int(mask.sum())
            if count:
                angles[mask] = rng.vonmises(
                    float(_to_angle(self._means[component])),
                    float(self._kappas[component]),
                    count,
                )
        return _to_hours(angles)

    def sample_minute_of_day(self, rng: np.random.Generator) -> int:
        hour = float(self.sample_hour(rng, size=1)[0])
        return int(hour * 60) % MINUTES_PER_DAY

    def density(self, hours: np.ndarray | float) -> np.ndarray:
        """Density per hour, integrating to one across the day.

        The von Mises form is normalised in radians, so it has to be rescaled
        to the hour axis it is evaluated on. Without that it integrates to
        roughly six, which passes unnoticed while any comparison of densities
        is unaffected but makes the value itself meaningless.
        """
        angles = _to_angle(np.atleast_1d(hours))
        total = np.zeros_like(angles, dtype=float)
        for mean, kappa, weight in zip(self._means, self._kappas, self._weights):
            centred = angles - _to_angle(mean)
            total += weight * np.exp(kappa * np.cos(centred)) / (TWO_PI * _bessel_i0(kappa))
        return total * (TWO_PI / HOURS_PER_DAY)

    def _density_cutoff(self, coverage: float) -> float:
        """Density bounding the smallest region that holds `coverage` mass."""
        densities = self.density(_GRID)
        order = np.argsort(densities)[::-1]
        cumulative = np.cumsum(densities[order])
        cumulative /= cumulative[-1]
        index = int(np.searchsorted(cumulative, coverage))
        return float(densities[order][min(index, len(order) - 1)])

    def contains(self, hour: float) -> bool:
        """Whether an hour sits in the usual range.

        The interval is the densest region holding the configured mass, not a
        mean plus a spread. On a circle the latter is wrong: an interval around
        a mean of 23:30 has to wrap past midnight, and a linear one would
        exclude the hours either side while including the middle of the day.
        """
        return bool(self.density(float(hour))[0] >= self._cutoff)

    def contains_timestamp(self, minutes: int) -> bool:
        return self.contains((minutes % MINUTES_PER_DAY) / 60.0)


def circular_mean_hour(hours: np.ndarray) -> float:
    angles = _to_angle(np.asarray(hours, dtype=float))
    return float(_to_hours(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())))


def resultant_length(hours: np.ndarray) -> float:
    """Concentration on the circle: 0 for flat, 1 for a spike."""
    angles = _to_angle(np.asarray(hours, dtype=float))
    return float(np.hypot(np.cos(angles).mean(), np.sin(angles).mean()))
