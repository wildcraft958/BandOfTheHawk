"""Time of day.

Transaction time is circular: 23:00 and 01:00 are two hours apart, not
twenty-two. A linear hour bucket cannot represent that, and any spread or
confidence interval computed on the raw hour is wrong across midnight. The von
Mises distribution is the circular analogue of the normal and handles it
directly.

The same fitted object does two jobs. It draws hours for a generated
population, and it answers whether a given transaction falls inside a holder's
usual hours, which is a feature in its own right. Fitting once and using it for
both keeps the generator and the feature consistent by construction.

Two components are the default. A single component misses the daytime shoulder
in the judge dataset badly, landing about seventeen times its noise floor,
while two components reach 2.3 and a third adds little.

One caution on evaluating this fit. The population's circular mean hour drifts
from 20.6 to 19.9 across the six months of the source, so a holdout split by
time charges any static fit for seasonal movement it was never meant to
describe. Comparisons belong on an entity split, which is also how the noise
floor itself was measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import i0

HOURS_PER_DAY = 24.0
TWO_PI = 2.0 * np.pi


def hours_to_angle(hours: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(hours) * TWO_PI / HOURS_PER_DAY


def angle_to_hours(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) % TWO_PI) * HOURS_PER_DAY / TWO_PI


@dataclass(frozen=True, slots=True)
class CircadianFit:
    """A von Mises mixture over hour of day."""

    means: tuple[float, ...]
    concentrations: tuple[float, ...]
    weights: tuple[float, ...]
    resultant_length: float
    n_samples: int

    @property
    def n_components(self) -> int:
        return len(self.means)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw hours in [0, 24)."""
        picks = rng.choice(self.n_components, size=size, p=np.asarray(self.weights))
        angles = np.empty(size)
        for component in range(self.n_components):
            mask = picks == component
            count = int(mask.sum())
            if count:
                angles[mask] = rng.vonmises(
                    hours_to_angle(self.means[component]),
                    self.concentrations[component],
                    count,
                )
        return np.asarray(angle_to_hours(angles))

    def density(self, hours: np.ndarray) -> np.ndarray:
        angles = np.asarray(hours_to_angle(hours))
        total = np.zeros_like(angles, dtype=float)
        for mean, kappa, weight in zip(self.means, self.concentrations, self.weights, strict=False):
            centred = angles - hours_to_angle(mean)
            total += weight * np.exp(kappa * np.cos(centred)) / (TWO_PI * i0(kappa))
        return total

    def contains(self, hour: float, coverage: float = 0.95) -> bool:
        """Whether an hour falls in the densest region holding `coverage` mass.

        Used as a detector feature. A holder who never transacts at 03:00 makes
        that hour unusual for them even though it is ordinary for someone else,
        which a population-level hour bucket cannot express.
        """
        grid = np.linspace(0.0, HOURS_PER_DAY, 480, endpoint=False)
        densities = self.density(grid)
        order = np.argsort(densities)[::-1]
        cumulative = np.cumsum(densities[order])
        cumulative /= cumulative[-1]
        cutoff = densities[order][np.searchsorted(cumulative, coverage)]
        return bool(self.density(np.asarray([hour]))[0] >= cutoff)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def circular_moments(hours: np.ndarray) -> tuple[float, float]:
    """Circular mean hour and resultant length.

    The resultant runs from 0 for a flat distribution to 1 for a spike, and is
    what a linear standard deviation gets wrong across midnight.
    """
    angles = hours_to_angle(np.asarray(hours, dtype=float))
    cos_mean = float(np.cos(angles).mean())
    sin_mean = float(np.sin(angles).mean())
    resultant = float(np.hypot(cos_mean, sin_mean))
    mean_hour = float(angle_to_hours(np.arctan2(sin_mean, cos_mean)))
    return mean_hour, resultant


def concentration_from_resultant(resultant: float) -> float:
    """Standard approximation of the von Mises concentration."""
    r = float(np.clip(resultant, 1e-6, 0.999999))
    if r < 0.53:
        return 2 * r + r**3 + 5 * r**5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1.0 / (r**3 - 4 * r**2 + 3 * r)


def fit_circadian(hours: np.ndarray, n_components: int = 1, seed: int = 0) -> CircadianFit:
    """Fit a von Mises mixture by expectation maximisation."""
    hours = np.asarray(hours, dtype=float) % HOURS_PER_DAY
    if len(hours) < 10:
        raise ValueError(f"need at least 10 observations, got {len(hours)}")

    mean_hour, resultant = circular_moments(hours)
    if n_components == 1:
        return CircadianFit(
            means=(mean_hour,),
            concentrations=(concentration_from_resultant(resultant),),
            weights=(1.0,),
            resultant_length=resultant,
            n_samples=len(hours),
        )

    rng = np.random.default_rng(seed)
    angles = hours_to_angle(hours)
    means = hours_to_angle(np.linspace(0, HOURS_PER_DAY, n_components, endpoint=False))
    kappas = np.full(n_components, max(concentration_from_resultant(resultant), 0.5))
    weights = np.full(n_components, 1.0 / n_components)

    for _ in range(60):
        responsibilities = np.empty((len(angles), n_components))
        for k in range(n_components):
            responsibilities[:, k] = (
                weights[k] * np.exp(kappas[k] * np.cos(angles - means[k])) / i0(kappas[k])
            )
        totals = responsibilities.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1e-12
        responsibilities /= totals

        weights = responsibilities.mean(axis=0)
        for k in range(n_components):
            weight_sum = responsibilities[:, k].sum()
            if weight_sum < 1e-9:
                means[k] = rng.uniform(0, TWO_PI)
                continue
            cos_k = float((responsibilities[:, k] * np.cos(angles)).sum() / weight_sum)
            sin_k = float((responsibilities[:, k] * np.sin(angles)).sum() / weight_sum)
            means[k] = float(np.arctan2(sin_k, cos_k))
            kappas[k] = concentration_from_resultant(float(np.hypot(cos_k, sin_k)))

    order = np.argsort(weights)[::-1]
    return CircadianFit(
        means=tuple(float(angle_to_hours(means[k])) for k in order),
        concentrations=tuple(float(kappas[k]) for k in order),
        weights=tuple(float(weights[k]) for k in order),
        resultant_length=resultant,
        n_samples=len(hours),
    )


@dataclass(frozen=True, slots=True)
class HierarchicalCircadianFit:
    """Hour as a habit each holder has, rather than one the population shares.

    The mixture above describes when transactions happen. It says nothing
    about whether a given holder keeps to their own hours, and a generator
    built from it alone gives every holder the population's curve: their
    apparent preferences then differ only by sampling, which is the amount
    defect in a circular coordinate.

    Two numbers describe the population instead of one: holders concentrate
    around their own preferred hour (`kappa_within`), and those preferred hours
    concentrate around a population mode (`kappa_between`). A third was tried
    and abandoned - see the fitting function.

    `kappa_within` is inflated slightly above the value the entity-level
    resultant implies, so that the generated marginal lands on the measured
    one. The correction is recorded as `marginal_gain` rather than folded in
    silently, because it is compensating for a shape mismatch rather than
    estimating anything.
    """

    population_mode_hour: float
    kappa_between: float
    kappa_within_mean: float
    marginal_gain: float
    hour_marginal_r: float
    hour_within_r: float
    hour_between_r: float
    n_entities: int
    n_events: int

    def as_dict(self) -> dict[str, float]:
        return {
            "population_mode_hour": self.population_mode_hour,
            "kappa_between": self.kappa_between,
            "kappa_within_mean": self.kappa_within_mean,
            "marginal_gain": self.marginal_gain,
            "hour_marginal_r": self.hour_marginal_r,
            "hour_within_r": self.hour_within_r,
            "hour_between_r": self.hour_between_r,
            "n_entities": float(self.n_entities),
            "n_events": float(self.n_events),
        }


def _simulate_marginal_r(
    kappa_between: float,
    kappa_within: float,
    n_entities: int,
    n_events: int,
    seed: int,
) -> float:
    """Marginal resultant a population with these parameters would produce."""
    rng = np.random.default_rng(seed)
    preferred = rng.vonmises(0.0, kappa_between, n_entities)
    angles = rng.vonmises(
        np.repeat(preferred, n_events), max(kappa_within, 1e-3)
    )
    return float(np.hypot(np.cos(angles).mean(), np.sin(angles).mean()))


def fit_hierarchical_circadian(
    frame,
    entity_column: str,
    hour_column: str,
    min_events: int = 10,
    seed: int = 0,
) -> HierarchicalCircadianFit:
    """Fit per-holder hour habits against all three measured resultants.

    For a von Mises within a von Mises the resultants multiply, so two
    parameters would suffice if the population were homogeneous. It is not:
    measured on the judge dataset the product of the within and between terms
    falls short of the marginal by between three and ten percent, and the
    shortfall moves with the cutoff. Event-weighting shrinks it without closing
    it, so a residual remains that is genuine heterogeneity rather than an
    artefact of how entities were weighted.

    Both parameters invert their measured resultants directly, and the
    within-holder one is then nudged until the generated marginal matches the
    measured marginal. What that nudge compensates for, and what it is not, is
    recorded in the body.
    """
    from .entity_stats import circular_entity_spread, resultant_to_kappa

    spread = circular_entity_spread(
        frame, entity_column, hour_column, min_events=min_events
    )
    kappa_between = resultant_to_kappa(spread.between_r)
    base_within = resultant_to_kappa(spread.within_r)

    # Calibrate the within-holder tightness so the generated marginal lands on
    # the measured one, and record how far it had to move.
    #
    # A spread on the tightness was tried here first, on the theory that
    # tighter holders transact more and so carry extra weight in the marginal.
    # Measurement says otherwise: busier entities keep looser hours, not
    # tighter (corrected R falls from 0.492 to 0.452 across activity bands,
    # correlating at -0.08 with log activity). Spreading the tightness also
    # moves the marginal the wrong way, monotonically, because the resultant
    # is concave in the concentration and spreading it lowers the mean.
    #
    # What remains is a shape mismatch rather than a missing parameter. Real
    # holders are not von Mises about a single hour - they have a midday and
    # an evening peak - so a single-component fit reproducing their resultant
    # under-delivers the marginal by a few percent. The gain compensates for
    # that, and is reported so nobody mistakes it for something measured.
    low, high = base_within, base_within * 4.0
    for _ in range(24):
        mid = 0.5 * (low + high)
        if _simulate_marginal_r(kappa_between, mid, 900, 30, seed + 17) < spread.marginal_r:
            low = mid
        else:
            high = mid
    kappa_within = 0.5 * (low + high)

    return HierarchicalCircadianFit(
        population_mode_hour=spread.marginal_mean,
        kappa_between=kappa_between,
        kappa_within_mean=kappa_within,
        marginal_gain=kappa_within / base_within if base_within else float("nan"),
        hour_marginal_r=spread.marginal_r,
        hour_within_r=spread.within_r,
        hour_between_r=spread.between_r,
        n_entities=spread.n_entities,
        n_events=spread.n_events,
    )
