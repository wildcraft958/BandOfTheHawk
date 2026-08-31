"""Per-entity structure, and the estimators that can see it.

A marginal metric pools every event and asks whether the shape matches, which
ignores which entity produced each one. A generator can match a pooled
distribution exactly while distributing it across entities entirely wrongly,
and a detector reading a value against an entity's own history reads precisely
what the pooled version gets wrong. Amount was found that way: off by a factor
of 43 per card while its marginal looked fine.

The statistics here are the per-entity view for the features amount does not
cover: hour of day, which is circular, and category and merchant, which are
categorical.

Every one of them needs a sampling correction, and for the same reason the
amount fit did. An entity seen k times looks concentrated purely for having
been seen k times: one event has a circular resultant of exactly 1 and a
Simpson index of exactly 1, whatever the process behind it. Uncorrected, the
estimate reads high and drifts with the cutoff, so a generator tuned against
it inherits the drift.

The corrections are different in form but identical in intent to the
`within**2 / k` term subtracted in `fit_heterogeneity`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24.0
TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------- circular


@dataclass(frozen=True, slots=True)
class CircularSpread:
    """How a population distributes a circular quantity, at three levels.

    The three are not independent: for a von Mises within a von Mises the
    resultants multiply, so `marginal_r` is approximately
    `within_r * between_r`. They are reported separately because a generator
    can hit any one of them while missing the others, and only the pair
    together says whether entities have their own habits or merely inherit
    the population's.
    """

    marginal_r: float
    marginal_mean: float
    within_r: float
    within_r_raw: float
    between_r: float
    preferred: np.ndarray
    n_entities: int
    n_events: int
    min_events: int

    def summary(self) -> dict[str, float]:
        return {
            "marginal_r": self.marginal_r,
            "marginal_mean": self.marginal_mean,
            "within_r": self.within_r,
            "within_r_raw": self.within_r_raw,
            "between_r": self.between_r,
            "n_entities": float(self.n_entities),
            "n_events": float(self.n_events),
        }


def fisher_corrected_r(resultants: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Resultant length with the small-sample bias removed.

    An entity seen once has a resultant of exactly 1: its single event points
    somewhere, and one vector is perfectly concentrated by construction. The
    same effect at k events inflates R by roughly 1/k even when the underlying
    process is uniform, so the raw mean confounds concentration with how much
    history an entity happens to have.

    Under a uniform null E[R^2] = 1/k, which gives the correction below. On the
    judge dataset it moves the estimate from 0.565 to 0.492 at a five-event
    cutoff, and -- the part that matters -- makes it stop drifting: raw R falls
    to 0.496 as the cutoff rises to twenty while the corrected value holds near
    0.48. Without it, a generator tuned at one cutoff is wrong at every other.
    """
    counts = np.asarray(counts, dtype=float)
    resultants = np.asarray(resultants, dtype=float)
    corrected = (counts * resultants**2 - 1.0) / np.maximum(counts - 1.0, 1e-12)
    return np.sqrt(np.clip(corrected, 0.0, None))


def circular_entity_spread(
    frame: pd.DataFrame,
    entity_column: str,
    value_column: str,
    period: float = HOURS_PER_DAY,
    min_events: int = 10,
) -> CircularSpread:
    """Marginal, within-entity, and between-entity concentration.

    Entities below `min_events` are dropped from the within and between
    estimates but still count towards the marginal, which needs no history per
    entity and is more accurate for including them.
    """
    values = frame[[entity_column, value_column]].dropna()
    angles = values[value_column].to_numpy(dtype=float) * TWO_PI / period

    cos_total, sin_total = np.cos(angles).mean(), np.sin(angles).mean()
    marginal_r = float(np.hypot(cos_total, sin_total))
    marginal_mean = float((np.arctan2(sin_total, cos_total) % TWO_PI) * period / TWO_PI)

    work = pd.DataFrame(
        {"entity": values[entity_column].to_numpy(), "cos": np.cos(angles), "sin": np.sin(angles)}
    )
    grouped = work.groupby("entity", observed=True).agg(
        cos=("cos", "sum"), sin=("sin", "sum"), n=("cos", "size")
    )
    usable = grouped[grouped["n"] >= min_events]
    if usable.empty:
        raise ValueError(f"no entity has {min_events} or more observations")

    counts = usable["n"].to_numpy(dtype=float)
    raw = np.hypot(usable["cos"].to_numpy(), usable["sin"].to_numpy()) / counts
    corrected = fisher_corrected_r(raw, counts)

    # Each entity's preferred angle, then how tightly those agree with one
    # another. This is the between-entity term, and it is the one a marginal
    # comparison is blind to: a population whose members all peak at different
    # hours and one where they peak together can share a marginal exactly.
    preferred_angles = np.arctan2(usable["sin"].to_numpy(), usable["cos"].to_numpy())
    between_r = float(
        np.hypot(np.cos(preferred_angles).mean(), np.sin(preferred_angles).mean())
    )

    return CircularSpread(
        marginal_r=marginal_r,
        marginal_mean=marginal_mean,
        within_r=float(corrected.mean()),
        within_r_raw=float(raw.mean()),
        between_r=between_r,
        preferred=(preferred_angles % TWO_PI) * period / TWO_PI,
        n_entities=len(usable),
        n_events=len(angles),
        min_events=min_events,
    )


def resultant_to_kappa(resultant: float, upper: float = 400.0) -> float:
    """Invert A(kappa) = I1(kappa)/I0(kappa) by bisection.

    A is strictly increasing from 0 to 1, so bisection is exact enough and
    carries no dependency. Written here rather than imported so the fit and
    the runtime generator agree on the inverse by construction.
    """
    if not 0.0 < resultant < 1.0:
        return 0.0 if resultant <= 0.0 else upper

    low, high = 0.0, upper
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _bessel_ratio(mid) < resultant:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _bessel_ratio(kappa: float) -> float:
    """I1(kappa)/I0(kappa), the mean resultant of a von Mises."""
    if kappa <= 0:
        return 0.0
    # Series for both orders, sharing terms. Stable well past the range any
    # circadian fit reaches.
    from scipy.special import i0e, i1e

    return float(i1e(kappa) / i0e(kappa))


# ------------------------------------------------------------ categorical


@dataclass(frozen=True, slots=True)
class ConcentrationSpread:
    """Whether entities favour particular values, or merely inherit the mix.

    The ratio is the whole point. A ratio near one says every entity draws
    from the same curve, which is the defect this module exists to detect;
    above one says entities have their own habits.

    The null is measured rather than assumed. Shuffling values across entities
    destroys any habit while preserving both the marginal and each entity's
    event count, and the resulting ratio is not exactly one even after the
    unbiased correction, because entity sizes are heterogeneous.
    """

    marginal_simpson: float
    within_simpson: float
    within_simpson_raw: float
    ratio: float
    null_ratio_mean: float
    null_ratio_sd: float
    n_entities: int
    n_events: int

    @property
    def z_against_null(self) -> float:
        """How far the observed ratio sits above chance, in null sigmas."""
        if not np.isfinite(self.null_ratio_sd) or self.null_ratio_sd <= 0:
            return float("nan")
        return (self.ratio - self.null_ratio_mean) / self.null_ratio_sd

    def summary(self) -> dict[str, float]:
        return {
            "marginal_simpson": self.marginal_simpson,
            "within_simpson": self.within_simpson,
            "within_simpson_raw": self.within_simpson_raw,
            "ratio": self.ratio,
            "null_ratio_mean": self.null_ratio_mean,
            "z_against_null": self.z_against_null,
            "n_entities": float(self.n_entities),
            "n_events": float(self.n_events),
        }


def unbiased_simpson(counts: np.ndarray) -> float:
    """Probability two distinct draws from one entity land on the same value.

    The plug-in form, sum of squared shares, is biased upward by roughly
    (1 - S)/n: an entity seen twice reads as concentrated if its two events
    happen to match. Drawing without replacement removes it exactly.

    On a source with fifteen hundred events per entity the difference is
    invisible. On generated traffic at ten events per entity it is most of the
    signal, which is why the estimator matters more here than where it was
    first measured.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n < 2:
        return float("nan")
    return float((counts * (counts - 1.0)).sum() / (n * (n - 1.0)))


def categorical_entity_concentration(
    frame: pd.DataFrame,
    entity_column: str,
    value_column: str,
    min_events: int = 10,
    n_shuffles: int = 20,
    seed: int = 0,
) -> ConcentrationSpread:
    """Per-entity concentration against the population mix and a shuffled null.

    The marginal uses the pooled plug-in and the per-entity term uses the
    unbiased form. That asymmetry is deliberate: the marginal is computed over
    every event at once, where the bias is negligible, while the per-entity
    term is computed over a handful and where it is not. Using one estimator
    for both would either inflate the numerator or deflate the denominator.
    """
    values = frame[[entity_column, value_column]].dropna()
    entities = values[entity_column].to_numpy()
    observed_values = values[value_column].to_numpy()

    shares = pd.Series(observed_values).value_counts(normalize=True).to_numpy()
    marginal = float((shares**2).sum())

    def within(sample: np.ndarray) -> tuple[float, float]:
        work = pd.DataFrame({"entity": entities, "value": sample})
        sizes = work.groupby("entity", observed=True)["value"].size()
        keep = set(sizes[sizes >= min_events].index)
        if not keep:
            raise ValueError(f"no entity has {min_events} or more observations")

        subset = work[work["entity"].isin(keep)]
        counts = subset.groupby(["entity", "value"], observed=True).size()
        unbiased, plugin = [], []
        for _, group in counts.groupby(level=0, observed=True):
            per_entity = group.to_numpy(dtype=float)
            unbiased.append(unbiased_simpson(per_entity))
            total = per_entity.sum()
            plugin.append(float(((per_entity / total) ** 2).sum()))
        return float(np.nanmean(unbiased)), float(np.nanmean(plugin))

    within_unbiased, within_plugin = within(observed_values)

    # The null: same marginal, same event counts, no entity habits. Measured
    # rather than assumed to be one.
    rng = np.random.default_rng(seed)
    null_ratios = []
    for _ in range(n_shuffles):
        shuffled, _ = within(rng.permutation(observed_values))
        null_ratios.append(shuffled / marginal if marginal else float("nan"))
    null = np.asarray(null_ratios, dtype=float)

    sizes = values.groupby(entity_column, observed=True).size()
    return ConcentrationSpread(
        marginal_simpson=marginal,
        within_simpson=within_unbiased,
        within_simpson_raw=within_plugin,
        ratio=within_unbiased / marginal if marginal else float("nan"),
        null_ratio_mean=float(null.mean()) if len(null) else float("nan"),
        null_ratio_sd=float(null.std(ddof=1)) if len(null) > 1 else float("nan"),
        n_entities=int((sizes >= min_events).sum()),
        n_events=len(values),
    )


# ------------------------------------------------------------------ matching


def matched_by_event_count(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    entity_column: str,
    statistic,
    bands: tuple[tuple[int, int], ...] = ((5, 9), (10, 19), (20, 49), (50, 10**9)),
) -> pd.DataFrame:
    """Compare a per-entity statistic within bands of equal history length.

    Comparing every entity at once mixes sparse with dense, and every
    statistic here varies with how many events an entity has. A real
    population whose median is two events against a generated one whose
    activity tiers put it elsewhere would then report a difference in census
    as a difference in behaviour -- misleading in both directions, since sparse
    entities scatter more whatever the model.

    `statistic` takes a frame and returns a float.
    """
    def slice_of(frame: pd.DataFrame, low: int, high: int) -> pd.DataFrame:
        sizes = frame.groupby(entity_column, observed=True).size()
        keep = sizes[(sizes >= low) & (sizes <= high)].index
        return frame[frame[entity_column].isin(keep)]

    rows = []
    for low, high in bands:
        left, right = slice_of(real, low, high), slice_of(generated, low, high)
        if left.empty or right.empty:
            continue
        real_value, generated_value = statistic(left), statistic(right)
        rows.append(
            {
                "band": f"{low}-{high if high < 10**9 else '+'}",
                "real": real_value,
                "generated": generated_value,
                "ratio": generated_value / real_value if real_value else float("nan"),
                "n_real": int(left[entity_column].nunique()),
                "n_generated": int(right[entity_column].nunique()),
            }
        )
    return pd.DataFrame(rows)
