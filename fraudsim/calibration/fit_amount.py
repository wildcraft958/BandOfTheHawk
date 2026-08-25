"""Transaction amount.

A single lognormal underestimates how often a legitimate purchase is very
large, which matters because the legitimate large purchase is one of the hard
negatives the defender has to survive. The body is lognormal and the tail is
Pareto, spliced at a quantile chosen so the pieces meet.

The tail index is estimated with the Hill estimator over the top few percent.
An index below one implies an infinite mean, which is a sign the sample is
contaminated rather than heavy tailed, so the fit reports it either way.

On what this model can and cannot reach. Fitted against the judge dataset, a
pooled fit lands around four times its noise floor and stays there under
tuning. That is not a defect in the estimator. Resampling the source
distribution directly scores about a seventh of the same floor, so the floor is
not sampling noise: it was measured between two disjoint sets of cardholders
and therefore contains real differences between populations. One pooled
distribution has no way to express per-entity heterogeneity, so the remaining
distance closes by fitting per archetype, not by adjusting these parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AmountFit:
    """Lognormal body spliced to a Pareto tail.

    The tail is truncated. An unbounded Pareto with an index near 1.8 produces
    occasional draws several times larger than anything in the source, which
    shows up immediately as a widened distance even though every summary
    statistic looks right. Real amounts are bounded by credit lines and
    acceptance limits, so the fitted maximum is carried and enforced.
    """

    lognormal_mu: float
    lognormal_sigma: float
    tail_threshold: float
    tail_index: float
    tail_fraction: float
    upper_bound: float
    whole_number_share: float
    median: float
    mean: float
    n_samples: int

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw amounts, splicing at the fitted threshold."""
        draws = rng.lognormal(self.lognormal_mu, self.lognormal_sigma, size)
        in_tail = rng.random(size) < self.tail_fraction
        n_tail = int(in_tail.sum())
        if n_tail:
            # Inverse-transform a Pareto restricted to [threshold, upper_bound].
            u = rng.random(n_tail)
            ceiling = (self.tail_threshold / self.upper_bound) ** self.tail_index
            scaled = 1.0 - u * (1.0 - ceiling)
            draws[in_tail] = self.tail_threshold * scaled ** (-1.0 / self.tail_index)
        draws = np.clip(draws, 0.01, self.upper_bound)
        return self._quantize(draws, rng)

    def _quantize(self, draws: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Snap a share of draws to whole currency units.

        Amounts are prices, not continuous quantities. Around half of real
        transactions land on a whole number and specific price points recur
        thousands of times. A continuous draw has no such structure, which is a
        difference a detector could learn from nothing but the cents digit.
        """
        whole = rng.random(len(draws)) < self.whole_number_share
        draws[whole] = np.maximum(1.0, np.round(draws[whole]))
        draws[~whole] = np.round(draws[~whole], 2)
        return draws

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def hill_index(values: np.ndarray, tail_fraction: float = 0.05) -> tuple[float, float]:
    """Hill tail index and the threshold it was measured above."""
    ordered = np.sort(np.asarray(values, dtype=float))
    ordered = ordered[ordered > 0]
    k = max(2, int(len(ordered) * tail_fraction))
    tail = ordered[-k:]
    threshold = float(tail[0])
    if threshold <= 0:
        return float("nan"), threshold
    index = 1.0 / float(np.mean(np.log(tail / threshold) + 1e-12))
    return index, threshold


def fit_amount(values: np.ndarray, tail_fraction: float = 0.05) -> AmountFit:
    """Fit the body by log-moments and the tail by Hill."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 100:
        raise ValueError(f"need at least 100 positive amounts, got {len(values)}")

    cutoff = float(np.quantile(values, 1.0 - tail_fraction))
    body = values[values <= cutoff]
    logs = np.log(body)

    index, threshold = hill_index(values, tail_fraction)
    return AmountFit(
        lognormal_mu=float(logs.mean()),
        lognormal_sigma=float(logs.std(ddof=1)),
        tail_threshold=threshold,
        tail_index=float(index),
        tail_fraction=float(tail_fraction),
        upper_bound=float(values.max()),
        whole_number_share=float(np.mean(values == np.floor(values))),
        median=float(np.median(values)),
        mean=float(values.mean()),
        n_samples=len(values),
    )


def fit_amount_by_group(
    values: np.ndarray, groups: np.ndarray, min_samples: int = 500
) -> dict[str, AmountFit]:
    """Fit one amount model per group, skipping groups with too little data."""
    out: dict[str, AmountFit] = {}
    for key in np.unique(groups):
        subset = values[groups == key]
        if len(subset) >= min_samples:
            out[str(key)] = fit_amount(subset)
    return out
