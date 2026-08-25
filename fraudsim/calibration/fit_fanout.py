"""Shared-attribute fan-out.

Assigning a shared attribute independently per row gives a Poisson-binomial
degree distribution, whose variance cannot exceed its mean. The measured
variance-to-mean ratio is in the hundreds, so independent assignment is ruled
out before anything is generated. Degrees are drawn first and entities are
matched to them afterwards.

The measured tail is not physical. Its Hill index sits at about one, which
implies an infinite mean, and the source key is a configuration fingerprint
rather than a device identifier: everyone on the same operating system,
browser, and screen size collapses into one key. So the fit describes a crowd
sharing a fingerprint, and the household-scale sharing a real device shows is a
separate, much smaller distribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FanoutFit:
    """A truncated power law over degrees, plus the targets it reproduces."""

    exponent: float
    minimum: int
    maximum: int
    share_singleton: float
    target_mean: float
    target_variance_to_mean: float
    target_share_shared: float
    target_p99: float
    n_nodes: int

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a degree sequence.

        A share of nodes are singletons; the rest follow a power law truncated
        at the observed maximum. Truncation matters because an untruncated draw
        at this exponent produces degrees far past anything in the source.
        """
        degrees = np.ones(size, dtype=int)
        shared = rng.random(size) >= self.share_singleton
        n_shared = int(shared.sum())
        if n_shared:
            support = np.arange(max(self.minimum, 2), self.maximum + 1)
            weights = support.astype(float) ** (-self.exponent)
            weights /= weights.sum()
            degrees[shared] = rng.choice(support, size=n_shared, p=weights)
        return degrees

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _stats(degrees: np.ndarray) -> dict[str, float]:
    degrees = np.asarray(degrees, dtype=float)
    mean = float(degrees.mean())
    variance = float(degrees.var(ddof=1)) if len(degrees) > 1 else 0.0
    return {
        "mean": mean,
        "variance_to_mean": variance / mean if mean else float("nan"),
        "share_shared": float((degrees > 1).mean()),
        "p99": float(np.quantile(degrees, 0.99)),
        "max": float(degrees.max()),
    }


def fit_fanout(
    degrees: np.ndarray,
    exponent_grid: tuple[float, ...] = (1.4, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5),
    seed: int = 0,
) -> FanoutFit:
    """Search the exponent that reproduces the observed body and shoulder.

    The exponent is searched rather than estimated from a tail index, whose own
    estimate is unstable at this sample size and, on the judge dataset, implies
    an infinite mean.
    """
    degrees = np.asarray(degrees, dtype=float)
    degrees = degrees[degrees >= 1]
    if len(degrees) < 50:
        raise ValueError(f"need at least 50 nodes, got {len(degrees)}")

    observed = _stats(degrees)
    share_singleton = float((degrees == 1).mean())
    maximum = int(degrees.max())

    rng = np.random.default_rng(seed)
    best: tuple[float, float] | None = None
    for exponent in exponent_grid:
        candidate = FanoutFit(
            exponent=exponent,
            minimum=1,
            maximum=maximum,
            share_singleton=share_singleton,
            target_mean=observed["mean"],
            target_variance_to_mean=observed["variance_to_mean"],
            target_share_shared=observed["share_shared"],
            target_p99=observed["p99"],
            n_nodes=len(degrees),
        )
        drawn = _stats(candidate.sample(len(degrees), rng))
        # Scored on mean and the 99th percentile, in relative terms.
        #
        # Dispersion is deliberately not in the objective. It is enormously
        # sensitive to the largest few degrees, so scoring it drags the fit
        # towards exponents that overshoot the tail by an order of magnitude
        # while leaving the mean less than half of target. Mean and p99 pin the
        # body and the shoulder, and the dispersion that results is checked
        # afterwards against the independence bound rather than optimised.
        loss = abs(drawn["mean"] - observed["mean"]) / max(observed["mean"], 1e-9) + abs(
            drawn["p99"] - observed["p99"]
        ) / max(observed["p99"], 1e-9)
        if best is None or loss < best[1]:
            best = (exponent, loss)

    return FanoutFit(
        exponent=float(best[0]),
        minimum=1,
        maximum=maximum,
        share_singleton=share_singleton,
        target_mean=observed["mean"],
        target_variance_to_mean=observed["variance_to_mean"],
        target_share_shared=observed["share_shared"],
        target_p99=observed["p99"],
        n_nodes=len(degrees),
    )


def household_fanout(
    n_devices: int,
    mean_size: float,
    rng: np.random.Generator,
    maximum: int = 8,
) -> np.ndarray:
    """Degrees for real devices, at household scale.

    Kept apart from the fingerprint fit on purpose. A device is a physical
    object that a mitigation may block, so its sharing has to stay within what
    a household plausibly does. The long tail belongs to the fingerprint crowd,
    which is never a mitigation target.
    """
    degrees = rng.poisson(max(mean_size - 1.0, 0.0), n_devices) + 1
    return np.clip(degrees, 1, maximum)
