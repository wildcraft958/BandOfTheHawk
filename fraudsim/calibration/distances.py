"""Distribution distances and the degradation ratio.

Wasserstein-1 is the primary continuous metric rather than Kolmogorov-Smirnov.
KS keys on the single point of maximum separation between two CDFs and is
largely blind to the tail, which is exactly where the amount distribution
matters and where the legitimate large purchase lives.

Every reported number is a ratio against a noise floor measured between two
halves of real data, so a distance in seconds and a dimensionless correlation
gap become comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def w1(left: np.ndarray, right: np.ndarray) -> float:
    """Wasserstein-1. Tail-sensitive, in the units of the samples."""
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    return float(wasserstein_distance(np.asarray(left, float), np.asarray(right, float)))


def circular_w1(left: np.ndarray, right: np.ndarray, period: float = 24.0) -> float:
    """Wasserstein-1 on a circle, in the units of the samples.

    The linear form is wrong on any periodic quantity and does not announce
    it. Hours 23.5 and 0.5 are one apart, and `w1` calls them twenty-three, so
    a generator whose evening peak sits an hour late scores worse than one
    whose peak is at dawn. It returns a plausible number either way.

    On a circle there is no fixed origin for a cumulative distribution to
    start from, so the distance is the minimum over where the circle is cut:

        min over c of  the mean of |F_left(x) - F_right(x) - c|

    which is minimised at the median of the CDF difference. Note this is a
    shift of the *difference between the two distributions*, not a rotation of
    one sample. Minimising over rotations of the data would make the distance
    translation-invariant, and two populations peaking twelve hours apart
    would score zero.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) == 0 or len(right) == 0:
        return float("nan")

    # Shared grid over the circle, so the two samples need not match in length.
    grid = np.linspace(0.0, period, 512, endpoint=False)
    fl = np.searchsorted(np.sort(left % period), grid, side="right") / len(left)
    fr = np.searchsorted(np.sort(right % period), grid, side="right") / len(right)

    delta = fl - fr
    shift = float(np.median(delta))
    return float(np.mean(np.abs(delta - shift)) * period)


def ks(left: np.ndarray, right: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic. Secondary; reported, not relied upon."""
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    return float(ks_2samp(left, right).statistic)


def jsd(left: np.ndarray, right: np.ndarray, bins: int | None = None) -> float:
    """Jensen-Shannon divergence between two categorical or binned samples."""
    left = np.asarray(left)
    right = np.asarray(right)
    if len(left) == 0 or len(right) == 0:
        return float("nan")

    if bins is None:
        categories = np.union1d(np.unique(left), np.unique(right))
        p = np.array([(left == c).sum() for c in categories], dtype=float)
        q = np.array([(right == c).sum() for c in categories], dtype=float)
    else:
        lo = min(left.min(), right.min())
        hi = max(left.max(), right.max())
        edges = np.linspace(lo, hi, bins + 1)
        p = np.histogram(left, bins=edges)[0].astype(float)
        q = np.histogram(right, bins=edges)[0].astype(float)

    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (_kl(p, m) + _kl(q, m)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    categories = np.union1d(np.unique(left), np.unique(right))
    p = np.array([(left == c).mean() for c in categories], dtype=float)
    q = np.array([(right == c).mean() for c in categories], dtype=float)
    return float(0.5 * np.abs(p - q).sum())


@dataclass(frozen=True, slots=True)
class Degradation:
    """One sub-metric expressed against its noise floor."""

    name: str
    observed: float
    floor: float

    @property
    def ratio(self) -> float:
        if self.floor == 0 or not np.isfinite(self.floor):
            return float("inf") if self.observed > 0 else float("nan")
        return self.observed / self.floor

    @property
    def verdict(self) -> str:
        ratio = self.ratio
        if not np.isfinite(ratio):
            return "undefined"
        if ratio <= 1.5:
            return "indistinguishable"
        if ratio <= 3.0:
            return "close"
        if ratio <= 10.0:
            return "structural gap"
        return "not reproduced"

    def row(self) -> str:
        return (
            f"  {self.name:<34} {self.observed:>12.4f} {self.floor:>12.4f} "
            f"{self.ratio:>9.2f}  {self.verdict}"
        )


class DegradationReport:
    """A set of sub-metrics plus their equal-weighted composite."""

    __slots__ = ("_entries", "title")

    def __init__(self, title: str) -> None:
        self.title = title
        self._entries: list[Degradation] = []

    def add(self, name: str, observed: float, floor: float) -> Degradation:
        entry = Degradation(name=name, observed=observed, floor=floor)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[Degradation, ...]:
        return tuple(self._entries)

    def composite(self) -> float:
        ratios = [e.ratio for e in self._entries if np.isfinite(e.ratio)]
        return float(np.mean(ratios)) if ratios else float("nan")

    def failures(self, threshold: float = 3.0) -> tuple[Degradation, ...]:
        return tuple(e for e in self._entries if np.isfinite(e.ratio) and e.ratio > threshold)

    def render(self) -> str:
        header = (
            f"{self.title}\n"
            f"  {'sub-metric':<34} {'observed':>12} {'floor':>12} {'ratio':>9}  verdict\n"
            f"  {'-' * 34} {'-' * 12} {'-' * 12} {'-' * 9}  {'-' * 18}"
        )
        body = "\n".join(entry.row() for entry in self._entries)
        composite = self.composite()
        return f"{header}\n{body}\n  composite (equal-weighted): {composite:.2f}"
