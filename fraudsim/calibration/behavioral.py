"""Behavioral pattern measurements.

Three of these test properties that row-independent sampling provably cannot
reproduce, so they are worth measuring on real data before a generator exists:

    inter-event times      drawing each timestamp independently forces the
                           within-entity lag-1 autocorrelation to be at most
                           zero, whatever distribution is used
    burst structure        the same independence flattens burst length and
                           active lifetime
    attribute fan-out      assigning a shared attribute per row from a marginal
                           forces a Poisson-binomial degree distribution, whose
                           variance cannot exceed its mean

Measured targets for this dataset sit well inside those bounds but are not
large: lag-1 autocorrelation averages about +0.037 and burstiness about +0.066.
Weak, positive, and therefore still out of reach of independent sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class InterEventStats:
    """Gap distribution and the autocorrelation that independence rules out."""

    gaps: np.ndarray
    mean_autocorrelation: float
    median_autocorrelation: float
    share_positive: float
    mean_burstiness: float
    n_entities: int

    def summary(self) -> dict[str, float]:
        return {
            "n_gaps": float(len(self.gaps)),
            "n_entities": float(self.n_entities),
            "gap_median": float(np.median(self.gaps)) if len(self.gaps) else float("nan"),
            "mean_autocorrelation": self.mean_autocorrelation,
            "median_autocorrelation": self.median_autocorrelation,
            "share_positive_autocorrelation": self.share_positive,
            "mean_burstiness": self.mean_burstiness,
        }


@dataclass(frozen=True, slots=True)
class BurstStats:
    """Burst lengths at several gap thresholds, plus active lifetime."""

    burst_lengths: dict[int, np.ndarray]
    active_lifetimes: np.ndarray

    def summary(self) -> dict[str, float]:
        out = {
            "lifetime_median": float(np.median(self.active_lifetimes))
            if len(self.active_lifetimes)
            else float("nan")
        }
        for threshold, lengths in self.burst_lengths.items():
            out[f"burst_mean_at_{threshold}s"] = float(np.mean(lengths)) if len(lengths) else 0.0
        return out


@dataclass(frozen=True, slots=True)
class FanoutStats:
    """Degree distribution of a shared attribute."""

    degrees: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.degrees)) if len(self.degrees) else float("nan")

    @property
    def variance(self) -> float:
        return float(np.var(self.degrees, ddof=1)) if len(self.degrees) > 1 else float("nan")

    @property
    def variance_to_mean(self) -> float:
        """At most one when each row picks its attribute independently."""
        mean = self.mean
        return self.variance / mean if mean else float("nan")

    @property
    def share_shared(self) -> float:
        return float(np.mean(self.degrees > 1)) if len(self.degrees) else float("nan")

    def hill_index(self, tail_start: int = 2) -> float:
        """Hill tail index. Below one implies an infinite mean, which points at
        a measurement artefact rather than a physical population."""
        tail = self.degrees[self.degrees >= tail_start].astype(float)
        if len(tail) < 2:
            return float("nan")
        return float(1.0 / np.mean(np.log(tail / tail_start + 1e-12) + 1e-12))

    def summary(self) -> dict[str, float]:
        return {
            "n_nodes": float(len(self.degrees)),
            "mean": self.mean,
            "variance": self.variance,
            "variance_to_mean": self.variance_to_mean,
            "share_shared": self.share_shared,
            "max": float(np.max(self.degrees)) if len(self.degrees) else float("nan"),
            "p99": float(np.quantile(self.degrees, 0.99)) if len(self.degrees) else float("nan"),
        }


def inter_event_stats(
    frame: pd.DataFrame,
    entity_column: str,
    time_column: str,
    min_events: int = 10,
) -> InterEventStats:
    """Gaps, lag-1 autocorrelation, and burstiness per entity."""
    ordered = frame.sort_values([entity_column, time_column])
    all_gaps: list[np.ndarray] = []
    autocorrelations: list[float] = []
    burstiness: list[float] = []

    for _, group in ordered.groupby(entity_column, observed=True, sort=False):
        times = group[time_column].to_numpy(dtype=float)
        if len(times) < min_events:
            continue
        gaps = np.diff(times)
        gaps = gaps[gaps > 0]
        if len(gaps) < 4:
            continue
        all_gaps.append(gaps)

        lead, lag = gaps[:-1], gaps[1:]
        if lead.std() > 0 and lag.std() > 0:
            rho = float(np.corrcoef(lead, lag)[0, 1])
            if np.isfinite(rho):
                autocorrelations.append(rho)

        mean, sd = gaps.mean(), gaps.std()
        if mean + sd > 0:
            burstiness.append(float((sd - mean) / (sd + mean)))

    gaps = np.concatenate(all_gaps) if all_gaps else np.empty(0)
    rhos = np.asarray(autocorrelations)
    return InterEventStats(
        gaps=gaps,
        mean_autocorrelation=float(rhos.mean()) if len(rhos) else float("nan"),
        median_autocorrelation=float(np.median(rhos)) if len(rhos) else float("nan"),
        share_positive=float((rhos > 0).mean()) if len(rhos) else float("nan"),
        mean_burstiness=float(np.mean(burstiness)) if burstiness else float("nan"),
        n_entities=len(all_gaps),
    )


def burst_stats(
    frame: pd.DataFrame,
    entity_column: str,
    time_column: str,
    thresholds: tuple[int, ...] = (60, 300, 1800),
    min_events: int = 2,
) -> BurstStats:
    """Burst lengths at each gap threshold and per-entity active lifetime."""
    ordered = frame.sort_values([entity_column, time_column])
    lengths: dict[int, list[int]] = {t: [] for t in thresholds}
    lifetimes: list[float] = []

    for _, group in ordered.groupby(entity_column, observed=True, sort=False):
        times = group[time_column].to_numpy(dtype=float)
        if len(times) < min_events:
            continue
        lifetimes.append(float(times[-1] - times[0]))
        gaps = np.diff(times)
        for threshold in thresholds:
            run = 1
            for gap in gaps:
                if gap <= threshold:
                    run += 1
                else:
                    lengths[threshold].append(run)
                    run = 1
            lengths[threshold].append(run)

    return BurstStats(
        burst_lengths={t: np.asarray(v, dtype=float) for t, v in lengths.items()},
        active_lifetimes=np.asarray(lifetimes, dtype=float),
    )


def fanout_stats(
    frame: pd.DataFrame, attribute_column: str, entity_column: str
) -> FanoutStats:
    """Distinct entities per shared attribute value."""
    degrees = frame.groupby(attribute_column, observed=True)[entity_column].nunique()
    return FanoutStats(degrees=degrees.to_numpy(dtype=float))


def fraud_rate_by_fanout(
    frame: pd.DataFrame,
    attribute_column: str,
    entity_column: str,
    label_column: str,
    buckets: tuple[int, ...] = (1, 2, 5, 10, 50, 100_000),
) -> pd.DataFrame:
    """Fraud rate across fan-out bands.

    A flat profile means sharing is ordinary behaviour. A profile that climbs
    with degree means sharing was stamped in as a fraud signal, which makes the
    source unusable as a benign anchor.
    """
    degrees = frame.groupby(attribute_column, observed=True)[entity_column].nunique()
    rates = frame.groupby(attribute_column, observed=True)[label_column].mean()
    table = pd.DataFrame({"degree": degrees, "fraud_rate": rates})
    table["band"] = pd.cut(table["degree"], bins=[0, *buckets], include_lowest=True)
    return (
        table.groupby("band", observed=True)
        .agg(nodes=("fraud_rate", "size"), fraud_rate=("fraud_rate", "mean"))
        .round(4)
    )
