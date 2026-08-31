"""Arrival timing, fitted as a slowly drifting rate.

Two candidate models were fitted first and both failed, which is what pointed
at the right one.

A pooled Hawkes kernel failed its goodness-of-fit gate outright. Its fitted
decay came out at several days, meaning the excitation term was standing in for
the ninefold spread of per-entity rates rather than describing any burst.

A session model then reproduced burst structure but landed at negative lag-1
autocorrelation, on the wrong side of the target.

Decomposing the real correlation explains both. Raw consecutive gaps correlate
at about +0.06, but after dividing each gap by a local rolling median the
correlation is -0.004. Nothing survives detrending, so the signal is not short
range clustering at all: it is each entity's own rate wandering across its
lifetime. Neighbouring gaps look alike because they are drawn under a similar
rate, not because one event triggers the next.

So the model is a renewal process whose rate follows a slow random walk. Two
parameters control the outcome: how far the rate wanders, and how much it
persists between draws.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ArrivalFit:
    """Per-entity rate spread plus the drift that couples consecutive gaps."""

    rate_log_mean: float
    rate_log_sigma: float
    drift_sigma: float
    drift_persistence: float
    gap_shape: float
    target_autocorrelation: float
    target_burstiness: float
    target_gap_median: float
    n_entities: int
    n_gaps: int

    def sample_rate(self, rng: np.random.Generator) -> float:
        return float(np.exp(rng.normal(self.rate_log_mean, self.rate_log_sigma)))

    def sample_gaps(self, n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
        """Gaps for one entity, under a rate that drifts as it goes.

        The drift term is an AR(1) in log space. Its persistence is what makes
        neighbouring gaps resemble each other; without it the process reduces to
        a plain renewal draw and the autocorrelation collapses to zero.
        """
        gaps = np.empty(n)
        drift = 0.0
        base = 1.0 / max(rate, 1e-12)
        for i in range(n):
            drift = self.drift_persistence * drift + rng.normal(0.0, self.drift_sigma)
            scale = base * float(np.exp(drift))
            gaps[i] = rng.gamma(self.gap_shape, scale / self.gap_shape)
        return gaps

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _lag1(values: np.ndarray) -> float:
    if len(values) < 4:
        return float("nan")
    lead, lag = values[:-1], values[1:]
    if lead.std() == 0 or lag.std() == 0:
        return float("nan")
    rho = float(np.corrcoef(lead, lag)[0, 1])
    return rho if np.isfinite(rho) else float("nan")


def measure_targets(sequences: list[np.ndarray], min_gaps: int = 6) -> dict[str, float]:
    """Autocorrelation, burstiness, and gap median, over usable sequences."""
    rhos: list[float] = []
    bursts: list[float] = []
    medians: list[float] = []
    for times in sequences:
        gaps = np.diff(np.asarray(times, float))
        gaps = gaps[gaps > 0]
        if len(gaps) < min_gaps:
            continue
        rho = _lag1(gaps)
        if np.isfinite(rho):
            rhos.append(rho)
        mean, sd = gaps.mean(), gaps.std()
        if mean + sd > 0:
            bursts.append(float((sd - mean) / (sd + mean)))
        medians.append(float(np.median(gaps)))
    return {
        "autocorrelation": float(np.mean(rhos)) if rhos else float("nan"),
        "burstiness": float(np.mean(bursts)) if bursts else float("nan"),
        "gap_median": float(np.median(medians)) if medians else float("nan"),
        "n_entities": float(len(medians)),
    }


def fit_arrival(
    sequences: list[np.ndarray],
    min_events: int = 10,
    drift_grid: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.7, 0.9),
    persistence_grid: tuple[float, ...] = (0.3, 0.5, 0.7, 0.85, 0.95),
    shape_grid: tuple[float, ...] = (0.5, 0.8, 1.0, 1.3, 1.8, 2.5),
    seed: int = 0,
) -> ArrivalFit:
    """Fit the rate spread directly, then search the shape and drift together.

    Shape is searched rather than taken from the pooled coefficient of
    variation. The pooled figure is inflated by the rate spread across
    entities, and feeding it in leaves the per-gap noise so wide that it buries
    the drift the model exists to express: autocorrelation stays negative for
    every drift setting. Searching the three jointly against both targets
    avoids that.

    Autocorrelation and burstiness are weighted by their own noise floors, so
    neither dominates purely because it happens to be measured on a larger
    numeric scale.
    """
    usable = [np.asarray(s, float) for s in sequences if len(s) >= min_events]
    if not usable:
        raise ValueError("no sequence long enough to fit")

    rates: list[float] = []
    all_gaps: list[np.ndarray] = []
    for times in usable:
        span = times[-1] - times[0]
        if span > 0:
            rates.append(len(times) / span)
        gaps = np.diff(times)
        gaps = gaps[gaps > 0]
        if len(gaps):
            all_gaps.append(gaps)

    log_rates = np.log(np.asarray(rates))
    pooled = np.concatenate(all_gaps)
    targets = measure_targets(usable)

    def build(drift_sigma: float, persistence: float, shape: float) -> ArrivalFit:
        return ArrivalFit(
            rate_log_mean=float(log_rates.mean()),
            rate_log_sigma=float(log_rates.std(ddof=1)),
            drift_sigma=float(drift_sigma),
            drift_persistence=float(persistence),
            gap_shape=float(shape),
            target_autocorrelation=targets["autocorrelation"],
            target_burstiness=targets["burstiness"],
            target_gap_median=targets["gap_median"],
            n_entities=len(usable),
            n_gaps=len(pooled),
        )

    # Scale each residual by the spread of its own target so the two are
    # comparable rather than ordered by numeric magnitude.
    rho_scale = max(abs(targets["autocorrelation"]), 1e-3)
    burst_scale = max(abs(targets["burstiness"]), 1e-3)

    rng = np.random.default_rng(seed)
    best: tuple[ArrivalFit, float] | None = None
    for shape in shape_grid:
        for drift_sigma in drift_grid:
            for persistence in persistence_grid:
                candidate = build(drift_sigma, persistence, shape)
                observed = measure_targets(_simulate(candidate, 300, 30, rng))
                loss = (
                    abs(observed["autocorrelation"] - targets["autocorrelation"]) / rho_scale
                    + abs(observed["burstiness"] - targets["burstiness"]) / burst_scale
                )
                if best is None or loss < best[1]:
                    best = (candidate, loss)

    return best[0]


def _simulate(
    fit: ArrivalFit, n_entities: int, events: int, rng: np.random.Generator
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _ in range(n_entities):
        rate = fit.sample_rate(rng)
        gaps = fit.sample_gaps(events - 1, rate, rng)
        out.append(np.concatenate([[0.0], np.cumsum(gaps)]))
    return out


def simulate_arrival(
    fit: ArrivalFit, n_entities: int, events_per_entity: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Generate sequences from a fitted arrival model."""
    return _simulate(fit, n_entities, events_per_entity, rng)
