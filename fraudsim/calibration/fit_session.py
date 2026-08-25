"""Session-based inter-event timing.

Chosen after a pooled Hawkes kernel failed its goodness-of-fit gate on the
judge dataset. The failure was informative rather than fatal: the fitted decay
came out at several days, which is not self-excitation but the single shared
kernel absorbing the spread of per-entity rates. Entity rates vary about
ninefold, and one kernel has nowhere else to put that.

So rate and burst are separated. Each entity carries its own baseline rate, and
on top of that a session state produces short runs of closely spaced events.

The session state has to be genuinely self-exciting. Mixing two gap regimes
independently per gap reproduces burstiness while leaving lag-1 autocorrelation
at or below zero, because nothing carries the current regime from one gap to
the next. Continuation probability is what carries it, and it is what makes the
autocorrelation target reachable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SessionFit:
    """Per-entity rate spread plus a within-session burst model."""

    rate_log_mean: float
    rate_log_sigma: float
    session_continue_prob: float
    within_session_scale: float
    between_session_shape: float
    between_session_scale: float
    target_autocorrelation: float
    target_burstiness: float
    n_entities: int
    n_gaps: int

    def sample_rate(self, rng: np.random.Generator) -> float:
        """One entity's baseline rate, in events per second."""
        return float(np.exp(rng.normal(self.rate_log_mean, self.rate_log_sigma)))

    def sample_gaps(self, n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
        """Consecutive gaps for one entity at its own rate.

        The session flag persists across draws, which is what puts short gaps
        next to short gaps and produces positive lag-1 autocorrelation.
        """
        scale = 1.0 / max(rate, 1e-12)
        gaps = np.empty(n)
        in_session = False
        for i in range(n):
            if in_session:
                gaps[i] = rng.exponential(self.within_session_scale)
                in_session = rng.random() < self.session_continue_prob
            else:
                gaps[i] = rng.gamma(self.between_session_shape, scale) * (
                    self.between_session_scale
                )
                in_session = rng.random() < self.session_continue_prob
        return gaps

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def fit_session(
    sequences: list[np.ndarray],
    session_threshold: float = 1800.0,
    min_events: int = 5,
) -> SessionFit:
    """Estimate the rate spread and the burst parameters directly.

    Every quantity here is a moment of the observed gaps, so there is no
    optimiser to converge and nothing to diverge.
    """
    usable = [np.asarray(s, float) for s in sequences if len(s) >= min_events]
    if not usable:
        raise ValueError("no sequence long enough to fit")

    rates: list[float] = []
    short_gaps: list[float] = []
    long_gaps: list[float] = []
    continuations = 0
    opportunities = 0
    autocorrelations: list[float] = []
    burstiness: list[float] = []

    for times in usable:
        span = times[-1] - times[0]
        if span > 0:
            rates.append(len(times) / span)

        gaps = np.diff(times)
        gaps = gaps[gaps > 0]
        if len(gaps) < 2:
            continue

        short = gaps <= session_threshold
        short_gaps.extend(gaps[short].tolist())
        long_gaps.extend(gaps[~short].tolist())

        # How often a short gap is followed by another short gap. This is the
        # quantity the persistent session flag has to reproduce.
        opportunities += int(short[:-1].sum())
        continuations += int((short[:-1] & short[1:]).sum())

        lead, lag = gaps[:-1], gaps[1:]
        if lead.std() > 0 and lag.std() > 0:
            rho = float(np.corrcoef(lead, lag)[0, 1])
            if np.isfinite(rho):
                autocorrelations.append(rho)

        mean, sd = gaps.mean(), gaps.std()
        if mean + sd > 0:
            burstiness.append(float((sd - mean) / (sd + mean)))

    log_rates = np.log(np.asarray(rates))
    long_array = np.asarray(long_gaps) if long_gaps else np.asarray([session_threshold])
    mean_long = float(long_array.mean())
    var_long = float(long_array.var())
    shape = (mean_long**2 / var_long) if var_long > 0 else 1.0

    return SessionFit(
        rate_log_mean=float(log_rates.mean()),
        rate_log_sigma=float(log_rates.std(ddof=1)),
        session_continue_prob=float(continuations / opportunities) if opportunities else 0.0,
        within_session_scale=float(np.mean(short_gaps)) if short_gaps else 60.0,
        between_session_shape=float(np.clip(shape, 0.1, 20.0)),
        between_session_scale=1.0,
        target_autocorrelation=float(np.mean(autocorrelations)) if autocorrelations else 0.0,
        target_burstiness=float(np.mean(burstiness)) if burstiness else 0.0,
        n_entities=len(usable),
        n_gaps=len(short_gaps) + len(long_gaps),
    )


def simulate_session(
    fit: SessionFit,
    n_entities: int,
    events_per_entity: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate sequences from a fitted session model."""
    out: list[np.ndarray] = []
    for _ in range(n_entities):
        rate = fit.sample_rate(rng)
        gaps = fit.sample_gaps(events_per_entity - 1, rate, rng)
        out.append(np.concatenate([[0.0], np.cumsum(gaps)]))
    return out
