"""Inter-event timing.

Two candidate processes behind one interface, because the measured signal is
weak enough that the more elaborate one may not earn its cost:

    Hawkes      self-exciting with an exponential kernel; each event lifts the
                rate of the next, which is what produces positive lag-1
                autocorrelation between consecutive gaps
    session     renewal process with a session state; simpler, and adequate if
                the fitted branching ratio comes out near zero

The likelihood is Ozaki's closed form. The inner sum over past events is
quadratic if written directly, so it uses Ogata's recursion, which turns it
into a single accumulator pass and makes fitting across thousands of entities
practical.

Goodness of fit is the time-rescaling theorem: under the fitted model the
compensator between consecutive events is unit exponential. A Kolmogorov-
Smirnov test against that is the gate. It tests our own fit before any
synthetic data exists, which is cheaper than discovering the same failure
downstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest


@dataclass(frozen=True, slots=True)
class HawkesFit:
    """Exponential-kernel Hawkes parameters and its goodness of fit."""

    mu: float
    alpha: float
    beta: float
    log_likelihood: float
    ks_statistic: float
    ks_pvalue: float
    n_entities: int
    n_events: int
    converged: bool

    @property
    def branching_ratio(self) -> float:
        """Expected direct offspring per event. Must stay below one."""
        return self.alpha / self.beta if self.beta > 0 else float("inf")

    @property
    def is_stable(self) -> bool:
        return self.branching_ratio < 1.0

    @property
    def passes_gate(self, level: float = 0.05) -> bool:
        return self.ks_pvalue > level

    def as_dict(self) -> dict[str, float]:
        payload = {k: float(v) for k, v in asdict(self).items() if not isinstance(v, bool)}
        payload["branching_ratio"] = float(self.branching_ratio)
        payload["converged"] = float(self.converged)
        return payload


def _excitation(times: np.ndarray, beta: float) -> np.ndarray:
    """Ogata's recursion for the per-event excitation sum.

    Written directly this is a double sum over all prior events. The kernel is
    exponential, so the term for one event factorises out of the next, and the
    whole sum becomes a single accumulator carried forward.
    """
    out = np.zeros(len(times))
    carried = 0.0
    for i in range(1, len(times)):
        carried = np.exp(-beta * (times[i] - times[i - 1])) * (1.0 + carried)
        out[i] = carried
    return out


def _negative_log_likelihood(params: np.ndarray, sequences: list[np.ndarray]) -> float:
    """Ozaki's log-likelihood, summed over entities. Parameters are logged so
    the optimiser works unconstrained while the values stay positive."""
    mu, alpha, beta = np.exp(params)
    if not np.isfinite([mu, alpha, beta]).all() or beta <= 0:
        return 1e12

    total = 0.0
    for times in sequences:
        if len(times) < 2:
            continue
        horizon = times[-1] - times[0]
        shifted = times - times[0]

        compensator = mu * horizon + (alpha / beta) * np.sum(
            1.0 - np.exp(-beta * (shifted[-1] - shifted))
        )
        intensity = mu + alpha * _excitation(shifted, beta)
        if np.any(intensity <= 0):
            return 1e12
        total += float(np.sum(np.log(intensity)) - compensator)

    return -total if np.isfinite(total) else 1e12


def _rescaled_gaps(sequences: list[np.ndarray], mu: float, alpha: float, beta: float) -> np.ndarray:
    """Compensator increments between consecutive events.

    Under a correct fit these are independent unit exponentials, which is the
    statement the goodness-of-fit test checks.
    """
    out: list[float] = []
    for times in sequences:
        if len(times) < 3:
            continue
        shifted = times - times[0]
        for i in range(1, len(shifted)):
            span = mu * (shifted[i] - shifted[i - 1])
            prior = shifted[:i]
            decayed = np.exp(-beta * (shifted[i - 1] - prior)) - np.exp(
                -beta * (shifted[i] - prior)
            )
            out.append(float(span + (alpha / beta) * np.sum(decayed)))
    return np.asarray(out)


def fit_hawkes(
    sequences: list[np.ndarray],
    max_entities: int | None = 2000,
    seed: int = 0,
) -> HawkesFit:
    """Fit one shared kernel across entities.

    Fitting per entity is not viable here: the median entity has two events,
    so a per-entity estimate would be noise. Pooling gives one kernel whose
    branching ratio is the parameter of interest.
    """
    usable = [np.asarray(s, dtype=float) for s in sequences if len(s) >= 3]
    if not usable:
        raise ValueError("no sequence has enough events to fit")

    if max_entities is not None and len(usable) > max_entities:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(usable), max_entities, replace=False)
        usable = [usable[i] for i in picks]

    spans = [s[-1] - s[0] for s in usable if s[-1] > s[0]]
    scale = float(np.median(spans)) if spans else 1.0
    counts = float(np.mean([len(s) for s in usable]))

    start = np.log([max(counts / max(scale, 1.0), 1e-8), 0.2, 2.0 / max(scale / counts, 1e-8)])
    result = minimize(
        _negative_log_likelihood,
        start,
        args=(usable,),
        method="L-BFGS-B",
        options={"maxiter": 300},
    )

    mu, alpha, beta = np.exp(result.x)
    rescaled = _rescaled_gaps(usable, mu, alpha, beta)
    if len(rescaled) > 5:
        test = kstest(rescaled, "expon")
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    else:
        statistic, pvalue = float("nan"), float("nan")

    return HawkesFit(
        mu=float(mu),
        alpha=float(alpha),
        beta=float(beta),
        log_likelihood=float(-result.fun),
        ks_statistic=statistic,
        ks_pvalue=pvalue,
        n_entities=len(usable),
        n_events=int(sum(len(s) for s in usable)),
        converged=bool(result.success),
    )


def simulate_hawkes(
    mu: float,
    alpha: float,
    beta: float,
    horizon: float,
    rng: np.random.Generator,
    max_events: int = 500,
) -> np.ndarray:
    """Ogata thinning. Used to check a fit reproduces its own target."""
    times: list[float] = []
    now = 0.0
    while now < horizon and len(times) < max_events:
        upper = mu + alpha * sum(np.exp(-beta * (now - t)) for t in times[-50:])
        now += rng.exponential(1.0 / max(upper, 1e-12))
        if now >= horizon:
            break
        intensity = mu + alpha * sum(np.exp(-beta * (now - t)) for t in times[-50:])
        if rng.random() <= intensity / max(upper, 1e-12):
            times.append(now)
    return np.asarray(times)


def sequences_from_frame(frame, entity_column: str, time_column: str, min_events: int = 5):
    """Split a frame into one time-ordered array per entity."""
    ordered = frame.sort_values([entity_column, time_column])
    out: list[np.ndarray] = []
    for _, group in ordered.groupby(entity_column, observed=True, sort=False):
        times = group[time_column].to_numpy(dtype=float)
        if len(times) >= min_events:
            out.append(times)
    return out
