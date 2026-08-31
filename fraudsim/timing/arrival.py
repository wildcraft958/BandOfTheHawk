"""Arrival timing at runtime.

A renewal draw under a rate that drifts as it goes. Two alternatives were
fitted first and neither survived: a self-exciting kernel failed its
goodness-of-fit gate, and a session model landed at negative lag-1
autocorrelation against a positive target.

Decomposing the real signal explained both. Raw consecutive gaps correlate at
about +0.06, but after dividing each by a local rolling median the correlation
vanishes. Nothing survives detrending, so neighbouring gaps resemble each other
because they were drawn under a similar rate, not because one event triggered
the next. Both rejected models describe short-range clustering, which is why
neither could reach it.

Each entity carries its own drift state, so the coupling is per entity rather
than global. That is the point: a shared kernel has nowhere to put the ninefold
spread of rates across entities and ends up absorbing it into the excitation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..clock import SECONDS_PER_MINUTE
from ..config.behavior import ArrivalConfig
from ..world.entities import ActivityTier, Archetype


@dataclass(slots=True)
class ArrivalState:
    """One entity's rate and where its drift has wandered to."""

    base_scale_seconds: float
    drift: float = 0.0

    def reset(self) -> None:
        self.drift = 0.0


class DriftingRateProcess:
    """Draws gaps for entities, each with its own drifting rate."""

    __slots__ = ("_config", "_shape")

    def __init__(self, config: ArrivalConfig) -> None:
        self._config = config
        self._shape = config.gap_shape

    def new_state(
        self,
        rng: np.random.Generator,
        rate_multiplier: float = 1.0,
    ) -> ArrivalState:
        """A fresh entity, drawn from the population's rate spread."""
        rate = float(np.exp(rng.normal(self._config.rate_log_mean, self._config.rate_log_sigma)))
        rate = max(rate * max(rate_multiplier, 1e-6), 1e-12)
        return ArrivalState(base_scale_seconds=1.0 / rate)

    def next_gap_seconds(self, state: ArrivalState, rng: np.random.Generator) -> float:
        """Seconds until this entity's next event.

        The drift is an AR(1) in log space, advanced once per draw. Its
        persistence is what makes neighbouring gaps resemble each other; with
        persistence at zero this reduces to a plain renewal process and the
        autocorrelation collapses.
        """
        state.drift = (
            self._config.drift_persistence * state.drift
            + rng.normal(0.0, self._config.drift_sigma)
        )
        scale = state.base_scale_seconds * float(np.exp(state.drift))
        return float(rng.gamma(self._shape, scale / self._shape))

    def next_gap_minutes(self, state: ArrivalState, rng: np.random.Generator) -> int:
        """Gap rounded to the clock's resolution, never below one tick."""
        return max(1, int(round(self.next_gap_seconds(state, rng) / SECONDS_PER_MINUTE)))

    def sample_gaps(
        self, state: ArrivalState, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        return np.array(
            [self.next_gap_seconds(state, rng) for _ in range(count)], dtype=float
        )


class ArrivalScheduler:
    """Holds arrival state for a whole population."""

    __slots__ = ("_process", "_states", "_rate_scale")

    def __init__(
        self,
        config: ArrivalConfig,
        archetype_rate_scale: dict[Archetype, float] | None = None,
    ) -> None:
        self._process = DriftingRateProcess(config)
        self._states: dict[int, ArrivalState] = {}
        self._rate_scale = archetype_rate_scale or {}

    def register(
        self,
        entity_id: int,
        rng: np.random.Generator,
        archetype: Archetype | None = None,
        activity_multiplier: float = 1.0,
    ) -> ArrivalState:
        multiplier = activity_multiplier
        if archetype is not None:
            multiplier *= self._rate_scale.get(archetype, 1.0)
        state = self._process.new_state(rng, rate_multiplier=multiplier)
        self._states[entity_id] = state
        return state

    def next_gap_minutes(self, entity_id: int, rng: np.random.Generator) -> int:
        state = self._states.get(entity_id)
        if state is None:
            raise KeyError(f"entity {entity_id} was never registered with a rate")
        return self._process.next_gap_minutes(state, rng)

    def state(self, entity_id: int) -> ArrivalState | None:
        return self._states.get(entity_id)

    def __len__(self) -> int:
        return len(self._states)


def lag1_autocorrelation(gaps: np.ndarray) -> float:
    """Correlation between consecutive gaps.

    Independent draws put this at or below zero whatever the distribution, so a
    positive value is evidence of coupling rather than of a particular shape.
    """
    gaps = np.asarray(gaps, dtype=float)
    if len(gaps) < 4:
        return float("nan")
    lead, lag = gaps[:-1], gaps[1:]
    if lead.std() == 0 or lag.std() == 0:
        return float("nan")
    value = float(np.corrcoef(lead, lag)[0, 1])
    return value if np.isfinite(value) else float("nan")


def burstiness(gaps: np.ndarray) -> float:
    """Coefficient running from -1 for a regular series to +1 for a bursty one."""
    gaps = np.asarray(gaps, dtype=float)
    if len(gaps) < 2:
        return float("nan")
    mean, sd = gaps.mean(), gaps.std()
    return float((sd - mean) / (sd + mean)) if mean + sd > 0 else float("nan")
