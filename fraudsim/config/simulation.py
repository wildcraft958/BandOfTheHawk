"""Root configuration and the merge that tracks where each value came from.

Two sources feed the simulation. YAML carries the design choices and the values
no data settles; the calibration artifact carries what was estimated from real
data. They are merged, never blended: a field supplied by both raises unless
the override is explicit, and the result records the origin of every field it
resolved.

That record is the point. A number fitted from data and a number chosen because
nothing measures it support very different claims, and once they are in the same
object nothing distinguishes them unless something tracked it on the way in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Mapping

from pydantic import Field

from .base import (
    Provenance,
    ProvenanceError,
    ProvenanceLedger,
    StrictModel,
    deep_merge,
    load_yaml,
)
from .behavior import BehaviorConfig
from .engine import EngineConfig
from .world import PopulationConfig, WarmStartConfig

# Which artifact entry maps onto which place in the config tree.
FITTED_ROUTES: dict[str, tuple[str, ...]] = {
    "amount": ("behavior", "amount"),
    "amount_heterogeneity": ("behavior", "amount"),
    "arrival": ("behavior", "arrival"),
    "circadian": ("behavior", "circadian"),
    "fingerprint_fanout": ("population", "fanout"),
}

# Artifact keys that describe a fit rather than parameterise the simulation.
FITTED_METADATA = {
    "median", "mean", "n_samples", "n_entities", "n_gaps", "n_nodes",
    "grand_mean", "between_share", "n_events", "total_sd",
    "minimum", "resultant_length", "log_likelihood", "ks_statistic",
    "ks_pvalue", "converged", "branching_ratio",
}

SWEPT_ROUTES: dict[str, tuple[str, ...]] = {
    "device_household_mean": ("population", "devices", "household_mean"),
    "device_household_max": ("population", "devices", "household_max"),
    "geo_home_radius_km": ("population", "geo", "home_radius_km"),
    "merchant_popularity_exponent": ("population", "merchants", "popularity_exponent"),
    "amount_by_category_spread": ("behavior", "amount", "category_spread"),
    "recovery_chain_probability": ("behavior", "hard_negatives", "recovery_chain_probability"),
}


class SimulationConfig(StrictModel):
    """Everything needed to build and run a world."""

    seed: Annotated[int, Field(ge=0)] = 0
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    warm_start: WarmStartConfig = Field(default_factory=WarmStartConfig)

    @classmethod
    def from_yaml(
        cls, path: Path | str, overrides: Mapping[str, Any] | None = None
    ) -> "SimulationConfig":
        payload = load_yaml(path)
        if overrides:
            payload = deep_merge(payload, overrides)
        return cls.model_validate(payload)


class ResolvedConfig:
    """A validated config plus the provenance of every field that was set."""

    __slots__ = ("config", "ledger", "artifact_fingerprint")

    def __init__(
        self,
        config: SimulationConfig,
        ledger: ProvenanceLedger,
        artifact_fingerprint: str | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.artifact_fingerprint = artifact_fingerprint

    def provenance_of(self, path: str) -> Provenance | None:
        return self.ledger._entries.get(path)

    def render(self) -> str:
        lines = ["resolved configuration"]
        if self.artifact_fingerprint:
            lines.append(f"  artifact split  {self.artifact_fingerprint[:16]}")
        lines += ["", self.ledger.table()]

        grouped = self.ledger.by_origin()
        for origin in (Provenance.FITTED, Provenance.SWEPT):
            paths = grouped[origin]
            if paths:
                lines += ["", f"  {origin.value}"]
                lines += [f"    {path}" for path in paths]
        return "\n".join(lines)


def _set_path(payload: dict[str, Any], route: tuple[str, ...], value: Any) -> None:
    cursor = payload
    for key in route[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[route[-1]] = value


def _get_path(payload: Mapping[str, Any], route: tuple[str, ...]) -> Any:
    cursor: Any = payload
    for key in route:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def resolve(
    yaml_path: Path | str | None = None,
    artifact: Any | None = None,
    overrides: Mapping[str, Any] | None = None,
    allow_override: tuple[str, ...] = (),
) -> ResolvedConfig:
    """Build a config from YAML and a fitted artifact, tracking origins.

    A field the artifact supplies and the YAML also sets raises, because
    silently preferring either would erase the distinction between a
    measurement and a choice. Naming the path in `allow_override` keeps the
    configured value and records it as a choice, which is what deliberately
    departing from a fit amounts to.
    """
    ledger = ProvenanceLedger()
    payload: dict[str, Any] = {}

    if yaml_path is not None:
        payload = load_yaml(yaml_path)
        for path in _walk(payload):
            ledger.record(path, Provenance.FREE)

    fingerprint: str | None = None
    if artifact is not None:
        fingerprint = getattr(artifact, "split_fingerprint", None)

        for name, route in FITTED_ROUTES.items():
            fitted = artifact.fitted.get(name)
            if not isinstance(fitted, Mapping):
                continue
            for key, value in fitted.items():
                if key in FITTED_METADATA:
                    continue
                field_route = (*route, key)
                path = ".".join(field_route)
                if _get_path(payload, field_route) is not None:
                    # Configured and fitted at once. Naming the path in
                    # allow_override says the configured value is intended, so
                    # it stands and stays marked as a choice rather than a
                    # measurement. Without that, refuse instead of silently
                    # preferring either one.
                    if path in allow_override:
                        ledger.record(path, Provenance.FREE)
                        continue
                    ledger.merge_field(path, Provenance.FITTED, override=False)
                _set_path(payload, field_route, value)
                ledger.record(path, Provenance.FITTED)

        for name, route in SWEPT_ROUTES.items():
            entry = artifact.swept.get(name)
            if not isinstance(entry, Mapping):
                continue
            path = ".".join(route)
            if _get_path(payload, route) is not None:
                if path in allow_override:
                    ledger.record(path, Provenance.FREE)
                    continue
                ledger.merge_field(path, Provenance.SWEPT, override=False)
                continue
            _set_path(payload, route, entry["value"])
            ledger.record(path, Provenance.SWEPT)

    if overrides:
        payload = deep_merge(payload, overrides)
        for path in _walk(overrides):
            ledger.record(path, Provenance.FREE)

    return ResolvedConfig(
        config=SimulationConfig.model_validate(payload),
        ledger=ledger,
        artifact_fingerprint=fingerprint,
    )


def _walk(payload: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Every leaf path in a nested mapping."""
    paths: list[str] = []
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            paths.extend(_walk(value, prefix=f"{path}."))
        else:
            paths.append(path)
    return paths


__all__ = [
    "SimulationConfig",
    "ResolvedConfig",
    "resolve",
    "Provenance",
    "ProvenanceError",
    "FITTED_ROUTES",
    "SWEPT_ROUTES",
]
