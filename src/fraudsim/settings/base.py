"""Configuration foundations.

Every parameter carries a provenance tag. Measured values come from the fitted
artifact, everything else from YAML, and the two are merged rather than blended:
a field supplied by both sources raises unless the override is explicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class Provenance(Enum):
    """Where a parameter's value came from. Never blend these in a writeup."""

    FITTED = "fitted"      # estimated from real data
    CITED = "cited"        # taken from published literature
    SWEPT = "swept"        # unmeasurable, reported across a range
    FREE = "free"          # pure design choice, tuned until sensible


class StrictModel(BaseModel):
    """Base for every config model: typo-proof and immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ProvenanceError(RuntimeError):
    """Raised when a fitted value would silently overwrite a configured one."""


class ProvenanceLedger:
    """Records the origin of each resolved parameter."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, Provenance] = {}

    def record(self, path: str, origin: Provenance) -> None:
        self._entries[path] = origin

    def merge_field(self, path: str, incoming: Provenance, *, override: bool) -> None:
        existing = self._entries.get(path)
        if existing is not None and existing is not incoming and not override:
            raise ProvenanceError(
                f"{path!r} already set from {existing.value}; refusing to replace with "
                f"{incoming.value}. Mark the config entry override=true to allow it."
            )
        self._entries[path] = incoming

    def by_origin(self) -> dict[Provenance, list[str]]:
        grouped: dict[Provenance, list[str]] = {origin: [] for origin in Provenance}
        for path, origin in self._entries.items():
            grouped[origin].append(path)
        for paths in grouped.values():
            paths.sort()
        return grouped

    def counts(self) -> dict[str, int]:
        grouped = self.by_origin()
        return {origin.value: len(paths) for origin, paths in grouped.items()}

    def table(self) -> str:
        lines = ["provenance          count", "-" * 26]
        for origin, count in self.counts().items():
            lines.append(f"{origin:<20}{count:>6}")
        lines.append("-" * 26)
        lines.append(f"{'total':<20}{len(self._entries):>6}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML config into a plain mapping."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay wins, recursing into nested mappings."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged
