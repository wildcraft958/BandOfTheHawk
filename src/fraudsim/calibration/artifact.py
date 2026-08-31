"""The fitted-parameter artifact.

One JSON file is the entire contract between calibration and the simulation.
Nothing in the runtime package imports anything from this tier; it reads this
file and validates it.

Every entry carries where it came from. Values estimated from data are marked
fitted; values that no data settles are marked swept and travel with the range
they are swept over. Keeping the two apart in the artifact itself means a later
report cannot quietly present a swept assumption as a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_VERSION = 1


@dataclass
class FittedParams:
    """Fitted models plus the provenance needed to interpret them."""

    source: str
    split_fingerprint: str
    split_seed: int
    created_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    version: int = ARTIFACT_VERSION
    fitted: dict[str, Any] = field(default_factory=dict)
    swept: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    noise_floors: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    rejected: dict[str, Any] = field(default_factory=dict)

    def add_fitted(self, name: str, payload: dict[str, Any]) -> None:
        self.fitted[name] = payload

    def add_swept(self, name: str, value: float, low: float, high: float, reason: str) -> None:
        """Record a value no data settles, with the range it is swept over."""
        if not low <= value <= high:
            raise ValueError(f"{name}: default {value} lies outside its sweep [{low}, {high}]")
        self.swept[name] = {"value": value, "low": low, "high": high, "reason": reason}

    def add_diagnostic(self, name: str, payload: dict[str, Any]) -> None:
        """Record a check that was run, passing or failing, with its numbers."""
        self.diagnostics[name] = payload

    def add_rejection(self, name: str, reason: str, payload: dict[str, Any]) -> None:
        """Record a model that was fitted and then ruled out.

        Kept in the artifact rather than discarded, because knowing which
        alternatives were tried and why they failed is part of justifying the
        one that was kept.
        """
        self.rejected[name] = {"reason": reason, **payload}

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "split_fingerprint": self.split_fingerprint,
            "split_seed": self.split_seed,
            "created_utc": self.created_utc,
            "fitted": self.fitted,
            "swept": self.swept,
            "targets": self.targets,
            "noise_floors": self.noise_floors,
            "diagnostics": self.diagnostics,
            "rejected": self.rejected,
        }

    @classmethod
    def load(cls, path: Path | str) -> FittedParams:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"artifact version {version} does not match the expected {ARTIFACT_VERSION}"
            )
        return cls(
            source=payload["source"],
            split_fingerprint=payload["split_fingerprint"],
            split_seed=payload["split_seed"],
            created_utc=payload["created_utc"],
            version=version,
            fitted=payload.get("fitted", {}),
            swept=payload.get("swept", {}),
            targets=payload.get("targets", {}),
            noise_floors=payload.get("noise_floors", {}),
            diagnostics=payload.get("diagnostics", {}),
            rejected=payload.get("rejected", {}),
        )

    def counts(self) -> dict[str, int]:
        return {
            "fitted_models": len(self.fitted),
            "fitted_parameters": sum(
                len(v) for v in self.fitted.values() if isinstance(v, dict)
            ),
            "swept": len(self.swept),
            "targets": len(self.targets),
            "noise_floors": len(self.noise_floors),
            "checks": len(self.diagnostics),
            "rejected_models": len(self.rejected),
        }

    def render(self) -> str:
        lines = [
            f"fitted parameters  version {self.version}",
            f"  source      {self.source}",
            f"  split       seed={self.split_seed} fingerprint={self.split_fingerprint[:16]}",
            f"  created     {self.created_utc}",
            "",
        ]
        for name, count in self.counts().items():
            lines.append(f"  {name:<22}{count:>6}")

        if self.fitted:
            lines += ["", "  fitted models"]
            for name in sorted(self.fitted):
                lines.append(f"    {name}")

        if self.swept:
            lines += ["", f"  {'swept parameter':<26}{'value':>10}{'range':>20}"]
            for name in sorted(self.swept):
                entry = self.swept[name]
                span = f"[{entry['low']:g}, {entry['high']:g}]"
                lines.append(f"    {name:<24}{entry['value']:>10.4g}{span:>20}")

        if self.diagnostics:
            lines += ["", "  checks run"]
            for name in sorted(self.diagnostics):
                lines.append(f"    {name}")

        if self.rejected:
            lines += ["", "  models tried and ruled out"]
            for name in sorted(self.rejected):
                lines.append(f"    {name}: {self.rejected[name].get('reason', '')}")

        return "\n".join(lines)
