"""Noise floors.

Each floor is a distance measured between two entity-disjoint halves of the
real data: the irreducible divergence between two samples of the same thing.
Everything a generator later produces is reported as a ratio against these, so
they are computed once and recorded with the split fingerprint that produced
them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .behavioral import burst_stats, fanout_stats, inter_event_stats
from .distances import circular_w1, jsd, w1
from .entity_stats import circular_entity_spread
from .splits import EntitySplit, entity_level_split


@dataclass(frozen=True, slots=True)
class NoiseFloors:
    """Denominators for every degradation ratio, plus the split that made them."""

    split_fingerprint: str
    seed: int
    left_rows: int
    right_rows: int
    left_entities: int
    right_entities: int
    floors: dict[str, float] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)

    def floor(self, name: str) -> float:
        if name not in self.floors:
            raise KeyError(f"no floor recorded for {name!r}; available: {sorted(self.floors)}")
        return self.floors[name]

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "NoiseFloors":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def render(self) -> str:
        lines = [
            "noise floors",
            f"  split       seed={self.seed} fingerprint={self.split_fingerprint[:16]}",
            f"  left        {self.left_rows:>9,} rows / {self.left_entities:>7,} entities",
            f"  right       {self.right_rows:>9,} rows / {self.right_entities:>7,} entities",
            "",
            f"  {'metric':<34}{'floor':>14}",
            f"  {'-' * 34}{'-' * 14}",
        ]
        for name in sorted(self.floors):
            lines.append(f"  {name:<34}{self.floors[name]:>14.6f}")
        if self.targets:
            lines += ["", f"  {'measured target':<34}{'value':>14}", f"  {'-' * 34}{'-' * 14}"]
            for name in sorted(self.targets):
                lines.append(f"  {name:<34}{self.targets[name]:>14.6f}")
        return "\n".join(lines)


class NoiseFloorBuilder:
    """Computes every floor from one entity-level split."""

    def __init__(
        self,
        frame: pd.DataFrame,
        entity_column: str,
        time_column: str,
        amount_column: str,
        seed: int = 0,
        min_events: int = 10,
    ) -> None:
        self.frame = frame
        self.entity_column = entity_column
        self.time_column = time_column
        self.amount_column = amount_column
        self.seed = seed
        self.min_events = min_events
        self.split: EntitySplit = entity_level_split(frame, entity_column, seed=seed)
        if not self.split.is_disjoint():
            raise RuntimeError("split leaked entities across halves")

    def build(self) -> NoiseFloors:
        left, right = self.split.left, self.split.right
        floors: dict[str, float] = {}
        targets: dict[str, float] = {}

        amounts_left = left[self.amount_column].to_numpy(float)
        amounts_right = right[self.amount_column].to_numpy(float)
        floors["amount_w1"] = w1(amounts_left, amounts_right)
        floors["amount_log_w1"] = w1(np.log1p(amounts_left), np.log1p(amounts_right))
        targets["amount_median"] = float(np.median(amounts_left))

        hours_left = self._hours(left)
        hours_right = self._hours(right)
        floors["hour_jsd"] = jsd(hours_left.astype(int), hours_right.astype(int))
        floors["hour_w1"] = circular_w1(hours_left, hours_right)

        # Per-entity hour structure. The marginal floors above say whether the
        # population's hours match; these say whether they are distributed
        # across entities the same way, which is a question the marginal
        # cannot reach.
        circ_left = self._circular(left)
        circ_right = self._circular(right)
        floors["hour_within_r_gap"] = abs(circ_left.within_r - circ_right.within_r)
        floors["hour_between_r_gap"] = abs(circ_left.between_r - circ_right.between_r)
        floors["preferred_hour_w1"] = circular_w1(circ_left.preferred, circ_right.preferred)
        targets["hour_marginal_r"] = circ_left.marginal_r
        targets["hour_within_r"] = circ_left.within_r
        targets["hour_between_r"] = circ_left.between_r

        sizes_left = left.groupby(self.entity_column, observed=True).size().to_numpy(float)
        sizes_right = right.groupby(self.entity_column, observed=True).size().to_numpy(float)
        floors["entity_activity_w1"] = w1(sizes_left, sizes_right)
        targets["entity_activity_median"] = float(np.median(sizes_left))

        ie_left = self._inter_event(left)
        ie_right = self._inter_event(right)
        floors["inter_event_w1"] = w1(ie_left.gaps, ie_right.gaps)
        floors["inter_event_log_w1"] = w1(np.log1p(ie_left.gaps), np.log1p(ie_right.gaps))
        floors["autocorrelation_gap"] = abs(
            ie_left.mean_autocorrelation - ie_right.mean_autocorrelation
        )
        floors["burstiness_gap"] = abs(ie_left.mean_burstiness - ie_right.mean_burstiness)
        targets["autocorrelation_mean"] = ie_left.mean_autocorrelation
        targets["burstiness_mean"] = ie_left.mean_burstiness
        targets["autocorrelation_share_positive"] = ie_left.share_positive

        # Per-band targets, because the pooled figure above is only comparable
        # against a population with the same history lengths.
        #
        # Lag-1 autocorrelation carries a small-sample bias of about -1/(n-1):
        # with five gaps per entity it is computed against a mean estimated
        # from those same five, which forces it negative. Real entities with
        # five to nine events measure -0.131; the same entities with fifty or
        # more measure +0.070. Same population, same process, opposite sign.
        #
        # Comparing generated traffic against the pooled target is therefore a
        # comparison of census rather than behaviour, and it reads as a
        # generator inverting the autocorrelation when it is doing no such
        # thing.
        for low, high in ((5, 9), (10, 19), (20, 49), (50, 10**9)):
            sizes = left.groupby(self.entity_column, observed=True).size()
            keep = sizes[(sizes >= low) & (sizes <= high)].index
            band = left[left[self.entity_column].isin(keep)]
            if band.empty:
                continue
            stats = inter_event_stats(
                band, self.entity_column, self.time_column, min_events=5
            )
            label = f"{low}_{high}" if high < 10**9 else f"{low}_plus"
            targets[f"autocorrelation_events_{label}"] = stats.mean_autocorrelation
            targets[f"burstiness_events_{label}"] = stats.mean_burstiness

        burst_left = self._bursts(left)
        burst_right = self._bursts(right)
        floors["active_lifetime_w1"] = w1(
            burst_left.active_lifetimes, burst_right.active_lifetimes
        )
        for threshold in burst_left.burst_lengths:
            floors[f"burst_length_w1_{threshold}s"] = w1(
                burst_left.burst_lengths[threshold], burst_right.burst_lengths[threshold]
            )
            targets[f"burst_mean_{threshold}s"] = float(
                np.mean(burst_left.burst_lengths[threshold])
            )

        return NoiseFloors(
            split_fingerprint=self.split.fingerprint(),
            seed=self.seed,
            left_rows=len(left),
            right_rows=len(right),
            left_entities=self.split.entity_counts[0],
            right_entities=self.split.entity_counts[1],
            floors={k: float(v) for k, v in floors.items()},
            targets={k: float(v) for k, v in targets.items()},
        )

    def _hours(self, frame: pd.DataFrame) -> np.ndarray:
        """Hour of day, derived in one place.

        The floor and the fit have to agree about the epoch, and computing it
        twice is how they stop agreeing.
        """
        return (frame[self.time_column].to_numpy(float) / 3600.0) % 24

    def _circular(self, frame: pd.DataFrame):
        work = frame[[self.entity_column]].copy()
        work["_hour"] = self._hours(frame)
        return circular_entity_spread(
            work, self.entity_column, "_hour", min_events=self.min_events
        )

    def _inter_event(self, frame: pd.DataFrame):
        return inter_event_stats(
            frame, self.entity_column, self.time_column, min_events=self.min_events
        )

    def _bursts(self, frame: pd.DataFrame):
        return burst_stats(frame, self.entity_column, self.time_column)


def fanout_floor(
    frame: pd.DataFrame,
    attribute_column: str,
    entity_column: str,
    seed: int = 0,
) -> dict[str, float]:
    """Floor for the fan-out degree distribution, split by attribute."""
    split = entity_level_split(frame, attribute_column, seed=seed)
    left = fanout_stats(split.left, attribute_column, entity_column)
    right = fanout_stats(split.right, attribute_column, entity_column)
    return {
        "fanout_w1": w1(left.degrees, right.degrees),
        "fanout_variance_to_mean_left": left.variance_to_mean,
        "fanout_variance_to_mean_right": right.variance_to_mean,
        "fanout_share_shared_left": left.share_shared,
    }
