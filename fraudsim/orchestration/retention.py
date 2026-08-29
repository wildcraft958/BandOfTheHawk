"""Asymmetric retention across rounds.

The two sides of the loop age differently, and the design is explicit that this
must be adopted rather than approximated. Fraud is rare and every example is
signal, so all of it is kept, from every round. Benign traffic drifts and old
benign teaches a world that no longer exists, so only the recent rounds are
kept. Upweighting the classes would change the loss; this changes what the model
has ever seen, which is the stronger and more honest version.

Catastrophic forgetting is the failure this guards against. A defender refit only
on recent rounds forgets the early attacks and starts losing to attackers it once
beat — which shows up as the columns of the arms-race chart going non-monotone.
Keeping all fraud is the fix, and keeping it here rather than in the model makes
the discipline visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..defender.table import FeatureTable


def _concat(tables: list[FeatureTable]) -> FeatureTable:
    """Stack aligned tables into one. All must share a column order."""
    tables = [t for t in tables if len(t) > 0]
    if not tables:
        empty = FeatureTable(
            X=np.zeros((0, 0)),
            y=np.zeros(0),
            columns=(),
            event_type=np.array([], dtype=object),
            is_warm_start=np.zeros(0, dtype=bool),
            episode_id=np.zeros(0, dtype=np.int64),
            group=np.zeros(0, dtype=np.int64),
            events=np.array([], dtype=object),
        )
        return empty
    columns = tables[0].columns
    for t in tables:
        if t.columns != columns:
            raise ValueError("cannot concatenate tables with different columns")
    return FeatureTable(
        X=np.vstack([t.X for t in tables]),
        y=np.concatenate([t.y for t in tables]),
        columns=columns,
        event_type=np.concatenate([t.event_type for t in tables]),
        is_warm_start=np.concatenate([t.is_warm_start for t in tables]),
        episode_id=np.concatenate([t.episode_id for t in tables]),
        group=np.concatenate([t.group for t in tables]),
        events=np.concatenate([t.events for t in tables]),
    )


@dataclass
class RetentionBuffer:
    """Holds per-round tables and assembles the training set asymmetrically.

    Fraud from every round, benign from the last `benign_rounds`. The columns are
    fixed from the first table added, so every round must be extracted against
    the same schema — which it is, since the schema is deterministic given the
    event types present.
    """

    benign_rounds: int = 2
    _fraud: list[FeatureTable] = field(default_factory=list)
    _benign: list[FeatureTable] = field(default_factory=list)

    def add(self, table: FeatureTable) -> None:
        """Split a round's table into its fraud and benign halves and store them."""
        fraud_mask = table.y == 1.0
        benign_mask = table.y == 0.0
        self._fraud.append(_mask(table, fraud_mask))
        self._benign.append(_mask(table, benign_mask))

    def training_table(self) -> FeatureTable:
        """All fraud ever seen, plus benign from the recent rounds only."""
        recent_benign = self._benign[-self.benign_rounds :]
        return _concat(self._fraud + recent_benign)

    @property
    def n_rounds(self) -> int:
        return len(self._fraud)


def _mask(table: FeatureTable, mask: np.ndarray) -> FeatureTable:
    return FeatureTable(
        X=table.X[mask],
        y=table.y[mask],
        columns=table.columns,
        event_type=table.event_type[mask],
        is_warm_start=table.is_warm_start[mask],
        episode_id=table.episode_id[mask],
        group=table.group[mask],
        events=table.events[mask],
    )
