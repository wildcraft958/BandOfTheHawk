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
        return FeatureTable(
            X=np.zeros((0, 0)),
            y=np.zeros(0),
            columns=(),
            event_type=np.array([], dtype=object),
            is_warm_start=np.zeros(0, dtype=bool),
            episode_id=np.zeros(0, dtype=np.int64),
            group=np.zeros(0, dtype=np.int64),
            events=np.array([], dtype=object),
        )
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
    # Fraud is kept for a bounded number of refits rather than forever. Keeping
    # everything is right for guarding against catastrophic forgetting, but it
    # also means a detector never loses a pattern, and an attacker that abandons
    # a tactic can never profitably return to it. Real fraud models are retrained
    # on windows and do age out attack patterns that stopped occurring. None
    # keeps everything, which is the original behaviour.
    fraud_rounds: int | None = None
    # The prevalence the assembled training set is held to, by subsampling the
    # fraud. None keeps every retained example, which is the original behaviour.
    #
    # This exists because the live phase produces fraud and benign in whatever
    # ratio the training loop happens to generate them, and that ratio was 42%
    # against a design that specifies 0.5%. A detector fitted at 42% is solving
    # a different and far easier problem than the deployed one — at that balance
    # nearly any split separates the classes — so the defender's dominance was
    # substantially an artefact of the mixture rather than a property of the
    # model or of the attacker it faced.
    #
    # Reaching 0.5% by generating benign instead would need roughly six hundred
    # thousand benign rows per refit window against three thousand fraud, which
    # is not tractable. Subsampling the fraud reaches the same ratio at the cost
    # of discarding positives, and the cost is worth paying: prevalence becomes
    # a number that is stated and controlled rather than an accident of how busy
    # the attacker happened to be.
    target_prevalence: float | None = None
    seed: int = 0
    _fraud: list[FeatureTable] = field(default_factory=list)
    _benign: list[FeatureTable] = field(default_factory=list)

    def add(self, table: FeatureTable) -> None:
        """Split a round's table into its fraud and benign halves and store them."""
        fraud_mask = table.y == 1.0
        benign_mask = table.y == 0.0
        self._fraud.append(_mask(table, fraud_mask))
        self._benign.append(_mask(table, benign_mask))

    def training_table(self) -> FeatureTable:
        """The retained fraud, plus benign from the recent rounds only."""
        recent_benign = self._benign[-self.benign_rounds :]
        fraud = (
            self._fraud if self.fraud_rounds is None
            else self._fraud[-self.fraud_rounds :]
        )
        table = _concat(fraud + recent_benign)
        return self._to_prevalence(table)

    def _to_prevalence(self, table: FeatureTable) -> FeatureTable:
        """Thin the fraud until it is `target_prevalence` of the whole.

        Only ever removes positives. Adding benign would mean inventing traffic
        the world did not produce, and dropping benign would throw away the
        negatives that make the problem hard. Where there is already too little
        fraud to need thinning, the table is returned untouched — the target is
        a ceiling on prevalence, not a quota to be met.
        """
        target = self.target_prevalence
        if target is None or not 0.0 < target < 1.0 or len(table) == 0:
            return table

        is_fraud = table.y == 1.0
        n_fraud = int(is_fraud.sum())
        n_benign = int((table.y == 0.0).sum())
        if n_fraud == 0 or n_benign == 0:
            return table

        keep_fraud = int(round(target * n_benign / (1.0 - target)))
        if keep_fraud >= n_fraud:
            return table

        rng = np.random.default_rng(self.seed + len(self._fraud))
        fraud_idx = np.flatnonzero(is_fraud)
        chosen = rng.choice(fraud_idx, size=max(1, keep_fraud), replace=False)
        mask = table.y != 1.0
        mask[chosen] = True
        return _mask(table, mask)

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
