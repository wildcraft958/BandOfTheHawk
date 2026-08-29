"""Splitting a table for training, without leaking an entity across the split.

A row split puts the same card in both train and test, so the model is tested on
entities it trained on and the number that comes out is optimistic — the same
failure the fidelity floor guards against, one layer down. The split here is by
entity: a card (or holder) lands wholly in one side.

Fraud is stamped per episode, so an episode must not straddle the split either,
or the model is evaluated on the second half of an attack whose first half it
trained on. Splitting by the group id keeps both an entity and, since an episode
acts on one entity, its episode on the same side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .table import FeatureTable


@dataclass(frozen=True, slots=True)
class Split:
    """A train/test partition of a table, aligned to the source rows."""

    train: FeatureTable
    test: FeatureTable


def _subset(table: FeatureTable, mask: np.ndarray) -> FeatureTable:
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


def entity_split(table: FeatureTable, test_fraction: float = 0.3, seed: int = 0) -> Split:
    """Partition by entity, keeping only labelled rows.

    Unlabelled rows cannot train or test a supervised model and are dropped
    here rather than silently scored as one class. The entities are shuffled and
    cut, so the test set is a sample of held-out cards rather than a time slice.
    """
    labelled = table.labelled_mask
    table = _subset(table, labelled)

    groups = np.unique(table.group)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    test_groups = set(groups[:n_test].tolist())

    in_test = np.array([g in test_groups for g in table.group], dtype=bool)
    return Split(train=_subset(table, ~in_test), test=_subset(table, in_test))
