"""Entity-level splitting.

Every noise floor is measured between two halves of the real data, so the split
decides what the floor means. Splitting by row puts the same card in both
halves, which drives the floor towards zero and inflates every degradation
ratio computed against it. Splitting by entity keeps a card's whole history on
one side.

The halves are equal on purpose. Sampling variability scales as 1/sqrt(n), so a
smaller half has a larger floor and therefore reports a smaller ratio; the equal
split minimises the floor and gives the most demanding number. Reported ratios
are upper bounds across split choices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class EntitySplit:
    """Two entity-disjoint halves of one real dataset."""

    left: pd.DataFrame
    right: pd.DataFrame
    entity_column: str
    seed: int

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.left), len(self.right)

    @property
    def entity_counts(self) -> tuple[int, int]:
        return (
            self.left[self.entity_column].nunique(),
            self.right[self.entity_column].nunique(),
        )

    def is_disjoint(self) -> bool:
        left = set(self.left[self.entity_column].unique())
        right = set(self.right[self.entity_column].unique())
        return left.isdisjoint(right)

    def fingerprint(self) -> str:
        """Stable digest of the entity assignment, recorded with any fit."""
        entities = sorted(self.left[self.entity_column].unique().tolist())
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(self.seed).encode())
        for entity in entities:
            digest.update(str(entity).encode())
        return digest.hexdigest()

    def summary(self) -> dict[str, object]:
        left_rows, right_rows = self.sizes
        left_entities, right_entities = self.entity_counts
        return {
            "left_rows": left_rows,
            "right_rows": right_rows,
            "left_entities": left_entities,
            "right_entities": right_entities,
            "row_balance": round(left_rows / max(1, left_rows + right_rows), 4),
            "disjoint": self.is_disjoint(),
            "fingerprint": self.fingerprint(),
        }


def entity_level_split(
    frame: pd.DataFrame,
    entity_column: str,
    seed: int = 0,
    left_fraction: float = 0.5,
) -> EntitySplit:
    """Assign whole entities to one side or the other."""
    if not 0.0 < left_fraction < 1.0:
        raise ValueError("left_fraction must lie strictly between 0 and 1")

    entities = frame[entity_column].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(entities)
    cut = int(round(len(shuffled) * left_fraction))
    left_entities = set(shuffled[:cut].tolist())

    mask = frame[entity_column].isin(left_entities)
    return EntitySplit(
        left=frame[mask],
        right=frame[~mask],
        entity_column=entity_column,
        seed=seed,
    )


def row_level_split(
    frame: pd.DataFrame, entity_column: str, seed: int = 0
) -> EntitySplit:
    """Deliberately wrong split, kept so tests can show the difference.

    Never use this for a noise floor. It exists to demonstrate that a row split
    leaks entities across halves and collapses the floor.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(len(frame)) < 0.5
    return EntitySplit(
        left=frame[mask],
        right=frame[~mask],
        entity_column=entity_column,
        seed=seed,
    )
