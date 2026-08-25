"""The noise floor is only meaningful if the split is entity-disjoint."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudsim.calibration.splits import entity_level_split, row_level_split


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for entity in range(400):
        for _ in range(int(rng.integers(1, 12))):
            rows.append({"entity": f"e{entity}", "value": float(rng.normal(50, 10))})
    return pd.DataFrame(rows)


def test_entity_split_is_disjoint(frame: pd.DataFrame) -> None:
    split = entity_level_split(frame, "entity", seed=0)
    assert split.is_disjoint()
    left, right = split.entity_counts
    assert left + right == frame["entity"].nunique()


def test_entity_split_preserves_every_row(frame: pd.DataFrame) -> None:
    split = entity_level_split(frame, "entity", seed=0)
    assert sum(split.sizes) == len(frame)


def test_entity_split_is_reproducible(frame: pd.DataFrame) -> None:
    first = entity_level_split(frame, "entity", seed=3)
    second = entity_level_split(frame, "entity", seed=3)
    assert first.fingerprint() == second.fingerprint()


def test_different_seeds_give_different_assignments(frame: pd.DataFrame) -> None:
    assert (
        entity_level_split(frame, "entity", seed=1).fingerprint()
        != entity_level_split(frame, "entity", seed=2).fingerprint()
    )


def test_row_split_leaks_entities(frame: pd.DataFrame) -> None:
    """Guards the reason entity splitting exists."""
    split = row_level_split(frame, "entity", seed=0)
    assert not split.is_disjoint()


def test_left_fraction_is_respected(frame: pd.DataFrame) -> None:
    split = entity_level_split(frame, "entity", seed=0, left_fraction=0.25)
    left, right = split.entity_counts
    assert abs(left / (left + right) - 0.25) < 0.05


def test_invalid_fraction_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        entity_level_split(frame, "entity", left_fraction=0.0)
    with pytest.raises(ValueError):
        entity_level_split(frame, "entity", left_fraction=1.0)
