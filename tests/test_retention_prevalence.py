"""The prevalence the defender is fitted at, and why it is controlled.

The live phase produces fraud and benign in whatever ratio the training loop
happens to generate them. Left alone that ratio was 42%, against a design that
specifies 0.5%. A detector fitted at 42% is not a stricter version of the
deployed one -- it is solving a different and far easier problem, because at that
balance almost any split separates the classes. Most of the defender's apparent
invincibility was the mixture rather than the model.

Two things had to change. The world had to keep living between refits, since the
warm start ran once before training and nothing after, so every event in a refit
window came from the attacker. And the assembled training set has to be held to
a stated share, since generating enough benign to reach 0.5% honestly would take
roughly six hundred thousand rows per window against three thousand fraud.

Subsampling only ever drops positives. Inventing benign would be fabricating
traffic the world did not produce; dropping benign would discard the negatives
that make the problem hard.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.defender.table import FeatureTable
from fraudsim.orchestration.retention import RetentionBuffer


def _table(n_fraud: int, n_benign: int, seed: int = 0) -> FeatureTable:
    n = n_fraud + n_benign
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(n_fraud), np.zeros(n_benign)])
    return FeatureTable(
        X=rng.standard_normal((n, 4)),
        y=y,
        columns=("a", "b", "c", "d"),
        event_type=np.array(["auth"] * n, dtype=object),
        is_warm_start=np.zeros(n, dtype=bool),
        episode_id=np.arange(n, dtype=np.int64),
        group=np.arange(n, dtype=np.int64),
        events=np.array([None] * n, dtype=object),
    )


def _prevalence(table: FeatureTable) -> float:
    return float((table.y == 1.0).sum()) / max(len(table), 1)


def test_without_a_target_every_example_is_kept():
    """The original behaviour, unchanged where no target is asked for."""
    buffer = RetentionBuffer(benign_rounds=3)
    buffer.add(_table(n_fraud=400, n_benign=600))

    table = buffer.training_table()
    assert len(table) == 1000
    assert int((table.y == 1.0).sum()) == 400


def test_fraud_is_thinned_to_the_target_share():
    """42% was the observed ratio; 2% is what a stated target holds it to."""
    buffer = RetentionBuffer(benign_rounds=3, target_prevalence=0.02)
    buffer.add(_table(n_fraud=4000, n_benign=6000))

    table = buffer.training_table()
    assert _prevalence(table) == pytest.approx(0.02, abs=0.002)
    # Only positives were removed.
    assert int((table.y == 0.0).sum()) == 6000


def test_benign_is_never_removed_and_never_invented():
    """Subsampling must not touch the negatives that make the problem hard."""
    buffer = RetentionBuffer(benign_rounds=3, target_prevalence=0.005)
    buffer.add(_table(n_fraud=5000, n_benign=2000))

    table = buffer.training_table()
    assert int((table.y == 0.0).sum()) == 2000, "benign count must be untouched"
    assert len(table) < 7000, "fraud must actually have been thinned"


def test_the_target_is_a_ceiling_not_a_quota():
    """Too little fraud to need thinning must be left exactly as it is.

    Meeting a target from below would mean duplicating positives, which teaches
    a model that a handful of attacks are more common than they were.
    """
    buffer = RetentionBuffer(benign_rounds=3, target_prevalence=0.50)
    buffer.add(_table(n_fraud=10, n_benign=1000))

    table = buffer.training_table()
    assert int((table.y == 1.0).sum()) == 10
    assert len(table) == 1010


def test_a_round_with_no_fraud_survives_subsampling():
    """An all-benign round is ordinary, not an error."""
    buffer = RetentionBuffer(benign_rounds=3, target_prevalence=0.02)
    buffer.add(_table(n_fraud=0, n_benign=500))

    table = buffer.training_table()
    assert len(table) == 500
    assert int((table.y == 1.0).sum()) == 0


def test_subsampling_is_deterministic_under_seed():
    """Two buffers with the same seed and rounds must agree, row for row."""
    a = RetentionBuffer(benign_rounds=3, target_prevalence=0.02, seed=7)
    b = RetentionBuffer(benign_rounds=3, target_prevalence=0.02, seed=7)
    a.add(_table(n_fraud=3000, n_benign=5000, seed=1))
    b.add(_table(n_fraud=3000, n_benign=5000, seed=1))

    assert np.array_equal(a.training_table().X, b.training_table().X)


def test_asymmetric_retention_still_holds_with_a_target():
    """Thinning must not undo the retention it is layered on top of.

    Benign ages out after `benign_rounds`; fraud is kept for longer. The target
    changes how much fraud is presented, never which rounds it comes from.
    """
    buffer = RetentionBuffer(benign_rounds=2, fraud_rounds=4, target_prevalence=0.10)
    for _ in range(5):
        buffer.add(_table(n_fraud=200, n_benign=400))

    table = buffer.training_table()
    # Two rounds of benign retained, all of it.
    assert int((table.y == 0.0).sum()) == 800
    assert _prevalence(table) == pytest.approx(0.10, abs=0.01)
