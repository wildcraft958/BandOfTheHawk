"""Saving and loading a fitted defender.

A run that takes hours produces a detector worth keeping. Without persistence the
weights vanish when the process exits, and re-scoring events with the final
defender, comparing defenders across runs, or resuming from one all mean
retraining from scratch.

The defenders hold fitted third-party models -- gradient-boosted trees and
logistic regressions -- so they are pickled through joblib rather than given a
bespoke serialisation. What is written alongside them is the column order, which
is what makes a loaded model usable: a matrix built with a different column order
would score silently and wrongly, so the order is stored and checked on load.
"""

from __future__ import annotations

import json
from pathlib import Path


def save_defender(defender, path: Path | str) -> Path:
    """Write a fitted defender and its column order.

    Two files: the pickled object, and a small sidecar naming the columns and the
    kind. The sidecar is readable without unpickling, so a run's output can be
    inspected without importing the code that produced it.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(defender, path)

    columns = getattr(defender, "columns", None)
    if columns is None:
        # The mixture holds its columns on the experts rather than on itself.
        bank = getattr(defender, "bank", None)
        if bank is not None and bank.experts:
            columns = bank.experts[0].columns

    sidecar = path.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "kind": type(defender).__name__,
                "n_columns": len(columns) if columns else 0,
                "columns": list(columns) if columns else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_defender(path: Path | str, expected_columns: tuple[str, ...] | None = None):
    """Load a fitted defender, refusing a column-order mismatch.

    A model scored against a differently ordered matrix produces numbers that
    look plausible and mean nothing, so the check is an error rather than a
    warning.
    """
    import joblib

    path = Path(path)
    defender = joblib.load(path)

    if expected_columns is not None:
        columns = getattr(defender, "columns", None)
        if columns is None:
            bank = getattr(defender, "bank", None)
            if bank is not None and bank.experts:
                columns = bank.experts[0].columns
        if columns is not None and tuple(columns) != tuple(expected_columns):
            raise ValueError(
                f"column mismatch: the defender was fitted on {len(columns)} columns "
                f"and is being used with {len(expected_columns)}"
            )
    return defender
