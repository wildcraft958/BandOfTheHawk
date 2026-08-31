"""Fit distributions to real data and produce the calibration artifact.

Every fitted value carries provenance (measured vs. swept). The artifact
is the single contract between calibration and the runtime simulation:
one JSON file, validated on load.

`FittedParams` is plain JSON and belongs to the runtime tier, which reads the
artifact on every run. `run_calibration` is what produces it, and that needs
pandas and pyarrow to read the judge datasets. Importing the second eagerly here
put pandas on the import path of every module that wanted the first, so
`python -m fraudsim.engine.cli demo` required the calibration extra despite the
import firewall promising the simulation runs on numpy alone. It is resolved
lazily instead, so the name still works and the dependency arrives only when
something actually calibrates.
"""

from typing import TYPE_CHECKING, Any

from .artifact import FittedParams

if TYPE_CHECKING:  # for type checkers, which do not execute the lazy path
    from .pipeline import run_calibration

__all__ = [
    "FittedParams",
    "run_calibration",
]


def __getattr__(name: str) -> Any:
    """Resolve `run_calibration` on first use (PEP 562)."""
    if name == "run_calibration":
        from .pipeline import run_calibration

        return run_calibration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
