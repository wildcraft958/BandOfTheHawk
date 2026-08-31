"""Fit distributions to real data and produce the calibration artifact.

Every fitted value carries provenance (measured vs. swept). The artifact
is the single contract between calibration and the runtime simulation:
one JSON file, validated on load.
"""

from .artifact import FittedParams
from .pipeline import run_calibration

__all__ = [
    "FittedParams",
    "run_calibration",
]
