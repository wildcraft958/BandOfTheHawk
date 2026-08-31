"""YAML loading, artifact merge, and profile selection.

Two sources feed the simulation: YAML for design choices, the calibration
artifact for values estimated from data. They merge with provenance
tracking so every parameter records where it came from.
"""

from .base import Provenance, StrictModel, load_yaml
from .simulation import SimulationConfig

__all__ = [
    "Provenance",
    "SimulationConfig",
    "StrictModel",
    "load_yaml",
]
