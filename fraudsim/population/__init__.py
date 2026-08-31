"""Population construction and warm-start.

Builds the entity graph in dependency order (households, holders, cards,
merchants, devices), then generates backdated history so derived fields
and rolling windows start populated rather than empty.
"""

from .builder import PopulationBuilder
from .warmstart import WarmStartRunner

__all__ = [
    "PopulationBuilder",
    "WarmStartRunner",
]
