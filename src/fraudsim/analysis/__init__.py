"""Post-run diagnostics: graph snapshots and per-entity reports.

Built on demand after a run, never during one. The only place in the
package permitted to import networkx.
"""

from .graph_snapshot import GraphSnapshot, Projection

__all__ = [
    "GraphSnapshot",
    "Projection",
]
