"""Entity graph: holders, cards, devices, merchants, and the edges between them.

Slotted dataclasses for nodes, payload dicts for edges, redundant adjacency
indices for O(1) traversal. Every mutation goes through EntityGraph methods
so the indices stay consistent.
"""

from .entities import (
    Account,
    ActivityTier,
    Archetype,
    Card,
    Cardholder,
    CategoryCluster,
    Device,
    FingerprintBucket,
    Merchant,
    Payee,
    RiskTier,
)
from .edges import AddedEdge, BindMethod, ProvisionedEdge, TransactsEdge, UsedByEdge
from .graph import EntityGraph, GraphInvariantError

__all__ = [
    "Account",
    "ActivityTier",
    "AddedEdge",
    "Archetype",
    "BindMethod",
    "Card",
    "Cardholder",
    "CategoryCluster",
    "Device",
    "EntityGraph",
    "FingerprintBucket",
    "GraphInvariantError",
    "Merchant",
    "Payee",
    "ProvisionedEdge",
    "RiskTier",
    "TransactsEdge",
    "UsedByEdge",
]
