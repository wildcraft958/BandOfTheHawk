"""Archetypes and activity tiers.

An archetype shapes what a holder buys and when. An activity tier shapes how
often, and matters more than it looks: the judge dataset has a median of two
transactions per entity and 39.5% with exactly one. A uniformly active
population would give every history-dependent feature more to work with than it
would ever have in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..world.entities import ActivityTier, Archetype, CategoryCluster

# Category preference per archetype, as multipliers on the population mix.
# These are design choices: the taxonomy source's per-category amounts are
# inverted and its geography is a ring, so nothing here could be fitted from it.
CATEGORY_AFFINITY: dict[Archetype, dict[CategoryCluster, float]] = {
    Archetype.COMMUTER: {
        CategoryCluster.FUEL_TRANSIT: 2.6,
        CategoryCluster.DINING: 1.5,
        CategoryCluster.GROCERY: 1.2,
        CategoryCluster.TRAVEL: 0.4,
    },
    Archetype.HOMEBODY: {
        CategoryCluster.GROCERY: 2.4,
        CategoryCluster.HEALTH: 1.4,
        CategoryCluster.TRAVEL: 0.15,
        CategoryCluster.FUEL_TRANSIT: 0.5,
    },
    Archetype.ONLINE_HEAVY: {
        CategoryCluster.ONLINE: 3.0,
        CategoryCluster.ENTERTAINMENT: 1.8,
        CategoryCluster.FUEL_TRANSIT: 0.3,
    },
    Archetype.TRAVELLER: {
        CategoryCluster.TRAVEL: 4.0,
        CategoryCluster.DINING: 1.6,
        CategoryCluster.FUEL_TRANSIT: 1.3,
        CategoryCluster.GROCERY: 0.5,
    },
    Archetype.SENIOR: {
        CategoryCluster.HEALTH: 2.6,
        CategoryCluster.GROCERY: 1.6,
        CategoryCluster.ONLINE: 0.3,
        CategoryCluster.ENTERTAINMENT: 0.5,
    },
    Archetype.BUSINESS: {
        CategoryCluster.TRAVEL: 2.2,
        CategoryCluster.DINING: 1.8,
        CategoryCluster.RETAIL: 1.3,
        CategoryCluster.HEALTH: 0.5,
    },
}

# Multiplier on a holder's baseline transaction rate.
ARCHETYPE_RATE_SCALE: dict[Archetype, float] = {
    Archetype.COMMUTER: 1.3,
    Archetype.HOMEBODY: 0.7,
    Archetype.ONLINE_HEAVY: 1.6,
    Archetype.TRAVELLER: 1.1,
    Archetype.SENIOR: 0.45,
    Archetype.BUSINESS: 2.2,
}

# Shift on the log amount, so an archetype spends more or less per transaction.
ARCHETYPE_AMOUNT_SHIFT: dict[Archetype, float] = {
    Archetype.COMMUTER: -0.10,
    Archetype.HOMEBODY: -0.05,
    Archetype.ONLINE_HEAVY: 0.05,
    Archetype.TRAVELLER: 0.30,
    Archetype.SENIOR: 0.10,
    Archetype.BUSINESS: 0.45,
}

# How far from home an archetype ranges, as a multiple of the base radius.
ARCHETYPE_GEO_SCALE: dict[Archetype, float] = {
    Archetype.COMMUTER: 1.4,
    Archetype.HOMEBODY: 0.5,
    Archetype.ONLINE_HEAVY: 0.7,
    Archetype.TRAVELLER: 3.0,
    Archetype.SENIOR: 0.6,
    Archetype.BUSINESS: 1.8,
}


@dataclass(frozen=True, slots=True)
class ArchetypeProfile:
    """Everything an archetype changes about a holder's behaviour."""

    archetype: Archetype
    rate_scale: float
    amount_shift: float
    geo_scale: float
    category_weights: np.ndarray

    def sample_category(self, rng: np.random.Generator) -> CategoryCluster:
        index = int(rng.choice(len(CLUSTER_ORDER), p=self.category_weights))
        return CLUSTER_ORDER[index]


CLUSTER_ORDER: tuple[CategoryCluster, ...] = tuple(CategoryCluster)


def build_profiles(base_mix: dict[str, float]) -> dict[Archetype, ArchetypeProfile]:
    """Combine the population category mix with each archetype's preferences."""
    base = np.array([base_mix.get(cluster.value, 0.0) for cluster in CLUSTER_ORDER], dtype=float)
    if base.sum() <= 0:
        raise ValueError("category mix is empty")
    base /= base.sum()

    profiles: dict[Archetype, ArchetypeProfile] = {}
    for archetype in Archetype:
        affinity = CATEGORY_AFFINITY.get(archetype, {})
        weights = base * np.array(
            [affinity.get(cluster, 1.0) for cluster in CLUSTER_ORDER], dtype=float
        )
        weights /= weights.sum()
        profiles[archetype] = ArchetypeProfile(
            archetype=archetype,
            rate_scale=ARCHETYPE_RATE_SCALE[archetype],
            amount_shift=ARCHETYPE_AMOUNT_SHIFT[archetype],
            geo_scale=ARCHETYPE_GEO_SCALE[archetype],
            category_weights=weights,
        )
    return profiles


class ActivitySampler:
    """Assigns activity tiers and the rate multiplier each implies."""

    __slots__ = ("_tiers", "_weights", "_multipliers")

    def __init__(self, tier_weights: dict[str, float], multipliers: dict[str, float]) -> None:
        self._tiers = tuple(ActivityTier(name) for name in tier_weights)
        weights = np.array(list(tier_weights.values()), dtype=float)
        self._weights = weights / weights.sum()
        self._multipliers = {ActivityTier(k): float(v) for k, v in multipliers.items()}

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        picks = rng.choice(len(self._tiers), size=size, p=self._weights)
        return np.array([self._tiers[i] for i in picks], dtype=object)

    def multiplier(self, tier: ActivityTier) -> float:
        return self._multipliers[tier]


class ArchetypeSampler:
    """Draws archetypes at the configured population shares."""

    __slots__ = ("_archetypes", "_weights")

    def __init__(self, weights: dict[str, float]) -> None:
        self._archetypes = tuple(Archetype(name) for name in weights)
        values = np.array(list(weights.values()), dtype=float)
        self._weights = values / values.sum()

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        picks = rng.choice(len(self._archetypes), size=size, p=self._weights)
        return np.array([self._archetypes[i] for i in picks], dtype=object)
