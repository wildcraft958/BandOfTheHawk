"""Where a card shops, and what it buys.

Both were population draws. Every card picked a merchant uniformly from the
whole roster, which made 99.7% of transactions the card's first at that
merchant and left the feature saying nothing, and category fell out of whichever
merchant was picked, so no card had a mix of its own.

A card now draws its category first and its merchant within that category,
which is the order the behaviour actually has: someone decides to buy petrol
and then goes to their usual station. Layering loyalty onto merchant alone
would leave category an uncontrolled by-product of which merchants happened to
land in the preferred set.

Neither concentration can be fitted. The judge dataset carries no merchant
entity at all, and the taxonomy source is itself a generator whose typical card
visits 572 of its 693 merchants with a top-1 share of 0.67% - a population with
no habits, whose small per-card excess is geographic rather than loyalty. Both
are therefore swept, across ranges whose low end reproduces the uniform draw
this replaces, so a claim can be stated as holding across a range that contains
the behaviour it corrects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..settings.behavior import LoyaltyConfig
from ..world.entities import Archetype, CategoryCluster


@dataclass(slots=True)
class CategoryProfile:
    """One card's own mix over the category clusters."""

    weights: np.ndarray

    def pick(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.weights), p=self.weights))


@dataclass(slots=True)
class MerchantProfile:
    """One card's usual merchants, per category.

    Kept per category rather than as one flat list, because that is the shape
    the habit has: a cardholder has a usual supermarket and a usual petrol
    station, not a single top-twelve spanning both.
    """

    preferred: dict[int, np.ndarray]
    weights: dict[int, np.ndarray]
    loyalty: float

    def pick(
        self,
        cluster: int,
        fallback: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """A merchant in this category: usually a regular, sometimes not.

        At zero loyalty this is the uniform draw it replaces, which is what
        makes the sweep contain the null.
        """
        regulars = self.preferred.get(cluster)
        if regulars is not None and len(regulars) and rng.random() < self.loyalty:
            return int(rng.choice(regulars, p=self.weights[cluster]))
        if len(fallback) == 0:
            return int(rng.choice(regulars)) if regulars is not None and len(regulars) else -1
        return int(fallback[rng.integers(0, len(fallback))])


class LoyaltyModel:
    """Category mixes and merchant habits for a population of cards."""

    __slots__ = ("_by_cluster", "_categories", "_config", "_merchants", "_popularity")

    def __init__(
        self,
        config: LoyaltyConfig,
        merchants_by_cluster: dict[int, np.ndarray],
        popularity_by_cluster: dict[int, np.ndarray],
    ) -> None:
        self._config = config
        self._by_cluster = merchants_by_cluster
        self._popularity = popularity_by_cluster
        self._categories: dict[int, CategoryProfile] = {}
        self._merchants: dict[int, MerchantProfile] = {}

    def register(
        self,
        card_id: int,
        rng: np.random.Generator,
        archetype_weights: np.ndarray,
    ) -> tuple[CategoryProfile, MerchantProfile]:
        """Give a card its own mix and its own regulars, once.

        The category mix is Dirichlet about the archetype's weights. The
        concentration is the swept knob and has the right limits: large means
        every card of an archetype shares its mix, which is the defect, and
        small means cards specialise sharply.

        Regulars are drawn without replacement weighted by popularity, so the
        popular merchants appear in many cards' sets. That reproduces
        population-level concentration and per-card loyalty from one mechanism
        rather than two that could disagree.
        """
        alpha = np.clip(archetype_weights, 1e-6, None) * self._config.category_concentration
        category = CategoryProfile(weights=rng.dirichlet(alpha))
        self._categories[card_id] = category

        # The set size is the card's whole roster, split across categories by
        # how much it uses each. Sizing it per category instead gave every
        # card twelve regulars in each of eight clusters - about a hundred
        # merchants against a median of five transactions - so it never
        # revisited any of them and the habit existed only on paper.
        total = max(1, int(rng.poisson(self._config.merchant_preferred_set_mean)))
        allocation = rng.multinomial(total, category.weights)

        preferred: dict[int, np.ndarray] = {}
        weights: dict[int, np.ndarray] = {}
        for cluster, options in self._by_cluster.items():
            if len(options) == 0:
                continue
            # At least one regular wherever the card shops at all, so a
            # category it uses rarely still has somewhere usual to go.
            size = min(len(options), max(1, int(allocation[cluster])))
            popularity = self._popularity[cluster]
            chosen = rng.choice(len(options), size=size, replace=False, p=popularity)
            preferred[cluster] = options[chosen]
            share = popularity[chosen]
            weights[cluster] = share / share.sum()

        merchant = MerchantProfile(
            preferred=preferred, weights=weights, loyalty=self._config.merchant_loyalty
        )
        self._merchants[card_id] = merchant
        return category, merchant

    def category(self, card_id: int) -> CategoryProfile | None:
        return self._categories.get(card_id)

    def merchant(self, card_id: int) -> MerchantProfile | None:
        return self._merchants.get(card_id)

    def pick_merchant(
        self,
        card_id: int,
        rng: np.random.Generator,
        fallback: np.ndarray,
        cluster: int | None = None,
    ) -> int | None:
        """A merchant for this card, category first.

        Returns None where the card has no profile, so the caller can fall
        back to whatever it did before rather than being handed a merchant
        drawn from nothing.
        """
        category = self._categories.get(card_id)
        profile = self._merchants.get(card_id)
        if category is None or profile is None:
            return None
        if cluster is None:
            cluster = category.pick(rng)
        picked = profile.pick(cluster, fallback, rng)
        return None if picked < 0 else picked

    def __len__(self) -> int:
        return len(self._merchants)


def cluster_index(cluster: CategoryCluster) -> int:
    return list(CategoryCluster).index(cluster)


def archetype_weights(
    profiles: dict[Archetype, object], archetype: Archetype
) -> np.ndarray:
    """The archetype's category weights, or a flat mix if it has none."""
    profile = profiles.get(archetype)
    weights = getattr(profile, "category_weights", None)
    if weights is None:
        return np.full(len(CategoryCluster), 1.0 / len(CategoryCluster))
    return np.asarray(weights, dtype=float)


def clusters_from_graph(
    graph, popularity_exponent: float
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Merchants grouped by category, with a popularity weight each.

    Derived from the graph rather than threaded through from the builder. Each
    merchant already carries the popularity rank it was created with, so the
    weights can be rebuilt here without the graph having to hold the
    distribution that generated it - it carries state, not the machinery that
    produced the state.

    Ranks are re-derived within each category. A merchant that is fiftieth
    overall but the second in its own category is the second choice a card
    picking that category faces, and using the global rank would make whole
    categories uniformly unpopular.
    """
    order = list(CategoryCluster)
    by_cluster: dict[int, list[int]] = {index: [] for index in range(len(order))}
    ranks: dict[int, list[int]] = {index: [] for index in range(len(order))}

    for merchant_id, merchant in graph.merchants.items():
        index = order.index(merchant.category)
        by_cluster[index].append(int(merchant_id))
        ranks[index].append(int(merchant.popularity_rank))

    merchants: dict[int, np.ndarray] = {}
    weights: dict[int, np.ndarray] = {}
    for index, ids in by_cluster.items():
        if not ids:
            merchants[index] = np.empty(0, dtype=int)
            weights[index] = np.empty(0, dtype=float)
            continue
        ordering = np.argsort(np.asarray(ranks[index]))
        ordered = np.asarray(ids, dtype=int)[ordering]
        local = np.arange(1, len(ordered) + 1, dtype=float) ** (-popularity_exponent)
        merchants[index] = ordered
        weights[index] = local / local.sum()
    return merchants, weights
