"""Graph metrics.

Built on demand, never during a run. The simulator keeps its own adjacency
indices because it touches them millions of times; a graph library is the wrong
shape for that but the right shape for the questions asked here once a run is
over.

This module is the only place in the package permitted to import networkx.
Degree work skips it entirely and reads the adjacency directly, since that is
the hot metric during a parameter sweep and building a graph object to count
neighbours would be wasteful.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np

from ..world.graph import EntityGraph


class Projection(Enum):
    """Which view of the world to build."""

    DEVICE_CARD = "device_card"
    FINGERPRINT_CARD = "fingerprint_card"
    ACCOUNT_PAYEE = "account_payee"
    CARD_MERCHANT = "card_merchant"
    # Entities linked when they share any attribute. This is the view the
    # clustering and triangle counts are taken on.
    ENTITY_PROJECTION = "entity_projection"


@dataclass(frozen=True, slots=True)
class DegreeSummary:
    """A degree distribution and the statistics that describe its shape."""

    name: str
    degrees: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.degrees)

    @property
    def mean(self) -> float:
        return float(self.degrees.mean()) if len(self.degrees) else float("nan")

    @property
    def variance_to_mean(self) -> float:
        """At most one where each row picks its attribute independently.

        Anything above that rules out independent assignment, whatever
        distribution it drew from, which is why this is reported rather than a
        tail index that is unstable at these sample sizes.
        """
        if len(self.degrees) < 2:
            return float("nan")
        mean = self.mean
        return float(self.degrees.var(ddof=1) / mean) if mean else float("nan")

    @property
    def share_shared(self) -> float:
        return float((self.degrees > 1).mean()) if len(self.degrees) else float("nan")

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.degrees, q)) if len(self.degrees) else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "n_nodes": float(self.n_nodes),
            "mean": self.mean,
            "variance_to_mean": self.variance_to_mean,
            "share_shared": self.share_shared,
            "p50": self.quantile(0.5),
            "p99": self.quantile(0.99),
            "max": float(self.degrees.max()) if len(self.degrees) else float("nan"),
        }


@dataclass(frozen=True, slots=True)
class MotifSummary:
    """Local structure in the entity projection."""

    n_nodes: int
    n_edges: int
    clustering: float
    triangles: int
    components: int
    largest_component: int

    def as_dict(self) -> dict[str, float]:
        return {
            "n_nodes": float(self.n_nodes),
            "n_edges": float(self.n_edges),
            "clustering": self.clustering,
            "triangles": float(self.triangles),
            "components": float(self.components),
            "largest_component": float(self.largest_component),
        }


class GraphSnapshot:
    """Metrics over a finished world."""

    def __init__(self, graph: EntityGraph) -> None:
        self.graph = graph

    # ------------------------------------------------------------- degrees

    def device_card_degrees(self) -> DegreeSummary:
        """Cards per physical device, which stays at household scale."""
        return DegreeSummary(
            name="device_card",
            degrees=np.asarray(self.graph.fanout_distribution(), dtype=float),
        )

    def fingerprint_card_degrees(self) -> DegreeSummary:
        """Cards reachable through a shared configuration signature.

        This is the quantity a naive reading mistakes for device fan-out. It is
        heavy tailed among ordinary holders because a signature groups
        strangers, not because anyone owns an unusual number of cards.
        """
        return DegreeSummary(
            name="fingerprint_card",
            degrees=np.asarray(self.graph.bucket_fanout_distribution(), dtype=float),
        )

    def card_device_degrees(self) -> DegreeSummary:
        """Devices per card, which is what a payment-method rule counts."""
        degrees = [len(self.graph.devices_of_card(c)) for c in self.graph.cards]
        return DegreeSummary(name="card_device", degrees=np.asarray(degrees, dtype=float))

    def card_merchant_degrees(self) -> DegreeSummary:
        degrees = [len(self.graph.merchants_of_card(c)) for c in self.graph.cards]
        return DegreeSummary(
            name="card_merchant", degrees=np.asarray(degrees, dtype=float)
        )

    # -------------------------------------------------------- projections

    def to_networkx(self, projection: Projection) -> nx.Graph:
        """Build a graph object for the metrics that need one."""
        if projection is Projection.DEVICE_CARD:
            return self._bipartite(
                ((f"d{d}", f"c{c}") for (c, d) in self.graph.provisioned)
            )
        if projection is Projection.FINGERPRINT_CARD:
            edges = []
            for bucket_id in self.graph.buckets:
                for device_id in self.graph.devices_of_bucket(bucket_id):
                    for card_id in self.graph.cards_of_device(device_id):
                        edges.append((f"f{bucket_id}", f"c{card_id}"))
            return self._bipartite(edges)
        if projection is Projection.ACCOUNT_PAYEE:
            return self._bipartite(
                ((f"a{a}", f"p{p}") for (a, p) in self.graph.added)
            )
        if projection is Projection.CARD_MERCHANT:
            return self._bipartite(
                ((f"c{c}", f"m{m}") for (c, m) in self.graph.transacts)
            )
        if projection is Projection.ENTITY_PROJECTION:
            return self._entity_projection()
        raise ValueError(f"unknown projection {projection}")

    @staticmethod
    def _bipartite(edges) -> nx.Graph:
        graph = nx.Graph()
        graph.add_edges_from(edges)
        return graph

    def _entity_projection(self, max_degree: int = 64) -> nx.Graph:
        """Cards linked when they sit behind the same device.

        Signatures are left out and a degree ceiling applies. A signature can
        cover a thousand cards, and projecting one produces a clique of half a
        million edges that says nothing except that many strangers run the same
        browser. The ceiling keeps the projection about devices people share.
        """
        projection = nx.Graph()
        projection.add_nodes_from(f"c{card_id}" for card_id in self.graph.cards)

        for device_id in self.graph.devices:
            cards = sorted(self.graph.cards_of_device(device_id))
            if len(cards) < 2 or len(cards) > max_degree:
                continue
            for index, left in enumerate(cards):
                for right in cards[index + 1 :]:
                    projection.add_edge(f"c{left}", f"c{right}")
        return projection

    # --------------------------------------------------------------- motifs

    def motifs(self, projection: Projection = Projection.ENTITY_PROJECTION) -> MotifSummary:
        graph = self.to_networkx(projection)
        components = list(nx.connected_components(graph)) if len(graph) else []
        return MotifSummary(
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges(),
            clustering=float(nx.average_clustering(graph)) if len(graph) else 0.0,
            triangles=sum(nx.triangles(graph).values()) // 3 if len(graph) else 0,
            components=len(components),
            largest_component=max((len(c) for c in components), default=0),
        )

    # --------------------------------------------------------------- report

    def render(self) -> str:
        lines = ["graph metrics", ""]
        for summary in (
            self.device_card_degrees(),
            self.card_device_degrees(),
            self.fingerprint_card_degrees(),
            self.card_merchant_degrees(),
        ):
            stats = summary.as_dict()
            lines.append(f"  {summary.name}")
            for key in ("n_nodes", "mean", "variance_to_mean", "share_shared", "p99", "max"):
                lines.append(f"    {key:<20}{stats[key]:>12.3f}")
            lines.append("")

        motifs = self.motifs()
        lines.append("  entity projection, cards sharing a device")
        for key, value in motifs.as_dict().items():
            lines.append(f"    {key:<20}{value:>12.3f}")
        return "\n".join(lines)


def compare_degrees(
    generated: DegreeSummary, reference: dict[str, float]
) -> dict[str, tuple[float, float]]:
    """Generated statistics beside the ones measured on real data."""
    stats = generated.as_dict()
    return {
        key: (stats[key], reference[key])
        for key in ("mean", "share_shared", "p99", "variance_to_mean")
        if key in reference and key in stats
    }
