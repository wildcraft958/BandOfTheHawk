"""Graph metrics over a finished world."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "networkx", reason='install the "analysis" extra'
)

import numpy as np


from fraudsim.analysis.graph_snapshot import (
    DegreeSummary,
    GraphSnapshot,
    Projection,
    compare_degrees,
)
from fraudsim.settings.simulation import SimulationConfig
from fraudsim.engine.simulator import Simulator
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.population.builder import PopulationBuilder
from fraudsim.population.warmstart import WarmStartRunner
from fraudsim.protocols import AlwaysApproveScorer


@pytest.fixture(scope="module")
def world():
    config = SimulationConfig.model_validate({"population": {"n_holders": 900}})
    graph, report = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    simulator = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
    WarmStartRunner(simulator, config, seed=3).run()
    return graph, config, report


@pytest.fixture
def snapshot(world):
    return GraphSnapshot(world[0])


def test_devices_stay_at_household_scale(snapshot) -> None:
    """Blocking a device is only proportionate if it is not a crowd."""
    summary = snapshot.device_card_degrees()
    assert summary.mean < 4.0
    assert summary.degrees.max() <= 20


def test_signatures_carry_the_heavy_tail(snapshot) -> None:
    """A signature groups strangers running the same configuration, so its
    reach is large for reasons unrelated to anyone owning many cards."""
    signatures = snapshot.fingerprint_card_degrees()
    devices = snapshot.device_card_degrees()
    assert signatures.variance_to_mean > 20.0
    assert signatures.variance_to_mean > devices.variance_to_mean * 20
    assert signatures.degrees.max() > devices.degrees.max() * 10


def test_device_dispersion_rules_out_independent_assignment(snapshot) -> None:
    """Independent assignment caps variance at the mean, whatever it draws
    from, so anything above that is evidence of structure."""
    assert snapshot.fingerprint_card_degrees().variance_to_mean > 1.0


def test_devices_per_card_is_distinct_from_cards_per_device(snapshot) -> None:
    """These are different questions. A rule counting payment methods wants the
    former; the latter is the shared-signature fan-out."""
    per_card = snapshot.card_device_degrees()
    per_device = snapshot.device_card_degrees()
    assert per_card.n_nodes != per_device.n_nodes
    assert per_card.mean < 5.0


def test_merchant_reach_appears_only_after_transactions(snapshot) -> None:
    summary = snapshot.card_merchant_degrees()
    assert summary.mean > 0.0
    assert summary.degrees.max() > 1


def test_projections_are_bipartite_and_non_empty(snapshot) -> None:
    for projection in (
        Projection.DEVICE_CARD,
        Projection.FINGERPRINT_CARD,
        Projection.CARD_MERCHANT,
    ):
        graph = snapshot.to_networkx(projection)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0


def test_entity_projection_links_cards_that_share_a_device(snapshot) -> None:
    graph = snapshot.to_networkx(Projection.ENTITY_PROJECTION)
    assert graph.number_of_nodes() == len(snapshot.graph.cards)
    assert graph.number_of_edges() > 0


def test_entity_projection_excludes_signatures(snapshot) -> None:
    """Projecting a signature covering a thousand cards makes a clique of half
    a million edges that says only that many strangers share a browser."""
    graph = snapshot.to_networkx(Projection.ENTITY_PROJECTION)
    assert all(str(node).startswith("c") for node in graph.nodes)
    degrees = [d for _, d in graph.degree()]
    assert max(degrees) < 100


def test_motifs_report_real_local_structure(snapshot) -> None:
    motifs = snapshot.motifs()
    assert motifs.n_nodes > 0
    assert motifs.n_edges > 0
    assert 0.0 <= motifs.clustering <= 1.0
    assert motifs.triangles > 0
    assert motifs.largest_component >= 2


def test_unknown_projection_is_refused(snapshot) -> None:
    with pytest.raises(ValueError, match="unknown projection"):
        snapshot.to_networkx("not_a_projection")


def test_degree_summary_handles_an_empty_distribution() -> None:
    summary = DegreeSummary(name="empty", degrees=np.asarray([], dtype=float))
    assert summary.n_nodes == 0
    assert np.isnan(summary.mean)
    assert np.isnan(summary.variance_to_mean)


def test_degree_summary_matches_a_hand_computed_case() -> None:
    summary = DegreeSummary(name="fixed", degrees=np.asarray([1, 1, 2, 4], dtype=float))
    assert summary.mean == pytest.approx(2.0)
    assert summary.share_shared == pytest.approx(0.5)
    assert summary.variance_to_mean == pytest.approx(np.var([1, 1, 2, 4], ddof=1) / 2.0)


def test_comparison_pairs_generated_against_measured(snapshot, world) -> None:
    _, config, _ = world
    target = config.population.fanout
    reference = {
        "mean": target.target_mean,
        "share_shared": target.target_share_shared,
        "p99": target.target_p99,
        "variance_to_mean": target.target_variance_to_mean,
    }
    paired = compare_degrees(snapshot.fingerprint_card_degrees(), reference)
    assert set(paired) == set(reference)
    generated, measured = paired["mean"]
    assert generated == pytest.approx(measured, rel=0.5)


def test_render_covers_every_projection(snapshot) -> None:
    text = snapshot.render()
    for name in ("device_card", "card_device", "fingerprint_card", "card_merchant"):
        assert name in text
    assert "clustering" in text
