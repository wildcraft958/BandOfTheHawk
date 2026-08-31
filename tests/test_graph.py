"""Adjacency indices are redundant state, so every mutation is checked against
a re-derivation from the edge payloads."""

from __future__ import annotations

import pytest

from fraudsim.world.edges import AddedEdge, AddMethod, BindMethod, ProvisionedEdge
from fraudsim.world.entities import (
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
from fraudsim.world.graph import EntityGraph, GraphInvariantError


def make_graph(n_cards: int = 3, n_devices: int = 2) -> EntityGraph:
    graph = EntityGraph()
    graph.add_holder(
        Cardholder(
            holder_id=1, home_lat=0.0, home_lon=0.0, city_pop=1000, age_years=40,
            job_code=1, tenure_days=100, archetype=Archetype.COMMUTER,
            activity_tier=ActivityTier.REGULAR, household_id=1,
        )
    )
    graph.add_bucket(FingerprintBucket(bucket_id=1, os_code=1, browser_code=1, screen_code=1))
    graph.add_account(Account(account_id=1, holder_id=1, opened_ts=0, balance=500.0))
    for card_id in range(1, n_cards + 1):
        graph.add_card(
            Card(card_id=card_id, holder_id=1, issued_ts=0, credit_line=1000.0, bin_tier=1)
        )
    for device_id in range(1, n_devices + 1):
        graph.add_device(
            Device(
                device_id=device_id, bucket_id=1, first_seen_ts=0, household_id=1,
                os_code=1, browser_code=1, app_version=1, ip_asn=1,
            )
        )
    graph.add_merchant(
        Merchant(
            merchant_id=1, category=CategoryCluster.GROCERY, avg_ticket=40.0,
            chargeback_rate=0.001, risk_tier=RiskTier.LOW, is_high_liquidity=False,
            is_card_not_present=False, popularity_rank=1,
        )
    )
    graph.add_payee(Payee(payee_id=1, target_account_id=1, first_added_ts=0))
    return graph


def bind(graph: EntityGraph, card_id: int, device_id: int) -> bool:
    return graph.bind_device(
        ProvisionedEdge(
            card_id=card_id, device_id=device_id, bind_ts=0, bind_method=BindMethod.SELF_SERVICE
        )
    )


def test_binding_updates_both_directions() -> None:
    graph = make_graph()
    bind(graph, 1, 1)
    bind(graph, 2, 1)
    assert graph.device_card_count(1) == 2
    assert graph.devices_of_card(1) == {1}
    assert graph.cards_of_device(1) == {1, 2}
    graph.check_invariants()


def test_duplicate_binding_rejected() -> None:
    graph = make_graph()
    assert bind(graph, 1, 1) is True
    assert bind(graph, 1, 1) is False
    assert graph.device_card_count(1) == 1


def test_unbind_clears_both_directions() -> None:
    graph = make_graph()
    bind(graph, 1, 1)
    assert graph.unbind_device(1, 1) is True
    assert graph.device_card_count(1) == 0
    assert graph.devices_of_card(1) == frozenset()
    assert graph.unbind_device(1, 1) is False
    graph.check_invariants()


def test_unknown_endpoints_rejected() -> None:
    graph = make_graph()
    with pytest.raises(KeyError):
        bind(graph, 99, 1)
    with pytest.raises(KeyError):
        bind(graph, 1, 99)


def test_corrupted_index_is_detected() -> None:
    graph = make_graph()
    bind(graph, 1, 1)
    graph._cards_of_device[1].add(99)
    with pytest.raises(GraphInvariantError, match="cards_of_device"):
        graph.check_invariants()


def test_dropped_index_entry_is_detected() -> None:
    graph = make_graph()
    bind(graph, 1, 1)
    graph._devices_of_card[1].clear()
    with pytest.raises(GraphInvariantError, match="devices_of_card"):
        graph.check_invariants()


def test_bucket_reach_exceeds_device_fanout() -> None:
    """A fingerprint groups unrelated devices; blocklisting one must not imply
    the whole bucket."""
    graph = make_graph(n_cards=4, n_devices=2)
    bind(graph, 1, 1)
    bind(graph, 2, 1)
    bind(graph, 3, 2)
    bind(graph, 4, 2)
    assert graph.device_card_count(1) == 2
    assert graph.device_card_count(2) == 2
    assert graph.bucket_card_count(1) == 4
    graph.check_invariants()


def test_transaction_edge_accumulates() -> None:
    graph = make_graph()
    graph.record_transaction(1, 1, 50.0, ts=10)
    edge = graph.record_transaction(1, 1, 25.0, ts=20)
    assert edge.count == 2
    assert edge.total_amount == 75.0
    assert edge.first_ts == 10
    assert edge.last_ts == 20
    assert graph.merchants_of_card(1) == {1}
    graph.check_invariants()


def test_payee_attach_and_detach() -> None:
    graph = make_graph()
    edge = AddedEdge(account_id=1, payee_id=1, add_ts=0, add_method=AddMethod.APP)
    assert graph.attach_payee(edge) is True
    assert graph.attach_payee(edge) is False
    assert graph.payees_of_account(1) == {1}
    assert graph.detach_payee(1, 1) is True
    assert graph.payees_of_account(1) == frozenset()
    graph.check_invariants()


def test_device_usage_updates_both_directions() -> None:
    graph = make_graph()
    graph.record_device_usage(1, 1, ts=5)
    graph.record_device_usage(1, 1, ts=9)
    assert graph.accounts_of_device(1) == {1}
    assert graph.used_by[(1, 1)].count == 2
    graph.check_invariants()


def test_fanout_distribution_matches_bindings() -> None:
    graph = make_graph(n_cards=3, n_devices=2)
    bind(graph, 1, 1)
    bind(graph, 2, 1)
    bind(graph, 3, 2)
    assert sorted(graph.fanout_distribution()) == [1, 2]
