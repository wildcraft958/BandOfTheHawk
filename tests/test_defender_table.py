"""The training table must describe behaviour, never leak identity or the label.

These tests build a small real world, run a few authorisations through the same
step method the simulation uses, stamp an episode, and check the matrix that
comes out: the label is absent from the columns, entity ids are gone, nullable
fields carry their missing flags, and warm-start rows are excluded.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.defender.table import MISSING_SUFFIX, build_table
from fraudsim.engine.actions import Action, ActionName
from fraudsim.engine.simulator import Actor, ActorKind, Simulator
from fraudsim.engine.stages import Stage
from fraudsim.features.builder import EventBuilder
from fraudsim.features.schema import EventType
from fraudsim.features.state import FeatureStateStore
from fraudsim.population.builder import PopulationBuilder
from fraudsim.protocols import AlwaysApproveScorer
from fraudsim.settings.simulation import SimulationConfig

# Fields that must never appear as a matrix column, whatever the event.
LEAKY = {"is_fraud", "episode_id", "event_id", "card_id", "merchant_id", "device_id", "ip_asn"}


@pytest.fixture
def logged():
    """A log with a handful of authorisations under one closed episode."""
    config = SimulationConfig.model_validate({"population": {"n_holders": 300}})
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    simulator = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())

    card_id, _ = next(iter(graph.provisioned))
    merchants = list(graph.merchants)
    simulator.register_actor(
        Actor(
            actor_id=1,
            kind=ActorKind.ADVERSARIAL,
            holder_id=graph.cards[card_id].holder_id,
            cards=[card_id],
            stage=Stage.BOUND,
        )
    )
    simulator.open_episode(1)
    for i in range(6):
        simulator.step(
            1,
            Action(
                name=ActionName.ATTEMPT_AUTH,
                target_id=card_id,
                secondary_id=merchants[i % len(merchants)],
                amount=40.0 + i,
                delay_minutes=30,
            ),
        )
    simulator.close_episode(1)
    return simulator.log


def test_no_leaky_columns(logged):
    table = build_table(logged, exclude_warm_start=False)
    assert not (set(table.columns) & LEAKY)


def test_label_returned_separately_and_aligned(logged):
    table = build_table(logged, exclude_warm_start=False)
    # The episode was adversarial, so every labelled row is positive.
    assert table.y.shape[0] == table.X.shape[0]
    assert set(np.unique(table.y[table.labelled_mask])) <= {1.0}
    assert table.labelled_mask.sum() > 0


def test_nullable_fields_carry_a_missing_flag(logged):
    table = build_table(logged, exclude_warm_start=False)
    # amount_vs_median is None on a card's first sight of a merchant, so its
    # flag must exist and be set on at least one row.
    assert "amount_vs_median" in table.columns
    flag = "amount_vs_median" + MISSING_SUFFIX
    assert flag in table.columns
    col = table.columns.index(flag)
    assert table.X[:, col].max() == 1.0


def test_missing_value_is_neutral_not_zero(logged):
    # Where amount_vs_median is missing it is filled with 1.0 (equal to median),
    # never 0.0, which would claim the amount was far below the median.
    table = build_table(logged, exclude_warm_start=False)
    val = table.columns.index("amount_vs_median")
    flag = table.columns.index("amount_vs_median" + MISSING_SUFFIX)
    missing_rows = table.X[:, flag] == 1.0
    assert np.all(table.X[missing_rows, val] == 1.0)


def test_warm_start_rows_excluded_by_default(logged):
    for event in logged.events:
        event.is_warm_start = True
    excluded = build_table(logged, exclude_warm_start=True)
    included = build_table(logged, exclude_warm_start=False)
    assert len(excluded) == 0
    assert len(included) > 0


def test_per_expert_view_selects_by_event_type(logged):
    table = build_table(logged, exclude_warm_start=False)
    auth_only = table.view(frozenset({EventType.AUTH_ATTEMPT}))
    assert len(auth_only) == len(table)  # every row here is an auth
    assert all(et is EventType.AUTH_ATTEMPT for et in auth_only.event_type)
    empty = table.view(frozenset({EventType.KYC_SUBMIT}))
    assert len(empty) == 0


def test_empty_log_yields_empty_table():
    from fraudsim.features.schema import EventLog

    table = build_table(EventLog())
    assert len(table) == 0
    assert table.columns == ()
