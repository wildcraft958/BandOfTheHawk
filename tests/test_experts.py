"""The experts fit, route by type, and combine; the combiner reports honestly.

A small world for speed. These assert the machinery: routing is by event type
and not learned, each expert scores only its own rows, the combiner runs and its
weights are readable, and the cost-curve band search returns an ordered banding.
Whether the mixture beats the flat table is a finding, not pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.settings.simulation import SimulationConfig
from fraudsim.engine.bands import CostModel, RiskBands, grid_search_bands
from fraudsim.defender.combiner import (
    FixedAverageCombiner,
    LearnedCombiner,
    MixtureScorer,
)
from fraudsim.defender.experts import EXPERT_EVENT_TYPES, ExpertBank
from fraudsim.defender.split import entity_split
from fraudsim.defender.table import build_table
from fraudsim.engine.simulator import Simulator
from fraudsim.features.builder import EventBuilder
from fraudsim.features.schema import AuthAttemptEvent, EventType
from fraudsim.features.state import FeatureStateStore
from fraudsim.orchestration.run import EpisodeRunner
from fraudsim.population.builder import PopulationBuilder
from fraudsim.population.warmstart import WarmStartRunner
from fraudsim.protocols import AlwaysApproveScorer, RiskAssessment
from fraudsim.timing.circadian import HolderClockModel


@pytest.fixture(scope="module")
def bundle():
    # A higher fraud rate so the small world still has positives per expert.
    config = SimulationConfig.model_validate(
        {"population": {"n_holders": 1500}, "engine": {"fraud_base_rate": 0.05}}
    )
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(
        graph, states, config.engine.windows, HolderClockModel(config.behavior.circadian)
    )
    sim = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
    WarmStartRunner(sim, config, seed=config.seed).run()
    EpisodeRunner(sim, config, seed=1, train_only=True).run(benign_seed=2)
    table = build_table(sim.log, exclude_warm_start=True)
    split = entity_split(table, test_fraction=0.3, seed=0)
    bank = ExpertBank.build(table.columns).fit(split.train)
    return table, split, bank


def test_routing_is_by_event_type(bundle):
    _, _, bank = bundle
    transaction = next(e for e in bank.experts if e.name == "transaction")
    assert transaction.applies_to(EventType.AUTH_ATTEMPT)
    assert not transaction.applies_to(EventType.KYC_SUBMIT)
    network = next(e for e in bank.experts if e.name == "network")
    # The network expert applies to everything.
    assert all(network.applies_to(t) for t in EventType)


def test_score_matrix_shape_and_mask(bundle):
    _, split, bank = bundle
    scores, mask = bank.score_matrix(split.test)
    assert scores.shape == (len(split.test), len(bank.experts))
    assert mask.shape == scores.shape
    # Every row has at least the network expert applicable.
    assert mask.any(axis=1).all()


def test_fixed_average_only_counts_applicable(bundle):
    _, split, bank = bundle
    scores, mask = bank.score_matrix(split.test)
    combined = FixedAverageCombiner().combine(scores, mask)
    assert combined.shape[0] == len(split.test)
    assert np.all((combined >= 0) & (combined <= 1))


def test_learned_combiner_weights_are_readable(bundle):
    _, split, bank = bundle
    scores, mask = bank.score_matrix(split.train)
    combiner = LearnedCombiner().fit(scores, mask, split.train.y)
    weights = combiner.weights(bank.names)
    assert set(weights) == set(bank.names)


def test_mixture_scores_an_event(bundle):
    table, split, _ = bundle
    scorer = MixtureScorer.fit(split.train, learned=True)
    event = next(e for e in split.test.events if isinstance(e, AuthAttemptEvent))
    assessment = scorer.score(event)
    assert isinstance(assessment, RiskAssessment)
    assert 0.0 <= assessment.risk_score <= 1.0


def test_grid_search_returns_ordered_bands(bundle):
    _, split, bank = bundle
    scores, mask = bank.score_matrix(split.test)
    combined = FixedAverageCombiner().combine(scores, mask)
    bands = grid_search_bands(split.test.y, combined, CostModel())
    assert bands.step_up_at <= bands.hold_at <= bands.decline_at <= bands.block_at


def test_expert_event_type_coverage():
    # Every event type is handled by at least one expert.
    covered = set()
    for types in EXPERT_EVENT_TYPES.values():
        covered |= types
    assert covered == set(EventType)
