"""The baseline fits, scores, and answers the open question honestly.

A small world, so the suite stays fast, but the whole path: collect labelled
traffic, split by entity, fit the tree, and run the per-entity ablation. The
tests assert the machinery is sound — the split does not leak, the metrics are
in range, the ablation produces a number — not that the number takes a
particular value, since that is a finding, not an invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.settings.simulation import SimulationConfig
from fraudsim.defender.baseline import PER_ENTITY_FEATURES, GBDTBaseline
from fraudsim.defender.metrics import DetectionMetrics, pr_auc, recall_at_fpr
from fraudsim.defender.split import entity_split
from fraudsim.defender.table import build_table
from fraudsim.engine.simulator import Simulator
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.orchestration.run import EpisodeRunner
from fraudsim.population.builder import PopulationBuilder
from fraudsim.population.warmstart import WarmStartRunner
from fraudsim.protocols import AlwaysApproveScorer
from fraudsim.timing.circadian import HolderClockModel


@pytest.fixture(scope="module")
def fitted():
    config = SimulationConfig.model_validate({"population": {"n_holders": 1500}})
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
    model = GBDTBaseline(table.columns).fit(split.train)
    return table, split, model


def test_split_does_not_leak_entities(fitted):
    _, split, _ = fitted
    train_groups = set(split.train.group.tolist())
    test_groups = set(split.test.group.tolist())
    assert not (train_groups & test_groups)


def test_both_classes_present_in_each_side(fitted):
    _, split, _ = fitted
    assert split.train.y.sum() > 0 and (1 - split.train.y).sum() > 0
    assert split.test.y.sum() > 0 and (1 - split.test.y).sum() > 0


def test_baseline_beats_chance(fitted):
    _, split, model = fitted
    scores = model.predict_scores(split.test.X)
    metrics = DetectionMetrics.compute(split.test.y, scores)
    # A fit model should rank fraud above the base rate; PR-AUC well over
    # prevalence is the weakest claim that still means "it learned something".
    prevalence = split.test.y.mean()
    assert metrics.pr_auc > prevalence


def test_ablation_produces_a_delta(fitted):
    table, split, full = fitted
    ablated = GBDTBaseline(table.columns).fit(split.train, drop_columns=PER_ENTITY_FEATURES)
    d_full = pr_auc(split.test.y, full.predict_scores(split.test.X))
    d_abl = pr_auc(split.test.y, ablated.predict_scores(split.test.X))
    # The ablation runs and yields a comparable number; its sign is a finding,
    # not something the test pins down.
    assert 0.0 <= d_abl <= 1.0
    assert 0.0 <= d_full <= 1.0


def test_feature_importance_ranks_columns(fitted):
    _, _, model = fitted
    importance = model.feature_importance()
    assert len(importance) == len(model.columns)
    assert importance[0][1] >= importance[-1][1]


def test_metrics_edge_cases():
    # No positives → PR-AUC 0, recall 0, not a crash.
    y = np.zeros(10)
    s = np.random.default_rng(0).random(10)
    assert pr_auc(y, s) == 0.0
    assert recall_at_fpr(y, s, 0.01) == 0.0


def test_scorer_facade_scores_events(fitted):
    _, split, model = fitted
    from fraudsim.features.schema import AuthAttemptEvent
    from fraudsim.protocols import RiskAssessment

    event = next(e for e in split.test.events if isinstance(e, AuthAttemptEvent))
    assessment = model.score(event)
    assert isinstance(assessment, RiskAssessment)
    assert 0.0 <= assessment.risk_score <= 1.0
