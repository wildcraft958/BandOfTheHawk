"""Adversarial episodes land at prevalence, produce labelled fraud, keep invariants.

A smaller world than the CLI uses, so the suite stays fast, but the same path:
warm start for benign traffic, then the runner for fraud, checked for the three
properties that matter -- the share is near target, the episodes reach
monetisation, and the graph is still consistent after all the mutations.
"""

from __future__ import annotations

import pytest

from fraudsim.defender.table import build_table
from fraudsim.engine.simulator import Simulator
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.orchestration.run import EpisodeRunner
from fraudsim.population.builder import PopulationBuilder
from fraudsim.population.warmstart import WarmStartRunner
from fraudsim.protocols import AlwaysApproveScorer
from fraudsim.settings.simulation import SimulationConfig
from fraudsim.timing.circadian import HolderClockModel


@pytest.fixture(scope="module")
def run_bundle():
    config = SimulationConfig.model_validate({"population": {"n_holders": 1200}})
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(
        graph, states, config.engine.windows, HolderClockModel(config.behavior.circadian)
    )
    sim = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
    WarmStartRunner(sim, config, seed=config.seed).run()
    runner = EpisodeRunner(sim, config, seed=7, train_only=True)
    report = runner.run(benign_seed=config.seed + 2)
    return sim, config, report


def test_prevalence_near_target(run_bundle):
    _, config, report = run_bundle
    target = config.engine.fraud_base_rate
    # Fraud is a share of authorisations, and lands within a burst of target.
    assert report.fraud_auths > 0
    assert abs(report.fraud_auth_share - target) < target  # within 100% relative, i.e. same order


def test_episodes_reach_monetized(run_bundle):
    _, _, report = run_bundle
    assert report.reached_monetized > 0
    assert report.episodes > 0


def test_fraud_is_labelled_and_extractable(run_bundle):
    sim, _, _ = run_bundle
    table = build_table(sim.log, exclude_warm_start=True)
    positives = (table.y == 1.0).sum()
    negatives = (table.y == 0.0).sum()
    # Both classes present, and fraud is the minority.
    assert positives > 0
    assert negatives > positives


def test_invariants_hold_after_mutations(run_bundle):
    sim, _, _ = run_bundle
    sim.graph.check_invariants()  # raises on inconsistency


def test_top_sequences_recorded(run_bundle):
    _, _, report = run_bundle
    assert report.top_sequences
    seq, count = report.top_sequences[0]
    assert count >= 1
    assert ">" in seq or seq  # a non-empty action chain


def test_holdouts_excluded_when_train_only(run_bundle):
    _, _, report = run_bundle
    assert "sim_swap" not in report.per_vertical
    assert "refund_abuse" not in report.per_vertical
