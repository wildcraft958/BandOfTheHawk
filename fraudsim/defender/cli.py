"""Defender baseline entry point, and the answer to the open question.

    python -m fraudsim.defender.cli baseline --holders 4000

Builds a world, collects benign and scripted-fraud traffic, extracts the flat
table, splits it by entity, and fits the gradient-boosted baseline. Then it does
the one thing the whole benign-fidelity effort was in service of: it drops the
per-entity features and refits, and reports whether the model was any worse
without them. If the delta is negligible, five experts resting on those features
are resting on nothing, and it is better to learn that here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..calibration.artifact import FittedParams
from ..config.simulation import resolve
from ..engine.simulator import Simulator
from ..features.builder import EventBuilder
from ..features.state import FeatureStateStore
from ..population.builder import PopulationBuilder
from ..population.warmstart import WarmStartRunner
from ..protocols import AlwaysApproveScorer
from ..rules.engine import VelocityRuleScorer
from ..timing.circadian import HolderClockModel
from ..orchestration.run import EpisodeRunner
from .baseline import PER_ENTITY_FEATURES, GBDTBaseline
from .metrics import DetectionMetrics
from .bands import CostModel, grid_search_bands
from .combiner import FixedAverageCombiner, LearnedCombiner, MixtureScorer
from .experts import ExpertBank
from .split import entity_split
from .table import build_table

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "simulation.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "fitted_params.json"


def _collect(config, holders: int | None):
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(
        graph, states, config.engine.windows, HolderClockModel(config.behavior.circadian)
    )
    sim = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
    WarmStartRunner(sim, config, seed=config.seed).run()
    EpisodeRunner(sim, config, seed=config.seed + 1, train_only=True).run(
        benign_seed=config.seed + 2
    )
    return sim


def cmd_baseline(args: argparse.Namespace) -> int:
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    overrides: dict = {}
    if args.holders:
        overrides["population"] = {"n_holders": args.holders}
    if args.fraud_rate:
        # H.6 is a measurement, not the deployed rate. Answering whether the
        # per-entity features carry signal needs enough positives for a stable
        # PR-AUC delta, so the experiment may run at a higher prevalence than the
        # 0.5% a real system sees. Stated, not hidden.
        overrides["engine"] = {"fraud_base_rate": args.fraud_rate}
    config = resolve(args.config, artifact=artifact, overrides=overrides or None).config

    sim = _collect(config, args.holders)
    table = build_table(sim.log, exclude_warm_start=True)
    split = entity_split(table, test_fraction=0.3, seed=config.seed)

    print("defender baseline")
    print(f"  train rows          {len(split.train):>10,}  ({int(split.train.y.sum()):,} fraud)")
    print(f"  test rows           {len(split.test):>10,}  ({int(split.test.y.sum()):,} fraud)")

    # D_0: the rule engine, the published baseline the tree must beat.
    d0 = VelocityRuleScorer(config.engine.rules)
    d0_scores = _rule_scores(d0, split.test)
    print()
    print(DetectionMetrics.compute(split.test.y, d0_scores).render("D_0 rule engine"))

    # The full tree.
    full = GBDTBaseline(table.columns).fit(split.train)
    full_scores = full.predict_scores(split.test.X)
    full_metrics = DetectionMetrics.compute(split.test.y, full_scores)
    print()
    print(full_metrics.render("GBDT full"))

    # The ablation: the same tree without the per-entity features.
    ablated = GBDTBaseline(table.columns).fit(split.train, drop_columns=PER_ENTITY_FEATURES)
    ablated_scores = ablated.predict_scores(split.test.X)
    ablated_metrics = DetectionMetrics.compute(split.test.y, ablated_scores)
    print()
    print(ablated_metrics.render("GBDT without per-entity features"))

    delta = full_metrics.pr_auc - ablated_metrics.pr_auc
    print("\n  the open question (H.6)")
    print(f"    per-entity PR-AUC lift   {delta:>+8.4f}")
    verdict = (
        "the per-entity features carry signal"
        if delta > 0.01
        else "the per-entity features add little here"
    )
    print(f"    verdict                  {verdict}")

    print("\n  top features by gain")
    for name, gain in full.feature_importance()[:12]:
        marker = "  <- per-entity" if name in PER_ENTITY_FEATURES else ""
        print(f"    {name:<32}{gain:>12.1f}{marker}")
    return 0


def _rule_scores(scorer, test_table):
    """Score each test row with the rule engine, aligned row for row.

    The table carries its source events, so the rule scorer — which reads an
    event, not a matrix — is run over exactly the events the test rows came
    from. A row whose event the rules do not apply to (anything but an auth)
    gets a neutral zero, which is the engine declining to judge it rather than a
    fabricated score.
    """
    import numpy as np

    from ..features.schema import AuthAttemptEvent

    scores = np.zeros(len(test_table.y), dtype=float)
    for i, event in enumerate(test_table.events):
        if isinstance(event, AuthAttemptEvent):
            scores[i] = scorer.score(event).risk_score
    return scores


def cmd_mixture(args: argparse.Namespace) -> int:
    """Fit the stacked experts and report against the flat baseline.

    Three numbers: the flat GBDT (the fallback), the experts under a fixed
    average (the honest combiner baseline), and the experts under the learned
    combiner (the mixture). The delta between the last two says whether the
    learned combination earned its place; the delta to the first says whether
    the mixture beat the flat table at all. Either way it is reported.
    """
    artifact = FittedParams.load(args.artifact) if args.artifact.exists() else None
    overrides: dict = {}
    if args.holders:
        overrides["population"] = {"n_holders": args.holders}
    if args.fraud_rate:
        overrides["engine"] = {"fraud_base_rate": args.fraud_rate}
    config = resolve(args.config, artifact=artifact, overrides=overrides or None).config

    sim = _collect(config, args.holders)
    table = build_table(sim.log, exclude_warm_start=True)
    split = entity_split(table, test_fraction=0.3, seed=config.seed)

    print("mixture of experts")
    print(f"  train rows          {len(split.train):>10,}  ({int(split.train.y.sum()):,} fraud)")
    print(f"  test rows           {len(split.test):>10,}  ({int(split.test.y.sum()):,} fraud)")

    # Flat baseline for comparison.
    full = GBDTBaseline(table.columns).fit(split.train)
    flat_scores = full.predict_scores(split.test.X)
    print()
    print(DetectionMetrics.compute(split.test.y, flat_scores).render("flat GBDT"))

    # Experts, fit once; scored two ways.
    bank = ExpertBank.build(table.columns).fit(split.train)
    train_scores, train_mask = bank.score_matrix(split.train)
    test_scores, test_mask = bank.score_matrix(split.test)

    avg = FixedAverageCombiner()
    avg_pred = avg.combine(test_scores, test_mask)
    print()
    print(DetectionMetrics.compute(split.test.y, avg_pred).render("experts + fixed average"))

    learned = LearnedCombiner().fit(train_scores, train_mask, split.train.y)
    learned_pred = learned.combine(test_scores, test_mask)
    learned_metrics = DetectionMetrics.compute(split.test.y, learned_pred)
    print()
    print(learned_metrics.render("experts + learned combiner"))

    print("\n  learned combiner weights")
    for name, w in learned.weights(bank.names).items():
        print(f"    {name:<14}{w:>+8.3f}")

    from .metrics import pr_auc
    lift = pr_auc(split.test.y, learned_pred) - pr_auc(split.test.y, avg_pred)
    print(f"\n  learned vs fixed-average PR-AUC lift   {lift:>+8.4f}")
    verdict = "learned combination helps" if lift > 0.01 else "fixed average was enough"
    print(f"  verdict                                {verdict}")

    # Bands grid-searched against the cost curve, on the learned scores.
    bands = grid_search_bands(split.test.y, learned_pred, CostModel())
    print("\n  cost-curve bands")
    print(f"    step_up  {bands.step_up_at:.2f}   hold {bands.hold_at:.2f}"
          f"   decline {bands.decline_at:.2f}   block {bands.block_at:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.defender")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = subparsers.add_parser("baseline", help="fit the baseline and answer H.6")
    base.add_argument("--holders", type=int, default=None)
    base.add_argument(
        "--fraud-rate",
        type=float,
        default=None,
        help="prevalence for the H.6 measurement (higher than deployment, for enough positives)",
    )
    base.set_defaults(func=cmd_baseline)

    mix = subparsers.add_parser("mixture", help="fit experts + combiner, report vs baseline")
    mix.add_argument("--holders", type=int, default=None)
    mix.add_argument("--fraud-rate", type=float, default=None)
    mix.set_defaults(func=cmd_mixture)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
