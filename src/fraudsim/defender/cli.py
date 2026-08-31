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

from ..logs import emit
from ..cli import add_scale_flags, base_parser, load_config
from ..engine.bands import CostModel, grid_search_bands
from ..orchestration.run import EpisodeRunner
from ..population.factory import build_warm_world
from ..rules.engine import VelocityRuleScorer
from .baseline import PER_ENTITY_FEATURES, GBDTBaseline
from .combiner import FixedAverageCombiner, LearnedCombiner
from .experts import ExpertBank
from .metrics import DetectionMetrics
from .split import entity_split
from .table import build_table


def _collect(config, holders: int | None):
    world = build_warm_world(config)
    EpisodeRunner(world.simulator, config, seed=config.seed + 1, train_only=True).run(
        benign_seed=config.seed + 2
    )
    return world.simulator


def cmd_baseline(args: argparse.Namespace) -> int:
    # --fraud-rate here is a measurement, not the deployed rate. H.6 asks
    # whether the per-entity features carry signal, and a stable PR-AUC delta
    # needs enough positives, so the experiment may run at a higher prevalence
    # than the 0.5% a real system sees. Stated, not hidden.
    config = load_config(args)

    sim = _collect(config, args.holders)
    table = build_table(sim.log, exclude_warm_start=True)
    split = entity_split(table, test_fraction=config.detector.split.test_fraction,
                         seed=config.seed)

    emit("defender baseline  [STATIC BENCHMARK vs the scripted red team]")
    emit("  a fixed adversary and fixed data, so architectures can be compared and")
    emit("  the per-entity ablation means something. NOT a claim about the learned")
    emit("  attacker -- that is the co-adaptation curve.")
    emit(f"  train rows          {len(split.train):>10,}  ({int(split.train.y.sum()):,} fraud)")
    emit(f"  test rows           {len(split.test):>10,}  ({int(split.test.y.sum()):,} fraud)")

    # D_0: the rule engine, the published baseline the tree must beat.
    d0 = VelocityRuleScorer(config.engine.rules)
    d0_scores = _rule_scores(d0, split.test)
    emit()
    emit(DetectionMetrics.compute(split.test.y, d0_scores).render("D_0 rule engine"))

    # The full tree.
    full = GBDTBaseline(table.columns).fit(split.train)
    full_scores = full.predict_scores(split.test.X)
    full_metrics = DetectionMetrics.compute(split.test.y, full_scores)
    emit()
    emit(full_metrics.render("GBDT full"))

    # The ablation: the same tree without the per-entity features.
    ablated = GBDTBaseline(table.columns).fit(split.train, drop_columns=PER_ENTITY_FEATURES)
    ablated_scores = ablated.predict_scores(split.test.X)
    ablated_metrics = DetectionMetrics.compute(split.test.y, ablated_scores)
    emit()
    emit(ablated_metrics.render("GBDT without per-entity features"))

    delta = full_metrics.pr_auc - ablated_metrics.pr_auc
    emit("\n  the open question (H.6)")
    emit(f"    per-entity PR-AUC lift   {delta:>+8.4f}")
    verdict = (
        "the per-entity features carry signal"
        if delta > 0.01
        else "the per-entity features add little here"
    )
    emit(f"    verdict                  {verdict}")

    emit("\n  top features by gain")
    for name, gain in full.feature_importance()[:12]:
        marker = "  <- per-entity" if name in PER_ENTITY_FEATURES else ""
        emit(f"    {name:<32}{gain:>12.1f}{marker}")
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
    config = load_config(args)

    sim = _collect(config, args.holders)
    table = build_table(sim.log, exclude_warm_start=True)
    split = entity_split(table, test_fraction=config.detector.split.test_fraction,
                         seed=config.seed)

    emit("mixture of experts  [STATIC BENCHMARK vs the scripted red team]")
    emit("  the same fixed data as the baseline, so the two are comparable. NOT a")
    emit("  claim about the learned attacker.")
    emit(f"  train rows          {len(split.train):>10,}  ({int(split.train.y.sum()):,} fraud)")
    emit(f"  test rows           {len(split.test):>10,}  ({int(split.test.y.sum()):,} fraud)")

    # Flat baseline for comparison.
    full = GBDTBaseline(table.columns).fit(split.train)
    flat_scores = full.predict_scores(split.test.X)
    emit()
    emit(DetectionMetrics.compute(split.test.y, flat_scores).render("flat GBDT"))

    # Experts, fit once; scored two ways.
    bank = ExpertBank.build(table.columns).fit(split.train)
    train_scores, train_mask = bank.score_matrix(split.train)
    test_scores, test_mask = bank.score_matrix(split.test)

    avg = FixedAverageCombiner()
    avg_pred = avg.combine(test_scores, test_mask)
    emit()
    emit(DetectionMetrics.compute(split.test.y, avg_pred).render("experts + fixed average"))

    learned = LearnedCombiner().fit(train_scores, train_mask, split.train.y)
    learned_pred = learned.combine(test_scores, test_mask)
    learned_metrics = DetectionMetrics.compute(split.test.y, learned_pred)
    emit()
    emit(learned_metrics.render("experts + learned combiner"))

    emit("\n  learned combiner weights")
    for name, w in learned.weights(bank.names).items():
        emit(f"    {name:<14}{w:>+8.3f}")

    from .metrics import pr_auc
    lift = pr_auc(split.test.y, learned_pred) - pr_auc(split.test.y, avg_pred)
    emit(f"\n  learned vs fixed-average PR-AUC lift   {lift:>+8.4f}")
    verdict = "learned combination helps" if lift > 0.01 else "fixed average was enough"
    emit(f"  verdict                                {verdict}")

    # Bands grid-searched against the cost curve, on the learned scores.
    cost = config.detector.cost
    bands = grid_search_bands(
        split.test.y, learned_pred, CostModel.from_config(cost), search=cost,
    )
    emit("\n  cost-curve bands")
    emit(f"    step_up  {bands.step_up_at:.2f}   hold {bands.hold_at:.2f}"
          f"   decline {bands.decline_at:.2f}   block {bands.block_at:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fraudsim.defender")
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = subparsers.add_parser("baseline", help="fit the baseline and answer H.6")
    add_scale_flags(base, fraud_rate=True)
    base.add_argument(
        "--fraud-rate",
        type=float,
        default=None,
        help="prevalence for the H.6 measurement (higher than deployment, for enough positives)",
    )
    base.set_defaults(func=cmd_baseline)

    mix = subparsers.add_parser("mixture", help="fit experts + combiner, report vs baseline")
    add_scale_flags(mix, fraud_rate=True)
    mix.set_defaults(func=cmd_mixture)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
