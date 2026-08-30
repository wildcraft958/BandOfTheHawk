"""The flat baseline: one gradient-boosted tree on the whole table.

This is the first real detector and the comparison point everything later is
measured against. The order of work in the design is deliberate: a single tree
on the flat table first, because it produces a real number early and because a
mixture that cannot beat it is not worth its complexity. If the experts lose to
this, the honest result is the ablation.

It is also what answers the open question. Whether the per-entity features —
`amount_vs_median`, `is_first_txn_this_merchant`, `within_usual_hours` — carry
signal is not known until a model is asked to separate fraud with and without
them. That is what `feature_importance` and the ablation harness are for, and it
should be answered before five experts are built on top of an assumption.

XGBoost lives in the defender extra, imported inside the fit so the runtime path
never reaches it. The histogram tree method is used, which is the fast one on a
table this wide and takes a GPU without changing anything else.
"""

from __future__ import annotations

import numpy as np

from ..features.schema import AuthAttemptEvent
from ..protocols import RiskAction, RiskAssessment
from .table import FeatureTable


# Per-entity features from Part 0, the ones the open question is about. Named
# here so the ablation can drop exactly this set and no more.
PER_ENTITY_FEATURES = (
    "amount_vs_median",
    "amount_vs_median_missing",
    "is_first_txn_this_merchant",
    "within_usual_hours",
    "within_usual_hours_missing",
)


class GBDTBaseline:
    """A gradient-boosted tree fit on the flat feature table.

    Presents as a scorer once fit, so the simulator can run against it exactly
    as it runs against the rule engine. Before fitting, scoring raises rather
    than returning a default that would look like a judgement.
    """

    def __init__(self, columns: tuple[str, ...], bands=None):
        self.columns = columns
        self._model = None
        self._col_index = {c: i for i, c in enumerate(columns)}
        # The banding turns a score into an action and a graph mutation. Without
        # it this detects and never acts, which would leave the mitigation layer
        # dormant wherever the flat model is the defender in force.
        from .bands import RiskBands

        self.bands = bands or RiskBands()

    # ------------------------------------------------------------------- fit

    def fit(self, table: FeatureTable, drop_columns: tuple[str, ...] = ()) -> "GBDTBaseline":
        """Fit on the labelled rows, optionally without a set of columns.

        `drop_columns` is how the ablation removes the per-entity features: the
        same model, the same data, one feature group zeroed, so the delta in
        PR-AUC is attributable to those features and nothing else.
        """
        from xgboost import XGBClassifier  # lazy; defender extra

        X = self._prepare(table.X, drop_columns)
        y = table.y
        # The base rate is ~0.5%, so the positive class is upweighted to the
        # inverse of its frequency rather than left to drown.
        pos = max(1, int(y.sum()))
        neg = max(1, int((1 - y).sum()))
        scale = neg / pos

        self._model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=0,
            n_jobs=-1,
            eval_metric="aucpr",
        )
        self._model.fit(X, y)
        self._dropped = set(drop_columns)
        return self

    def _prepare(self, X: np.ndarray, drop_columns) -> np.ndarray:
        """Zero the dropped columns rather than remove them.

        Keeping the width fixed means the same fitted model scores events built
        with the full column set; the dropped features simply carry no signal
        because they were zero at fit time. Removing columns instead would force
        the scorer to know which ablation produced the model.
        """
        if not drop_columns:
            return X
        X = X.copy()
        for name in drop_columns:
            idx = self._col_index.get(name)
            if idx is not None:
                X[:, idx] = 0.0
        return X

    # --------------------------------------------------------------- predict

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("baseline is not fit")
        Xp = self._prepare(X, tuple(getattr(self, "_dropped", ())))
        return self._model.predict_proba(Xp)[:, 1]

    # ------------------------------------------------------- feature importance

    def feature_importance(self) -> list[tuple[str, float]]:
        """Gain-based importance, highest first.

        The direct read on the open question: where the per-entity features rank
        tells whether the model found them useful, before the ablation confirms
        it on the metric.
        """
        if self._model is None:
            raise RuntimeError("baseline is not fit")
        # The booster's own gain, not `feature_importances_`. The latter is
        # normalised to sum to one, which turns every number into a small
        # fraction and makes the ranking hard to read against a run with a
        # different feature count; gain is the actual reduction in loss the
        # splits on a feature achieved.
        booster = self._model.get_booster()
        scores = booster.get_score(importance_type="gain")
        # The booster names features f0, f1, ... in column order; a feature that
        # was never split on is absent, which is a gain of zero.
        pairs = [
            (name, float(scores.get(f"f{i}", 0.0)))
            for i, name in enumerate(self.columns)
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    # ---------------------------------------------------------- scorer facade

    def score(self, event: AuthAttemptEvent) -> RiskAssessment:
        """Score one event, so the simulator can run against this detector.

        Builds the one-row matrix the same way the training table was built, so
        an event is scored on exactly the features the model was fit on.
        """
        from .table import build_table
        from ..features.schema import EventLog

        log = EventLog()
        log.append(event)
        row = build_table(log, exclude_warm_start=False)
        # Align the row's columns to the model's; a column the row lacks is zero.
        X = np.zeros((1, len(self.columns)))
        for i, name in enumerate(row.columns):
            j = self._col_index.get(name)
            if j is not None:
                X[0, j] = row.X[0, i]
        score = float(self.predict_scores(X)[0])
        action, mitigations = self.bands.decide(score, event)
        return RiskAssessment(
            risk_score=score, action=action, mitigations=tuple(mitigations)
        )


def _action_for(score: float) -> RiskAction:
    """A default banding, the same shape the mitigation layer will refine.

    Placeholder thresholds — the real boundaries are grid-searched against the
    business cost curve in the mitigation phase. Here they only let the baseline
    act as a scorer end to end.
    """
    if score >= 0.95:
        return RiskAction.BLOCK
    if score >= 0.8:
        return RiskAction.DECLINE
    if score >= 0.6:
        return RiskAction.HOLD
    if score >= 0.3:
        return RiskAction.STEP_UP
    return RiskAction.APPROVE
