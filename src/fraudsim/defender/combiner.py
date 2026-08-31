"""Combining the experts' opinions into one score.

Which experts apply to an event is already decided — a schema fact. What is not
decided is how much each applicable opinion should count, and that is what this
learns. An authorisation minutes after a device bind that followed a password
reset is scored by the transaction expert (an ordinary purchase), the binding
expert (a recovery chain) and the network expert (a device on many cards) all at
once; how to weigh those three is a real question with a learnable answer.

This is stacking: a small model over the applicable experts' scores. It is
reported against the honest baseline of simply averaging them, because if the
learned combination gives no lift over a fixed average, that is a clean result
worth stating rather than a mixture dressed up as progress.

Where an expert does not apply, its score is masked to a neutral value before it
reaches the combiner, so an expert never contributes to an event outside its
remit.
"""

from __future__ import annotations

import numpy as np

from ..features.schema import EventLog
from ..protocols import RiskAction, RiskAssessment
from .experts import ExpertBank
from .table import build_table


def _masked(scores: np.ndarray, mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Expert scores with the inapplicable ones replaced by a neutral fill."""
    out = scores.copy()
    out[~mask] = fill
    return out


class FixedAverageCombiner:
    """The honest baseline: mean of the applicable experts.

    No parameters, so it cannot overfit and cannot flatter itself. The learned
    combiner has to beat this to justify its existence.
    """

    def combine(self, scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        applicable = mask.sum(axis=1)
        summed = (scores * mask).sum(axis=1)
        return np.where(applicable > 0, summed / np.maximum(applicable, 1), 0.0)


class LearnedCombiner:
    """A logistic model over the masked expert scores.

    Small on purpose: five inputs, so it learns a weighting rather than a second
    detector. Fit on the same labels the experts saw, over the whole table so it
    sees every event type's mix of applicable experts.
    """

    def __init__(self) -> None:
        self._model = None

    def fit(self, scores: np.ndarray, mask: np.ndarray, y: np.ndarray) -> "LearnedCombiner":
        from sklearn.linear_model import LogisticRegression  # lazy; defender extra

        X = _masked(scores, mask)
        if y.sum() == 0 or (1 - y).sum() == 0:
            self._model = None
            self._constant = float(y.mean()) if len(y) else 0.0
            return self
        self._model = LogisticRegression(max_iter=2000, class_weight="balanced")
        self._model.fit(X, y)
        return self

    def combine(self, scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.full(scores.shape[0], getattr(self, "_constant", 0.0))
        return self._model.predict_proba(_masked(scores, mask))[:, 1]

    def weights(self, expert_names: list[str]) -> dict[str, float]:
        """The learned weight on each expert, for reading whether one matters.

        A combiner that collapsed to near-equal weights is reporting that the
        fixed average was enough; a spread means the learned weighting found
        something. Either is a result.
        """
        if self._model is None:
            return {name: 0.0 for name in expert_names}
        return {name: float(w) for name, w in zip(expert_names, self._model.coef_[0])}


class MixtureScorer:
    """The whole defender: experts, a combiner, and the risk banding.

    Presents as a RiskScorer, so the simulator runs against it exactly as it
    runs against the rule engine or the flat baseline. Built by fitting the
    expert bank and the combiner on a training table, then handed to the
    simulator frozen for a round.
    """

    def __init__(self, bank: ExpertBank, combiner, bands=None) -> None:
        self.bank = bank
        self.combiner = combiner
        self.bands = bands  # optional RiskBands; a default banding is used if None

    @classmethod
    def fit(cls, table, learned: bool = True, bands=None) -> "MixtureScorer":
        """Fit the experts and combiner, and attach the banding.

        The bands are what turn a score into a mitigation, so a scorer built
        without them detects and never acts — the graph write-back would sit
        dormant and the loop would not actually close. A default banding is
        attached unless the caller supplies one it has searched.
        """
        from ..engine.bands import RiskBands

        bank = ExpertBank.build(table.columns).fit(table)
        scores, mask = bank.score_matrix(table)
        combiner = LearnedCombiner().fit(scores, mask, table.y) if learned else FixedAverageCombiner()
        return cls(bank, combiner, bands=bands or RiskBands())

    def predict_scores(self, table) -> np.ndarray:
        scores, mask = self.bank.score_matrix(table)
        return self.combiner.combine(scores, mask)

    def score(self, event) -> RiskAssessment:
        """Score one event by building its one-row view and running the bank."""
        log = EventLog()
        log.append(event)
        row = build_table(log, exclude_warm_start=False)
        # Align the row to the bank's columns.
        X = np.zeros((1, len(self.bank.experts[0].columns)))
        col_index = self.bank.experts[0]._col_index
        for i, name in enumerate(row.columns):
            j = col_index.get(name)
            if j is not None:
                X[0, j] = row.X[0, i]
        row = _replace_X(row, X, self.bank.experts[0].columns)
        risk = float(self.predict_scores(row)[0])
        if self.bands is not None:
            action, mitigations = self.bands.decide(risk, event)
        else:
            action, mitigations = _default_action(risk), ()
        return RiskAssessment(risk_score=risk, action=action, mitigations=tuple(mitigations))


def _replace_X(row, X, columns):
    """A single-row table realigned to the bank's column order."""
    from .table import FeatureTable

    return FeatureTable(
        X=X,
        y=row.y,
        columns=columns,
        event_type=row.event_type,
        is_warm_start=row.is_warm_start,
        episode_id=row.episode_id,
        group=row.group,
        events=row.events,
    )


def _default_action(score: float) -> RiskAction:
    if score >= 0.95:
        return RiskAction.BLOCK
    if score >= 0.8:
        return RiskAction.DECLINE
    if score >= 0.6:
        return RiskAction.HOLD
    if score >= 0.3:
        return RiskAction.STEP_UP
    return RiskAction.APPROVE
