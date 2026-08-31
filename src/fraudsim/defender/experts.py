"""Five experts over structurally different feature spaces.

The justification for separate experts is not "many attack types" — it is that
the event types have genuinely different feature spaces. A KYC submission has no
amount; an authorisation has no liveness score; a dispute carries a text
embedding that appears nowhere else. Flatten them into one table and most cells
are null, and the null pattern itself becomes a shortcut a single model learns
instead of behaviour.

**Routing is a schema fact, not a learned gate.** Which expert can score an event
is decided by its type, a dictionary lookup, and there is nothing to learn there
— a gate trained to predict it would reach near-perfect accuracy and contribute
nothing. What is learned is the *combination* of the applicable experts'
opinions, and that lives in the combiner, not here. This is stacking, and naming
it that is more honest than dressing a lookup as a mixture.

Each expert is a small model over its own view of the table. The transaction and
identity experts are trees; the binding expert is a calibrated logistic model
over the recovery-chain features it keys on; the text expert reads the scores the
generative layer already computed; the network expert reads the graph-derived
fan-out features that are stamped on the event, never the live graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.schema import EventType
from ..features.columns import (
    CARD_N_DEVICES,
    DEVICE_AGE_DAYS,
    DEVICE_N_CARDS,
    DEVICE_NEW_TO_CARD,
    TEXT_SCORE_COLUMNS,
)
from ..settings.detector import DetectorConfig, LogisticConfig, TreeConfig
from .table import FeatureTable

# Which event types each expert is responsible for. An event is scored by every
# expert whose set contains its type; most belong to one, some to two.
EXPERT_EVENT_TYPES: dict[str, frozenset[EventType]] = {
    "identity": frozenset({EventType.KYC_SUBMIT, EventType.IVR_CALL}),
    "binding": frozenset(
        {
            EventType.DEVICE_BIND,
            EventType.SIM_CHANGE,
            EventType.AUTH_RESET,
            EventType.PAYEE_ADD,
            EventType.SUPPORT_TICKET,
        }
    ),
    "transaction": frozenset({EventType.AUTH_ATTEMPT, EventType.TRANSFER}),
    "text": frozenset(
        {EventType.DISPUTE_FILED, EventType.REFUND_REQUEST, EventType.SUPPORT_TICKET}
    ),
    # The network expert applies to everything: fan-out is a property of any
    # event, and it is the one expert with a universal feature space.
    "network": frozenset(EventType),
}

# The graph-derived columns the network expert is allowed to read. These are
# stamped on the event by the builder, which may read the graph; the expert
# reads only the event, so the observation boundary holds.
NETWORK_FEATURES = (
    DEVICE_N_CARDS,
    CARD_N_DEVICES,
    DEVICE_NEW_TO_CARD,
    DEVICE_AGE_DAYS,
)

# The text expert reads the generated-text signal: the scalar scores plus the
# embedding columns the artifact stamped on the event. The embedding names are
# discovered from the table at fit time, since their count depends on the
# embedding model, so only the scalar names are fixed here.
TEXT_SCORE_FEATURES = TEXT_SCORE_COLUMNS


class Expert:
    """One model over one view of the table.

    Fit on the rows for its event types, scores the same. An expert that never
    saw a positive in training returns a constant, which the combiner learns to
    discount rather than trust.
    """

    def __init__(self, name: str, columns: tuple[str, ...], kind: str = "tree",
                 tree: TreeConfig | None = None,
                 logistic: LogisticConfig | None = None) -> None:
        self.name = name
        self.columns = columns
        self.kind = kind
        # The expert profile is deliberately shallower than the flat baseline:
        # each sees one event type, so it has fewer rows to spend depth on.
        self.tree = tree or DetectorConfig().expert
        self.logistic = logistic or LogisticConfig()
        self.event_types = EXPERT_EVENT_TYPES[name]
        self._model = None
        self._constant = 0.0
        self._col_index = {c: i for i, c in enumerate(columns)}

    def applies_to(self, event_type: EventType) -> bool:
        return event_type in self.event_types

    # ------------------------------------------------------------------- fit

    def fit(self, table: FeatureTable) -> "Expert":
        view = table.view(self.event_types)
        if len(view) == 0 or view.y.sum() == 0 or (1 - view.y).sum() == 0:
            # Nothing to separate. The expert becomes the base rate, an honest
            # constant rather than a model pretending to a decision.
            self._constant = float(view.y.mean()) if len(view) else 0.0
            self._model = None
            return self

        X = self._select(view.X)
        y = view.y
        pos = max(1, int(y.sum()))
        neg = max(1, int((1 - y).sum()))

        if self.kind == "linear":
            from sklearn.linear_model import LogisticRegression  # lazy; defender extra

            lg = self.logistic
            self._model = LogisticRegression(
                max_iter=lg.max_iter, class_weight=lg.class_weight, C=lg.C
            )
            self._model.fit(X, y)
        else:
            from xgboost import XGBClassifier  # lazy; defender extra

            p = self.tree
            self._model = XGBClassifier(
                n_estimators=p.n_estimators,
                max_depth=p.max_depth,
                learning_rate=p.learning_rate,
                scale_pos_weight=neg / pos,
                min_child_weight=p.min_child_weight,
                subsample=p.subsample,
                colsample_bytree=p.colsample_bytree,
                reg_lambda=p.reg_lambda,
                tree_method=p.tree_method,
                random_state=p.random_state,
                n_jobs=p.n_jobs,
                eval_metric=p.eval_metric,
            )
            self._model.fit(X, y)
        return self

    def _select(self, X: np.ndarray) -> np.ndarray:
        """The columns this expert reads.

        The network expert is restricted to the fan-out columns, since that is
        its whole remit and the observation boundary depends on it reading
        nothing else. Every other expert reads the full row and lets the model
        weight it.
        """
        if self.name == "network":
            idx = [self._col_index[c] for c in NETWORK_FEATURES if c in self._col_index]
            return X[:, idx] if idx else np.zeros((X.shape[0], 1))
        if self.name == "text":
            # The scalar text scores plus every embedding column, which are named
            # emb_0, emb_1, ... The text expert reads meaning and surface stats
            # together; nothing else on the row is its concern.
            names = [c for c in TEXT_SCORE_FEATURES if c in self._col_index]
            names += [c for c in self.columns if c.startswith("emb_")]
            idx = [self._col_index[c] for c in names]
            return X[:, idx] if idx else np.zeros((X.shape[0], 1))
        return X

    # --------------------------------------------------------------- predict

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.full(X.shape[0], self._constant)
        Xs = self._select(X)
        if self.kind == "linear":
            return self._model.predict_proba(Xs)[:, 1]
        return self._model.predict_proba(Xs)[:, 1]


@dataclass
class ExpertBank:
    """The five experts, fit together and scored together.

    Holds the per-expert models and produces, for a set of rows, a matrix of
    expert scores plus an applicability mask — which experts had a right to
    judge each row. The combiner consumes exactly this.
    """

    experts: list[Expert]

    @classmethod
    def build(cls, columns: tuple[str, ...],
              detector: DetectorConfig | None = None) -> "ExpertBank":
        d = detector or DetectorConfig()
        return cls(
            experts=[
                Expert("identity", columns, kind="tree", tree=d.expert, logistic=d.logistic),
                Expert("binding", columns, kind="linear", tree=d.expert, logistic=d.logistic),
                Expert("transaction", columns, kind="tree", tree=d.expert, logistic=d.logistic),
                Expert("text", columns, kind="tree", tree=d.expert, logistic=d.logistic),
                Expert("network", columns, kind="tree", tree=d.expert, logistic=d.logistic),
            ]
        )

    def fit(self, table: FeatureTable) -> "ExpertBank":
        for expert in self.experts:
            expert.fit(table)
        return self

    def score_matrix(self, table: FeatureTable) -> tuple[np.ndarray, np.ndarray]:
        """Per-expert scores and the applicability mask, aligned to the rows.

        The score matrix is (rows, experts); the mask marks where an expert
        applies. Where it does not, its score is still produced but the mask
        tells the combiner to ignore it, so an expert never votes on an event
        outside its remit.
        """
        n = len(table)
        scores = np.zeros((n, len(self.experts)))
        mask = np.zeros((n, len(self.experts)), dtype=bool)
        for j, expert in enumerate(self.experts):
            applies = np.array(
                [expert.applies_to(et) for et in table.event_type], dtype=bool
            )
            mask[:, j] = applies
            # An expert whose remit covers none of these rows is not run at all.
            # Its scores would be masked out by the combiner anyway, and the
            # single-row path — one event scored as it happens, five experts, a
            # scikit-learn call each — is dominated by per-call overhead rather
            # than by arithmetic. Skipping the four that do not apply is most of
            # the cost of scoring one event.
            if not applies.any():
                continue
            scores[:, j] = expert.predict_scores(table.X)
        return scores, mask

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.experts]
