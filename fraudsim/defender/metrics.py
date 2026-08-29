"""The metrics a detection task is actually judged on.

At a 0.5% base rate, accuracy and plain ROC-AUC flatter a model that never fires
— predicting benign everywhere scores 99.5% accurate and says nothing. What an
issuer cares about is caught fraud at a tolerable false-positive rate and the
precision of a fixed alert budget an investigator can actually work through.

These are numpy, no sklearn, so they sit beside the table on the runtime side of
the import firewall and the same numbers can be computed anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, by the trapezoid rule.

    Primary for imbalanced detection: it ignores the true negatives that ROC-AUC
    lets dominate at this base rate, and rewards ranking the rare positives high.
    """
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    total_pos = y.sum()
    if total_pos == 0:
        return 0.0
    recall = tp / total_pos
    precision = tp / np.maximum(tp + fp, 1)
    # Prepend the (recall=0, precision=1) origin so the curve is anchored.
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Reported alongside PR-AUC, never instead of it.

    Kept because it is the number reviewers expect, with the caveat that at a
    0.5% base rate it is optimistic — a model can look excellent here while
    missing most fraud at any usable threshold.
    """
    order = np.argsort(-scores)
    y = y_true[order]
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return 0.5
    tpr = np.cumsum(y) / pos
    fpr = np.cumsum(1 - y) / neg
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapezoid(tpr, fpr))


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    """Fraud caught when the false-positive rate is held at a budget.

    The operational number: at 0.1% and 1% FPR, what share of fraud is above the
    threshold. This is what a review team's capacity actually buys.
    """
    order = np.argsort(-scores)
    y = y_true[order]
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return 0.0
    fpr = np.cumsum(1 - y) / neg
    tpr = np.cumsum(y) / pos
    allowed = fpr <= fpr_target
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision of the top-k alerts — a fixed investigation budget."""
    if k <= 0:
        return 0.0
    order = np.argsort(-scores)[:k]
    return float(y_true[order].mean())


@dataclass
class DetectionMetrics:
    """The full set, for one model on one evaluation split."""

    pr_auc: float
    roc_auc: float
    recall_at_0p1: float
    recall_at_1: float
    precision_at_budget: float
    n_positives: int
    n_total: int

    @classmethod
    def compute(
        cls, y_true: np.ndarray, scores: np.ndarray, alert_budget: int = 100
    ) -> "DetectionMetrics":
        return cls(
            pr_auc=pr_auc(y_true, scores),
            roc_auc=roc_auc(y_true, scores),
            recall_at_0p1=recall_at_fpr(y_true, scores, 0.001),
            recall_at_1=recall_at_fpr(y_true, scores, 0.01),
            precision_at_budget=precision_at_k(y_true, scores, alert_budget),
            n_positives=int(y_true.sum()),
            n_total=int(y_true.shape[0]),
        )

    def render(self, label: str = "") -> str:
        head = f"  {label}" if label else "  metrics"
        return "\n".join(
            [
                head,
                f"    PR-AUC              {self.pr_auc:>8.4f}",
                f"    ROC-AUC            {self.roc_auc:>8.4f}",
                f"    recall @0.1% FPR   {self.recall_at_0p1:>8.4f}",
                f"    recall @1% FPR     {self.recall_at_1:>8.4f}",
                f"    precision @budget  {self.precision_at_budget:>8.4f}",
                f"    positives          {self.n_positives:>8,} / {self.n_total:,}",
            ]
        )
