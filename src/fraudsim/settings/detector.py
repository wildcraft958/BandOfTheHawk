"""Defender parameters: the tree models, the combiner, the cost curve, the bands.

About twenty-five literals sat inside `defender/baseline.py`,
`defender/experts.py`, `defender/combiner.py` and `engine/bands.py`, none of them
reachable without editing Python. Two of them were the same numbers written
twice: the risk-band thresholds appear in `engine/bands.py` as RiskBands defaults
and again in `defender/combiner.py` as a fallback, and `engine/bands.py`
hardcoded a review cost that `configs/simulation.yaml` also declares.

Every default here is the value the code used before it moved.

Nothing in this file may import sklearn or xgboost. It is plain data on the
runtime side of the import firewall; the defender tier reads it and fits.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import PositiveFloat, StrictModel, UnitInterval


class TreeConfig(StrictModel):
    """Gradient-boosted tree settings, shared shape for the flat model and experts."""

    n_estimators: Annotated[int, Field(ge=1, le=10_000)] = 200
    max_depth: Annotated[int, Field(ge=1, le=32)] = 6
    learning_rate: PositiveFloat = 0.05
    min_child_weight: Annotated[float, Field(ge=0.0)] = 5.0
    subsample: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    colsample_bytree: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    reg_lambda: Annotated[float, Field(ge=0.0)] = 1.0
    tree_method: Literal["hist", "exact", "approx", "auto"] = "hist"
    random_state: Annotated[int, Field(ge=0)] = 0
    n_jobs: int = -1
    eval_metric: str = "aucpr"


class LogisticConfig(StrictModel):
    """The linear experts and the combiner that stacks them."""

    max_iter: Annotated[int, Field(ge=1)] = 2000
    class_weight: Literal["balanced"] | None = "balanced"
    C: PositiveFloat = 1.0


class BandConfig(StrictModel):
    """Score thresholds separating approve, step up, hold, decline and block."""

    step_up_at: UnitInterval = 0.30
    hold_at: UnitInterval = 0.60
    decline_at: UnitInterval = 0.80
    block_at: UnitInterval = 0.95

    @model_validator(mode="after")
    def _ascending(self) -> "BandConfig":
        edges = (self.step_up_at, self.hold_at, self.decline_at, self.block_at)
        if list(edges) != sorted(edges):
            raise ValueError(f"bands must ascend, got {edges}")
        return self


class CostConfig(StrictModel):
    """What each defensive action costs, so bands land where the three trade off.

    review_cost duplicated `engine.channel.manual_review_cost`, which is declared
    in the YAML and was being ignored.
    """

    friction_cost: PositiveFloat = 5.0
    review_cost: PositiveFloat = 8.0
    # Threshold grid searched when fitting bands against the cost curve.
    search_steps: Annotated[int, Field(ge=2, le=100)] = 9
    search_low: UnitInterval = 0.10
    search_high: UnitInterval = 0.95


class SplitConfig(StrictModel):
    """How the labelled table is divided, by entity so no holder spans both sides."""

    test_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.3


class MetricsConfig(StrictModel):
    """What detection quality is reported against.

    PR-AUC over accuracy, because at a 0.5% base rate accuracy is a constant.
    """

    alert_budget: Annotated[int, Field(ge=1)] = 100
    false_positive_budgets: tuple[UnitInterval, ...] = (0.001, 0.01)


class DetectorConfig(StrictModel):
    """Everything the defender's fitting reads."""

    baseline: TreeConfig = Field(default_factory=TreeConfig)
    expert: TreeConfig = Field(
        default_factory=lambda: TreeConfig(
            n_estimators=150, max_depth=4, min_child_weight=3.0,
            subsample=0.9, colsample_bytree=0.9,
        )
    )
    logistic: LogisticConfig = Field(default_factory=LogisticConfig)
    bands: BandConfig = Field(default_factory=BandConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
