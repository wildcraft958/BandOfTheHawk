"""Simulator and feature configuration."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from .base import PositiveFloat, StrictModel, UnitInterval


class WindowConfig(StrictModel):
    """Rolling aggregates over a card's recent activity.

    Compound criteria matter more than the plain windows. Published feature
    work finds that counting within a window conditioned on a second matching
    attribute carries most of the lift, and that a window keyed on the card
    alone leaves it on the table.

    Geography is deliberately not one of the criteria. The only source with
    merchant locations places them in a ring around each customer, so a
    geographic bucket here would make an uncalibrated value load-bearing for a
    third of the highest-lift features. Merchant risk tier stands in its place.
    """

    windows_seconds: tuple[int, ...] = (3600, 86_400, 604_800)
    compound_criteria: tuple[str, ...] = ("category_cluster", "entry_mode", "merchant_risk_tier")
    max_events_per_window: Annotated[int, Field(ge=16, le=4096)] = 512

    @model_validator(mode="after")
    def _windows_increase(self) -> WindowConfig:
        if list(self.windows_seconds) != sorted(self.windows_seconds):
            raise ValueError("windows_seconds must be ascending")
        if len(set(self.compound_criteria)) != len(self.compound_criteria):
            raise ValueError("compound_criteria must be distinct")
        return self

    @property
    def n_compound_features(self) -> int:
        """Windows times criteria, for a count and a sum in each cell."""
        return len(self.windows_seconds) * len(self.compound_criteria) * 2


class VelocityRuleConfig(StrictModel):
    """Thresholds for the canonical rule set.

    Used two ways: as a naive baseline to compare a learned detector against,
    and as the measurement behind the hard-negative target. The share of
    ordinary traffic these trip is what makes that target verifiable rather
    than asserted.
    """

    # Thresholds are tuned against generated traffic rather than taken from the
    # source they name. The published set fixes which quantity each rule reads,
    # not where the line sits, and a line drawn before there was traffic to draw
    # it against says nothing. These land the combined rate inside its target
    # with no single rule carrying more than half of it.
    txn_count_1h: Annotated[int, Field(ge=1, le=100)] = 3
    distinct_merchants_24h: Annotated[int, Field(ge=1, le=100)] = 5
    amount_sum_24h: PositiveFloat = 1000.0
    new_account_txn_count: Annotated[int, Field(ge=1, le=50)] = 3
    new_account_days: Annotated[int, Field(ge=1, le=90)] = 7
    distinct_payment_methods_7d: Annotated[int, Field(ge=1, le=50)] = 4
    amount_ratio_30d: PositiveFloat = 6.0
    declines_1h: Annotated[int, Field(ge=1, le=50)] = 2
    distinct_ips_24h: Annotated[int, Field(ge=1, le=50)] = 3


class ChannelConfig(StrictModel):
    """Control thresholds and outcome rates on the payment path.

    These are operating points taken from published sources rather than fitted,
    and the ones sourced from vendor material are swept rather than cited.
    """

    base_decline_rate: UnitInterval = 0.03
    voice_similarity_threshold: UnitInterval = 0.85
    liveness_threshold: UnitInterval = 0.90
    payee_cooling_off_hours: Annotated[int, Field(ge=0, le=168)] = 24


class EpisodeConfig(StrictModel):
    """Bounds on a single actor's run.

    Every action carries a cost and the episode is capped, because an
    unbounded actor with free actions finds a degenerate loop rather than a
    strategy.
    """

    max_actions: Annotated[int, Field(ge=1, le=500)] = 40
    max_hours: Annotated[int, Field(ge=1, le=8760)] = 720
    max_value_per_merchant: PositiveFloat = 2000.0
    # Per-episode jitter on the decision thresholds, the last of the five
    # anti-reward-hacking controls. A fixed boundary is a number a policy can
    # find and then sit just underneath, which is memorising one detector rather
    # than learning to evade detection. Drawn once per episode, so the amount is
    # stable while an attacker acts and different the next time it tries — the
    # attacker cannot binary-search a threshold that moves between attempts.
    #
    # Zero disables it, which is what the static benchmarks want: a fixed
    # adversary against a fixed detector needs a fixed operating point for its
    # numbers to be comparable.
    threshold_jitter: Annotated[float, Field(ge=0.0, le=0.2)] = 0.03


class EngineConfig(StrictModel):
    """Simulator behaviour."""

    windows: WindowConfig = Field(default_factory=WindowConfig)
    rules: VelocityRuleConfig = Field(default_factory=VelocityRuleConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    fraud_base_rate: Annotated[float, Field(ge=0.0001, le=0.2)] = 0.005
