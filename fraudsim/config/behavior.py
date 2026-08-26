"""Spending behaviour configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import PositiveFloat, StrictModel, UnitInterval


class AmountConfig(StrictModel):
    """Lognormal body spliced to a truncated Pareto tail.

    The tail is bounded. An unbounded draw at this index reaches several times
    past anything in the source, which widens every distance while leaving each
    summary statistic looking correct.

    The whole-number share exists because amounts are prices. Around half of
    real transactions land on a whole currency unit and single price points
    recur tens of thousands of times; a continuous draw has none of that, and
    the cents digit alone would separate it from real data.
    """

    # A card's own level, and the spread around it.
    #
    # These replace a single shared curve. With one curve the only thing
    # separating two cards is sampling, so the spread of per-card means comes
    # out at about half what real cards show. A marginal comparison cannot see
    # that: pooling every transaction ignores which card produced it, and a
    # detector reading an amount against a card's own history reads exactly
    # what pooling gets wrong.
    level_mean: float = 4.44
    between_sd: PositiveFloat = 0.605
    within_sd: PositiveFloat = 0.706

    # Kept for the pooled marginal, which the fidelity comparison still uses.
    lognormal_mu: float = 4.245
    lognormal_sigma: PositiveFloat = 0.803
    tail_threshold: PositiveFloat = 435.0
    tail_index: Annotated[float, Field(ge=1.05, le=3.0)] = 1.79
    tail_fraction: UnitInterval = 0.05
    upper_bound: PositiveFloat = 31937.39
    whole_number_share: UnitInterval = 0.516
    category_spread: Annotated[float, Field(ge=0.05, le=1.0)] = 0.35


class ArrivalConfig(StrictModel):
    """Gaps between an entity's transactions.

    A renewal draw under a rate that drifts as it goes. Two alternatives were
    fitted first and neither survived: a self-exciting kernel failed its
    goodness-of-fit gate, and a session model landed at negative lag-1
    autocorrelation against a positive target.

    Decomposing the real signal explained both. Raw consecutive gaps correlate
    at about +0.06, but after dividing each by a local rolling median the
    correlation vanishes. Nothing survives detrending, so neighbouring gaps
    resemble each other because they were drawn under a similar rate, not
    because one event triggered the next. Both rejected models describe
    short-range clustering, which is why neither could reach it.
    """

    model: Literal["drifting_rate"] = "drifting_rate"
    rate_log_mean: float = -12.99
    rate_log_sigma: PositiveFloat = 0.93
    drift_sigma: Annotated[float, Field(ge=0.0, le=2.0)] = 0.35
    drift_persistence: UnitInterval = 0.70
    gap_shape: Annotated[float, Field(ge=0.05, le=10.0)] = 0.8
    target_autocorrelation: float = 0.0378
    target_burstiness: float = 0.0719
    target_gap_median: PositiveFloat = 279436.5


class CircadianConfig(StrictModel):
    """Hour of day as a von Mises mixture.

    Hour is circular, so 23:00 and 01:00 are two hours apart rather than
    twenty-two, and a linear bucket cannot express that. Two components are the
    default because one misses the daytime shoulder badly.
    """

    means: tuple[float, ...] = (23.5, 16.9)
    concentrations: tuple[float, ...] = (1.62, 2.29)
    weights: tuple[float, ...] = (0.55, 0.45)
    confidence: UnitInterval = 0.95
    min_history_days: Annotated[int, Field(ge=1, le=90)] = 7

    @model_validator(mode="after")
    def _components_align(self) -> "CircadianConfig":
        if not len(self.means) == len(self.concentrations) == len(self.weights):
            raise ValueError("means, concentrations, and weights must be the same length")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {total}")
        if any(not 0.0 <= m < 24.0 for m in self.means):
            raise ValueError("means must lie in [0, 24)")
        return self


class CategoryConfig(StrictModel):
    """Merchant category mix and which categories are card-not-present."""

    mix: dict[str, UnitInterval] = Field(
        default_factory=lambda: {
            "grocery": 0.1297,
            "fuel_transit": 0.1016,
            "dining": 0.0708,
            "retail": 0.3343,
            "online": 0.1227,
            "entertainment": 0.0727,
            "health": 0.1367,
            "travel": 0.0313,
        }
    )
    card_not_present_share: dict[str, UnitInterval] = Field(
        default_factory=lambda: {
            "grocery": 0.12,
            "fuel_transit": 0.02,
            "dining": 0.10,
            "retail": 0.25,
            "online": 0.95,
            "entertainment": 0.45,
            "health": 0.20,
            "travel": 0.70,
        }
    )

    @model_validator(mode="after")
    def _mix_sums_to_one(self) -> "CategoryConfig":
        total = sum(self.mix.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"category mix must sum to 1, got {total}")
        missing = set(self.mix) - set(self.card_not_present_share)
        if missing:
            raise ValueError(f"categories without a card-not-present share: {sorted(missing)}")
        return self


class HardNegativeConfig(StrictModel):
    """Legitimate behaviour that looks suspicious.

    Without these the false-positive rate is meaningless. The target is that a
    small share of ordinary traffic trips a naive rule engine, which is a
    calibration objective to tune towards rather than an input.

    The recovery chain is the important one and the least knowable. No
    published figure exists for how often a legitimate holder resets a
    password, calls support, and rebinds a device within an hour, and the
    binding detector keys on exactly that sequence. It is swept across three
    orders of magnitude, and any claim about it has to hold across the range.
    """

    naive_rule_trip_target: UnitInterval = 0.065
    new_device_rate_yearly: PositiveFloat = 0.6
    travel_rate_yearly: PositiveFloat = 2.0
    large_purchase_share: UnitInterval = 0.01
    dispute_rate: UnitInterval = 0.0008
    password_reset_rate_yearly: PositiveFloat = 0.35
    recovery_chain_probability: Annotated[float, Field(ge=1e-8, le=1e-3)] = 1e-5
    gift_card_share: UnitInterval = 0.02
    # One session becomes several transactions minutes apart, so its share of
    # slots is not its share of events. Set against the combined target rather
    # than read off a source, since nothing reports how often a holder shops in
    # one sitting.
    shopping_session_share: UnitInterval = 0.012


class BehaviorConfig(StrictModel):
    """Everything that determines what ordinary activity looks like."""

    amount: AmountConfig = Field(default_factory=AmountConfig)
    arrival: ArrivalConfig = Field(default_factory=ArrivalConfig)
    circadian: CircadianConfig = Field(default_factory=CircadianConfig)
    categories: CategoryConfig = Field(default_factory=CategoryConfig)
    hard_negatives: HardNegativeConfig = Field(default_factory=HardNegativeConfig)
