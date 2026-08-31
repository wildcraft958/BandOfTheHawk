"""Population and world-structure configuration."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from .base import PositiveFloat, StrictModel, UnitInterval


class FanoutConfig(StrictModel):
    """How a shared configuration signature spreads across cardholders.

    A fingerprint is not a device. Everyone running the same operating system,
    browser, and screen size collapses into one signature, so the crowd behind
    it is large and heavy tailed. Generating it faithfully matters because a
    graph detector given one card per device would score perfectly for a
    trivial reason.
    """

    exponent: Annotated[float, Field(ge=1.0, le=3.5)] = 1.8
    maximum: Annotated[int, Field(ge=2, le=5000)] = 1642
    share_singleton: UnitInterval = 0.516
    target_mean: PositiveFloat = 8.14
    target_share_shared: UnitInterval = 0.484
    target_p99: PositiveFloat = 138.78
    # Carried so a generated population can be checked against what was
    # measured. Not optimised: dispersion turns on the largest few degrees.
    target_variance_to_mean: PositiveFloat = 229.17


class DeviceConfig(StrictModel):
    """Physical devices, which are what a mitigation may block.

    Sharing here stays at household scale on purpose. Blocking a device has to
    be a proportionate action, and it would not be if a device stood for
    hundreds of unrelated holders.
    """

    household_mean: Annotated[float, Field(ge=1.0, le=6.0)] = 2.1
    household_max: Annotated[int, Field(ge=2, le=20)] = 8
    emulator_share: UnitInterval = 0.01
    age_days_median: PositiveFloat = 180.0
    age_days_spread: PositiveFloat = 1.2

    @model_validator(mode="after")
    def _mean_within_max(self) -> DeviceConfig:
        if self.household_mean > self.household_max:
            raise ValueError("household_mean cannot exceed household_max")
        return self


class HouseholdConfig(StrictModel):
    """Households are what make device sharing organic rather than stamped in."""

    mean_size: Annotated[float, Field(ge=1.0, le=8.0)] = 2.5
    single_occupant_share: UnitInterval = 0.28


class ActivityConfig(StrictModel):
    """Most holders transact rarely.

    The judge dataset has a median of two transactions per entity and 39.5%
    with exactly one. A uniformly active population would give every
    history-dependent feature more to work with than it deserves.
    """

    tier_weights: dict[str, UnitInterval] = Field(
        default_factory=lambda: {
            "dormant": 0.40,
            "occasional": 0.35,
            "regular": 0.20,
            "heavy": 0.05,
        }
    )
    tier_rate_multipliers: dict[str, PositiveFloat] = Field(
        default_factory=lambda: {
            "dormant": 0.15,
            "occasional": 0.6,
            "regular": 1.0,
            "heavy": 4.0,
        }
    )

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ActivityConfig:
        total = sum(self.tier_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"tier_weights must sum to 1, got {total}")
        missing = set(self.tier_weights) - set(self.tier_rate_multipliers)
        if missing:
            raise ValueError(f"tiers without a rate multiplier: {sorted(missing)}")
        return self


class GeoConfig(StrictModel):
    """Distance from home.

    Swept rather than fitted. The taxonomy source places merchants in a ring
    around each customer, with a tenth-percentile distance of 35km and no local
    spend at all, so nothing about its geography can be taken at face value.
    """

    home_radius_km: Annotated[float, Field(ge=1.0, le=60.0)] = 12.0
    travel_share: UnitInterval = 0.04
    travel_distance_km: PositiveFloat = 400.0


class MerchantConfig(StrictModel):
    """Merchant population and how traffic concentrates across it.

    The popularity exponent is swept. The taxonomy source's merchant traffic is
    nearly flat, and the judge dataset carries no merchant entity at all.
    """

    count: Annotated[int, Field(ge=10, le=100_000)] = 2000
    popularity_exponent: Annotated[float, Field(ge=0.3, le=3.0)] = 1.2
    high_liquidity_share: UnitInterval = 0.06
    high_risk_share: UnitInterval = 0.08
    chargeback_rate_mean: UnitInterval = 0.002


class PopulationConfig(StrictModel):
    """Sizes and structure of the generated world."""

    n_holders: Annotated[int, Field(ge=10, le=5_000_000)] = 20_000
    cards_per_holder_mean: Annotated[float, Field(ge=1.0, le=10.0)] = 2.0
    accounts_per_holder_mean: Annotated[float, Field(ge=1.0, le=5.0)] = 1.2
    devices_per_holder_mean: Annotated[float, Field(ge=1.0, le=6.0)] = 1.6

    # Left unset by default, and derived from the fan-out target instead.
    #
    # A fixed count silently contradicts that target: the cards a signature
    # reaches is devices-per-signature times cards-per-device, so choosing the
    # count also chooses the reach. Setting 500 signatures against 32,000
    # devices forces a reach near 134 whatever the degrees say, and the
    # measured data had over nine thousand signatures across a comparable
    # number of rows. Signatures are common, not rare.
    fingerprint_count: Annotated[int, Field(ge=10, le=1_000_000)] | None = None

    households: HouseholdConfig = Field(default_factory=HouseholdConfig)
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    fanout: FanoutConfig = Field(default_factory=FanoutConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)
    merchants: MerchantConfig = Field(default_factory=MerchantConfig)

    archetype_weights: dict[str, UnitInterval] = Field(
        default_factory=lambda: {
            "commuter": 0.24,
            "homebody": 0.20,
            "online_heavy": 0.18,
            "traveller": 0.08,
            "senior": 0.16,
            "business": 0.14,
        }
    )

    @model_validator(mode="after")
    def _archetype_weights_sum_to_one(self) -> PopulationConfig:
        total = sum(self.archetype_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"archetype_weights must sum to 1, got {total}")
        return self

    def resolved_fingerprint_count(self) -> int:
        """How many configuration signatures the population needs.

        Derived from the fan-out target unless set explicitly, since the count
        and the reach are two views of the same quantity and picking one fixes
        the other.
        """
        if self.fingerprint_count is not None:
            return self.fingerprint_count
        n_devices = max(1, int(self.n_holders * self.devices_per_holder_mean))
        cards_reached = n_devices * self.devices.household_mean
        return max(2, int(round(cards_reached / max(self.fanout.target_mean, 1e-6))))


class WarmStartConfig(StrictModel):
    """Backdated history generated before the observation window opens.

    A graph starting cold leaves device age, tenure, and every prior count
    degenerate through the burn-in, which quietly corrupts the earliest events.
    Published work on login risk finds novelty features settle after four to
    eight events, so ten is enough rather than months of history.
    """

    events_per_entity: Annotated[int, Field(ge=0, le=200)] = 10
    lookback_days: Annotated[int, Field(ge=1, le=1095)] = 180
    tenure_days_median: PositiveFloat = 400.0
    tenure_days_spread: PositiveFloat = 1.1
