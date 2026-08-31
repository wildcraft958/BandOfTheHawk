"""Fan-out is a gate rather than a checkpoint.

A benign population where one device carries one card would let a graph
detector score perfectly for a reason that says nothing about detection, so
this is checked before anything downstream is built on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.population.fanout import (
    CardDeviceAssigner,
    FingerprintDegreeSampler,
    HouseholdDeviceSampler,
    independent_assignment_degrees,
    summarise,
)
from fraudsim.settings.world import DeviceConfig, FanoutConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(4242)


def test_independent_assignment_cannot_exceed_unit_dispersion(rng) -> None:
    """The bound the generation order exists to escape."""
    degrees = independent_assignment_degrees(80_000, 10_000, rng)
    assert summarise(degrees).variance_to_mean < 1.5


def test_generated_fanout_is_heavy_tailed(rng) -> None:
    degrees = FingerprintDegreeSampler(FanoutConfig()).sample(9412, rng)
    assert summarise(degrees).variance_to_mean > 50.0


def test_generated_fanout_matches_its_targets(rng) -> None:
    config = FanoutConfig()
    summary = summarise(FingerprintDegreeSampler(config).sample(9412, rng))
    assert summary.mean == pytest.approx(config.target_mean, rel=0.25)
    assert summary.share_shared == pytest.approx(config.target_share_shared, abs=0.05)
    assert summary.p99 == pytest.approx(config.target_p99, rel=0.35)


def test_fanout_respects_its_ceiling(rng) -> None:
    """Untruncated, this exponent reaches far past anything observed."""
    config = FanoutConfig(maximum=200)
    assert summarise(FingerprintDegreeSampler(config).sample(20_000, rng)).maximum <= 200


def test_devices_stay_at_household_scale(rng) -> None:
    """Blocking a device has to be proportionate, so it cannot stand for a crowd."""
    config = DeviceConfig()
    summary = summarise(HouseholdDeviceSampler(config).sample(12_000, rng))
    assert summary.maximum <= config.household_max
    assert summary.mean == pytest.approx(config.household_mean, rel=0.15)


def test_device_and_fingerprint_scales_stay_apart(rng) -> None:
    devices = summarise(HouseholdDeviceSampler(DeviceConfig()).sample(12_000, rng))
    fingerprints = summarise(FingerprintDegreeSampler(FanoutConfig()).sample(9412, rng))
    assert fingerprints.maximum > devices.maximum * 50


def test_assignment_fills_the_requested_degrees(rng) -> None:
    cards = np.arange(2000)
    households = cards % 700
    degrees = np.array([1, 2, 3, 5, 8], dtype=np.int64)
    assigned = CardDeviceAssigner(rng).assign(degrees, cards, households)
    assert len(assigned) == len(degrees)
    for wanted, got in zip(degrees.tolist(), assigned, strict=False):
        assert len(got) <= wanted


def test_assignment_prefers_cards_from_one_household(rng) -> None:
    """Sharing should reflect people who live together."""
    cards = np.arange(600)
    households = cards // 3
    degrees = np.full(60, 3, dtype=np.int64)
    assigned = CardDeviceAssigner(rng).assign(degrees, cards, households)

    same_household = 0
    for group in assigned:
        if len(group) > 1 and len(set(households[group].tolist())) == 1:
            same_household += 1
    assert same_household > len(assigned) * 0.5


def test_assignment_rejects_mismatched_inputs(rng) -> None:
    with pytest.raises(ValueError, match="same length"):
        CardDeviceAssigner(rng).assign(
            np.array([1]), np.arange(10), np.arange(5)
        )
