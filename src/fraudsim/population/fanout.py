"""Device and fingerprint assignment.

The order here is the whole point. Degrees are drawn first and cards are
matched onto them afterwards. Letting each card pick a device independently
from a marginal gives a Poisson-binomial degree distribution, whose variance
cannot exceed its mean, and the measured dispersion is in the hundreds. No
amount of tuning recovers from picking independently; it has to be avoided by
construction.

Two structures, deliberately separate.

    device       a physical object, shared at household scale, and the thing a
                 mitigation may block
    fingerprint  a configuration signature that many unrelated devices share,
                 carrying the heavy tail

Keeping them apart is what makes blocking a device a proportionate action. The
measured tail reaches over a thousand cards behind a single signature, and its
tail index sits near one, implying an infinite mean. That is a property of
grouping strangers by their operating system and screen size, not of any device
anyone owns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..settings.world import DeviceConfig, FanoutConfig


@dataclass(frozen=True, slots=True)
class FanoutSummary:
    """Realised degree statistics, for checking against what was measured."""

    n_nodes: int
    mean: float
    variance_to_mean: float
    share_shared: float
    p99: float
    maximum: int

    def as_dict(self) -> dict[str, float]:
        return {
            "n_nodes": float(self.n_nodes),
            "mean": self.mean,
            "variance_to_mean": self.variance_to_mean,
            "share_shared": self.share_shared,
            "p99": self.p99,
            "max": float(self.maximum),
        }


def summarise(degrees: np.ndarray) -> FanoutSummary:
    degrees = np.asarray(degrees, dtype=float)
    if len(degrees) == 0:
        raise ValueError("no degrees to summarise")
    mean = float(degrees.mean())
    variance = float(degrees.var(ddof=1)) if len(degrees) > 1 else 0.0
    return FanoutSummary(
        n_nodes=len(degrees),
        mean=mean,
        variance_to_mean=variance / mean if mean else float("nan"),
        share_shared=float((degrees > 1).mean()),
        p99=float(np.quantile(degrees, 0.99)),
        maximum=int(degrees.max()),
    )


class FingerprintDegreeSampler:
    """Draws how many cardholders sit behind each configuration signature."""

    __slots__ = ("_config", "_support", "_weights")

    def __init__(self, config: FanoutConfig) -> None:
        self._config = config
        self._support = np.arange(2, config.maximum + 1)
        weights = self._support.astype(float) ** (-config.exponent)
        self._weights = weights / weights.sum()

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """A degree per signature.

        Most signatures cover one holder. The rest follow a power law truncated
        at the observed maximum; untruncated, this exponent reaches well past
        anything real.
        """
        degrees = np.ones(size, dtype=np.int64)
        shared = rng.random(size) >= self._config.share_singleton
        n_shared = int(shared.sum())
        if n_shared:
            degrees[shared] = rng.choice(self._support, size=n_shared, p=self._weights)
        return degrees


class HouseholdDeviceSampler:
    """Draws how many cards a physical device carries.

    Bounded at household scale. A device standing for hundreds of unrelated
    holders would make blocking it a disproportionate action, and the long tail
    belongs to the fingerprint it shares, not to the device itself.
    """

    __slots__ = ("_config",)

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        extra = rng.poisson(max(self._config.household_mean - 1.0, 0.0), size)
        return np.clip(extra + 1, 1, self._config.household_max).astype(np.int64)


class CardDeviceAssigner:
    """Matches cards onto a drawn degree sequence.

    Cards from the same household are preferred when filling a device, so
    sharing reflects people who live together rather than an arbitrary pairing.
    Households run out before the degrees do, and the remainder is filled from
    the wider pool, which is what produces the occasional device carrying
    unrelated cards without making that the norm.
    """

    __slots__ = ("_rng",)

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def assign(
        self,
        degrees: np.ndarray,
        card_ids: np.ndarray,
        card_households: np.ndarray,
    ) -> list[np.ndarray]:
        """Return the cards assigned to each device."""
        if len(card_ids) != len(card_households):
            raise ValueError("card_ids and card_households must be the same length")

        by_household: dict[int, list[int]] = {}
        for card_id, household in zip(card_ids.tolist(), card_households.tolist(), strict=False):
            by_household.setdefault(int(household), []).append(int(card_id))
        for cards in by_household.values():
            self._rng.shuffle(cards)

        households = np.array(sorted(by_household), dtype=np.int64)
        self._rng.shuffle(households)

        assignments: list[np.ndarray] = []
        cursor = 0
        for degree in degrees.tolist():
            chosen: list[int] = []
            attempts = 0
            while len(chosen) < degree and attempts < len(households):
                household = int(households[(cursor + attempts) % len(households)])
                pool = by_household.get(household, [])
                take = min(degree - len(chosen), len(pool))
                if take > 0:
                    chosen.extend(pool[:take])
                attempts += 1
            cursor = (cursor + 1) % max(len(households), 1)
            assignments.append(np.asarray(chosen, dtype=np.int64))
        return assignments


def independent_assignment_degrees(
    n_cards: int, n_devices: int, rng: np.random.Generator
) -> np.ndarray:
    """Degrees from letting each card pick a device independently.

    Kept as the counterexample. This is what the ordering above exists to
    avoid, and its dispersion cannot exceed one however the marginal is chosen.
    """
    picks = rng.integers(0, n_devices, size=n_cards)
    return np.bincount(picks, minlength=n_devices).astype(np.int64)
