"""Transaction amounts.

Each card carries its own level, drawn once and kept. Without one every card
draws from the same curve, so the only thing separating two cards is sampling
noise, and the spread of per-card means comes out at roughly half what real
cards show.

That difference does not appear in a marginal comparison. Pooling every
transaction and checking the shape ignores which card produced it, so a
generator can match the overall distribution while distributing it across cards
wrongly. Any detector reading a card's amount relative to its own history is
reading exactly what the pooled version gets wrong.

Two adjustments sit on top of the level. The body is lognormal but the tail is
Pareto and bounded, because an unbounded draw at the fitted index reaches past
anything real. And around half of amounts land on a whole currency unit, since
amounts are prices; a continuous draw has none of that, and the cents digit
alone would separate it from real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..settings.behavior import AmountConfig
from ..world.entities import Archetype

# How each archetype shifts a card's level, in log space, as a share of the
# fitted between-card spread rather than an amount added to it.
#
# Adding an absolute shift double-counts. The between-card spread was measured
# across real cardholders, who already are a mixture of habits, so whatever
# separates a business traveller from a pensioner is inside it. Layering a
# further shift on top raised the generated spread from 0.379 to 0.740 against
# a target of 0.605: too little became too much.
#
# Expressed as shares that average to zero, these redistribute the fitted
# spread across archetypes instead of enlarging it.
ARCHETYPE_LEVEL_TILT: dict[Archetype, float] = {
    Archetype.COMMUTER: -0.25,
    Archetype.HOMEBODY: -0.15,
    Archetype.ONLINE_HEAVY: 0.05,
    Archetype.TRAVELLER: 0.45,
    Archetype.SENIOR: 0.15,
    Archetype.BUSINESS: 0.70,
}


@dataclass(slots=True)
class AmountProfile:
    """One card's spending level."""

    level: float
    within_sd: float

    def sample(self, config: AmountConfig, rng: np.random.Generator) -> float:
        """Draw an amount for this card."""
        value = float(np.exp(rng.normal(self.level, self.within_sd)))

        if rng.random() < config.tail_fraction:
            # Pareto beyond the splice point, truncated at the observed
            # maximum. Untruncated, this index reaches several times past
            # anything in the source.
            u = float(rng.random())
            ceiling = (config.tail_threshold / config.upper_bound) ** config.tail_index
            scaled = 1.0 - u * (1.0 - ceiling)
            value = config.tail_threshold * scaled ** (-1.0 / config.tail_index)

        value = float(np.clip(value, 0.01, config.upper_bound))
        if rng.random() < config.whole_number_share:
            return max(1.0, float(round(value)))
        return round(value, 2)


class AmountModel:
    """Amount levels for a population of cards."""

    __slots__ = ("_config", "_profiles")

    def __init__(self, config: AmountConfig) -> None:
        self._config = config
        self._profiles: dict[int, AmountProfile] = {}

    def register(
        self,
        card_id: int,
        rng: np.random.Generator,
        archetype: Archetype | None = None,
    ) -> AmountProfile:
        """Give a card its own level, once.

        The level is drawn from a between-card spread measured on real data
        rather than assumed. Where that spread is zero the model reduces to a
        single shared curve, which is the behaviour this exists to replace.
        """
        spread = max(self._config.between_sd, 1e-9)
        tilt = ARCHETYPE_LEVEL_TILT.get(archetype, 0.0) if archetype else 0.0

        # The tilt moves where this archetype sits inside the fitted spread and
        # the remainder is drawn around it, so the two together reproduce that
        # spread rather than exceeding it.
        residual = float(np.sqrt(max(1.0 - tilt**2, 0.05)))
        level = float(
            rng.normal(self._config.level_mean + tilt * spread, spread * residual)
        )

        profile = AmountProfile(level=level, within_sd=self._config.within_sd)
        self._profiles[card_id] = profile
        return profile

    def profile(self, card_id: int) -> AmountProfile | None:
        return self._profiles.get(card_id)

    def sample(self, card_id: int, rng: np.random.Generator) -> float:
        profile = self._profiles.get(card_id)
        if profile is None:
            raise KeyError(f"card {card_id} has no amount profile; register it first")
        return profile.sample(self._config, rng)

    def __len__(self) -> int:
        return len(self._profiles)


def level_spread(amounts_by_card: dict[int, list[float]], min_events: int = 5) -> float:
    """Spread of per-card mean log amount.

    The statistic that distinguishes a population of different cardholders from
    one distribution sampled repeatedly.
    """
    means = [
        float(np.mean(np.log(values)))
        for values in amounts_by_card.values()
        if len(values) >= min_events and min(values) > 0
    ]
    return float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
