"""How much cardholders differ from one another.

A single amount distribution says every card is drawn from the same curve, so
the only reason two cards differ is sampling. Real cards differ far more than
that: their mean log amount has a spread of about 0.69, while a pooled
generator produces 0.38, and all of the latter is noise.

That gap is invisible to a marginal metric. Pooling every transaction and
comparing the result ignores which card made it, so a generator can match the
overall shape exactly while distributing it across cards entirely wrongly. The
statistic that sees it is the spread of per-entity means, and it is measured
here alongside the parameters that reproduce it.

The decomposition is the useful part. Total variance splits into a between-card
component, which says cards have different habits, and a within-card component,
which says a card's own purchases vary. A pooled fit collapses the first into
the second and gets the total right by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class HeterogeneityFit:
    """A per-card level plus the spread around it."""

    grand_mean: float
    between_sd: float
    within_sd: float
    between_share: float
    n_entities: int
    n_events: int

    @property
    def total_sd(self) -> float:
        """What a pooled fit would see, with the two components collapsed."""
        return float(np.sqrt(self.between_sd**2 + self.within_sd**2))

    def sample_level(self, rng: np.random.Generator) -> float:
        """One card's own amount level, in log space."""
        return float(rng.normal(self.grand_mean, self.between_sd))

    def sample_amount(self, level: float, rng: np.random.Generator) -> float:
        return float(np.exp(rng.normal(level, self.within_sd)))

    def as_dict(self) -> dict[str, float]:
        payload = {k: float(v) for k, v in asdict(self).items()}
        payload["total_sd"] = self.total_sd
        return payload


def fit_heterogeneity(
    frame: pd.DataFrame,
    entity_column: str,
    value_column: str,
    min_events: int = 5,
) -> HeterogeneityFit:
    """Split the spread of a value into between-card and within-card parts.

    Only cards with enough history contribute to the between-card estimate. A
    card seen once has a mean equal to its single observation, and treating
    that as evidence about its habits would inflate the between-card term with
    what is really within-card noise.
    """
    values = frame[frame[value_column] > 0]
    logs = np.log(values[value_column].to_numpy(float))
    grouped = pd.DataFrame({"entity": values[entity_column].to_numpy(), "log": logs})

    stats = grouped.groupby("entity", observed=True)["log"].agg(["mean", "std", "size"])
    usable = stats[stats["size"] >= min_events]
    if usable.empty:
        raise ValueError(f"no entity has {min_events} or more observations")

    within = float(np.sqrt(np.nanmedian(usable["std"].to_numpy() ** 2)))

    # The observed scatter of entity means is not the between-entity spread. An
    # entity seen k times has a mean carrying sampling noise of within/sqrt(k)
    # on top of its true level, so the raw scatter is both combined. Subtracting
    # the sampling term leaves the part that is actually about entities
    # differing.
    #
    # Skipping this reads high, and a generator built on the inflated figure
    # produces a spread that stays wide however many events a card accumulates,
    # where a real one narrows.
    observed = float(usable["mean"].var(ddof=1))
    sampling = float(np.mean(within**2 / usable["size"].to_numpy()))
    between = float(np.sqrt(max(observed - sampling, 1e-6)))
    total = between**2 + within**2

    return HeterogeneityFit(
        grand_mean=float(usable["mean"].mean()),
        between_sd=between,
        within_sd=within,
        between_share=float(between**2 / total) if total else float("nan"),
        n_entities=int(len(usable)),
        n_events=int(len(logs)),
    )


def entity_level_spread(
    frame: pd.DataFrame,
    entity_column: str,
    value_column: str,
    min_events: int = 5,
) -> np.ndarray:
    """Mean log value per entity.

    The statistic a marginal comparison cannot see. Two datasets can share a
    pooled distribution exactly while disagreeing completely about how it is
    spread across the entities that produced it.
    """
    values = frame[frame[value_column] > 0]
    logs = np.log(values[value_column].to_numpy(float))
    grouped = pd.DataFrame({"entity": values[entity_column].to_numpy(), "log": logs})
    stats = grouped.groupby("entity", observed=True)["log"].agg(["mean", "size"])
    return stats[stats["size"] >= min_events]["mean"].to_numpy(dtype=float)
