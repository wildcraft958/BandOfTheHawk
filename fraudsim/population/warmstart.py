"""Backdated history.

A world starting cold leaves device age, tenure, and every prior count
degenerate through the burn-in, which quietly corrupts the earliest events a
detector would train on. Published work on login risk finds novelty features
settle after four to eight events, so ten per entity is enough rather than
months of traffic.

History runs through the real step method rather than being written directly
into the state. Two things follow from that. Derived fields like a card's usual
amount are genuinely realised rather than asserted, and the rolling windows
arrive populated rather than empty. It also means the warm start exercises the
same code the run does, so a bug here surfaces here.

Hard negatives are injected during this phase: ordinary behaviour that looks
suspicious. Without them a false-positive rate has nothing to measure, and the
rules keyed on travel, sessions, and new devices never fire on anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.simulation import SimulationConfig
from ..engine.actions import Action, ActionName
from ..engine.simulator import Actor, ActorKind, Simulator
from ..engine.stages import Stage
from ..ids import ActorId, CardId
from ..timing.arrival import ArrivalState, DriftingRateProcess
from ..world.entities import ActivityTier, Archetype

MINUTES_PER_DAY = 1440
SECONDS_PER_MINUTE = 60


@dataclass
class WarmStartReport:
    """What the history contained, against what it was meant to contain."""

    n_entities: int
    n_events: int
    events_per_entity: dict[str, float] = field(default_factory=dict)
    hard_negatives: dict[str, int] = field(default_factory=dict)
    dormant_share: float = 0.0
    cards_with_median: float = 0.0

    def render(self) -> str:
        lines = [
            "warm start",
            f"  entities            {self.n_entities:>10,}",
            f"  events              {self.n_events:>10,}",
            "",
            "  events per entity",
        ]
        for name in ("min", "p10", "median", "p90", "max"):
            lines.append(f"    {name:<16}{self.events_per_entity.get(name, 0):>10.1f}")
        lines += [
            "",
            f"  dormant share       {self.dormant_share:>10.3f}",
            f"  cards with a median {self.cards_with_median:>10.3f}",
            "",
            "  hard negatives injected",
        ]
        for name in sorted(self.hard_negatives):
            lines.append(f"    {name:<16}{self.hard_negatives[name]:>10,}")
        return "\n".join(lines)


class WarmStartRunner:
    """Fills the world with history by running it."""

    def __init__(
        self,
        simulator: Simulator,
        config: SimulationConfig,
        seed: int = 0,
    ) -> None:
        self.simulator = simulator
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._process = DriftingRateProcess(config.behavior.arrival)
        self._injected: dict[str, int] = {
            "travel": 0,
            "large_purchase": 0,
            "session": 0,
            "new_device": 0,
            "gift_card": 0,
        }

    def run(self) -> WarmStartReport:
        graph = self.simulator.graph
        warm = self.config.warm_start
        self.simulator.builder.set_warm_start(True)

        cards = self._eligible_cards()
        schedule = self._build_schedule(cards, warm.events_per_entity)

        # History is backdated, so the clock has to start behind the
        # observation window. Leaving it at the origin makes every event land
        # at the same instant, since the clock refuses to move backwards, and
        # the resulting traffic looks like one enormous burst.
        if schedule:
            self.simulator.clock.rewind_to(schedule[0][0])

        merchants = list(graph.merchants)
        liquid = [m for m in merchants if graph.merchants[m].is_high_liquidity] or merchants
        amount = self.config.behavior.amount
        negatives = self.config.behavior.hard_negatives

        counts: dict[int, int] = {}
        for ts, card_id in schedule:
            actor_id = self._actor_for(card_id)
            merchant_id, value = self._choose(
                card_id, merchants, liquid, amount, negatives, ts
            )
            self.simulator.clock.advance_to(max(self.simulator.clock.now, ts))
            self.simulator.step(
                actor_id,
                Action(
                    name=ActionName.ATTEMPT_AUTH,
                    target_id=card_id,
                    secondary_id=merchant_id,
                    amount=value,
                    entry_mode=int(self.rng.integers(0, 4)),
                ),
            )
            counts[card_id] = counts.get(card_id, 0) + 1

        self.simulator.builder.set_warm_start(False)
        return self._report(counts, cards)

    # ------------------------------------------------------------ schedule

    def _eligible_cards(self) -> list[CardId]:
        """Cards that could actually transact.

        A card with no binding has nothing to authorise through, so including
        one would only produce failures.
        """
        graph = self.simulator.graph
        return [c for c in graph.cards if graph.devices_of_card(c)]

    def _build_schedule(
        self, cards: list[CardId], events_per_entity: int
    ) -> list[tuple[int, CardId]]:
        """Draw each card's own timeline, then merge them in time order.

        The windows evict from the front assuming it is the oldest entry, so
        the merged stream has to be sorted. Interleaving without sorting would
        hand a card its own history out of order.
        """
        graph = self.simulator.graph
        activity = self.config.population.activity
        lookback = self.config.warm_start.lookback_days * MINUTES_PER_DAY

        schedule: list[tuple[int, CardId]] = []
        for card_id in cards:
            holder = graph.holders[graph.cards[card_id].holder_id]
            multiplier = activity.tier_rate_multipliers.get(
                holder.activity_tier.value, 1.0
            )
            count = self._event_count(holder.activity_tier, events_per_entity)
            if count == 0:
                continue

            state = self._process.new_state(self.rng, rate_multiplier=multiplier)
            elapsed = 0.0
            for _ in range(count):
                elapsed += self._process.next_gap_seconds(state, self.rng)
                minutes = int(elapsed / SECONDS_PER_MINUTE)
                if minutes > lookback:
                    break
                schedule.append((minutes - lookback, card_id))

        schedule.sort()
        return schedule

    def _event_count(self, tier: ActivityTier, base: int) -> int:
        """How much history one card gets.

        Most holders transact rarely, so a uniform count would hand every
        history-dependent feature more to work with than it would ever have.
        """
        if tier is ActivityTier.DORMANT:
            return int(self.rng.integers(0, max(2, base // 4)))
        if tier is ActivityTier.OCCASIONAL:
            return int(self.rng.integers(1, max(2, base // 2)))
        if tier is ActivityTier.HEAVY:
            return int(self.rng.integers(base, base * 3))
        return int(self.rng.integers(max(1, base // 2), base + 2))

    # -------------------------------------------------------------- actors

    def _actor_for(self, card_id: CardId) -> ActorId:
        """One actor per holder, created on first use."""
        graph = self.simulator.graph
        holder_id = graph.cards[card_id].holder_id
        actor_id = ActorId(int(holder_id))
        try:
            self.simulator.actor(actor_id)
        except KeyError:
            self.simulator.register_actor(
                Actor(
                    actor_id=actor_id,
                    kind=ActorKind.BENIGN,
                    stage=Stage.BOUND,
                    holder_id=int(holder_id),
                    cards=list(graph.cards_of_holder(holder_id)),
                )
            )
        return actor_id

    # ---------------------------------------------------- hard negatives

    def _choose(self, card_id, merchants, liquid, amount, negatives, ts):
        """Pick a merchant and an amount, sometimes an awkward one.

        These are the events that make a false-positive rate meaningful:
        ordinary behaviour that a naive rule would flag. Their rates are tuned
        towards a target rather than measured, since no source reports how
        often a legitimate holder does any of this.
        """
        roll = self.rng.random()
        base = float(
            np.clip(
                self.rng.lognormal(amount.lognormal_mu, amount.lognormal_sigma),
                1.0,
                amount.upper_bound,
            )
        )

        if roll < negatives.large_purchase_share:
            self._injected["large_purchase"] += 1
            return merchants[int(self.rng.integers(0, len(merchants)))], base * float(
                self.rng.uniform(8.0, 20.0)
            )

        if roll < negatives.large_purchase_share + negatives.gift_card_share:
            self._injected["gift_card"] += 1
            return liquid[int(self.rng.integers(0, len(liquid)))], base

        return merchants[int(self.rng.integers(0, len(merchants)))], base

    # -------------------------------------------------------------- report

    def _report(self, counts: dict[int, int], cards: list[CardId]) -> WarmStartReport:
        graph = self.simulator.graph
        values = np.array([counts.get(c, 0) for c in cards], dtype=float)
        with_median = sum(
            1 for c in cards if graph.cards[c].median_amount is not None
        )

        return WarmStartReport(
            n_entities=len(cards),
            n_events=int(values.sum()),
            events_per_entity={
                "min": float(values.min()) if len(values) else 0.0,
                "p10": float(np.quantile(values, 0.1)) if len(values) else 0.0,
                "median": float(np.median(values)) if len(values) else 0.0,
                "p90": float(np.quantile(values, 0.9)) if len(values) else 0.0,
                "max": float(values.max()) if len(values) else 0.0,
            },
            hard_negatives=dict(self._injected),
            dormant_share=float((values <= 2).mean()) if len(values) else 0.0,
            cards_with_median=with_median / max(len(cards), 1),
        )
