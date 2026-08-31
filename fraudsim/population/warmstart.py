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
from ..features.schema import EventType
from ..ids import ActorId, BucketId, CardId, DeviceId
from ..timing.arrival import DriftingRateProcess
from ..timing.circadian import HolderClockModel
from ..world.edges import BindMethod, ProvisionedEdge
from ..world.entities import ActivityTier, CategoryCluster, Device
from ..behavior.amount import AmountModel
from ..behavior.loyalty import LoyaltyModel, archetype_weights, clusters_from_graph
from ..population.archetypes import build_profiles
from ..clock import MINUTES_PER_DAY, SECONDS_PER_MINUTE
from .negatives import NegativeInjector, Plan

# Which action produces each non-payment event.
_ACTION_FOR_EVENT = {
    EventType.DEVICE_BIND: ActionName.ADD_DEVICE_SELFSERVE,
    EventType.AUTH_RESET: ActionName.RESET_PASSWORD,
    EventType.SUPPORT_TICKET: ActionName.OPEN_TICKET,
    EventType.DISPUTE_FILED: ActionName.OPEN_TICKET,
}


@dataclass
class WarmStartReport:
    """What the history contained, against what it was meant to contain."""

    n_entities: int
    n_events: int
    events_per_entity: dict[str, float] = field(default_factory=dict)
    hard_negatives: dict[str, int] = field(default_factory=dict)
    dormant_share: float = 0.0
    cards_with_median: float = 0.0
    live: bool = False

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
        self._amounts = AmountModel(config.behavior.amount)
        # Shared with the event builder where it has one, so a holder is
        # judged against the same hours they were generated with. A second
        # model would draw different preferences and the feature would be
        # measuring the disagreement between two populations.
        # `or` would not do: a freshly built model is empty and therefore
        # falsy, so the builder's model would be discarded and the holders
        # judged against hours they were never generated with.
        shared = getattr(simulator.builder, "clocks", None)
        self._clocks = shared if shared is not None else HolderClockModel(
            config.behavior.circadian
        )
        by_cluster, popularity = clusters_from_graph(
            simulator.graph, config.population.merchants.popularity_exponent
        )
        self._loyalty = LoyaltyModel(config.behavior.loyalty, by_cluster, popularity)
        self._profiles = build_profiles(
            config.behavior.categories.mix,
            config.population.archetype_weights,
        )
        self._injector = NegativeInjector(
            negatives=config.behavior.hard_negatives,
            amount=config.behavior.amount,
            geo=config.population.geo,
            rng=self.rng,
            amounts=self._amounts,
            loyalty=self._loyalty,
        )
        # Devices created during the warm start are numbered above anything the
        # population builder issued, so a replacement handset cannot collide
        # with a device that already exists.
        self._next_device = 10_000_000

    def run(self, live: bool = False, card_fraction: float = 1.0) -> WarmStartReport:
        """Fill the world with benign traffic.

        The default run backdates history and flags it, so training can exclude
        the feature-poor burn-in. A `live` run generates the same benign
        behaviour forward from the current clock and leaves it unflagged: this
        is the ordinary traffic that continues through the observation window,
        and it is the negative class a detector trains against. The behavioural
        machinery is identical either way — only the placement in time and the
        warm-start flag differ.
        """
        graph = self.simulator.graph
        warm = self.config.warm_start
        self.simulator.builder.set_warm_start(not live)

        cards = self._eligible_cards()
        # A live sweep may run over a sample of the population rather than all
        # of it. The full sweep is right for building history once, but the live
        # phase repeats it at every refit purely to supply a negative class, and
        # at twelve thousand holders that dominated the run. A sample of the
        # same cards behaves the same way; there is simply less of it.
        if live and 0.0 < card_fraction < 1.0 and cards:
            keep = max(1, int(len(cards) * card_fraction))
            idx = self.rng.choice(len(cards), size=keep, replace=False)
            cards = [cards[i] for i in sorted(idx)]
        # Each card gets its own amount level before any of them transacts, so
        # a card's purchases look like that card's rather than the population's.
        # Its holder gets their own hours for the same reason.
        for card_id in cards:
            holder = graph.holders[graph.cards[card_id].holder_id]
            self._amounts.register(int(card_id), self.rng, holder.archetype)
            self._clocks.require(int(holder.holder_id), self.rng)
            # Its own category mix and its own regulars, tilted by the
            # archetype's preferences. Without this every card draws from the
            # population's mix and shops at a merchant nobody has ever used.
            self._loyalty.register(
                int(card_id),
                self.rng,
                archetype_weights(self._profiles, holder.archetype),
            )

        schedule = self._build_schedule(cards, warm.events_per_entity)

        if live:
            # Forward of the current clock, not backdated. `_build_schedule`
            # returns times ending near zero (the observation start); shifting
            # them past `now` places the same shaped traffic into the live
            # window without any of it moving the clock backwards.
            lookback = warm.lookback_days * MINUTES_PER_DAY
            now = self.simulator.clock.now
            schedule = [(ts + lookback + now, card_id) for ts, card_id in schedule]
        elif schedule:
            # History is backdated, so the clock has to start behind the
            # observation window. Leaving it at the origin makes every event land
            # at the same instant, since the clock refuses to move backwards, and
            # the resulting traffic looks like one enormous burst.
            self.simulator.clock.rewind_to(schedule[0][0])

        merchants = list(graph.merchants)
        liquid = [m for m in merchants if graph.merchants[m].is_high_liquidity] or merchants
        travel = [
            m for m in merchants if graph.merchants[m].category is CategoryCluster.TRAVEL
        ] or merchants

        # Plans are expanded into timed events and the whole lot sorted before
        # any of it runs.
        #
        # Executing plan by plan does not work once time of day carries a
        # habit. A plan spans real time - a trip covers days, a session an
        # afternoon - so running one to completion pushes the clock past the
        # slots that follow it, and the clock only moves forward. Every later
        # event was then dragged to whatever hour the clock had reached,
        # which cost most of the rhythm the schedule had just been given and
        # inflated the gaps between a card's transactions tenfold. Both were
        # invisible while hours were uniform.
        planned: list[tuple[int, int, CardId, object]] = []
        for ts, card_id in schedule:
            self._injector.for_card(int(card_id))
            plan = self._injector.plan(merchants, liquid, travel)
            if plan.bind_device:
                planned.append((ts, 0, card_id, ("bind_device", None)))
            for event_type, offset in plan.bindings:
                planned.append((ts + offset, 1, card_id, ("binding", event_type)))
            for auth in plan.auths:
                planned.append((ts + auth.offset_minutes, 2, card_id, ("auth", auth)))

        # Ties break towards the device binding, so a plan that creates a
        # device and then spends through it still does so in that order.
        planned.sort(key=lambda row: (row[0], row[1]))

        counts: dict[int, int] = {}
        for ts, _, card_id, (kind, payload) in planned:
            actor_id = self._actor_for(card_id)
            counts[card_id] = counts.get(card_id, 0) + self._emit(
                kind, payload, actor_id, card_id, ts
            )

        self.simulator.builder.set_warm_start(False)
        report = self._report(counts, cards)
        report.live = live
        return report

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

            clock = self._clocks.require(int(holder.holder_id), self.rng)
            state = self._process.new_state(self.rng, rate_multiplier=multiplier)
            elapsed = 0.0
            for _ in range(count):
                elapsed += self._process.next_gap_seconds(state, self.rng)
                minutes = int(elapsed / SECONDS_PER_MINUTE)
                # The gap process decides which day an event lands on; the
                # holder's own clock decides the time of day. Without the
                # second step every hour is equally likely, and the generated
                # marginal comes out four times flatter than the real one
                # while no holder has any hours of their own.
                #
                # The perturbation is bounded by a day against a gap median of
                # roughly three, and the arrival targets it could disturb are
                # checked against their noise floors rather than assumed safe.
                day = minutes // MINUTES_PER_DAY
                minutes = day * MINUTES_PER_DAY + clock.sample_minute_of_day(self.rng)
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

    def _emit(self, kind: str, payload, actor_id: ActorId, card_id: CardId, ts: int) -> int:
        """Run one already-scheduled event.

        Offsets were folded in when the plan was expanded, so this places
        exactly what it is given and returns how many scoreable events it
        produced.
        """
        if kind == "bind_device":
            self._advance_to(ts)
            self._bind_new_device(card_id, self.simulator.clock.now)
            return 0

        if kind == "binding":
            self._advance_to(ts)
            self.simulator.step(
                actor_id,
                Action(
                    name=_ACTION_FOR_EVENT.get(payload, ActionName.RESET_PASSWORD),
                    target_id=int(card_id),
                ),
            )
            return 1

        self._advance_to(ts)
        self.simulator.step(
            actor_id,
            Action(
                name=ActionName.ATTEMPT_AUTH,
                target_id=card_id,
                secondary_id=payload.merchant_id,
                amount=payload.amount,
                entry_mode=int(self.rng.integers(0, 4)),
            ),
        )
        return 1

    def _advance_to(self, ts: int) -> None:
        self.simulator.clock.advance_to(max(self.simulator.clock.now, ts))

    def _bind_new_device(self, card_id: CardId, ts: int) -> DeviceId:
        """A replaced handset, bound to this card.

        Given its own signature rather than an existing one, since a new phone
        is rarely the same configuration as the old. The binding itself is
        emitted as an event by the action that follows, so a detector sees the
        sequence rather than only its consequence.
        """
        graph = self.simulator.graph
        device_id = DeviceId(self._next_device)
        self._next_device += 1

        bucket_id = BucketId(int(self.rng.choice(list(graph.buckets))))
        holder = graph.holders[graph.cards[card_id].holder_id]
        graph.add_device(
            Device(
                device_id=device_id,
                bucket_id=bucket_id,
                first_seen_ts=ts,
                household_id=holder.household_id,
                os_code=int(self.rng.integers(0, 12)),
                browser_code=int(self.rng.integers(0, 6)),
                app_version=int(self.rng.integers(1, 40)),
                ip_asn=int(self.rng.integers(0, 5000)),
            )
        )
        graph.bind_device(
            ProvisionedEdge(
                card_id=card_id,
                device_id=device_id,
                bind_ts=ts,
                bind_method=BindMethod.SELF_SERVICE,
            )
        )
        return device_id

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
            hard_negatives=dict(self._injector.counts),
            dormant_share=float((values <= 2).mean()) if len(values) else 0.0,
            cards_with_median=with_median / max(len(cards), 1),
        )
