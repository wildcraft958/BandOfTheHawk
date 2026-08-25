"""Population construction.

Builds the entity graph in dependency order: households, then holders, then the
cards and accounts they own, then merchants, then the device layer that binds
cards to hardware.

The device layer runs last because it needs every card to exist first. Degrees
are drawn before cards are matched onto them, which is what keeps the sharing
structure out of reach of independent assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.simulation import SimulationConfig
from ..ids import BucketId, CardId, DeviceId, EntityKind, HolderId, IdMinter
from ..rng import RngHub
from ..world.edges import BindMethod, ProvisionedEdge
from ..world.entities import (
    Account,
    ActivityTier,
    Archetype,
    Card,
    Cardholder,
    CategoryCluster,
    Device,
    FingerprintBucket,
    KycLevel,
    Merchant,
    RiskTier,
)
from ..world.graph import EntityGraph
from .archetypes import ActivitySampler, ArchetypeSampler, build_profiles
from .fanout import (
    CardDeviceAssigner,
    FingerprintDegreeSampler,
    HouseholdDeviceSampler,
    summarise,
)

DAY_MINUTES = 24 * 60


@dataclass(frozen=True, slots=True)
class PopulationReport:
    """What was built, and how the device layer compares to its targets."""

    counts: dict[str, int]
    fanout: dict[str, float]
    device_fanout: dict[str, float]
    archetype_mix: dict[str, float]
    activity_mix: dict[str, float]

    def render(self) -> str:
        lines = ["population"]
        for key, value in self.counts.items():
            lines.append(f"  {key:<22}{value:>10,}")

        lines += ["", "  fingerprint fan-out    generated     target"]
        for key in ("mean", "share_shared", "p99", "variance_to_mean", "max"):
            generated = self.fanout.get(key, float("nan"))
            target = self.fanout.get(f"target_{key}", float("nan"))
            lines.append(f"    {key:<20}{generated:>10.3f}{target:>11.3f}")

        lines += ["", "  device fan-out, kept at household scale"]
        for key in ("mean", "share_shared", "max"):
            lines.append(f"    {key:<20}{self.device_fanout.get(key, float('nan')):>10.3f}")

        lines += ["", "  archetypes"]
        for name, share in sorted(self.archetype_mix.items()):
            lines.append(f"    {name:<20}{share:>10.3f}")

        lines += ["", "  activity tiers"]
        for name, share in sorted(self.activity_mix.items()):
            lines.append(f"    {name:<20}{share:>10.3f}")
        return "\n".join(lines)


class PopulationBuilder:
    """Builds a world from a resolved configuration."""

    def __init__(self, config: SimulationConfig, hub: RngHub | None = None) -> None:
        self.config = config
        self.hub = hub or RngHub(config.seed)
        self.minter = IdMinter()
        self.profiles = build_profiles(config.behavior.categories.mix)
        self._holder_archetype: dict[HolderId, Archetype] = {}
        self._holder_activity: dict[HolderId, ActivityTier] = {}
        # Selection weights over merchants, held here rather than on the graph:
        # the graph carries state, not the distributions used to generate it.
        self.merchant_popularity: np.ndarray | None = None

    def build(self) -> tuple[EntityGraph, PopulationReport]:
        graph = EntityGraph()
        self._add_holders(graph)
        self._add_cards_and_accounts(graph)
        self._add_merchants(graph)
        fanout, device_fanout = self._add_device_layer(graph)
        graph.check_invariants()
        return graph, self._report(graph, fanout, device_fanout)

    # ------------------------------------------------------------- holders

    def _add_holders(self, graph: EntityGraph) -> None:
        population = self.config.population
        rng = self.hub.stream("holders")
        n = population.n_holders

        archetypes = ArchetypeSampler(population.archetype_weights).sample(n, rng)
        activity = ActivitySampler(
            population.activity.tier_weights, population.activity.tier_rate_multipliers
        ).sample(n, rng)

        # Households are what make device sharing organic rather than arbitrary.
        n_households = max(1, int(n / max(population.households.mean_size, 1.0)))
        households = rng.integers(0, n_households, size=n)

        # Age from the taxonomy source's demographic spread.
        ages = np.clip(rng.normal(47.0, 17.0, n), 18, 95).astype(int)
        city_pop = np.clip(rng.lognormal(9.5, 1.6, n), 50, 3_000_000).astype(int)
        tenure = np.clip(
            rng.lognormal(
                np.log(self.config.warm_start.tenure_days_median),
                self.config.warm_start.tenure_days_spread,
                n,
            ),
            1,
            8000,
        ).astype(int)

        latitudes = rng.uniform(25.0, 49.0, n)
        longitudes = rng.uniform(-124.0, -67.0, n)

        for index in range(n):
            holder_id = HolderId(self.minter.mint(EntityKind.HOLDER))
            graph.add_holder(
                Cardholder(
                    holder_id=holder_id,
                    home_lat=float(latitudes[index]),
                    home_lon=float(longitudes[index]),
                    city_pop=int(city_pop[index]),
                    age_years=int(ages[index]),
                    job_code=int(rng.integers(0, 475)),
                    tenure_days=int(tenure[index]),
                    archetype=archetypes[index],
                    activity_tier=activity[index],
                    household_id=int(households[index]),
                )
            )
            self._holder_archetype[holder_id] = archetypes[index]
            self._holder_activity[holder_id] = activity[index]

    # ------------------------------------------------- cards and accounts

    def _add_cards_and_accounts(self, graph: EntityGraph) -> None:
        population = self.config.population
        rng = self.hub.stream("cards")
        holder_ids = list(graph.holders)

        card_counts = rng.poisson(
            max(population.cards_per_holder_mean - 1.0, 0.0), len(holder_ids)
        ) + 1
        account_counts = rng.poisson(
            max(population.accounts_per_holder_mean - 1.0, 0.0), len(holder_ids)
        ) + 1

        for position, holder_id in enumerate(holder_ids):
            holder = graph.holders[holder_id]
            issued_ts = -int(holder.tenure_days) * DAY_MINUTES

            for _ in range(int(card_counts[position])):
                graph.add_card(
                    Card(
                        card_id=CardId(self.minter.mint(EntityKind.CARD)),
                        holder_id=holder_id,
                        issued_ts=issued_ts,
                        credit_line=float(np.clip(rng.lognormal(8.6, 0.8), 300, 60_000)),
                        bin_tier=int(rng.integers(0, 4)),
                    )
                )

            for _ in range(int(account_counts[position])):
                graph.add_account(
                    Account(
                        account_id=self.minter.mint(EntityKind.ACCOUNT),
                        holder_id=holder_id,
                        opened_ts=issued_ts,
                        balance=float(np.clip(rng.lognormal(7.5, 1.3), 0, 500_000)),
                        kyc_level=KycLevel.FULL,
                    )
                )

    # ----------------------------------------------------------- merchants

    def _add_merchants(self, graph: EntityGraph) -> None:
        merchants = self.config.population.merchants
        rng = self.hub.stream("merchants")
        categories = self.config.behavior.categories

        clusters = list(CategoryCluster)
        weights = np.array([categories.mix.get(c.value, 0.0) for c in clusters], dtype=float)
        weights /= weights.sum()

        # Traffic concentrates on a few merchants. The exponent is swept: the
        # taxonomy source's merchant traffic is nearly flat and the judge
        # dataset carries no merchant entity at all.
        ranks = np.arange(1, merchants.count + 1)
        popularity = ranks.astype(float) ** (-merchants.popularity_exponent)
        popularity /= popularity.sum()
        self.merchant_popularity = popularity

        picks = rng.choice(len(clusters), size=merchants.count, p=weights)
        for rank in range(merchants.count):
            cluster = clusters[int(picks[rank])]
            cnp_share = categories.card_not_present_share.get(cluster.value, 0.2)
            risk_roll = rng.random()
            risk = (
                RiskTier.HIGH
                if risk_roll < merchants.high_risk_share
                else RiskTier.MEDIUM
                if risk_roll < merchants.high_risk_share * 3
                else RiskTier.LOW
            )
            graph.add_merchant(
                Merchant(
                    merchant_id=self.minter.mint(EntityKind.MERCHANT),
                    category=cluster,
                    avg_ticket=float(np.clip(rng.lognormal(4.0, 0.7), 2, 5000)),
                    chargeback_rate=float(
                        np.clip(rng.exponential(merchants.chargeback_rate_mean), 0, 0.2)
                    ),
                    risk_tier=risk,
                    is_high_liquidity=bool(rng.random() < merchants.high_liquidity_share),
                    is_card_not_present=bool(rng.random() < cnp_share),
                    popularity_rank=rank,
                )
            )

    # ------------------------------------------------------- device layer

    def _add_device_layer(
        self, graph: EntityGraph
    ) -> tuple[dict[str, float], dict[str, float]]:
        population = self.config.population
        rng = self.hub.stream("devices")

        for _ in range(population.resolved_fingerprint_count()):
            graph.add_bucket(
                FingerprintBucket(
                    bucket_id=BucketId(self.minter.mint(EntityKind.FINGERPRINT_BUCKET)),
                    os_code=int(rng.integers(0, 12)),
                    browser_code=int(rng.integers(0, 6)),
                    screen_code=int(rng.integers(0, 20)),
                    is_common_configuration=bool(rng.random() < 0.2),
                )
            )

        n_devices = max(1, int(len(graph.holders) * population.devices_per_holder_mean))
        card_ids = np.array(sorted(graph.cards), dtype=np.int64)
        card_households = np.array(
            [graph.holders[graph.cards[CardId(int(c))].holder_id].household_id for c in card_ids],
            dtype=np.int64,
        )

        # Degrees first, then cards matched onto them. Reversing this order is
        # what caps dispersion at one, whatever the marginal.
        degrees = HouseholdDeviceSampler(population.devices).sample(n_devices, rng)
        assignments = CardDeviceAssigner(rng).assign(degrees, card_ids, card_households)

        bucket_ids = np.array(sorted(graph.buckets), dtype=np.int64)
        fingerprint_degrees = FingerprintDegreeSampler(population.fanout).sample(
            len(bucket_ids), rng
        )
        # A signature's drawn degree is how many cards it should reach in total,
        # not a weight. Since each device already carries several cards, the
        # devices per signature is that degree divided by the cards per device;
        # treating the degree as a weight instead lets reach compound and
        # overshoots the target by an order of magnitude.
        cards_per_device = max(float(degrees.mean()), 1.0)
        devices_per_bucket = np.maximum(
            1, np.round(fingerprint_degrees / cards_per_device)
        ).astype(np.int64)

        device_buckets = np.repeat(bucket_ids, devices_per_bucket)
        if len(device_buckets) >= n_devices:
            device_buckets = rng.permutation(device_buckets)[:n_devices]
        else:
            filler = rng.choice(bucket_ids, size=n_devices - len(device_buckets))
            device_buckets = np.concatenate([device_buckets, filler])
            rng.shuffle(device_buckets)

        ages = np.clip(
            rng.lognormal(
                np.log(population.devices.age_days_median),
                population.devices.age_days_spread,
                n_devices,
            ),
            0,
            3000,
        )
        emulators = rng.random(n_devices) < population.devices.emulator_share

        for index in range(n_devices):
            cards = assignments[index]
            if len(cards) == 0:
                continue
            device_id = DeviceId(self.minter.mint(EntityKind.DEVICE))
            household = int(card_households[np.searchsorted(card_ids, cards[0])])
            graph.add_device(
                Device(
                    device_id=device_id,
                    bucket_id=BucketId(int(device_buckets[index])),
                    first_seen_ts=-int(ages[index]) * DAY_MINUTES,
                    household_id=household,
                    os_code=int(rng.integers(0, 12)),
                    browser_code=int(rng.integers(0, 6)),
                    app_version=int(rng.integers(1, 40)),
                    ip_asn=int(rng.integers(0, 5000)),
                    is_emulator=bool(emulators[index]),
                )
            )
            for card in cards.tolist():
                graph.bind_device(
                    ProvisionedEdge(
                        card_id=CardId(int(card)),
                        device_id=device_id,
                        bind_ts=-int(ages[index]) * DAY_MINUTES,
                        bind_method=BindMethod.SELF_SERVICE,
                    )
                )

        device_summary = summarise(np.asarray(graph.fanout_distribution(), dtype=float))
        bucket_summary = summarise(
            np.asarray(graph.bucket_fanout_distribution(), dtype=float)
        )
        return bucket_summary.as_dict(), device_summary.as_dict()

    # -------------------------------------------------------------- report

    def _report(
        self,
        graph: EntityGraph,
        fanout: dict[str, float],
        device_fanout: dict[str, float],
    ) -> PopulationReport:
        target = self.config.population.fanout
        fanout = dict(fanout)
        fanout.update(
            {
                "target_mean": target.target_mean,
                "target_share_shared": target.target_share_shared,
                "target_p99": target.target_p99,
                "target_variance_to_mean": target.target_variance_to_mean,
                "target_max": float(target.maximum),
            }
        )

        total = max(len(graph.holders), 1)
        archetype_mix: dict[str, float] = {}
        activity_mix: dict[str, float] = {}
        for holder in graph.holders.values():
            archetype_mix[holder.archetype.value] = (
                archetype_mix.get(holder.archetype.value, 0.0) + 1.0 / total
            )
            activity_mix[holder.activity_tier.value] = (
                activity_mix.get(holder.activity_tier.value, 0.0) + 1.0 / total
            )

        return PopulationReport(
            counts=graph.summary(),
            fanout=fanout,
            device_fanout=device_fanout,
            archetype_mix=archetype_mix,
            activity_mix=activity_mix,
        )
