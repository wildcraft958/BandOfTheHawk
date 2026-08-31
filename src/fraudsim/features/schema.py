"""Event records.

An event is what the world reports about an action. It carries no indication of
who caused it: the same builder produces the row whether an ordinary holder or
an attacker acted, and any field distinguishing them would be a shortcut a
detector could learn instead of learning behaviour.

Fields that need history are optional. A card with no past has no median to
compare against, and reporting zero would be a claim rather than an absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class EventType(Enum):
    AUTH_ATTEMPT = "auth_attempt"
    DEVICE_BIND = "device_bind"
    PAYEE_ADD = "payee_add"
    AUTH_RESET = "auth_reset"
    SUPPORT_TICKET = "support_ticket"
    TRANSFER = "transfer"
    LIMIT_CHANGE = "limit_change"
    CASHOUT = "cashout"
    DISPUTE_FILED = "dispute_filed"
    REFUND_REQUEST = "refund_request"
    KYC_SUBMIT = "kyc_submit"
    IVR_CALL = "ivr_call"
    SIM_CHANGE = "sim_change"
    THREEDS_RESULT = "threeds_result"


class EntryMode(Enum):
    CHIP = 0
    CONTACTLESS = 1
    CARD_NOT_PRESENT = 2
    TOKEN = 3


@dataclass(slots=True)
class AuthAttemptEvent:
    """An authorisation, as the world reports it."""

    event_id: int
    ts: int
    card_id: int
    merchant_id: int
    device_id: int

    amount: float
    category_cluster: int
    entry_mode: int
    merchant_risk_tier: int
    is_high_liquidity: bool

    device_age_days: int
    device_new_to_card: bool
    device_n_cards: int
    card_n_devices: int
    ip_asn: int
    geo_distance_km: float

    auths_last_60s: int
    auths_last_1h: int
    auths_last_24h: int
    distinct_categories_1h: int
    distinct_merchants_24h: int
    distinct_ips_24h: int
    amount_sum_24h: float
    declines_last_1h: int
    seconds_since_last_auth: int | None

    is_first_txn_this_merchant: bool
    hour_of_day: int
    is_weekend: bool
    within_usual_hours: bool | None

    amount_vs_median: float | None
    account_age_days: int
    holder_tenure_days: int

    compound_features: tuple[float, ...] = ()
    compound_feature_names: tuple[str, ...] = ()

    # Stamped after an episode closes, never present at scoring time.
    is_fraud: bool | None = None
    episode_id: int | None = None
    is_warm_start: bool = False

    @property
    def event_type(self) -> EventType:
        return EventType.AUTH_ATTEMPT

    def scoring_fields(self) -> dict[str, object]:
        """Everything a scorer may see.

        The label and episode are excluded structurally rather than by
        convention, so a scorer cannot read them even by accident.
        """
        payload = {
            name: getattr(self, name)
            for name in self.__slots__
            if name
            not in {
                "is_fraud",
                "episode_id",
                "compound_features",
                "compound_feature_names",
            }
        }
        payload.update(zip(self.compound_feature_names, self.compound_features, strict=False))
        return payload


@dataclass(slots=True)
class BindingEvent:
    """A device binding, payee addition, or credential reset.

    These share a shape because what matters is the same in each: how new the
    target is, how often it has happened lately, and what preceded it. The
    preceded-by fields are the whole point. A reset, a call, and a binding are
    unremarkable alone, and it is the sequence that carries the signal.
    """

    event_id: int
    ts: int
    event_type_value: str
    actor_id: int
    target_id: int
    holder_id: int

    method: int
    channel: int
    time_since_account_open_days: int
    time_since_last_bind_days: int | None
    n_binds_30d: int
    hour_of_day: int

    hours_since_password_reset: float | None
    hours_since_support_call: float | None
    recovery_chain_within_1h: bool

    device_age_days: int | None = None
    device_n_cards: int | None = None

    # Text an action presented to a control, as an embedding plus the scalar
    # scores. Only the text verticals (dispute, ticket, refund) carry these; on
    # every other binding event they are empty, and the table treats an empty
    # embedding as a missing block. This is where the generative layer reaches
    # the text expert.
    text_embedding: tuple[float, ...] = ()
    text_scores: tuple[float, ...] = ()
    text_score_names: tuple[str, ...] = ()

    is_fraud: bool | None = None
    episode_id: int | None = None
    is_warm_start: bool = False

    @property
    def event_type(self) -> EventType:
        return EventType(self.event_type_value)

    def scoring_fields(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__slots__
            if name
            not in {
                "is_fraud",
                "episode_id",
                "event_type_value",
                "text_embedding",
                "text_scores",
                "text_score_names",
            }
        }
        # Text vectors expand into named columns the text expert reads, the same
        # way the compound window features do on an authorisation.
        payload.update({f"emb_{i}": v for i, v in enumerate(self.text_embedding)})
        payload.update(zip(self.text_score_names, self.text_scores, strict=False))
        return payload


class Labellable(Protocol):
    """What the log requires of an event: a fraud label it can stamp.

    The log was typed `list[object]`, which meant every read of `.is_fraud` was
    unchecked. Only the label is needed here, so only the label is demanded.
    """

    is_fraud: bool | None


@dataclass(slots=True)
class EventLog:
    """Append-only record of everything that happened.

    Labels are stamped after an episode closes rather than as events are
    written, because at the moment of scoring nothing knows the answer.
    """

    events: list[Labellable] = field(default_factory=list)
    _by_episode: dict[int, list[int]] = field(default_factory=dict)

    def append(self, event: Labellable) -> int:
        index = len(self.events)
        self.events.append(event)
        episode = getattr(event, "episode_id", None)
        if episode is not None:
            self._by_episode.setdefault(episode, []).append(index)
        return index

    def stamp_episode(self, episode_id: int, is_fraud: bool) -> int:
        """Label every event from one episode, once its outcome is known."""
        indices = self._by_episode.get(episode_id, [])
        for index in indices:
            self.events[index].is_fraud = is_fraud
        return len(indices)

    def stamp_unlabelled_benign(self) -> int:
        """Label everything still unlabelled as benign.

        Only adversarial episodes stamp fraud; the benign backdrop is never
        wrapped in an episode, so its events stay unlabelled. In this simulator
        an event that belonged to no adversarial episode is benign ground truth,
        which is exactly the negative class a detector trains against. Called
        once collection is complete, never mid-run -- a benign label is only true
        once no fraud episode can still claim the event.
        """
        stamped = 0
        for event in self.events:
            if getattr(event, "is_fraud", None) is None:
                event.is_fraud = False
                stamped += 1
        return stamped

    def labelled(self) -> list[object]:
        return [e for e in self.events if getattr(e, "is_fraud", None) is not None]

    def scoreable(self) -> list[object]:
        """Events outside the warm start.

        Warm-start rows are systematically feature-poorer, since the history
        they would draw on is exactly what they are creating. Training on them
        would learn that difference.
        """
        return [e for e in self.events if not getattr(e, "is_warm_start", False)]

    def truncate(self, n: int) -> None:
        """Roll back to the first *n* events, undoing later appends."""
        for event in self.events[n:]:
            episode = getattr(event, "episode_id", None)
            if episode is not None and episode in self._by_episode:
                self._by_episode[episode] = [
                    i for i in self._by_episode[episode] if i < n
                ]
                if not self._by_episode[episode]:
                    del self._by_episode[episode]
        del self.events[n:]

    def __len__(self) -> int:
        return len(self.events)
