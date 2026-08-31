"""From a log of events to a matrix a model can fit.

The event schema is a set of dataclasses with optional history fields; a model
needs a dense float array with a fixed column order. This module is the bridge,
and it exists to make three decisions once rather than in every model.

**Identity is not a feature.** A card id, a device id, an ip are labels for
entities, not descriptions of behaviour. Left in the matrix a model memorises
which entities were fraudulent in the training window and reports that as skill.
They are dropped, and only quantities that describe *this* event survive.

**Absence is not zero.** A card with no history has no median to compare
against, and its `amount_vs_median` is None. Filling that with zero states that
the amount matched the median exactly, which is a claim the world never made.
Every nullable field is split into a neutral-filled value column and a companion
`*_missing` flag, so a model can learn from the absence itself rather than from a
fabricated value.

**The label is never read.** The matrix is built from `scoring_fields()`, which
already excludes `is_fraud` and `episode_id` structurally. The label is returned
as a separate vector, aligned by row, and the two never share a column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.schema import EventLog, EventType

# Columns that name an entity rather than describe behaviour. A model that reads
# these separates the entities it saw in training from the ones it did not,
# which is memorisation wearing the clothes of detection.
_IDENTITY_FIELDS = frozenset(
    {
        "event_id",
        "ts",
        "card_id",
        "merchant_id",
        "device_id",
        "ip_asn",
        "holder_id",
        "actor_id",
        "target_id",
    }
)

# Fields declared optional on the event. Each becomes a value column plus a
# `<name>_missing` flag; the value where absent is filled with the neutral
# constant below rather than zero.
_NULLABLE_FILL: dict[str, float] = {
    # AuthAttemptEvent
    "seconds_since_last_auth": 0.0,
    "within_usual_hours": 0.0,
    "amount_vs_median": 1.0,  # a ratio; 1.0 is "equal to the median"
    # BindingEvent
    "time_since_last_bind_days": 0.0,
    "hours_since_password_reset": 0.0,
    "hours_since_support_call": 0.0,
    "device_age_days": 0.0,
    "device_n_cards": 0.0,
}

MISSING_SUFFIX = "_missing"


@dataclass(frozen=True, slots=True)
class FeatureTable:
    """A dense design matrix plus everything needed to align and slice it.

    The label lives here beside the matrix but never inside it. `event_type`
    carries which expert each row belongs to, so a per-expert view is a mask
    away without re-reading the log.
    """

    X: np.ndarray
    y: np.ndarray  # -1 where a row is unlabelled (outside a closed episode)
    columns: tuple[str, ...]
    event_type: np.ndarray  # dtype object, one EventType per row
    is_warm_start: np.ndarray
    episode_id: np.ndarray  # -1 where none
    group: np.ndarray  # entity id per row, for a leak-free grouped split; never a feature
    events: np.ndarray  # the source event objects, for per-event scorers; never a feature

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def labelled_mask(self) -> np.ndarray:
        """Rows with a stamped label, the only ones a supervised fit may use."""
        return self.y >= 0

    def view(self, event_types: frozenset[EventType]) -> FeatureTable:
        """The rows for one expert's event types, columns unchanged.

        Kept as the full column set so a per-expert model can drop the columns
        that are constant for its types itself, rather than this deciding for
        it. The alternative — narrowing columns here — bakes a feature-selection
        choice into the extractor where it cannot be measured.
        """
        mask = np.array([et in event_types for et in self.event_type], dtype=bool)
        return FeatureTable(
            X=self.X[mask],
            y=self.y[mask],
            columns=self.columns,
            event_type=self.event_type[mask],
            is_warm_start=self.is_warm_start[mask],
            episode_id=self.episode_id[mask],
            group=self.group[mask],
            events=self.events[mask],
        )


def _numeric(value: object, fill: float) -> tuple[float, float]:
    """One field to (value, is_missing).

    Booleans map to 0/1, None to the neutral fill with the flag raised, numbers
    pass through. A field that is neither is not a feature and the caller is
    expected to have dropped it.
    """
    if value is None:
        return fill, 1.0
    if isinstance(value, bool):
        return float(value), 0.0
    return float(value), 0.0


def _row_fields(event: object) -> dict[str, object]:
    """The scoreable fields of an event, with identity and non-numerics removed.

    Reads `scoring_fields()` so the label exclusion is the schema's, not ours.
    """
    fields = event.scoring_fields()
    return {
        name: value
        for name, value in fields.items()
        if name not in _IDENTITY_FIELDS
        and not isinstance(value, str)
    }


def _schema(events: list[object]) -> list[str]:
    """The union of feature names across every event, in stable order.

    Different event types carry different fields, so the flat matrix is their
    union. The order is sorted rather than first-appearance, and the difference
    is not cosmetic.

    First-appearance order is deterministic given one log and unstable across
    two. Two windows holding exactly the same event types in a different
    sequence produce the same column set in a different order, and the retention
    buffer — which concatenates a window against every window before it — then
    refuses them as incompatible despite their being identical. That failure
    surfaced only when a change to the attacker altered which event happened
    first; it was latent before, waiting for any reordering at all. Sorting
    depends on the field names alone, so a column order is a property of the
    schema rather than of the traffic that happened to arrive.
    """
    seen: set[str] = set()
    for event in events:
        seen.update(_row_fields(event))
    return sorted(seen)


def _columns(base: list[str]) -> tuple[list[str], list[str]]:
    """Expand the base fields into the final column list.

    A nullable field contributes two columns, its value and its missing flag,
    with the flag placed immediately after so the pair reads together.
    """
    columns: list[str] = []
    for name in base:
        columns.append(name)
        if name in _NULLABLE_FILL:
            columns.append(name + MISSING_SUFFIX)
    return columns, base


def build_table(log: EventLog, exclude_warm_start: bool = True) -> FeatureTable:
    """Extract a design matrix from an event log.

    Warm-start rows are excluded by default: they are feature-poorer by
    construction, since the history they would read is what they are creating,
    and a model trained on them learns that difference rather than fraud.
    """
    events = log.scoreable() if exclude_warm_start else list(log.events)
    if not events:
        return FeatureTable(
            X=np.zeros((0, 0)),
            y=np.zeros(0),
            columns=(),
            event_type=np.array([], dtype=object),
            is_warm_start=np.zeros(0, dtype=bool),
            episode_id=np.zeros(0, dtype=np.int64),
            group=np.zeros(0, dtype=np.int64),
            events=np.array([], dtype=object),
        )

    base = _schema(events)
    columns, base_order = _columns(base)
    index = {name: i for i, name in enumerate(columns)}

    X = np.zeros((len(events), len(columns)), dtype=np.float64)
    y = np.full(len(events), -1.0)
    types = np.empty(len(events), dtype=object)
    warm = np.zeros(len(events), dtype=bool)
    episodes = np.full(len(events), -1, dtype=np.int64)
    groups = np.full(len(events), -1, dtype=np.int64)
    source = np.array(events, dtype=object)

    for row, event in enumerate(events):
        fields = _row_fields(event)
        for name in base_order:
            if name not in fields:
                # A field this event type does not carry. Treat it as missing
                # where it is nullable, otherwise leave the neutral zero — the
                # column is constant-zero for this type and a per-expert view
                # drops it.
                if name in _NULLABLE_FILL:
                    X[row, index[name]] = _NULLABLE_FILL[name]
                    X[row, index[name + MISSING_SUFFIX]] = 1.0
                continue
            value, missing = _numeric(fields[name], _NULLABLE_FILL.get(name, 0.0))
            X[row, index[name]] = value
            if name in _NULLABLE_FILL:
                X[row, index[name + MISSING_SUFFIX]] = missing

        types[row] = event.event_type
        warm[row] = getattr(event, "is_warm_start", False)
        # The entity a row belongs to, so a split can keep an entity wholly in
        # one side. A card for an auth, a holder for a binding — read from the
        # event directly, never placed in X.
        gid = getattr(event, "card_id", None)
        if gid is None:
            gid = getattr(event, "holder_id", -1)
        groups[row] = int(gid)
        label = getattr(event, "is_fraud", None)
        if label is not None:
            y[row] = 1.0 if label else 0.0
        ep = getattr(event, "episode_id", None)
        if ep is not None:
            episodes[row] = ep

    return FeatureTable(
        X=X,
        y=y,
        columns=tuple(columns),
        event_type=types,
        is_warm_start=warm,
        episode_id=episodes,
        group=groups,
        events=source,
    )
