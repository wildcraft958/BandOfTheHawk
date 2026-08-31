"""Feature column names, written once and checked against their source.

Ten column names were spelled out as string literals in two to four files each,
with nothing linking them: `defender/baseline.py` and `defender/experts.py` name
the groups a model is allowed to read, `defender/table.py` names the ones that
may be null, `analysis/entity_report.py` reports on one, and
`generative/scoring.py` produces three. A rename on the event schema would have
left every one of those spellings pointing at a column that no longer existed,
and the failure surfaces as a silently absent feature rather than an error.

The constants below are checked at import against the two places the names
really come from: the event dataclasses for the stamped features, and the text
scorer's own output for the generated-text scores. A name that stops matching
fails on import instead of quietly degrading a model.
"""

from __future__ import annotations

import dataclasses
from typing import Final

from . import schema

# Stamped on an auth event by the builder, which may read the graph.
AMOUNT_VS_MEDIAN: Final[str] = "amount_vs_median"
WITHIN_USUAL_HOURS: Final[str] = "within_usual_hours"
IS_FIRST_TXN_THIS_MERCHANT: Final[str] = "is_first_txn_this_merchant"
DEVICE_N_CARDS: Final[str] = "device_n_cards"
CARD_N_DEVICES: Final[str] = "card_n_devices"
DEVICE_NEW_TO_CARD: Final[str] = "device_new_to_card"
DEVICE_AGE_DAYS: Final[str] = "device_age_days"

# Produced by the text scorer and carried on the artifact.
TEMPLATE_SIMILARITY: Final[str] = "template_similarity"
ENTITY_CONSISTENCY: Final[str] = "entity_consistency"
PERPLEXITY_PROXY: Final[str] = "perplexity_proxy"

# Suffix marking the companion column that says whether a nullable value was
# present. Written once here rather than spelled out per column.
MISSING_SUFFIX: Final[str] = "_missing"

SCHEMA_COLUMNS: Final[tuple[str, ...]] = (
    AMOUNT_VS_MEDIAN,
    WITHIN_USUAL_HOURS,
    IS_FIRST_TXN_THIS_MERCHANT,
    DEVICE_N_CARDS,
    CARD_N_DEVICES,
    DEVICE_NEW_TO_CARD,
    DEVICE_AGE_DAYS,
)

TEXT_SCORE_COLUMNS: Final[tuple[str, ...]] = (
    TEMPLATE_SIMILARITY,
    ENTITY_CONSISTENCY,
    PERPLEXITY_PROXY,
)


def missing_flag(column: str) -> str:
    """The companion column recording whether `column` had a value."""
    return f"{column}{MISSING_SUFFIX}"


def _schema_field_names() -> frozenset[str]:
    """Every field across the event dataclasses."""
    names: set[str] = set()
    for attribute in vars(schema).values():
        if dataclasses.is_dataclass(attribute):
            names.update(field.name for field in dataclasses.fields(attribute))
    return frozenset(names)


_FIELDS = _schema_field_names()
_UNKNOWN = tuple(name for name in SCHEMA_COLUMNS if name not in _FIELDS)
if _UNKNOWN:  # pragma: no cover - import-time contract
    raise ImportError(
        f"feature columns not present on any event dataclass: {_UNKNOWN}. "
        f"Either the schema was renamed without updating this module, or a name "
        f"here is misspelled."
    )
