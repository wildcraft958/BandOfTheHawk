"""Dataset access.

This is the only tier permitted to import pandas or pyarrow. Loaders read the
minimum set of columns and hand back plain frames; nothing here reaches into
the runtime package.

Two sources, with deliberately narrow roles:

    IEEE-CIS   the realism judge. Amount shape, inter-event timing, entity
               sequences, and device fan-out all come from here.
    Sparkov    taxonomy and demographics only. Its geo is an annulus around
               each customer, its per-category amounts are inverted, and its
               merchant popularity is nearly flat, so none of those are fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ..paths import DATASET_DIR

# card1 alone over-merges; pairing it with card2 and addr1 is the standard
# reconstruction of a cardholder in this dataset.
IEEE_ENTITY_COLUMNS = ("card1", "card2", "addr1")

# Fingerprint composite. This is a configuration signature shared by many
# unrelated devices, which is exactly why buckets exist as their own entity.
IEEE_FINGERPRINT_COLUMNS = ("DeviceInfo", "id_30", "id_31", "id_33")


class DatasetNotFound(FileNotFoundError):
    """Raised when a parquet file the calibration step needs is missing."""


@dataclass(frozen=True, slots=True)
class TransactionFrame:
    """Transactions with a reconstructed entity key."""

    data: pd.DataFrame
    entity_column: str
    time_column: str
    amount_column: str
    label_column: str

    def benign(self) -> pd.DataFrame:
        return self.data[self.data[self.label_column] == 0]

    def fraudulent(self) -> pd.DataFrame:
        return self.data[self.data[self.label_column] == 1]

    def entity_sizes(self, benign_only: bool = True) -> pd.Series:
        frame = self.benign() if benign_only else self.data
        return frame.groupby(self.entity_column, observed=True).size()

    def sequences(self, min_length: int, benign_only: bool = True) -> pd.DataFrame:
        """Entities with enough history to carry a timing signal, time-sorted."""
        frame = self.benign() if benign_only else self.data
        sizes = frame.groupby(self.entity_column, observed=True).size()
        keep = sizes[sizes >= min_length].index
        subset = frame[frame[self.entity_column].isin(keep)]
        return subset.sort_values([self.entity_column, self.time_column])


def _require(path: Path) -> Path:
    if not path.exists():
        raise DatasetNotFound(
            f"{path} is missing. Datasets are not committed; fetch them into Dataset/ first."
        )
    return path


class IeeeCisLoader:
    """Reads IEEE-CIS transactions and identity rows."""

    def __init__(self, root: Path | str = DATASET_DIR) -> None:
        self.root = Path(root) / "ieee-fraud-detection"

    def transactions(self, extra_columns: tuple[str, ...] = ()) -> TransactionFrame:
        columns = [
            "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
            *IEEE_ENTITY_COLUMNS, *extra_columns,
        ]
        path = _require(self.root / "train_transaction.parquet")
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
        frame["entity"] = _entity_key(frame, IEEE_ENTITY_COLUMNS)
        return TransactionFrame(
            data=frame,
            entity_column="entity",
            time_column="TransactionDT",
            amount_column="TransactionAmt",
            label_column="isFraud",
        )

    def identity(self) -> pd.DataFrame:
        path = _require(self.root / "train_identity.parquet")
        columns = ["TransactionID", "DeviceType", *IEEE_FINGERPRINT_COLUMNS]
        return pd.read_parquet(path, columns=columns)

    def fingerprint_to_entity(self) -> pd.DataFrame:
        """Fingerprint composite joined to entity keys and labels.

        Identity rows cover only about a fifth of transactions, so the fan-out
        measured here describes that subset rather than the whole population.
        """
        identity = self.identity()
        transactions = pd.read_parquet(
            _require(self.root / "train_transaction.parquet"),
            columns=["TransactionID", "isFraud", *IEEE_ENTITY_COLUMNS],
        )
        merged = transactions.merge(identity, on="TransactionID", how="inner")
        merged["entity"] = _entity_key(merged, IEEE_ENTITY_COLUMNS)
        merged["fingerprint"] = _entity_key(merged, IEEE_FINGERPRINT_COLUMNS)
        return merged

    def identity_coverage(self) -> float:
        total = pd.read_parquet(
            _require(self.root / "train_transaction.parquet"), columns=["TransactionID"]
        )
        return len(self.identity()) / len(total)


class SparkovLoader:
    """Reads Sparkov for taxonomy and demographics only."""

    CATEGORY_CLUSTERS: ClassVar[dict[str, str]] = {
        "grocery_pos": "grocery",
        "grocery_net": "grocery",
        "gas_transport": "fuel_transit",
        "food_dining": "dining",
        "shopping_pos": "retail",
        "home": "retail",
        "kids_pets": "retail",
        "shopping_net": "online",
        "misc_net": "online",
        "misc_pos": "retail",
        "entertainment": "entertainment",
        "health_fitness": "health",
        "personal_care": "health",
        "travel": "travel",
    }

    def __init__(self, root: Path | str = DATASET_DIR) -> None:
        self.root = Path(root) / "sparkov"

    def transactions(self) -> pd.DataFrame:
        columns = [
            "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
            "city_pop", "job", "dob", "is_fraud",
        ]
        frame = pd.read_parquet(_require(self.root / "fraudTrain.parquet"), columns=columns)
        frame["cluster"] = frame["category"].map(self.CATEGORY_CLUSTERS)
        # The _net suffix marks card-not-present, which survives the merge into
        # clusters and is worth keeping as its own flag.
        frame["is_card_not_present"] = frame["category"].str.endswith("_net")
        return frame

    def demographics(self) -> pd.DataFrame:
        frame = self.transactions()
        benign = frame[frame["is_fraud"] == 0]
        holders = benign.groupby("cc_num", observed=True).agg(
            city_pop=("city_pop", "first"),
            job=("job", "first"),
            dob=("dob", "first"),
        )
        birth = pd.to_datetime(holders["dob"], errors="coerce")
        holders["age_years"] = ((pd.Timestamp("2020-01-01") - birth).dt.days / 365.25).round()
        return holders.dropna(subset=["age_years"])

    def cluster_mix(self) -> pd.Series:
        frame = self.transactions()
        benign = frame[frame["is_fraud"] == 0]
        return benign["cluster"].value_counts(normalize=True).sort_index()


def _entity_key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    parts = [frame[column].astype("string").fillna("na") for column in columns]
    key = parts[0]
    for part in parts[1:]:
        key = key + "_" + part
    return key
