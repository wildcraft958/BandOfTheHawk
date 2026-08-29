"""The real-narrative reference corpus.

CFPB payment complaints are the human-written negative class and the reference
that template similarity is measured against. Only narratives dated 2022 or
earlier are used: 2023 onward carries the risk of partly machine-authored text,
and training a "human text" class on that would quietly undermine the very
result the text expert produces. The cut is stated, not silent.

This reads the parquet, so it lives in a tier where pandas is available and is
never imported by the simulation path.
"""

from __future__ import annotations

from pathlib import Path

# The contamination cut. 2023+ is a large share of the corpus and is held out.
MAX_YEAR = 2022


def load_reference(
    path: Path | str,
    limit: int | None = 2000,
    seed: int = 0,
    fraud_only: bool = False,
) -> list[str]:
    """Narratives up to the contamination cut, as plain strings.

    A subsample by default, since half a million narratives is far more than a
    similarity reference needs and the score is a max over them. Sampling is
    seeded so the reference is stable across runs.
    """
    import pandas as pd  # lazy; only this tier has pandas

    columns = ["narrative", "year", "is_fraud_issue"]
    df = pd.read_parquet(path, columns=columns)
    df = df[df["year"] <= MAX_YEAR]
    if fraud_only:
        df = df[df["is_fraud_issue"]]
    df = df[df["narrative"].notna() & (df["narrative"].str.len() > 20)]
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)
    return df["narrative"].tolist()
