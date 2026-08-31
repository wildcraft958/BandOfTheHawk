"""Read a multi-seed stealth ablation and say whether the posture head helped.

    python -m fraudsim.orchestration.ablation

Reads artifacts/ablation/{stealth,control}_s*.json, as written by running the
co-adaptation twice per seed -- once ordinarily, once with --stealth-frozen
--dump-size 1, which reproduces the attacker from before the posture head
existed.

Paired by seed, because the two arms share a world per seed and pairing removes
the between-world variance. Reported as an effect size with a bootstrap
interval rather than as a single number, because the first comparison of these
arms was a single run per arm showing a 1.5x gap that did not survive a rank
test (p = 0.98, the whole gap carried by one spike in eighteen points). One run
is one sample from a heavy-tailed distribution.

The measured answer, four seeds each: 33.5 against 37.3 mean post-refit
extraction, a paired difference of -3.8 with a 95% interval of [-15.0, +9.3].
The interval spans zero. The arms collapse alike.
"""

from __future__ import annotations

import json
import statistics as st
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..logs import emit
from ..paths import ABLATION_DIR

REFIT_AT = 6  # first defender refit, at --refit-every 6


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """One arm's extraction curve, split at the first defender refit.

    A five-key dict before, read by string at three call sites, so a renamed key
    would have failed in the middle of rendering rather than at import.
    """

    pre: float
    post: float
    post_median: float
    post_max: float
    tail: float


def load(arm: str) -> dict[int, dict[str, Any]]:
    """Every completed run of one arm, keyed by seed."""
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(ABLATION_DIR.glob(f"{arm}_s*.json")):
        seed = int(path.name.split("_s")[1].split(".")[0])
        out[seed] = json.loads(path.read_text())
    return out


def summarise(metrics: dict[str, Any]) -> ArmSummary:
    """One run's extraction, before and after the first refit."""
    curve = metrics["attacker_success"]
    return ArmSummary(
        pre=st.mean(curve[:REFIT_AT]),
        post=st.mean(curve[REFIT_AT:]),
        post_median=st.median(curve[REFIT_AT:]),
        post_max=max(curve[REFIT_AT:]),
        tail=st.mean(curve[-6:]),
    )


def bootstrap_paired(
    diffs: Sequence[float], n: int = 20_000, seed: int = 0
) -> np.ndarray:
    """A percentile interval on the paired difference, resampled by seed pair."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    return np.percentile(means, [2.5, 97.5])


def main() -> None:
    A, B = load("stealth"), load("control")
    seeds = sorted(set(A) & set(B))
    if not seeds:
        emit("no completed pairs yet")
        return

    emit(f"paired on {len(seeds)} seed(s): {seeds}\n")
    emit(f"{'seed':>5}{'stealth post':>14}{'control post':>14}{'diff':>10}")
    diffs = []
    for s in seeds:
        a, b = summarise(A[s]), summarise(B[s])
        diff = a.post - b.post
        diffs.append(diff)
        emit(f"{s:>5}{a.post:>14.1f}{b.post:>14.1f}{diff:>+10.1f}")

    emit(f"\n{'':>5}{'mean':>14}{'mean':>14}{'mean diff':>10}")
    emit(
        f"{'':>5}"
        f"{st.mean([summarise(A[s]).post for s in seeds]):>14.1f}"
        f"{st.mean([summarise(B[s]).post for s in seeds]):>14.1f}"
        f"{st.mean(diffs):>+10.1f}"
    )

    if len(seeds) >= 3:
        lo, hi = bootstrap_paired(diffs)
        emit(f"\n  95% bootstrap CI on the paired difference: [{lo:+.1f}, {hi:+.1f}]")
        verdict = (
            "stealth helps"
            if lo > 0
            else "stealth hurts"
            if hi < 0
            else "no detectable difference"
        )
        emit(f"  verdict: {verdict}")
        if lo <= 0 <= hi:
            emit("  (the interval spans zero: this design cannot separate the arms)")
    else:
        emit("\n  too few seeds for an interval; run more before concluding")

    emit("\n  recovery shape, stealth arm (mean extraction by phase)")
    for label, sl in (("pre-refit", slice(0, 6)), ("just after", slice(6, 12)),
                      ("mid", slice(12, 18)), ("late", slice(18, None))):
        vals = [st.mean(A[s]["attacker_success"][sl]) for s in seeds]
        emit(f"    {label:<12}{st.mean(vals):>10.1f}")

    emit("\n  final strategies")
    for arm, D in (("stealth", A), ("control", B)):
        for s in seeds:
            top = D[s].get("top_sequences") or []
            if top:
                emit(f"    {arm:<8} s{s}: {top[0]['sequence'][:88]}")


if __name__ == "__main__":
    main()
