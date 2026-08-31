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

import glob
import json
import os
import statistics as st

import numpy as np

REFIT_AT = 6  # first defender refit, at --refit-every 6


def load(arm):
    out = {}
    for path in sorted(glob.glob(f"artifacts/ablation/{arm}_s*.json")):
        seed = int(os.path.basename(path).split("_s")[1].split(".")[0])
        out[seed] = json.load(open(path))
    return out


def summarise(d):
    s = d["attacker_success"]
    return {
        "pre": st.mean(s[:REFIT_AT]),
        "post": st.mean(s[REFIT_AT:]),
        "post_median": st.median(s[REFIT_AT:]),
        "post_max": max(s[REFIT_AT:]),
        "tail": st.mean(s[-6:]),
    }


def bootstrap_paired(diffs, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    return np.percentile(means, [2.5, 97.5])


def main():
    A, B = load("stealth"), load("control")
    seeds = sorted(set(A) & set(B))
    if not seeds:
        print("no completed pairs yet")
        return

    print(f"paired on {len(seeds)} seed(s): {seeds}\n")
    print(f"{'seed':>5}{'stealth post':>14}{'control post':>14}{'diff':>10}")
    diffs = []
    for s in seeds:
        a, b = summarise(A[s]), summarise(B[s])
        diff = a["post"] - b["post"]
        diffs.append(diff)
        print(f"{s:>5}{a['post']:>14.1f}{b['post']:>14.1f}{diff:>+10.1f}")

    print(f"\n{'':>5}{'mean':>14}{'mean':>14}{'mean diff':>10}")
    print(
        f"{'':>5}"
        f"{st.mean([summarise(A[s])['post'] for s in seeds]):>14.1f}"
        f"{st.mean([summarise(B[s])['post'] for s in seeds]):>14.1f}"
        f"{st.mean(diffs):>+10.1f}"
    )

    if len(seeds) >= 3:
        lo, hi = bootstrap_paired(diffs)
        print(f"\n  95% bootstrap CI on the paired difference: [{lo:+.1f}, {hi:+.1f}]")
        verdict = (
            "stealth helps"
            if lo > 0
            else "stealth hurts"
            if hi < 0
            else "no detectable difference"
        )
        print(f"  verdict: {verdict}")
        if lo <= 0 <= hi:
            print("  (the interval spans zero: this design cannot separate the arms)")
    else:
        print("\n  too few seeds for an interval; run more before concluding")

    print("\n  recovery shape, stealth arm (mean extraction by phase)")
    for label, sl in (("pre-refit", slice(0, 6)), ("just after", slice(6, 12)),
                      ("mid", slice(12, 18)), ("late", slice(18, None))):
        vals = [st.mean(A[s]["attacker_success"][sl]) for s in seeds]
        print(f"    {label:<12}{st.mean(vals):>10.1f}")

    print("\n  final strategies")
    for arm, D in (("stealth", A), ("control", B)):
        for s in seeds:
            top = D[s].get("top_sequences") or []
            if top:
                print(f"    {arm:<8} s{s}: {top[0]['sequence'][:88]}")


if __name__ == "__main__":
    main()
