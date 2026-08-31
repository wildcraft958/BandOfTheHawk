"""Regenerate the paper's five measured figures from the stored run files.

    python make_figures.py                    # -> artifacts/figures/
    python make_figures.py --out paper/

Reads exactly what `python -m fraudsim.orchestration.ablation` reads:
artifacts/ablation/{stealth,control}_s*.json, written by the paired runs.

    for s in 1 2 3 4; do
      python main.py coadapt --profile gpu --seed $s
      python main.py control --profile gpu --seed $s
    done

The paired statistics come from that module rather than being recomputed here,
so a figure and the number quoted beside it cannot drift apart. That is the
whole point: every figure traces to a run artifact rather than to a
transcription.

The three architecture diagrams in the paper (simulation, defender, attacker)
are drawn by hand and are not regenerated.

Needs the analysis extra:  pip install -e ".[analysis]"
"""

from __future__ import annotations

import argparse
import logging
import statistics as st
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a server; must precede the pyplot import
import matplotlib.pyplot as plt
import numpy as np

from fraudsim.logs import configure, emit
from fraudsim.orchestration.ablation import REFIT_AT, bootstrap_paired, load, summarise
from fraudsim.paths import ARTIFACT_DIR

ARMS = ("stealth", "control")
COLOURS = {"stealth": "#6E63FF", "control": "#FB1E39"}


def _refit_marks(metrics: dict[str, Any]) -> list[int]:
    return [int(u) for u in metrics.get("defender_refits", [])]


def fig_armsrace(runs: dict[str, dict[int, Any]], seeds: list[int], out: Path) -> None:
    """Extraction per update, one panel per seed, both arms, refits marked."""
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.0 * len(seeds), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, seed in zip(axes, seeds, strict=True):
        for arm in ARMS:
            curve = runs[arm][seed]["attacker_success"]
            ax.plot(range(1, len(curve) + 1), curve, color=COLOURS[arm], label=arm, lw=1.6)
        for update in _refit_marks(runs["stealth"][seed]):
            ax.axvline(update, color="0.6", ls="--", lw=0.9)
        ax.set_title(f"seed {seed}", fontsize=10)
        ax.set_xlabel("update")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("value extracted per episode")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig2_armsrace.png", dpi=200)
    plt.close(fig)


def fig_ablation(diffs: list[float], seeds: list[int], out: Path) -> None:
    """Paired difference per seed against the mean and its bootstrap interval."""
    lo, hi = bootstrap_paired(diffs)
    mean = st.mean(diffs)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhspan(lo, hi, color="#6E63FF", alpha=0.15, label="95% bootstrap interval")
    ax.axhline(mean, color="#6E63FF", ls="--", lw=1.6, label=f"mean {mean:+.0f}")
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.bar([str(s) for s in seeds], diffs,
           color=["#04B492" if d > 0 else "#FB1E39" for d in diffs], width=0.55)
    ax.set_xlabel("seed")
    ax.set_ylabel("stealth minus control")
    ax.set_title("Paired difference in post-refit extraction", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig3_ablation.png", dpi=200)
    plt.close(fig)


def _distinct_at_each_refit(metrics: dict[str, Any]) -> list[int]:
    return [len(snap.get("sequences", [])) for snap in metrics.get("strategy_history", [])]


def fig_diversity(runs: dict[str, dict[int, Any]], seeds: list[int], out: Path) -> None:
    """Distinct converged strategies at each refit, averaged over seeds."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for arm in ARMS:
        series = [_distinct_at_each_refit(runs[arm][s]) for s in seeds]
        width = min(len(s) for s in series)
        if width == 0:
            continue
        mean = [st.mean([s[i] for s in series]) for i in range(width)]
        ax.plot(range(1, width + 1), mean, marker="o", color=COLOURS[arm], label=arm, lw=1.6)
    ax.set_xlabel("refit")
    ax.set_ylabel("distinct converged strategies")
    ax.set_title("Strategy diversity, averaged over seeds", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig4_diversity.png", dpi=200)
    plt.close(fig)


def fig_friction(runs: dict[str, dict[int, Any]], seeds: list[int], out: Path) -> None:
    """Genuine authorisations refused, per refit, as a percentage."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    n_arms = len(ARMS)
    for offset, arm in enumerate(ARMS):
        series = [runs[arm][s].get("false_positive_rate", []) for s in seeds]
        width = min((len(s) for s in series), default=0)
        if width == 0:
            continue
        mean = [100.0 * st.mean([s[i] for s in series]) for i in range(width)]
        x = np.arange(1, width + 1) + (offset - (n_arms - 1) / 2) * 0.35
        ax.bar(x, mean, width=0.35, color=COLOURS[arm], label=arm)
    ax.set_xlabel("refit")
    ax.set_ylabel("genuine authorisations refused (%)")
    ax.set_title("Defender friction", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig5_friction.png", dpi=200)
    plt.close(fig)


def fig_bandit(runs: dict[str, dict[int, Any]], seeds: list[int], out: Path) -> None:
    """Victim-selection posterior: coefficients as differences from a reference."""
    weights = [runs["stealth"][s]["selection"].get("weights") for s in seeds]
    weights = [w for w in weights if w]
    if not weights:
        emit("  no selection weights recorded; skipping fig6_bandit")
        return
    width = min(len(w) for w in weights)
    arr = np.array([w[:width] for w in weights], dtype=float)
    mean, spread = arr.mean(axis=0), arr.std(axis=0)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    idx = np.arange(width)
    ax.bar(idx, mean, yerr=spread, capsize=3,
           color=["#04B492" if m >= 0 else "#FB1E39" for m in mean])
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"f{i}" for i in idx], fontsize=8)
    ax.set_xlabel("dump-knowable feature (differences from the reference level)")
    ax.set_ylabel("posterior mean")
    ax.set_title("Victim-selection posterior, mean and spread over seeds", fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig6_bandit.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regenerate the paper's measured figures")
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR / "figures",
                        help="directory to write the PNGs into")
    args = parser.parse_args(argv)
    configure()
    # matplotlib chats at INFO about categorical units on every bar chart.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    runs = {arm: load(arm) for arm in ARMS}
    seeds = sorted(set(runs["stealth"]) & set(runs["control"]))
    if not seeds:
        emit("no completed pairs in artifacts/ablation/; run the paired arms first:")
        emit("  for s in 1 2 3 4; do")
        emit("    python main.py coadapt --profile gpu --seed $s")
        emit("    python main.py control --profile gpu --seed $s")
        emit("  done")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    diffs = [summarise(runs["stealth"][s]).post - summarise(runs["control"][s]).post
             for s in seeds]

    fig_armsrace(runs, seeds, args.out)
    fig_ablation(diffs, seeds, args.out)
    fig_diversity(runs, seeds, args.out)
    fig_friction(runs, seeds, args.out)
    fig_bandit(runs, seeds, args.out)

    lo, hi = bootstrap_paired(diffs)
    emit(f"\n  paired on {len(seeds)} seed(s): {seeds}")
    emit(f"  first refit at update {REFIT_AT}")
    emit(f"  mean paired difference {st.mean(diffs):+.0f}, 95% interval [{lo:+.0f}, {hi:+.0f}]")
    emit(f"  figures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
