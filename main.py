"""Run the system: the whole pipeline, or one stage at a time.

TWO WAYS TO RUN
---------------

    python main.py                      # the full pipeline, every stage in order
    python main.py <stage>              # one stage on its own

The stages, in the order the full run executes them:

    demo       build the benign world and warm it; prints the fidelity numbers
               (rule trip rate, fan-out, events per entity) — the sanity check
               that the simulation still matches its calibration
    text       generate the text pool with Qwen and embed it; prints the tier
               ladder. Uses the real models unless --mock is passed. The pool is
               written to artifacts/text_pool.json and REUSED on every later run
               — an existing pool is never regenerated, so the models load once
               and never again. Force a fresh corpus with:
                   python -m fraudsim.generative.cli build --qwen --embed --rebuild
    fraud      add scripted fraud at the base rate; prints prevalence and the
               top scripted action sequences
    baseline   fit the flat gradient-boosted detector; prints PR-AUC, recall at
               fixed false-positive budgets, and the per-entity ablation (H.6)
    mixture    fit the five experts and the combiner; prints each against the
               flat baseline, the learned combiner weights, and the cost-curve
               bands
    coadapt    THE SOLUTION. Warm-start the defender, then the actor, then the
               critic, then run live: the reinforcement-learning attacker adapts
               continuously while the defender refits every K updates. Prints the
               live curve, the zero-shot recall, and the attack strategies the
               policy found. Also writes a JSON of the same numbers for plotting.

SCALE
-----

    --profile quick     a fast smoke of everything            (~10 min)
    --profile default   a real run                            (~1-2 h)
    --profile server    large, for a GPU box                  (hours)

Scale only changes sizes — population, training length — never which stages run
or what they print.

LOGGING
-------

Every stage is wrapped in a banner naming it and its arguments, timed, and
followed by a summary table at the end, so one redirected stream is a complete
readable record:

    nohup python main.py --profile server > run.log 2>&1 &     # linux
    tail -f run.log

Output flushes at each stage boundary, so `tail -f` follows it live.

WHAT A RUN LEAVES BEHIND
------------------------

    artifacts/text_pool.json          the generated corpus and its embeddings,
                                      reused by every later run
    artifacts/coadapt_metrics.json    the live curve, the defender refit points,
                                      the zero-shot recalls, and the attacker's
                                      top action sequences, as data for plotting
    artifacts/checkpoints/            the trained attacker (actor + critic) and
                                      the final refitted defender, both loadable
"""

from __future__ import annotations

import argparse
import importlib
import time
import traceback
from datetime import datetime

PROFILES = ("quick", "default", "server")

# The pipeline, in order. Each entry is (stage name, module, argument builder).
# The module is the stage's own command-line entry point, so running a stage here
# and running it directly are the same code path.
STAGE_ORDER = ("demo", "text", "fraud", "baseline", "mixture", "coadapt")


def _scales(profile: str) -> dict:
    """Sizes per profile. Only sizes — never which stages run or what they print.

    `server` is set for a large-memory GPU box: a population big enough that the
    text experts and the per-entity statistics have real sample sizes, and a live
    phase long enough for the attacker and defender to trade several rounds.
    """
    return {
        "quick": dict(
            holders=600, fraud_rate=0.06, per_key=25,
            demo_episodes=40, bc_epochs=6, critic_rollouts=16, critic_epochs=8,
            updates=12, episodes_per_update=12, refit_every=4,
            hidden=128, minibatch=128, embed_dim=256,
            label_latency=2880, fraud_rounds=3,
        ),
        "default": dict(
            holders=3000, fraud_rate=0.02, per_key=150,
            demo_episodes=300, bc_epochs=10, critic_rollouts=48, critic_epochs=20,
            updates=60, episodes_per_update=48, refit_every=10,
            hidden=256, minibatch=256, embed_dim=256,
            label_latency=4320, fraud_rounds=4,
        ),
        "server": dict(
            holders=12000, fraud_rate=0.01, per_key=500,
            demo_episodes=800, bc_epochs=15, critic_rollouts=128, critic_epochs=40,
            updates=150, episodes_per_update=80, refit_every=12,
            hidden=512, minibatch=512, embed_dim=256,
            label_latency=4320, fraud_rounds=4,
        ),
    }[profile]


def _stage_args(stage: str, s: dict, use_models: bool) -> tuple[str, list[str]]:
    """The module and argument list for one stage at one scale."""
    holders = str(s["holders"])
    fraud_rate = str(s["fraud_rate"])

    if stage == "demo":
        return "fraudsim.engine.cli", ["demo", "--holders", holders]

    if stage == "text":
        args = ["build", "--per-key", str(s["per_key"])]
        if use_models:
            # The real generation and embedding models, which is the default.
            # The build skips itself if a pool already exists, so this loads
            # nothing on a run where the corpus is already there.
            args += ["--qwen", "--embed", "--embed-dim", str(s["embed_dim"])]
        return "fraudsim.generative.cli", args

    if stage == "fraud":
        return "fraudsim.orchestration.cli", ["run", "--holders", holders, "--train-only"]

    if stage == "baseline":
        return "fraudsim.defender.cli", [
            "baseline", "--holders", holders, "--fraud-rate", fraud_rate,
        ]

    if stage == "mixture":
        return "fraudsim.defender.cli", [
            "mixture", "--holders", holders, "--fraud-rate", fraud_rate,
        ]

    if stage == "coadapt":
        return "fraudsim.orchestration.cli", [
            "coadapt", "--holders", holders, "--fraud-rate", fraud_rate,
            "--learned",
            "--demo-episodes", str(s["demo_episodes"]),
            "--bc-epochs", str(s["bc_epochs"]),
            "--critic-rollouts", str(s["critic_rollouts"]),
            "--critic-epochs", str(s["critic_epochs"]),
            "--updates", str(s["updates"]),
            "--episodes-per-update", str(s["episodes_per_update"]),
            "--refit-every", str(s["refit_every"]),
            "--hidden", str(s["hidden"]),
            "--minibatch", str(s["minibatch"]),
            # The defender's handicaps: labels arrive late, and fraud examples
            # age out. Without them the detector sees every new tactic instantly
            # and keeps it forever, and there is no contest to observe.
            "--label-latency", str(s["label_latency"]),
            "--fraud-rounds", str(s["fraud_rounds"]),
        ]

    raise ValueError(f"unknown stage {stage!r}")


def _run_stage(name: str, module: str, argv: list[str]) -> tuple[bool, float]:
    """Run one stage, banner and timing around it.

    A stage that raises is reported and the run continues, so one failure does
    not cost the output of every stage after it. The exit code reflects whether
    anything failed.
    """
    bar = "=" * 78
    print(f"\n{bar}")
    print(f"  STAGE: {name}")
    print(f"  {module} {' '.join(argv)}")
    print(f"  start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(bar, flush=True)

    started = time.perf_counter()
    try:
        ok = importlib.import_module(module).main(argv) == 0
    except Exception:  # noqa: BLE001 — one stage must not sink the run
        traceback.print_exc()
        ok = False
    elapsed = time.perf_counter() - started
    print(f"\n  {name}: {'ok' if ok else 'FAILED'} in {elapsed:.1f}s", flush=True)
    return ok, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="run the whole pipeline, or one stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py                        the full pipeline\n"
            "  python main.py coadapt                only the live co-adaptation\n"
            "  python main.py --profile server       the full pipeline, GPU-box scale\n"
            "  python main.py text                   build the pool with Qwen + embeddings\n"
            "  python main.py --mock                 the pipeline with no models (no GPU)\n"
        ),
    )
    parser.add_argument(
        "stage", nargs="?", choices=STAGE_ORDER, default=None,
        help="a single stage to run; omit to run the whole pipeline in order",
    )
    parser.add_argument("--profile", choices=PROFILES, default="default")
    parser.add_argument(
        "--mock", action="store_true",
        help="build the text pool with the deterministic stand-in instead of the "
             "real models — for a machine without a GPU, or a quick check. The "
             "real Qwen generation and embeddings are the default.",
    )
    args = parser.parse_args(argv)

    scales = _scales(args.profile)
    stages = [args.stage] if args.stage else list(STAGE_ORDER)

    bar = "=" * 78
    print(bar)
    print("  fraudsim")
    print(f"  profile: {args.profile}    population: {scales['holders']:,}")
    print(f"  models:  {'deterministic stand-ins' if args.mock else 'real Qwen + embeddings'}")
    print(f"  stages:  {' -> '.join(stages)}")
    print(f"  started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(bar, flush=True)

    results = []
    for stage in stages:
        module, stage_argv = _stage_args(stage, scales, use_models=not args.mock)
        ok, elapsed = _run_stage(stage, module, stage_argv)
        results.append((stage, ok, elapsed))

    print(f"\n{bar}")
    print("  SUMMARY")
    print(bar)
    total = 0.0
    for name, ok, elapsed in results:
        total += elapsed
        print(f"  {name:<12}{'ok' if ok else 'FAILED':<8}{elapsed:>9.1f}s")
    print(f"  {'total':<12}{'':<8}{total:>9.1f}s")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\n  failed: {', '.join(failed)}")
    print(f"  finished: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
