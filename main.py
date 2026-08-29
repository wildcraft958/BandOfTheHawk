"""Run the whole system from one file.

    python main.py                 # the default profile, every stage
    python main.py --profile quick # a fast smoke of every stage
    python main.py --profile full  # a large, slow, publication-scale run
    python main.py --only coadapt  # a single stage
    python main.py --skip coadapt   # everything but the live co-adaptation

Launching it so it keeps running after you disconnect, and captures one log:

    # Linux / macOS / Git-Bash on a machine where nohup detaches:
    nohup python main.py --profile full > run.log 2>&1 &

    # Windows (PowerShell), the reliable detached form:
    Start-Process -NoNewWindow python "main.py --profile full" `
        -RedirectStandardOutput run.log -RedirectStandardError run.err

    # Windows (cmd):
    start /b python main.py --profile full > run.log 2>&1

Output is line-buffered at each stage boundary (flush=True), so `tail -f run.log`
follows the run stage by stage.

Each stage calls the same code its own command-line entry point calls — nothing
here reimplements a stage, it only sequences them and prints a banner around
each, so a single `nohup python main.py > run.log 2>&1 &` produces one readable
log of the entire pipeline.

The stages, in order:

    demo       build a benign world and warm it, show the fidelity numbers
    text_pool  generate the text pool with the mock generator (no model loaded)
    run        add scripted fraud at the base rate, print prevalence and sequences
    baseline   fit the flat GBDT, answer the open question H.6
    mixture    fit the experts and combiner, report against the baseline
    coadapt    warm-start the defender, actor and critic, then run live
               co-adaptation: the RL attacker and the defender improve against
               each other (needs the rl/defender extras; the heaviest stage)

A profile only sets default scales; every scale is still overridable by editing
the STAGES table below. The coadapt stage is heaviest and can be skipped with
--skip coadapt, though it is the solution the rest builds toward.
"""

from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime

# Each stage is (name, module-main, args-by-profile). The args are exactly what
# the stage's own CLI accepts, so the single file and the individual commands
# stay in lock step.
PROFILES = ("quick", "default", "full")


def _stages(profile: str) -> list[tuple[str, str, list[str]]]:
    # Scales per profile. quick is a smoke; default is a real but modest run;
    # full is publication-scale and slow.
    holders = {"quick": "600", "default": "3000", "full": "8000"}[profile]
    fraud_rate = {"quick": "0.06", "default": "0.02", "full": "0.01"}[profile]
    # Co-adaptation scales: the warm-start sizes and the live-phase length.
    coadapt = {
        "quick": ["--demo-episodes", "40", "--bc-epochs", "6", "--critic-rollouts", "16",
                  "--critic-epochs", "8", "--updates", "12", "--episodes-per-update", "12",
                  "--refit-every", "4", "--hidden", "128", "--minibatch", "128"],
        "default": ["--demo-episodes", "300", "--bc-epochs", "10", "--critic-rollouts", "48",
                    "--critic-epochs", "20", "--updates", "60", "--episodes-per-update", "48",
                    "--refit-every", "10"],
        "full": ["--demo-episodes", "600", "--bc-epochs", "12", "--critic-rollouts", "64",
                 "--critic-epochs", "30", "--updates", "120", "--episodes-per-update", "64",
                 "--refit-every", "12"],
    }[profile]

    return [
        ("demo", "fraudsim.engine.cli", ["demo", "--holders", holders]),
        ("text_pool", "fraudsim.generative.cli", ["build", "--per-key", "8"]),
        ("run", "fraudsim.orchestration.cli",
         ["run", "--holders", holders, "--train-only"]),
        ("baseline", "fraudsim.defender.cli",
         ["baseline", "--holders", holders, "--fraud-rate", fraud_rate]),
        ("mixture", "fraudsim.defender.cli",
         ["mixture", "--holders", holders, "--fraud-rate", fraud_rate]),
        ("coadapt", "fraudsim.orchestration.cli",
         ["coadapt", "--holders", holders, "--fraud-rate", fraud_rate, *coadapt]),
    ]


def _run_stage(name: str, module: str, argv: list[str]) -> tuple[bool, float]:
    """Call one stage's main, timing it and catching failures.

    A stage that raises does not abort the whole run: the failure is reported
    and the pipeline continues, so one slow or broken stage does not cost the
    output of every stage after it. The overall exit code reflects whether any
    stage failed.
    """
    import importlib

    banner = "=" * 78
    print(f"\n{banner}")
    print(f"  STAGE: {name}    ({module} {' '.join(argv)})")
    print(f"  start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(banner, flush=True)

    started = time.perf_counter()
    try:
        mod = importlib.import_module(module)
        code = mod.main(argv)
        ok = code == 0
    except Exception:  # noqa: BLE001 - a stage failure must not sink the run
        traceback.print_exc()
        ok = False
    elapsed = time.perf_counter() - started
    print(f"\n  {name}: {'ok' if ok else 'FAILED'} in {elapsed:.1f}s", flush=True)
    return ok, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the whole fraudsim pipeline")
    parser.add_argument("--profile", choices=PROFILES, default="default")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="run only these stages (by name)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=(),
        help="skip these stages (by name); coadapt is the heaviest",
    )
    args = parser.parse_args(argv)

    stages = _stages(args.profile)
    if args.only:
        wanted = set(args.only)
        stages = [s for s in stages if s[0] in wanted]
    if args.skip:
        skip = set(args.skip)
        stages = [s for s in stages if s[0] not in skip]

    print("=" * 78)
    print("  fraudsim — full pipeline")
    print(f"  profile: {args.profile}")
    print(f"  stages:  {', '.join(s[0] for s in stages)}")
    print(f"  started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78, flush=True)

    results: list[tuple[str, bool, float]] = []
    for name, module, stage_argv in stages:
        ok, elapsed = _run_stage(name, module, stage_argv)
        results.append((name, ok, elapsed))

    # A final summary, so the tail of the log says what happened at a glance.
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    total = 0.0
    for name, ok, elapsed in results:
        total += elapsed
        print(f"  {name:<12}{'ok' if ok else 'FAILED':<8}{elapsed:>8.1f}s")
    print(f"  {'total':<12}{'':<8}{total:>8.1f}s")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\n  failed stages: {', '.join(failed)}")
    print(f"  finished: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
