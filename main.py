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
    baseline   STATIC BENCHMARK against the scripted red team. Fits the flat
               gradient-boosted detector; prints PR-AUC, recall at fixed
               false-positive budgets, and the per-entity ablation (H.6). A fixed
               adversary on fixed data, which is what makes the ablation
               interpretable -- in the live phase the defender refits under you
               and there is no fixed point to measure at.
    mixture    STATIC BENCHMARK, same data as baseline. Fits the five experts and
               the combiner and reports both against the flat tree, so the
               mixture-versus-flat ablation is a like-for-like comparison.

               Neither of these is a claim about the learned attacker. Their
               models are reported and discarded; the co-adaptation stage fits
               its own defender from scratch.
    coadapt    THE SOLUTION. Warm-start the defender, then the actor, then the
               critic, then run live: the reinforcement-learning attacker adapts
               continuously while the defender refits every K updates. Prints the
               live curve, the zero-shot recall, and the attack strategies the
               policy found. Also writes a JSON of the same numbers for plotting.
    control    THE SAME RUN WITH STEALTH DISABLED. Identical in every other
               respect, with the attacker's posture head pinned to the loud
               setting and its dump cut to one card -- which is exactly the
               attacker that existed before those were added. Not part of the
               default pipeline; run it when the question is whether stealth
               earned its place:

                   python main.py control --profile server

               Read it against coadapt's curve -- but read several seeds of
               each, not one. A single run is one sample from a heavy-tailed
               distribution: the first comparison showed a 1.5x gap that looked
               convincing and was not (Mann-Whitney p = 0.98, the whole gap
               carried by one spike). Use --seed on the underlying command to
               repeat it.

               MEASURED, four seeds each, paired: mean post-refit extraction
               33.5 (stealth) against 37.3 (control), a paired difference of
               -3.8 with a 95% bootstrap interval of [-15.0, +9.3]. The interval
               spans zero and the sign favours the control. The two arms
               collapse alike, retaining about 2% of pre-refit extraction, and
               neither recovers.

               So the finding is the defender's, and it is the stronger claim:
               the attacker had the capability and it did not save it. The
               aged posture is not decorative -- it moves median device age on a
               fraudulent auth from 18.5 days to 327, and on a fitted detector
               that alone takes the mean fraud score from 0.972 to 0.554,
               under the 0.80 decline threshold. The policy adopted it (two of
               four seeds converge on [aged]) and still lost. A detector refit
               on the traffic beats a single-feature evasion because it simply
               moves to the features that remain.

SCALE
-----

    --profile quick     a fast smoke of everything            (~10 min)
    --profile default   a real run                            (~1-2 h)
    --profile gpu       large, sized to finish in a few hours  (~4 h)
    --profile server    the full thing; 150 updates at 12,000
                        holders is ~40 h, so use it only when
                        that is genuinely affordable

Scale only changes sizes — population, training length — never which stages run
or what they print.

LOGGING
-------

Two streams, on purpose. Progress — the stage banners, per-stage timings, and
everything the stages themselves report — goes to stderr. The final summary table
goes to stdout. So a redirect keeps the record complete, and a redirect of stdout
alone keeps just the summary:

    nohup python main.py --profile server > run.log 2>&1 &     # the full record
    tail -f run.log

    python main.py --profile quick > summary.txt               # summary only

Raise or lower the progress detail with --log-level, or GAUNTLET_LOG_LEVEL.

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
from datetime import datetime

from fraudsim.logs import configure, emit, get_logger
from fraudsim.paths import ARTIFACT_DIR

_log = get_logger("fraudsim.pipeline")

PROFILES = ("quick", "default", "gpu", "server")

# The pipeline, in order. Each entry is (stage name, module, argument builder).
# The module is the stage's own command-line entry point, so running a stage here
# and running it directly are the same code path.
STAGE_ORDER = ("demo", "text", "fraud", "baseline", "mixture", "coadapt")

# Runnable on its own but never part of the full pipeline: it is the control arm
# of an ablation, and folding it into the default run would double every run's
# cost to answer a question that is only asked once.
EXTRA_STAGES = ("control",)


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
            label_latency=2880, fraud_rounds=3, target_prevalence=0.02,
        ),
        "default": dict(
            holders=3000, fraud_rate=0.02, per_key=150,
            demo_episodes=300, bc_epochs=10, critic_rollouts=48, critic_epochs=20,
            updates=60, episodes_per_update=48, refit_every=10,
            hidden=256, minibatch=256, embed_dim=256,
            label_latency=4320, fraud_rounds=4, target_prevalence=0.02,
        ),
        # Sized to finish inside a working day on the GPU box. The full "server"
        # profile is ~19 min per update at twelve thousand holders and eighty
        # episodes, which is forty hours for a hundred and fifty updates -- fine
        # as a number, useless as a schedule. Halving the population and the
        # episodes per update quarters the per-update cost; sixty updates at a
        # refit every six still gives ten refit points, which is more of the
        # arms-race curve than the full profile's twelve at a fraction of the
        # time.
        "gpu": dict(
            holders=6000, fraud_rate=0.02, per_key=300,
            demo_episodes=400, bc_epochs=12, critic_rollouts=64, critic_epochs=25,
            updates=60, episodes_per_update=32, refit_every=6,
            hidden=256, minibatch=256, embed_dim=256,
            label_latency=4320, fraud_rounds=4, target_prevalence=0.02,
        ),
        "server": dict(
            holders=12000, fraud_rate=0.01, per_key=500,
            demo_episodes=800, bc_epochs=15, critic_rollouts=128, critic_epochs=40,
            updates=150, episodes_per_update=80, refit_every=12,
            hidden=512, minibatch=512, embed_dim=256,
            label_latency=4320, fraud_rounds=4, target_prevalence=0.02,
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

    if stage in ("coadapt", "control"):
        # The control arm differs in exactly two flags. Sharing the builder is
        # deliberate: an ablation whose arms are assembled separately drifts
        # apart, and then the comparison is measuring the drift.
        # Appended, never prepended: these are subcommand options, and argparse
        # rejects them before the subcommand name.
        ablation = (
            ["--stealth-frozen", "--dump-size", "1",
             # A separate metrics file, or the second arm silently overwrites
             # the first and the ablation has nothing left to compare.
             "--metrics", str(ARTIFACT_DIR / "control_metrics.json")]
            if stage == "control"
            else []
        )
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
            # The share of fraud the defender is fitted at. Left to the loop it
            # was whatever the attacker happened to produce -- forty-two percent,
            # against a design that specifies half of one. A detector fitted at
            # that balance is solving a much easier problem, and the contest was
            # decided by the mixture rather than by either side.
            "--target-prevalence", str(s["target_prevalence"]),
        ] + ablation

    raise ValueError(f"unknown stage {stage!r}")


def _run_stage(name: str, module: str, argv: list[str]) -> tuple[bool, float]:
    """Run one stage, banner and timing around it.

    A stage that raises is reported and the run continues, so one failure does
    not cost the output of every stage after it. The exit code reflects whether
    anything failed.
    """
    bar = "=" * 78
    _log.info("%s", bar)
    _log.info("STAGE: %s", name)
    _log.info("%s %s", module, " ".join(argv))
    _log.info("%s", bar)

    started = time.perf_counter()
    try:
        ok = importlib.import_module(module).main(argv) == 0
    except Exception:  # noqa: BLE001 — one stage must not sink the run
        _log.exception("stage %s raised", name)
        ok = False
    elapsed = time.perf_counter() - started
    _log.info("%s: %s in %.1fs", name, "ok" if ok else "FAILED", elapsed)
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
        "stage", nargs="?", choices=STAGE_ORDER + EXTRA_STAGES, default=None,
        help="a single stage to run; omit to run the whole pipeline in order. "
             "'control' is the stealth ablation's control arm and is never part "
             "of the full pipeline",
    )
    parser.add_argument("--profile", choices=PROFILES, default="default")
    parser.add_argument(
        "--log-level", default=None,
        help="diagnostic verbosity on stderr (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="build the text pool with the deterministic stand-in instead of the "
             "real models — for a machine without a GPU, or a quick check. The "
             "real Qwen generation and embeddings are the default.",
    )
    args = parser.parse_args(argv)
    configure(level=args.log_level)

    scales = _scales(args.profile)
    stages = [args.stage] if args.stage else list(STAGE_ORDER)

    bar = "=" * 78
    _log.info("%s", bar)
    _log.info("fraudsim")
    _log.info("profile: %s    population: %s", args.profile, f"{scales['holders']:,}")
    _log.info("models:  %s",
              "deterministic stand-ins" if args.mock else "real Qwen + embeddings")
    _log.info("stages:  %s", " -> ".join(stages))
    _log.info("%s", bar)

    results = []
    for stage in stages:
        module, stage_argv = _stage_args(stage, scales, use_models=not args.mock)
        ok, elapsed = _run_stage(stage, module, stage_argv)
        results.append((stage, ok, elapsed))

    emit(f"\n{bar}")
    emit("  SUMMARY")
    emit(bar)
    total = 0.0
    for name, ok, elapsed in results:
        total += elapsed
        emit(f"  {name:<12}{'ok' if ok else 'FAILED':<8}{elapsed:>9.1f}s")
    emit(f"  {'total':<12}{'':<8}{total:>9.1f}s")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        emit(f"\n  failed: {', '.join(failed)}")
    emit(f"  finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
