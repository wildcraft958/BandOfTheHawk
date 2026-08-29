"""Text-pool entry point.

    python -m fraudsim.generative.cli build --out artifacts/text_pool.json

Builds the pool with the mock generator by default — no model, runs anywhere.
Pass --qwen to build the real corpus on a machine that can hold the model; that
is the only path that loads one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cfpb import load_reference
from .pool import build_pool
from .scoring import TextScorer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "text_pool.json"
DEFAULT_CFPB = ROOT / "Dataset" / "complaints" / "cfpb_payments_all.parquet"


def _tier_report(pool, scorer) -> str:
    """Similarity and consistency by tier, to show the ladder is monotone."""
    from collections import defaultdict

    sim = defaultdict(list)
    con = defaultdict(list)
    for e in pool.entries:
        sim[e.tier].append(scorer.template_similarity(e.text))
        con[e.tier].append(scorer.entity_consistency(e.text, e.facts))
    lines = ["  tier   template_sim   entity_consistency"]
    for tier in sorted(sim):
        s = sum(sim[tier]) / len(sim[tier])
        c = sum(con[tier]) / len(con[tier])
        lines.append(f"    {tier}        {s:>8.3f}           {c:>8.3f}")
    return "\n".join(lines)


def cmd_build(args: argparse.Namespace) -> int:
    generator = None
    if args.qwen:
        from .pool import QwenGenerator

        # The only path that loads a model. Reached only on an explicit flag.
        generator = QwenGenerator()

    pool = build_pool(generator=generator, per_key=args.per_key, seed=args.seed)
    pool.save(args.out)

    reference: list[str] = []
    if args.cfpb.exists():
        reference = load_reference(args.cfpb, limit=args.reference_limit, seed=args.seed)
    scorer = TextScorer(reference=reference or [e.text for e in pool.entries])

    print(f"text pool  ({pool.generator_name})")
    print(f"  entries          {len(pool.entries):>8,}")
    print(f"  fingerprint      {pool.fingerprint[:16]}")
    print(f"  reference        {len(reference):>8,} CFPB narratives")
    print()
    print(_tier_report(pool, scorer))
    print(f"\n  written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.generative")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="generate and score the text pool")
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build.add_argument("--cfpb", type=Path, default=DEFAULT_CFPB)
    build.add_argument("--per-key", type=int, default=8)
    build.add_argument("--reference-limit", type=int, default=2000)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument(
        "--qwen",
        action="store_true",
        help="use the real model (loads Qwen; needs the generative extra and a capable machine)",
    )
    build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
