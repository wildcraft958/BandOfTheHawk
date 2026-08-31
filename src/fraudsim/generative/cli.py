"""Text-pool entry point.

    python -m fraudsim.generative.cli build --out artifacts/text_pool.json

An existing pool is reused rather than rebuilt: the corpus costs GPU hours to
generate and nothing downstream changes while it stays the same, so a build is
skipped unless --rebuild is passed. Pass --qwen --embed for the real corpus; the
stand-in generator runs anywhere and is what the tests use. A stand-in build will
not overwrite a real-model pool without --force.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..paths import DEFAULT_CFPB, DEFAULT_POOL
from .cfpb import load_reference
from .pool import build_pool
from .scoring import TextScorer


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
    # The pool is a cached artifact, not something to rebuild every run. It costs
    # GPU hours to generate for real and nothing downstream changes when it stays
    # the same, so an existing pool is reused unless a rebuild is asked for. The
    # check happens before any model is constructed, since loading a seven
    # billion parameter model and then deciding to skip would defeat the point.
    if args.out.exists() and not args.rebuild:
        import json as _json

        try:
            existing = _json.loads(args.out.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if existing and existing.get("entries"):
            print(f"text pool  ({existing.get('generator')})")
            print(f"  entries          {len(existing['entries']):>8,}")
            print(f"  embed model      {existing.get('embed_model')}  "
                  f"(dim {existing.get('embed_dim')})")
            print(f"  fingerprint      {str(existing.get('fingerprint'))[:16]}")
            print(f"\n  reusing {args.out} -- pass --rebuild to generate a new one")
            return 0

    # Rebuilding with the stand-in over a pool built with the real model would
    # replace a real corpus with placeholder text, silently, and every downstream
    # result would rest on it. Refuse unless that is what was asked for.
    if not args.qwen and args.out.exists():
        import json as _json

        try:
            existing = _json.loads(args.out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if existing.get("generator") not in (None, "mock") and not args.force:
            print(
                f"refusing to overwrite {args.out}: it was built with "
                f"'{existing.get('generator')}' and this build would replace it with "
                f"the stand-in.\n  Pass --qwen to rebuild it for real, --out to "
                f"write elsewhere, or --force to overwrite anyway."
            )
            return 1

    generator = None
    if args.qwen:
        from .pool import QwenGenerator

        # The only path that loads a model, and only once the checks above have
        # decided a build is actually needed.
        generator = QwenGenerator()

    embedder = None
    if args.embed:
        from .embed import Embedder

        # The real sentence-transformer. The one heavy load here, and only on the
        # flag; the default builds with the hash stand-in and downloads nothing.
        embedder = Embedder(truncate_dim=args.embed_dim)

    pool = build_pool(
        generator=generator, per_key=args.per_key, seed=args.seed, embedder=embedder,
        batch_size=args.batch_size,
    )
    pool.save(args.out)

    reference: list[str] = []
    if args.cfpb.exists():
        reference = load_reference(args.cfpb, limit=args.reference_limit, seed=args.seed)
    scorer = TextScorer(reference=reference or [e.text for e in pool.entries])

    print(f"text pool  ({pool.generator_name})")
    print(f"  entries          {len(pool.entries):>8,}")
    print(f"  fingerprint      {pool.fingerprint[:16]}")
    print(f"  embed model      {pool.embed_model}  (dim {pool.embed_dim})")
    print(f"  reference        {len(reference):>8,} CFPB narratives")
    print()
    print(_tier_report(pool, scorer))
    print(f"\n  written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudsim.generative")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="generate and score the text pool")
    build.add_argument("--out", type=Path, default=DEFAULT_POOL)
    build.add_argument("--cfpb", type=Path, default=DEFAULT_CFPB)
    build.add_argument("--per-key", type=int, default=8)
    build.add_argument("--reference-limit", type=int, default=2000)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument(
        "--qwen",
        action="store_true",
        help="use the real generation model (loads Qwen-2.5-7B; needs the generative extra)",
    )
    build.add_argument(
        "--embed",
        action="store_true",
        help="embed the text with Qwen3-Embedding-0.6B (semantic vectors for the text expert)",
    )
    build.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="prompts per forward pass when generating with the real model; raise "
             "it on a card with memory to spare, lower it if generation runs out",
    )
    build.add_argument(
        "--rebuild",
        action="store_true",
        help="regenerate the pool even though one already exists",
    )
    build.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing real-model pool with a stand-in build",
    )
    build.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Matryoshka output width, 32-1024 (default 256; the text expert fits "
             "on few rows, so the full 1024 is usually more columns than it can use)",
    )
    build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
