# Run logs

Raw stdout from the runs behind the reported figures. Committed because a number
without the run that produced it is a claim rather than a result.

Each log records the exact command it ran on its own third line, so any of these is
reproducible from the file itself.

| File | Finished | Profile | Stages | Wall time |
|------|----------|---------|--------|-----------|
| `server-run-2026-08-31-2200.log` | 2026-08-31 22:00 | server, 12,000 holders | demo, text, fraud, baseline, mixture, coadapt | 934.4s |
| `server-run-2026-08-31-2214-optimized.log` | 2026-08-31 22:14 | server, 12,000 holders | same six | 446.3s |
| `server-run-2026-09-01-refit-artifact.log` | 2026-09-01 00:07 | server, 12,000 holders | same six | 459.6s |
| `gpu-run-2026-09-01-table7.log` | 2026-09-01 00:13 | gpu, 6,000 holders | coadapt only | 91.0s |
| `ablation-4seed-2026-08-31.txt` | 2026-08-31 03:18 | local, 600 holders | stealth ablation | not recorded |

## What each one is

**The two 31 August server runs** report identical static detector figures. The second
reached them in roughly half the wall time after the world build dropped from 36.0s to
7.6s. Their co-adaptation series differ, so they are two runs rather than one run timed
twice.

**The 1 September server run** is a later run of the same six stages against changed code.
Its static figures moved: rule-engine PR-AUC 0.0353 against the previous 0.0281, GBDT
0.9961 against 0.9890.

**The GPU run** covers the co-adaptation stage alone at 6,000 holders.

**The stealth ablation** is four paired seeds at 600 holders, both arms sharing a world per
seed, 24 updates with a refit every 6. Per-seed extraction series are printed in full at the
foot of the file.

## Provenance of the figures in the top-level README

The static detection table in the repository README quotes PR-AUC 0.9879, recall 0.9727 at
0.1% FPR, rule-engine PR-AUC 0.0266, a combiner lift of +0.0108 and a per-entity lift of
+0.0003. None of those five values appears in any log in this directory. They come from a
30 August run that is not committed here.

The nearest figures that are committed here are the 31 August pair (0.0281, 0.9890, +0.0149,
+0.0005) and the 1 September run (0.0353, 0.9961, +0.0349, +0.0004). Neither reproduces the
README table, and the two are not interchangeable with each other either.
