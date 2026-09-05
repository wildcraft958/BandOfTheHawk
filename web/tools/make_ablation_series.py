#!/usr/bin/env python3
"""Pull the per-update series out of the committed ablation runs.

Unlike src/data/paper.ts, which is a hand transcription of the solution
document's Tables 5 and 6, this output is DERIVED: every value comes from a JSON
file tracked in the repository at artifacts/ablation/, so a checker can open the
same file and read the same number.

Only the fields the site actually draws are copied, because the full run files
carry strategy histories and checkpoint blobs that would bloat a single-file
build for no benefit.

Deliberately NOT copied: the `zero_shot` block. The solution document withholds
that measurement, because the held-out action stayed legal in the attacker's
action space, so the defender trained on the traffic it was then asked to
generalise to. Carrying it here would put it one careless import away from the
screen again.

Arm naming, verified rather than assumed: mean post-refit extraction is higher in
every `stealth_s*` run than in its paired `control_s*` run, so stealth is the
full attacker and control is the ablated one.

Usage:  python tools/make_ablation_series.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = HERE.parent
SRC = WEB.parent / "artifacts" / "ablation"
OUT = WEB / "src" / "data" / "ablation_runs.json"

ARM = {"stealth": "full", "control": "ablated"}
EXPECTED_UPDATES = 24
EXPECTED_REFITS = [5, 11, 17, 23]

KEEP = ("attacker_success", "entropy", "defender_refits", "false_positive_rate")


def main() -> None:
    files = sorted(SRC.glob("*.json"))
    if not files:
        raise SystemExit(f"no run files under {SRC}")

    runs = []
    for path in files:
        stem = path.stem
        arm_key, seed = stem.split("_s")
        payload = json.loads(path.read_text())

        series = payload["attacker_success"]
        refits = payload["defender_refits"]
        if len(series) != EXPECTED_UPDATES:
            raise SystemExit(f"{stem}: expected {EXPECTED_UPDATES} updates, got {len(series)}")
        if refits != EXPECTED_REFITS:
            raise SystemExit(f"{stem}: expected refits {EXPECTED_REFITS}, got {refits}")

        runs.append(
            {
                "arm": ARM[arm_key],
                "seed": int(seed),
                "file": f"artifacts/ablation/{path.name}",
                "extraction": [round(v, 1) for v in series],
                "entropy": [round(v, 4) for v in payload["entropy"]],
                "refits": refits,
                # One value per refit: the share of genuine authorisations the
                # defender then in force would refuse.
                "friction": payload.get("false_positive_rate", []),
            }
        )

    full = [r for r in runs if r["arm"] == "full"]
    ablated = [r for r in runs if r["arm"] == "ablated"]
    if len(full) != len(ablated):
        raise SystemExit("arms are not paired")

    out = {
        "source": "artifacts/ablation/*.json, tracked in the repository",
        "note": (
            "Derived from committed run files, not transcribed. These are not the "
            "runs the solution document reports: the direction agrees and the "
            "magnitude does not, so the document's figures stay the ones cited "
            "for the headline."
        ),
        "updates": EXPECTED_UPDATES,
        "refits": EXPECTED_REFITS,
        "seeds": len(full),
        "runs": runs,
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    peak = max(max(r["extraction"]) for r in runs)
    print(f"  wrote {OUT.relative_to(WEB)}  {OUT.stat().st_size:,} bytes")
    print(f"  {len(runs)} runs, {len(full)} paired seeds, {EXPECTED_UPDATES} updates each")
    print(f"  refits at {EXPECTED_REFITS}, peak extraction {peak:,.0f}")


if __name__ == "__main__":
    main()
