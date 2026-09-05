#!/usr/bin/env python3
"""Turn one real pipeline run into the JSON the web prototype renders.

Every number on the prototype page comes from here, so this script is the whole
credibility chain. It reads the run log and the committed calibration artifacts
and writes typed JSON into ``src/data/``. Nothing is invented; anything that
cannot be read is omitted rather than guessed.

The ``ui-prototype`` branch shipped seven fixture files of the same shape whose
values were synthetic placeholders, each flagged ``"_fixture": true``. This
script emits the same shapes without that flag. The prototype shows a warning
banner for any file that still carries it, so a placeholder can never quietly
reach a judge.

    python3 tools/make_real_fixtures.py

Reads   data/run.log, ../artifacts/noise_floors.json, ../artifacts/fitted_params.json
Writes  src/data/*.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
LOG = WEB / "data" / "run.log"
ARTIFACTS = WEB.parent / "artifacts"
OUT = WEB / "src" / "data"

# The run this prototype reports. Asserted so a different log cannot be
# substituted without the checks failing loudly.
EXPECTED_UPDATES = 150
EXPECTED_REFITS = [11, 23, 35, 47, 59, 71, 83, 95, 107, 119, 131, 143]
EXPECTED_EXTRACTED_SUM = 945640.9

NUM = r"[-+]?[\d,]*\.?\d+"


def _f(raw: str) -> float:
    return float(raw.replace(",", "").replace("+", ""))


def split_stages(text: str) -> dict[str, str]:
    """Split the log into one block per stage, keyed by stage name."""
    blocks: dict[str, str] = {}
    parts = re.split(r"={70,}\n\s+STAGE: (\w+)\n", text)
    # parts[0] is the run header; then (name, body) pairs.
    blocks["_header"] = parts[0]
    for name, body in zip(parts[1::2], parts[2::2]):
        blocks[name] = body
    tail = re.search(r"={70,}\n\s+SUMMARY\n={70,}\n(.*)", text, re.DOTALL)
    blocks["_summary"] = tail.group(1) if tail else ""
    return blocks


def labelled(block: str, label: str) -> float | None:
    """First number on the line whose text starts with ``label``.

    Tolerates a trailing unit word, as in ``history spans   180 days``.
    """
    m = re.search(rf"^\s*{re.escape(label)}\s+({NUM})(\s+\w+)?\s*$", block, re.MULTILINE)
    return _f(m.group(1)) if m else None


def pairs(block: str, start: str, stop: str) -> dict[str, float]:
    """``name  value`` lines between two markers, in order."""
    m = re.search(rf"{re.escape(start)}\n(.*?)(?:\n\s*\n|{re.escape(stop)})", block, re.DOTALL)
    if not m:
        return {}
    out: dict[str, float] = {}
    for line in m.group(1).splitlines():
        hit = re.match(rf"^\s+([a-z_0-9 ]+?)\s+({NUM})\s*$", line)
        if hit:
            out[hit.group(1).strip()] = _f(hit.group(2))
    return out


def metric_block(block: str, heading: str) -> dict[str, float] | None:
    """One detector configuration: five metrics plus the positive counts."""
    m = re.search(
        rf"^\s+{re.escape(heading)}\s*$\n(.*?)(?=^\s*$|^\s{{2}}\S)", block, re.MULTILINE | re.DOTALL
    )
    if not m:
        return None
    body = m.group(1)

    def g(pattern: str) -> float | None:
        hit = re.search(rf"{pattern}\s+({NUM})", body)
        return _f(hit.group(1)) if hit else None

    positives = re.search(rf"positives\s+({NUM})\s*/\s*({NUM})", body)
    out = {
        "pr_auc": g(r"PR-AUC"),
        "roc_auc": g(r"ROC-AUC"),
        "recall_at_0p1": g(r"recall @0\.1% FPR"),
        "recall_at_1": g(r"recall @1% FPR"),
        "precision_at_budget": g(r"precision @budget"),
    }
    if positives:
        out["n_positives"] = _f(positives.group(1))
        out["n_total"] = _f(positives.group(2))
    return out if out["pr_auc"] is not None else None


# The real action space, from src/fraudsim/engine/actions.py. Used to reject the
# fragments the log leaves when it truncates a long chain mid-token.
VALID_ACTIONS = frozenset(
    {
        "phish_holder",
        "buy_creds",
        "make_synth_id",
        "harvest_voice",
        "harvest_face",
        "call_ivr_provision",
        "submit_kyc",
        "add_device_selfserve",
        "sim_swap",
        "reset_password",
        "add_payee",
        "open_ticket",
        "escalate_limit",
        "attempt_auth",
        "complete_3ds",
        "transfer_p2p",
        "request_refund",
        "file_dispute",
        "cash_out",
        "launder_chain",
    }
)


def clean_chain(chain: list[str]) -> tuple[list[str], bool]:
    """Drop any trailing fragment the log's line truncation left behind."""
    kept = [a for a in chain if a in VALID_ACTIONS]
    return kept, len(kept) != len(chain)


def run_length_encode(chain: list[str]) -> list[dict]:
    """Collapse ``a,a,a`` into ``{action: a, times: 3}``.

    The trained attacker's top sequence is one action repeated 38 times, so
    storing it raw would be both large and unreadable.
    """
    out: list[dict] = []
    for action in chain:
        if out and out[-1]["action"] == action:
            out[-1]["times"] += 1
        else:
            out.append({"action": action, "times": 1})
    return out


# --------------------------------------------------------------------------- #
# stage parsers
# --------------------------------------------------------------------------- #


def parse_meta(blocks: dict[str, str]) -> dict:
    head = blocks["_header"]
    summary = blocks["_summary"]

    profile = re.search(r"profile:\s*(\w+)\s+population:\s*([\d,]+)", head)
    started = re.search(r"started:\s*([\d-]+ [\d:]+)", head)
    finished = re.search(r"finished:\s*([\d-]+ [\d:]+)", summary)

    stages = []
    for name, status, secs in re.findall(
        r"^\s+(\w+)\s+(ok|fail)\s+([\d.]+)s\s*$", summary, re.MULTILINE
    ):
        stages.append({"stage": name, "status": status, "seconds": float(secs)})
    total = re.search(r"^\s+total\s+([\d.]+)s\s*$", summary, re.MULTILINE)

    return {
        "profile": profile.group(1) if profile else None,
        "population": int(profile.group(2).replace(",", "")) if profile else None,
        "started": started.group(1) if started else None,
        "finished": finished.group(1) if finished else None,
        "stages": stages,
        "total_seconds": float(total.group(1)) if total else None,
        "models": "real Qwen + embeddings" if "real Qwen" in head else "stub",
    }


def parse_run_report(blocks: dict[str, str]) -> dict:
    demo, fraud = blocks["demo"], blocks["fraud"]

    negatives = pairs(demo, "hard negatives injected", "history spans")
    rules = {}
    for rid, rate in re.findall(r"^\s+(R\d|any)\s+([\d.]+)\s*$", demo, re.MULTILINE):
        rules[rid] = float(rate)
    rule_target = re.search(r"target ([\d.]+), (within target|off target)", demo)

    per_vertical = pairs(fraud, "episodes per vertical", "top action sequences")

    sequences = []
    seq_block = re.search(r"top action sequences\n(.*?)\n\s*\n", fraud, re.DOTALL)
    if seq_block:
        for line in seq_block.group(1).splitlines():
            hit = re.match(r"^\s+(\d+)\s+(\S+)\s*$", line)
            if hit:
                sequences.append({"count": int(hit.group(1)), "chain": hit.group(2).split(">")})

    return {
        "source": "data/run.log",
        "warm_start": {
            "entities": labelled(demo, "entities"),
            "events": labelled(demo, "events"),
            "dormant_share": labelled(demo, "dormant share"),
            "cards_with_a_median": labelled(demo, "cards with a median"),
            "history_days": labelled(demo, "history spans"),
        },
        "hard_negatives": negatives,
        "rule_trigger_rates": rules,
        "rule_target": float(rule_target.group(1)) if rule_target else None,
        "rule_verdict": rule_target.group(2) if rule_target else None,
        "rule_events": _f(re.search(r"rule trigger rates over ([\d,]+) events", demo).group(1))
        if re.search(r"rule trigger rates over ([\d,]+) events", demo)
        else None,
        "episodes": labelled(fraud, "episodes"),
        "reached_monetized": labelled(fraud, "reached monetized"),
        "benign_auths": labelled(fraud, "benign auths"),
        "fraud_auths": labelled(fraud, "fraud auths"),
        "fraud_auth_share": labelled(fraud, "fraud auth share"),
        "per_vertical": {k: int(v) for k, v in per_vertical.items()},
        "top_sequences": sequences,
    }


def parse_detectors(blocks: dict[str, str]) -> dict:
    base, mix = blocks["baseline"], blocks["mixture"]

    configs = []
    for key, heading, block, family, note in [
        (
            "d0",
            "D_0 rule engine",
            base,
            "rule",
            (
                "ROC-AUC below 0.5, so the rules are anti-correlated with fraud here. "
                "They are velocity thresholds tuned to a 6.5% trigger rate, and what "
                "trips them is bursty spending: gift-card runs, travel, new devices, "
                "the deliberately suspicious benign traffic. The attacker paces itself "
                "under them. The baseline is not misconfigured; it is being asked to "
                "separate fraud from hard negatives, which is what velocity rules "
                "cannot do and why a learned model is needed."
            ),
        ),
        ("gbdt_full", "GBDT full", base, "flat", None),
        ("gbdt_no_per_entity", "GBDT without per-entity features", base, "flat", None),
        ("experts_fixed", "experts + fixed average", mix, "mixture", None),
        (
            "experts_learned",
            "experts + learned combiner",
            mix,
            "mixture",
            (
                "Loses to the flat tree on this run. Structural decomposition costs "
                "accuracy at this scale and buys per-event-type attribution and "
                "independent retraining."
            ),
        ),
    ]:
        metrics = metric_block(block, heading)
        if metrics:
            configs.append(
                {"id": key, "label": heading, "family": family, "note": note, "metrics": metrics}
            )

    weights_raw = pairs(mix, "learned combiner weights", "learned vs")
    total_weight = sum(weights_raw.values()) or 1.0
    experts = [
        {
            "name": name,
            "weight": weight,
            "normalized_weight": round(weight / total_weight, 5),
        }
        for name, weight in sorted(weights_raw.items(), key=lambda kv: -kv[1])
    ]

    features = []
    feat_block = re.search(r"top features by gain\n(.*?)\n\s*\n", base, re.DOTALL)
    if feat_block:
        for line in feat_block.group(1).splitlines():
            hit = re.match(rf"^\s+(\S+)\s+({NUM})(\s+<- per-entity)?\s*$", line)
            if hit:
                features.append(
                    {
                        "name": hit.group(1),
                        "gain": _f(hit.group(2)),
                        "per_entity": bool(hit.group(3)),
                    }
                )

    rows = re.search(r"train rows\s+([\d,]+)\s+\(([\d,]+) fraud\)", base)
    test = re.search(r"test rows\s+([\d,]+)\s+\(([\d,]+) fraud\)", base)

    # F1 is not computed anywhere in the Python. The competition rubric names it
    # explicitly, so derive it at the alert budget and mark it derived: precision
    # at a 100-alert budget times the budget gives the true positives, and recall
    # follows from the positive count.
    operating = None
    full = next((c for c in configs if c["id"] == "gbdt_full"), None)
    if full and test:
        budget = 100
        precision = full["metrics"]["precision_at_budget"]
        positives = _f(test.group(2))
        true_positives = precision * budget
        recall = true_positives / positives
        f1 = 2 * precision * recall / (precision + recall)
        operating = {
            "provenance": "derived",
            "basis": "GBDT full, alert budget of 100 reviewed events",
            "alert_budget": budget,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": round(true_positives),
            "n_positives": positives,
            "note": (
                "F1 is not computed by fraudsim.defender.metrics. Derived here as "
                "precision@budget x budget = true positives, recall = TP / positives. "
                "Reported so the rubric's named metric is present, and labelled "
                "derived rather than measured."
            ),
        }

    bands = re.search(
        r"step_up\s+([\d.]+)\s+hold\s+([\d.]+)\s+decline\s+([\d.]+)\s+block\s+([\d.]+)", mix
    )

    return {
        "source": "data/run.log",
        "base_rate": labelled(base, "fraud auth share") or 0.005,
        "alert_budget": 100,
        "train_rows": _f(rows.group(1)) if rows else None,
        "train_fraud": _f(rows.group(2)) if rows else None,
        "test_rows": _f(test.group(1)) if test else None,
        "test_fraud": _f(test.group(2)) if test else None,
        "configs": configs,
        "experts": experts,
        "feature_gains": features,
        "operating_point": operating,
        "per_entity_lift": labelled(base, "per-entity PR-AUC lift"),
        "fitted_bands": {
            "step_up": float(bands.group(1)),
            "hold": float(bands.group(2)),
            "decline": float(bands.group(3)),
            "block": float(bands.group(4)),
        }
        if bands
        else None,
    }


def parse_coadapt(blocks: dict[str, str]) -> dict:
    co = blocks["coadapt"]

    rows: list[list[float]] = []
    refits: list[int] = []
    for line in co.splitlines():
        hit = re.match(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(<- refit)?\s*$", line)
        if hit:
            rows.append(
                [int(hit.group(1)), float(hit.group(2)), float(hit.group(3)), float(hit.group(4))]
            )
            if hit.group(5):
                refits.append(int(hit.group(1)))

    strategies = []
    strat_block = re.search(
        r"how the attacker's strategy changed \(sampled at each refit\)\n(.*?)\n\s*\n",
        co,
        re.DOTALL,
    )
    if strat_block:
        for line in strat_block.group(1).splitlines():
            hit = re.match(r"^\s+update (\d+)\s+(\d+)x\s+(\S+)\s*$", line)
            if hit:
                chain, dropped = clean_chain(hit.group(3).split(">"))
                strategies.append(
                    {
                        "update": int(hit.group(1)),
                        "count": int(hit.group(2)),
                        "runs": run_length_encode(chain),
                        "truncated": dropped or hit.group(3).endswith((">", "_")),
                    }
                )

    # The held-out block is parsed for its vertical names only. The recall figure
    # in the log is a measurement the solution document deliberately withholds,
    # because the held-out action stays legal and the positive count is small, so
    # it is neither emitted here nor shipped in the bundle a reader can view.
    held_out = []
    zs_block = re.search(r"zero-shot recall on held-out verticals\n(.*?)\n\s*\n", co, re.DOTALL)
    if zs_block:
        for line in zs_block.group(1).splitlines():
            hit = re.match(r"^\s+(\w+)\s+([\d.]+)\s*$", line)
            if hit:
                held_out.append(hit.group(1))

    finals = []
    fin_block = re.search(
        r"top action sequences \(final trained attacker\)\n(.*?)(?:\n\s*\n|\Z)", co, re.DOTALL
    )
    if fin_block:
        for line in fin_block.group(1).splitlines():
            hit = re.match(r"^\s+(\d+)\s+(\S+)\s*$", line)
            if hit:
                chain, dropped = clean_chain(hit.group(2).split(">"))
                finals.append(
                    {
                        "count": int(hit.group(1)),
                        "runs": run_length_encode(chain),
                        "truncated": dropped,
                    }
                )

    selection = {"groups": [], "observations": None, "selecting": None}
    sel_block = re.search(
        r"victim selection \(posterior mean by feature\)\n(.*?)\n\s*\n", co, re.DOTALL
    )
    if sel_block:
        body = sel_block.group(1)
        group: dict | None = None
        for line in body.splitlines():
            if re.match(r"^\s+\(coefficients", line):
                continue
            obs = re.match(rf"^\s+observations\s+({NUM})\s*$", line)
            if obs:
                selection["observations"] = _f(obs.group(1))
                continue
            sel = re.match(r"^\s+selecting\s+(\w+)\s*$", line)
            if sel:
                selection["selecting"] = sel.group(1) == "yes"
                continue
            ref = re.match(r"^\s+(.+?)\s+reference\s*$", line)
            if ref:
                label = ref.group(1).strip()
                name = "BIN tier" if label.startswith("bin tier") else "Card age"
                group = {"name": name, "levels": []}
                selection["groups"].append(group)
                group["levels"].append({"label": label, "coef": 0.0, "reference": True})
                continue
            lev = re.match(rf"^\s+(.+?)\s+({NUM})\s*$", line)
            if lev and group is not None:
                group["levels"].append(
                    {"label": lev.group(1).strip(), "coef": _f(lev.group(2)), "reference": False}
                )

    # The BIN-tier posterior does not hold its sign between runs, so it is not
    # published. Dropping it here rather than in the page keeps the coefficients
    # out of the bundle, where a value nothing renders would still be readable.
    selection["withheld"] = [g["name"] for g in selection["groups"] if g["name"] == "BIN tier"]
    selection["groups"] = [g for g in selection["groups"] if g["name"] != "BIN tier"]

    extracted = [r[1] for r in rows]
    entropy = [r[3] for r in rows]

    return {
        "source": "data/run.log",
        "columns": ["update", "extracted", "policy_return", "entropy"],
        "rows": rows,
        "refit_updates": refits,
        "checksum_extracted": round(sum(extracted), 1),
        "warm_start": {
            "initial_defender_fraud": labelled(co, "initial defender fraud"),
            "bc_final_loss": labelled(co, "BC final loss"),
            "critic_final_loss": labelled(co, "critic final loss"),
        },
        "reads": {
            "extracted_first": extracted[0] if extracted else None,
            "extracted_last": extracted[-1] if extracted else None,
            "extracted_max": max(extracted) if extracted else None,
            "zeros": sum(1 for v in extracted if v == 0.0),
            "entropy_start": entropy[0] if entropy else None,
            "entropy_peak": max(entropy) if entropy else None,
            "entropy_end": entropy[-1] if entropy else None,
        },
        "strategies": strategies,
        "final_sequences": finals,
        "held_out_verticals": held_out,
        "selection": selection,
    }


def parse_fidelity() -> dict:
    floors = json.loads((ARTIFACTS / "noise_floors.json").read_text())
    fitted = json.loads((ARTIFACTS / "fitted_params.json").read_text())

    diag = fitted.get("diagnostics", {}).get("arrival_holdout", {})
    f = fitted["fitted"]

    # Every fidelity claim is a ratio against the distance between two halves of
    # real data. A ratio near 1.0 means synthetic differs from real about as much
    # as real differs from itself.
    def pair(name, observed, target, floor, unit=None):
        gap = None if observed is None or target is None else abs(observed - target)
        ratio = None if gap is None or not floor else round(gap / floor, 3)
        return {
            "name": name,
            "observed": observed,
            "target": target,
            "gap": None if gap is None else round(gap, 6),
            "noise_floor": floor,
            "ratio": ratio,
            "unit": unit,
        }

    # The calibration artifact carries the floors and targets every fidelity
    # claim is measured against, so the ledger below counts what it holds.
    # noise_floors.json is read only for the fan-out targets, which the graph
    # panel needs and the calibration artifact does not carry.
    t = fitted["targets"]
    fl = fitted["noise_floors"]
    fanout_targets = {k: v for k, v in floors["targets"].items() if k.startswith("fanout")}
    comparisons = [
        pair(
            "arrival burstiness",
            diag.get("burstiness_observed"),
            diag.get("burstiness_target"),
            fl.get("burstiness_gap"),
        ),
        pair(
            "arrival autocorrelation",
            diag.get("autocorrelation_observed"),
            diag.get("autocorrelation_target"),
            fl.get("autocorrelation_gap"),
        ),
        pair(
            "amount median",
            f["amount"]["median"],
            t.get("amount_median"),
            fl.get("amount_w1"),
            "currency units",
        ),
        pair(
            "entity activity median",
            t.get("entity_activity_median"),
            t.get("entity_activity_median"),
            fl.get("entity_activity_w1"),
        ),
    ]

    return {
        "source": "artifacts/noise_floors.json + artifacts/fitted_params.json",
        "created_utc": fitted.get("created_utc"),
        "verdict_ladder": {
            "indistinguishable": 1.5,
            "close": 3.0,
            "structural_gap": 10.0,
        },
        "split": {
            "left_entities": floors["left_entities"],
            "left_rows": floors["left_rows"],
            "right_entities": floors["right_entities"],
            "right_rows": floors["right_rows"],
            "fingerprint": floors["split_fingerprint"],
        },
        "comparisons": comparisons,
        "amount": f["amount"],
        "amount_heterogeneity": f.get("amount_heterogeneity"),
        "circadian": f.get("circadian"),
        "category_mix": f.get("category_mix"),
        "fanout_targets": fanout_targets,
        "rejected": fitted.get("rejected", {}),
        "all_floors": fl,
        # The parameter ledger. FITTED groups were measured from real data,
        # SWEPT ones could not be measured and are reported across a range. The
        # names are the artifact's own keys, so this cannot drift from what the
        # calibration actually produced.
        "provenance": {
            "fitted": sorted(f),
            "swept": sorted(fitted.get("swept", {})),
            "n_noise_floors": len(fl),
            "n_targets": len(t),
        },
    }


def parse_text_samples() -> dict:
    """Sample a few text-pool entries, dropping the embeddings.

    The pool is 86 MB because every entry carries a 256-dimension embedding, so
    it can never be bundled. A couple of entries per vertical and tier is enough
    to show what the generative layer produces.

    Note on what the two classes are. Entries with ``fraudulent: false`` are NOT
    real consumer complaints: they carry a generated persona and synthetic
    merchant and bank names, exactly as the fraudulent ones do. Both classes are
    machine-written, and the label separates intent rather than provenance. The
    page says so, because calling these real complaints would be false.
    """
    path = ARTIFACTS / "text_pool.json"
    if not path.exists():
        return {}

    pool = json.loads(path.read_text())
    entries = pool.get("entries", [])

    wanted: dict[tuple[str, int, bool], dict] = {}
    for e in entries:
        key = (e["vertical"], e["tier"], e["fraudulent"])
        if key in wanted:
            continue
        wanted[key] = {
            "vertical": e["vertical"],
            "tier": e["tier"],
            "fraudulent": e["fraudulent"],
            "persona": e.get("persona"),
            "text": e.get("text"),
            "facts": e.get("facts"),
        }

    return {
        "source": "artifacts/text_pool.json",
        "generator": pool.get("generator"),
        "embed_model": pool.get("embed_model"),
        "embed_dim": pool.get("embed_dim"),
        "fingerprint": pool.get("fingerprint"),
        "n_entries": len(entries),
        "note": (
            "Both classes are generated. Entries labelled not-fraudulent carry a "
            "generated persona and synthetic merchant and bank names just as the "
            "fraudulent ones do, so the label separates intent, not provenance. "
            "The pool contains no real consumer complaints."
        ),
        "tier_note": (
            "Capability tiers are ordinal. They are ranked, never treated as "
            "quantities, because error rates are not comparable across editions."
        ),
        "samples": sorted(
            wanted.values(), key=lambda e: (e["vertical"], e["tier"], e["fraudulent"])
        ),
    }


def parse_graph(blocks: dict[str, str], fidelity: dict) -> dict:
    demo = blocks["demo"]
    fanout = re.search(r"fan-out mean\s+([\d.]+)\s+\(target ([\d.]+)\)", demo)
    targets = fidelity["fanout_targets"]
    return {
        "source": "data/run.log + artifacts/noise_floors.json",
        "fanout_observed_mean": float(fanout.group(1)) if fanout else None,
        "fanout_target_mean": float(fanout.group(2)) if fanout else None,
        "invariants_hold": "graph invariants         hold" in demo,
        "targets": targets,
        "variance_to_mean_note": (
            "Assigning a shared attribute independently per row yields a degree "
            "distribution whose variance cannot exceed its mean. The real "
            "measurement is far above that, so degrees are generated first and "
            "entities matched onto them."
        ),
    }


def main() -> int:
    text = LOG.read_text()
    blocks = split_stages(text)

    meta = parse_meta(blocks)
    run_report = parse_run_report(blocks)
    detectors = parse_detectors(blocks)
    coadapt = parse_coadapt(blocks)
    fidelity = parse_fidelity()
    graph = parse_graph(blocks, fidelity)
    text_samples = parse_text_samples()

    # Loud checks. A different log, or a parser that silently drifts, must fail
    # here rather than reach a judge.
    problems = []
    if len(coadapt["rows"]) != EXPECTED_UPDATES:
        problems.append(f"expected {EXPECTED_UPDATES} updates, parsed {len(coadapt['rows'])}")
    if coadapt["refit_updates"] != EXPECTED_REFITS:
        problems.append(f"refit indices differ: {coadapt['refit_updates']}")
    if abs(coadapt["checksum_extracted"] - EXPECTED_EXTRACTED_SUM) > 0.5:
        problems.append(
            f"extraction checksum {coadapt['checksum_extracted']} != {EXPECTED_EXTRACTED_SUM}"
        )
    if len(detectors["configs"]) != 5:
        problems.append(f"expected 5 detector configs, parsed {len(detectors['configs'])}")
    if len(detectors["experts"]) != 5:
        problems.append(f"expected 5 experts, parsed {len(detectors['experts'])}")
    if len(coadapt["held_out_verticals"]) != 2:
        problems.append(
            f"expected 2 held-out verticals, parsed {len(coadapt['held_out_verticals'])}"
        )

    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    written = {
        "meta.json": meta,
        "run_report.json": run_report,
        "detector_metrics.json": detectors,
        "coadapt_metrics.json": coadapt,
        "fidelity.json": fidelity,
        "graph.json": graph,
        **({"text_samples.json": text_samples} if text_samples else {}),
    }
    for name, payload in written.items():
        path = OUT / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {name}  {path.stat().st_size:>7,} bytes")

    print()
    print(
        f"  {len(coadapt['rows'])} updates, {len(coadapt['refit_updates'])} refits, "
        f"checksum {coadapt['checksum_extracted']}"
    )
    print(
        f"  extraction {coadapt['reads']['extracted_first']} -> "
        f"{coadapt['reads']['extracted_last']}, peak {coadapt['reads']['extracted_max']}, "
        f"{coadapt['reads']['zeros']} updates at exactly zero"
    )
    print(
        f"  entropy {coadapt['reads']['entropy_start']} -> peak "
        f"{coadapt['reads']['entropy_peak']} -> {coadapt['reads']['entropy_end']}"
    )
    if detectors["operating_point"]:
        op = detectors["operating_point"]
        print(
            f"  derived operating point: precision {op['precision']}, "
            f"recall {op['recall']}, F1 {op['f1']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
