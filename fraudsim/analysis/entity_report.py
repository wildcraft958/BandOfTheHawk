"""Per-entity structure, generated beside real.

The instrument, built before the generators it measures. A fix asserted is a
fix believed; a fix measured against the judge dataset is one that can be shown
to have moved the number it claimed to move, and shown not to have moved
anything else.

Every statistic here is one a marginal comparison cannot see. Hour is the
clearest case: a population whose members each keep tight hours but disagree
about which, and one where everybody shops at the same time, can share a
marginal exactly while being entirely different worlds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..calibration.entity_stats import (
    categorical_entity_concentration,
    circular_entity_spread,
)
from ..features.schema import EventType


def auth_frame(log) -> pd.DataFrame:
    """Authorisations from an event log, as a frame the estimators accept."""
    rows = [
        {
            "card": event.card_id,
            "merchant": event.merchant_id,
            "category": event.category_cluster,
            "hour": float(event.hour_of_day),
            "ts": event.ts,
            "first_at_merchant": event.is_first_txn_this_merchant,
            "within_usual_hours": event.within_usual_hours,
        }
        for event in log.events
        if getattr(event, "event_type", None) is EventType.AUTH_ATTEMPT
    ]
    return pd.DataFrame(rows)


def judge_frame(root: Path) -> pd.DataFrame:
    """Benign rows from the judge dataset, keyed by reconstructed cardholder."""
    from ..calibration.loaders import IeeeCisLoader

    frame = IeeeCisLoader(root).transactions()
    benign = frame.benign().copy()
    benign["hour"] = (benign[frame.time_column].to_numpy(float) / 3600.0) % 24
    return benign.rename(columns={frame.entity_column: "card"})


def render_entity_report(log, judge_root: Path | None = None, min_events: int = 5) -> str:
    """Generated statistics, against real ones where a real one exists.

    Merchant and category carry no real anchor: the judge dataset has no
    merchant entity, and the taxonomy source is a generator whose typical card
    visits 83% of the roster. Those rows report the generated value and the
    shuffled-null baseline, and say so, rather than inventing a target.
    """
    generated = auth_frame(log)
    if generated.empty:
        return "no authorisations in the log"

    lines = [
        "per-entity structure",
        f"  generated: {len(generated):,} auths / {generated['card'].nunique():,} cards"
        f"  (min_events={min_events})",
        "",
    ]

    real = None
    if judge_root is not None:
        real = judge_frame(judge_root)
        lines.append(f"  real:      {len(real):,} auths / {real['card'].nunique():,} entities")
        lines.append("")

    # ------------------------------------------------------------ hour
    gen_hour = circular_entity_spread(generated, "card", "hour", min_events=min_events)
    lines += [
        "  hour of day",
        f"    {'statistic':<22}{'generated':>12}{'real':>12}{'ratio':>9}",
        f"    {'-' * 22}{'-' * 12}{'-' * 12}{'-' * 9}",
    ]
    if real is not None:
        real_hour = circular_entity_spread(real, "card", "hour", min_events=min_events)
        for name, gen_value, real_value in (
            ("marginal R", gen_hour.marginal_r, real_hour.marginal_r),
            ("within-entity R", gen_hour.within_r, real_hour.within_r),
            ("between-entity R", gen_hour.between_r, real_hour.between_r),
            ("peak hour", gen_hour.marginal_mean, real_hour.marginal_mean),
        ):
            ratio = gen_value / real_value if real_value else float("nan")
            lines.append(
                f"    {name:<22}{gen_value:>12.4f}{real_value:>12.4f}{ratio:>9.2f}"
            )
    else:
        for name, value in (
            ("marginal R", gen_hour.marginal_r),
            ("within-entity R", gen_hour.within_r),
            ("between-entity R", gen_hour.between_r),
            ("peak hour", gen_hour.marginal_mean),
        ):
            lines.append(f"    {name:<22}{value:>12.4f}{'-':>12}{'-':>9}")
    lines.append("")

    # ------------------------------------------- category and merchant
    lines += [
        "  concentration, against a shuffled null rather than a real target",
        f"    {'feature':<22}{'ratio':>12}{'null':>12}{'z':>9}",
        f"    {'-' * 22}{'-' * 12}{'-' * 12}{'-' * 9}",
    ]
    underpowered = []
    for label, column in (("category", "category"), ("merchant", "merchant")):
        try:
            spread = categorical_entity_concentration(
                generated, "card", column, min_events=min_events, n_shuffles=8
            )
        except ValueError:
            lines.append(f"    {label:<22}{'no entity has enough history':>33}")
            continue
        lines.append(
            f"    {label:<22}{spread.ratio:>12.3f}{spread.null_ratio_mean:>12.3f}"
            f"{spread.z_against_null:>9.1f}"
        )
        # The statistic loses its power when a card cannot plausibly repeat a
        # value: with many values and few events each, it is estimated from
        # accidental collisions and its null wanders far from one. A wide null
        # alone is not enough to call it, though - a signal far outside even a
        # wide null is still a signal, and flagging that would be telling the
        # reader to ignore the strongest evidence on the page.
        if spread.null_ratio_sd > 0.05 and abs(spread.z_against_null) < 6.0:
            underpowered.append((label, spread.null_ratio_sd))

    lines += [
        "",
        "    a ratio at the null means every card draws from one shared curve,",
        "    which is the defect these statistics exist to detect",
    ]
    for label, sd in underpowered:
        lines.append(
            f"    {label}: null spreads +/-{sd:.2f}, so this row is underpowered -"
        )
        lines.append(
            "      too many values against too little history per card to detect"
        )
        lines.append("      concentration at all; read it once loyalty exists, not before")
    lines.append("")

    # ------------------------------------------------------- features
    first = float(generated["first_at_merchant"].mean())
    known = generated["within_usual_hours"].notna().mean()
    lines += [
        "  features that read per-entity history",
        f"    {'first at merchant':<22}{first:>12.4f}",
        f"    {'within usual hours set':<22}{known:>12.4f}",
    ]
    if known:
        lines.append(
            f"    {'within usual hours':<22}"
            f"{float(generated['within_usual_hours'].dropna().mean()):>12.4f}"
        )
    return "\n".join(lines)
