"""Behavioral metrics must detect the failure modes that independent sampling
produces, since that is the only reason they exist."""

from __future__ import annotations

import pytest

pd = pytest.importorskip(
    "pandas", reason='install the "calibration" extra'
)

import numpy as np


from fraudsim.calibration.behavioral import (
    burst_stats,
    fanout_stats,
    fraud_rate_by_fanout,
    inter_event_stats,
)
from fraudsim.calibration.distances import DegradationReport, jsd, total_variation, w1


def independent_timestamps(n_entities: int = 300, n_events: int = 30, seed: int = 0):
    """Each timestamp drawn independently: the autocorrelation must not exceed zero."""
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(n_entities):
        times = np.sort(rng.exponential(3600, n_events).cumsum())
        rows.extend({"entity": entity, "ts": float(t)} for t in times)
    return pd.DataFrame(rows)


def clustered_timestamps(n_entities: int = 300, seed: int = 0):
    """Self-exciting arrivals, where each event raises the rate of the next.

    Note this is not the same as merely mixing short and long gaps. A process
    with two gap regimes can reproduce burstiness exactly while showing zero or
    negative lag-1 autocorrelation, because nothing carries the current rate
    from one gap to the next. Only genuine self-excitation does that, which is
    why the two statistics are both measured.
    """
    rng = np.random.default_rng(seed)
    mu, alpha, beta = 1 / 3600, 0.6, 1 / 600
    rows = []
    for entity in range(n_entities):
        now, history = 0.0, []
        while len(history) < 60:
            excitation = sum(alpha * beta * np.exp(-beta * (now - s)) for s in history[-40:])
            now += float(rng.exponential(1 / max(mu + excitation, 1e-9)))
            history.append(now)
            rows.append({"entity": entity, "ts": now})
    return pd.DataFrame(rows)


def bimodal_gap_timestamps(n_entities: int = 300, seed: int = 1):
    """Two gap regimes chosen independently per gap: bursty but not self-exciting."""
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(n_entities):
        now = 0.0
        for _ in range(60):
            scale = 30.0 if rng.random() < 0.55 else 40_000.0
            now += float(rng.exponential(scale))
            rows.append({"entity": entity, "ts": now})
    return pd.DataFrame(rows)


def test_independent_sampling_cannot_produce_positive_autocorrelation() -> None:
    stats = inter_event_stats(independent_timestamps(), "entity", "ts", min_events=10)
    assert stats.mean_autocorrelation < 0.05


def test_self_exciting_arrivals_show_positive_autocorrelation() -> None:
    stats = inter_event_stats(clustered_timestamps(), "entity", "ts", min_events=10)
    assert stats.mean_autocorrelation > 0.1
    assert stats.share_positive > 0.5


def test_burstiness_separates_the_two_processes() -> None:
    flat = inter_event_stats(independent_timestamps(), "entity", "ts", min_events=10)
    bursty = inter_event_stats(clustered_timestamps(), "entity", "ts", min_events=10)
    assert bursty.mean_burstiness > flat.mean_burstiness


def test_burstiness_alone_does_not_imply_autocorrelation() -> None:
    """The reason both statistics are reported.

    Mixing two gap regimes reproduces burst structure without carrying any rate
    between consecutive gaps, so a timing model can look right on bursts and
    still fail the autocorrelation target.
    """
    bimodal = inter_event_stats(bimodal_gap_timestamps(), "entity", "ts", min_events=10)
    exciting = inter_event_stats(clustered_timestamps(), "entity", "ts", min_events=10)
    assert bimodal.mean_burstiness > 0.2
    assert bimodal.mean_autocorrelation < 0.05
    assert exciting.mean_autocorrelation > bimodal.mean_autocorrelation


def test_burst_lengths_grow_with_threshold() -> None:
    stats = burst_stats(clustered_timestamps(), "entity", "ts", thresholds=(60, 300, 1800))
    means = [stats.burst_lengths[t].mean() for t in (60, 300, 1800)]
    assert means[0] <= means[1] <= means[2]


def test_independent_attribute_assignment_is_thin_tailed() -> None:
    """Picking a shared attribute per row from a marginal bounds variance by mean."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "entity": np.arange(4000),
            "attribute": rng.integers(0, 800, 4000),
        }
    )
    assert fanout_stats(frame, "attribute", "entity").variance_to_mean < 1.5


def test_preferential_assignment_is_heavy_tailed() -> None:
    rng = np.random.default_rng(1)
    weights = 1.0 / np.arange(1, 801) ** 1.1
    weights /= weights.sum()
    frame = pd.DataFrame(
        {
            "entity": np.arange(4000),
            "attribute": rng.choice(800, size=4000, p=weights),
        }
    )
    assert fanout_stats(frame, "attribute", "entity").variance_to_mean > 3.0


def test_stamped_sharing_shows_a_climbing_fraud_profile() -> None:
    """A source where sharing implies fraud is unusable as a benign anchor."""
    rows = []
    for attribute in range(200):
        degree = 1 if attribute < 150 else int(10 + attribute % 30)
        for entity in range(degree):
            rows.append(
                {
                    "attribute": attribute,
                    "entity": f"{attribute}_{entity}",
                    "label": 0 if degree == 1 else 1,
                }
            )
    table = fraud_rate_by_fanout(pd.DataFrame(rows), "attribute", "entity", "label")
    rates = table["fraud_rate"].to_numpy()
    assert rates[0] < 0.1 and rates[-1] > 0.9


def test_w1_is_tail_sensitive() -> None:
    rng = np.random.default_rng(0)
    base = rng.lognormal(4.0, 1.0, 20_000)
    heavy = np.concatenate([base[:19_000], base[19_000:] * 40])
    assert w1(base, heavy) > 0


def test_jsd_bounds() -> None:
    a = np.array([0, 0, 1, 1, 2])
    assert jsd(a, a) == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= jsd(a, np.array([3, 3, 4, 4, 5])) <= 1.0


def test_total_variation_bounds() -> None:
    a = np.array([0, 0, 1])
    assert total_variation(a, a) == pytest.approx(0.0)
    assert total_variation(a, np.array([2, 2, 2])) == pytest.approx(1.0)


def test_degradation_ratio_and_verdicts() -> None:
    report = DegradationReport("demo")
    report.add("matched", observed=1.0, floor=1.0)
    report.add("broken", observed=30.0, floor=1.0)
    assert report.entries[0].verdict == "indistinguishable"
    assert report.entries[1].verdict == "not reproduced"
    assert report.composite() == pytest.approx(15.5)
    assert len(report.failures()) == 1
