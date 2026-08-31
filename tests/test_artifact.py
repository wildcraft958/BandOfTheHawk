"""The artifact is the only contract between calibration and the simulation, so
its provenance separation is enforced rather than assumed."""

from __future__ import annotations

import json

import pytest

from fraudsim.calibration.artifact import ARTIFACT_VERSION, FittedParams


def make() -> FittedParams:
    return FittedParams(source="test", split_fingerprint="abc123", split_seed=0)


def test_round_trip(tmp_path) -> None:
    params = make()
    params.add_fitted("amount", {"mu": 4.2, "sigma": 0.8})
    params.add_swept("radius_km", value=12.0, low=3.0, high=40.0, reason="not measurable")
    path = params.save(tmp_path / "fitted.json")
    reloaded = FittedParams.load(path)
    assert reloaded.fitted["amount"]["mu"] == 4.2
    assert reloaded.swept["radius_km"]["high"] == 40.0
    assert reloaded.split_fingerprint == "abc123"


def test_swept_default_must_lie_inside_its_range() -> None:
    params = make()
    with pytest.raises(ValueError, match="outside its sweep"):
        params.add_swept("bad", value=99.0, low=0.0, high=1.0, reason="typo")


def test_fitted_and_swept_stay_separate() -> None:
    """A swept assumption must never be readable as a measurement."""
    params = make()
    params.add_fitted("amount", {"mu": 4.2})
    params.add_swept("radius_km", value=12.0, low=3.0, high=40.0, reason="not measurable")
    payload = params.to_dict()
    assert set(payload["fitted"]) == {"amount"}
    assert set(payload["swept"]) == {"radius_km"}
    assert "radius_km" not in payload["fitted"]


def test_rejected_models_are_kept_with_their_reason() -> None:
    params = make()
    params.add_rejection("hawkes", reason="failed its gate", payload={"ks_pvalue": 0.0})
    assert params.rejected["hawkes"]["reason"] == "failed its gate"
    assert params.counts()["rejected_models"] == 1


def test_version_mismatch_is_refused(tmp_path) -> None:
    path = tmp_path / "old.json"
    payload = make().to_dict()
    payload["version"] = ARTIFACT_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        FittedParams.load(path)


def test_render_lists_every_section() -> None:
    params = make()
    params.add_fitted("amount", {"mu": 4.2})
    params.add_swept("radius_km", value=12.0, low=3.0, high=40.0, reason="not measurable")
    params.add_rejection("hawkes", reason="failed its gate", payload={})
    text = params.render()
    assert "amount" in text
    assert "radius_km" in text
    assert "ruled out" in text
