"""Configuration validates on load and keeps measurements apart from choices."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fraudsim.calibration.artifact import FittedParams
from fraudsim.settings.base import Provenance, ProvenanceError
from fraudsim.settings.behavior import CategoryConfig, CircadianConfig
from fraudsim.settings.engine import WindowConfig
from fraudsim.settings.simulation import SimulationConfig, resolve
from fraudsim.settings.world import ActivityConfig, DeviceConfig, PopulationConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "simulation.yaml"
ARTIFACT = ROOT / "artifacts" / "fitted_params.json"

# The artifact is generated rather than committed, since it is reproducible
# from the datasets. Tests that need it skip with a pointer instead of failing
# on a fresh clone.
needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="run: python -m fraudsim.calibration.cli fit",
)


def artifact() -> FittedParams:
    return FittedParams.load(ARTIFACT)


def test_defaults_are_valid() -> None:
    assert SimulationConfig().population.n_holders > 0


def test_unknown_key_is_refused() -> None:
    with pytest.raises(ValidationError, match="not_a_field"):
        SimulationConfig.model_validate({"population": {"not_a_field": 1}})


def test_out_of_range_is_refused() -> None:
    with pytest.raises(ValidationError, match="household_mean"):
        DeviceConfig(household_mean=99.0)


def test_household_mean_cannot_exceed_max() -> None:
    with pytest.raises(Exception, match="household_mean"):
        DeviceConfig(household_mean=5.0, household_max=3)


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(Exception, match="sum to 1"):
        ActivityConfig(tier_weights={"dormant": 0.5, "regular": 0.2},
                       tier_rate_multipliers={"dormant": 1.0, "regular": 1.0})
    with pytest.raises(Exception, match="sum to 1"):
        PopulationConfig(archetype_weights={"commuter": 0.5, "senior": 0.2})


def test_every_tier_needs_a_rate() -> None:
    with pytest.raises(Exception, match="rate multiplier"):
        ActivityConfig(
            tier_weights={"dormant": 0.5, "regular": 0.5},
            tier_rate_multipliers={"dormant": 1.0},
        )


def test_circadian_components_must_align() -> None:
    with pytest.raises(Exception, match="same length"):
        CircadianConfig(means=(1.0, 2.0), concentrations=(1.0,), weights=(0.5, 0.5))
    with pytest.raises(Exception, match=r"\[0, 24\)"):
        CircadianConfig(means=(25.0,), concentrations=(1.0,), weights=(1.0,))


def test_every_category_needs_a_cnp_share() -> None:
    with pytest.raises(Exception, match="card-not-present"):
        CategoryConfig(mix={"grocery": 1.0}, card_not_present_share={"retail": 0.2})


def test_windows_must_ascend() -> None:
    with pytest.raises(Exception, match="ascending"):
        WindowConfig(windows_seconds=(86_400, 3600))


def test_compound_criteria_are_distinct() -> None:
    with pytest.raises(Exception, match="distinct"):
        WindowConfig(compound_criteria=("entry_mode", "entry_mode"))


def test_compound_feature_count() -> None:
    """Three windows across three criteria, counted and summed in each cell."""
    assert WindowConfig().n_compound_features == 18


def test_geography_is_not_a_compound_criterion() -> None:
    """The only source with merchant locations is unusable, so a geographic
    bucket would put an uncalibrated value under a third of these features."""
    assert not any("geo" in c for c in WindowConfig().compound_criteria)


@needs_artifact
def test_resolve_tracks_where_each_value_came_from() -> None:
    resolved = resolve(CONFIG, artifact=artifact())
    counts = resolved.ledger.counts()
    assert counts["fitted"] > 0
    assert counts["swept"] > 0
    assert resolved.provenance_of("behavior.amount.tail_index") is Provenance.FITTED
    assert resolved.provenance_of("population.geo.home_radius_km") is Provenance.SWEPT


@needs_artifact
def test_fitted_values_reach_the_config() -> None:
    resolved = resolve(CONFIG, artifact=artifact())
    fitted = artifact().fitted["amount"]["tail_index"]
    assert resolved.config.behavior.amount.tail_index == pytest.approx(fitted)


@needs_artifact
def test_setting_a_fitted_field_in_yaml_is_refused(tmp_path) -> None:
    payload = yaml.safe_load(CONFIG.open(encoding="utf-8"))
    payload.setdefault("behavior", {}).setdefault("amount", {})["tail_index"] = 2.5
    path = tmp_path / "collide.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="tail_index"):
        resolve(path, artifact=artifact())


@needs_artifact
def test_explicit_override_keeps_the_configured_value_as_a_choice(tmp_path) -> None:
    """Departing from a fit on purpose is a choice, and has to be recorded as one."""
    payload = yaml.safe_load(CONFIG.open(encoding="utf-8"))
    payload.setdefault("behavior", {}).setdefault("amount", {})["tail_index"] = 2.5
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    resolved = resolve(path, artifact=artifact(),
                       allow_override=("behavior.amount.tail_index",))
    assert resolved.config.behavior.amount.tail_index == 2.5
    assert resolved.provenance_of("behavior.amount.tail_index") is Provenance.FREE


@needs_artifact
def test_swept_values_carry_their_default() -> None:
    resolved = resolve(CONFIG, artifact=artifact())
    entry = artifact().swept["geo_home_radius_km"]
    assert resolved.config.population.geo.home_radius_km == pytest.approx(entry["value"])


def test_config_is_immutable() -> None:
    config = SimulationConfig()
    with pytest.raises(ValidationError, match="frozen"):
        config.seed = 5


@needs_artifact
def test_render_separates_the_two_origins() -> None:
    text = resolve(CONFIG, artifact=artifact()).render()
    assert "fitted" in text and "swept" in text


def test_derived_counts_are_read_through_their_accessor() -> None:
    """Optional config fields are None until derived, so print them resolved.

    `fingerprint_count` is left unset in the YAML and computed from the fan-out
    target. Formatting the raw field crashes on the default config.
    """
    population = SimulationConfig().population
    assert population.fingerprint_count is None
    assert population.resolved_fingerprint_count() > 0


@pytest.mark.parametrize("command", ["show", "provenance"])
def test_settings_cli_runs(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    from fraudsim.settings.cli import main

    assert main([command]) == 0
    assert capsys.readouterr().out.strip()
