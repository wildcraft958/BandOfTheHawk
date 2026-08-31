"""Run profiles live in configs/profiles.yaml and cover every flag a stage needs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "configs" / "profiles.yaml"


def pipeline() -> ModuleType:
    """main.py loaded as a module, since it is a script at the repo root."""
    spec = importlib.util.spec_from_file_location("pipeline_main", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profiles_file_defines_the_advertised_profiles() -> None:
    names = set(yaml.safe_load(PROFILES.read_text()))
    assert names == {"quick", "ablation", "default", "gpu", "server"}


def test_argparse_choices_come_from_the_file() -> None:
    """Adding a profile should take one edit, not two."""
    assert set(pipeline().PROFILES) == set(yaml.safe_load(PROFILES.read_text()))


@pytest.mark.parametrize("profile", ["quick", "ablation", "default", "gpu", "server"])
def test_every_profile_supplies_every_key_a_stage_reads(profile: str) -> None:
    """A missing key would surface as a KeyError mid-run, after minutes of work."""
    main = pipeline()
    scales = main._scales(profile)
    for stage in main.STAGE_ORDER + main.EXTRA_STAGES:
        for use_models in (True, False):
            module, argv = main._stage_args(stage, scales, use_models)
            assert module.startswith("fraudsim.")
            assert all(isinstance(a, str) for a in argv)


def test_an_unknown_profile_fails_with_the_available_names() -> None:
    with pytest.raises(SystemExit) as caught:
        pipeline()._scales("enormous")
    assert "quick" in str(caught.value)


def test_sizes_increase_with_the_profile() -> None:
    main = pipeline()
    holders = [main._scales(p)["holders"] for p in ("quick", "default", "gpu", "server")]
    assert holders == sorted(holders)


def test_ablation_profile_refits_where_the_reader_splits() -> None:
    """The ablation profile and the ablation reader must agree on the first refit.

    `orchestration.ablation` splits every curve at REFIT_AT to form the pre and
    post means that the paired comparison is built from. If a run refits on a
    different cadence the split lands mid-block, and the reported difference is
    computed over the wrong updates while still looking entirely plausible.
    That is how this went unnoticed: the published comparison needs 600 holders
    and a refit every 6, and no profile supplied both, so the documented
    commands could not reproduce Section 8.
    """
    from fraudsim.orchestration.ablation import REFIT_AT

    profile = yaml.safe_load(PROFILES.read_text())["ablation"]
    assert profile["refit_every"] == REFIT_AT
    # And the run has to be long enough for a post-refit block to exist at all.
    assert profile["updates"] > REFIT_AT
