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
    assert names == {"quick", "default", "gpu", "server"}


def test_argparse_choices_come_from_the_file() -> None:
    """Adding a profile should take one edit, not two."""
    assert set(pipeline().PROFILES) == set(yaml.safe_load(PROFILES.read_text()))


@pytest.mark.parametrize("profile", ["quick", "default", "gpu", "server"])
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
