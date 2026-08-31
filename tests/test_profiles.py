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
    post means the paired comparison is built from. An arm run on a different
    refit cadence is summarised across the wrong updates, and reports a paired
    difference that looks entirely sound and is not.
    """
    from fraudsim.orchestration.ablation import REFIT_AT

    profile = yaml.safe_load(PROFILES.read_text())["ablation"]
    assert profile["refit_every"] == REFIT_AT
    # And the run has to be long enough for a post-refit block to exist at all.
    assert profile["updates"] > REFIT_AT


# The co-adaptation configuration reported in the solution document, Table 9.
# The paired comparison in Section 8 is only reproducible if `--profile ablation`
# resolves to exactly these, so they are pinned rather than described.
PUBLISHED_ABLATION = {
    "holders": 600,
    "updates": 24,
    "episodes_per_update": 12,
    "refit_every": 6,
    "label_latency": 2880,
    "fraud_rounds": 3,
    "target_prevalence": 0.02,
    "demo_episodes": 40,
    "bc_epochs": 6,
    "critic_rollouts": 16,
    "critic_epochs": 8,
    "hidden": 256,
    "minibatch": 256,
}


@pytest.mark.parametrize("key", sorted(PUBLISHED_ABLATION))
def test_ablation_profile_matches_the_published_table(key: str) -> None:
    profile = yaml.safe_load(PROFILES.read_text())["ablation"]
    assert profile[key] == PUBLISHED_ABLATION[key], (
        f"--profile ablation sets {key}={profile[key]}, but the solution document "
        f"reports {PUBLISHED_ABLATION[key]} for the runs behind Section 8."
    )


def test_published_dump_size_and_candidates_come_from_config() -> None:
    """Table 9's dump size and selection candidates are config, not profile keys.

    The ablation profile does not set them, so they have to resolve from
    configs/simulation.yaml or the published run cannot be reproduced from the
    profile alone.
    """
    from fraudsim.settings.simulation import resolve

    loop = resolve().config.training.loop
    assert loop.dump_size == 3
    assert loop.candidates == 5
