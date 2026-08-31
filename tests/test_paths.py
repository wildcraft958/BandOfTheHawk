"""Project locations resolve once, and the environment can move them."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from fraudsim import paths


def reloaded(monkeypatch: pytest.MonkeyPatch, **env: str) -> ModuleType:
    """The paths module re-imported under a patched environment."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(paths)


@pytest.fixture(autouse=True)
def _restore() -> Iterator[None]:
    """Leave the module as the rest of the suite expects to find it."""
    yield
    importlib.reload(paths)


def test_root_contains_the_configs_directory() -> None:
    assert (paths.PROJECT_ROOT / "configs").is_dir()
    assert paths.DEFAULT_CONFIG.is_file()


def test_defaults_hang_off_their_directory() -> None:
    assert paths.DEFAULT_ARTIFACT.parent == paths.ARTIFACT_DIR
    assert paths.DEFAULT_POOL.parent == paths.ARTIFACT_DIR
    assert paths.ABLATION_DIR.parent == paths.ARTIFACT_DIR
    assert paths.DEFAULT_CFPB.is_relative_to(paths.DATASET_DIR)


def test_artifact_dir_follows_its_environment_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = reloaded(monkeypatch, GAUNTLET_ARTIFACTS=str(tmp_path))
    assert p.ARTIFACT_DIR == tmp_path.resolve()
    assert p.DEFAULT_ARTIFACT == tmp_path.resolve() / "fitted_params.json"


def test_directories_move_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pointing artifacts elsewhere must not drag the config with it."""
    p = reloaded(monkeypatch, GAUNTLET_ARTIFACTS=str(tmp_path))
    assert p.CONFIG_DIR == p.PROJECT_ROOT / "configs"


def test_root_relocates_every_derived_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = reloaded(monkeypatch, GAUNTLET_ROOT=str(tmp_path))
    assert p.CONFIG_DIR == tmp_path.resolve() / "configs"
    assert p.ARTIFACT_DIR == tmp_path.resolve() / "artifacts"
    assert p.DATASET_DIR == tmp_path.resolve() / "Dataset"
