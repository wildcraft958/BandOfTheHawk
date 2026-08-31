"""Every filesystem location the project reads or writes, resolved once.

Thirteen modules used to derive the project root from their own `__file__`
depth, and nine of them repeated the same two default paths verbatim. Moving
the package changed the depth in all thirteen at once, which is the kind of
edit that is easy to get almost right.

Each location honours an environment variable so an installed copy, a CI job,
or a run against a dataset held elsewhere does not need the repo laid out in
any particular way:

    GAUNTLET_ROOT       project root (default: the repo containing this file)
    GAUNTLET_CONFIGS    configuration directory
    GAUNTLET_ARTIFACTS  generated and committed artifacts
    GAUNTLET_DATASET    real datasets, which are not committed
"""

from __future__ import annotations

import os
from pathlib import Path


def _from_env(name: str, default: Path) -> Path:
    """The path named by an environment variable, else the default."""
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


# src/fraudsim/paths.py -> src/fraudsim -> src -> project root
PROJECT_ROOT = _from_env("GAUNTLET_ROOT", Path(__file__).resolve().parents[2])

CONFIG_DIR = _from_env("GAUNTLET_CONFIGS", PROJECT_ROOT / "configs")
ARTIFACT_DIR = _from_env("GAUNTLET_ARTIFACTS", PROJECT_ROOT / "artifacts")
DATASET_DIR = _from_env("GAUNTLET_DATASET", PROJECT_ROOT / "Dataset")

DEFAULT_CONFIG = CONFIG_DIR / "simulation.yaml"
DEFAULT_LOGGING = CONFIG_DIR / "logging.yaml"

DEFAULT_ARTIFACT = ARTIFACT_DIR / "fitted_params.json"
DEFAULT_FLOORS = ARTIFACT_DIR / "noise_floors.json"
DEFAULT_POOL = ARTIFACT_DIR / "text_pool.json"
DEFAULT_METRICS = ARTIFACT_DIR / "coadapt_metrics.json"
DEFAULT_CHECKPOINTS = ARTIFACT_DIR / "checkpoints"
ABLATION_DIR = ARTIFACT_DIR / "ablation"

DEFAULT_CFPB = DATASET_DIR / "complaints" / "cfpb_payments_all.parquet"

__all__ = [
    "ABLATION_DIR",
    "ARTIFACT_DIR",
    "CONFIG_DIR",
    "DATASET_DIR",
    "DEFAULT_ARTIFACT",
    "DEFAULT_CFPB",
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_CONFIG",
    "DEFAULT_FLOORS",
    "DEFAULT_LOGGING",
    "DEFAULT_METRICS",
    "DEFAULT_POOL",
    "PROJECT_ROOT",
]
