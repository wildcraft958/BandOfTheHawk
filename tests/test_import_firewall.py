"""The runtime tier must not reach any of the heavy tiers.

A stray `import pandas` or `import torch` inside the simulation path would
resolve fine in a dev environment where every tier is installed, and only fail
much later in a clean install. This test makes that failure immediate.

The learned components (defender, attacker, generative) live behind the Protocol
seams and carry their own heavy dependencies. They are exempt from the check,
but the simulation path that runs without them must never import them.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "fraudsim"

FORBIDDEN = {
    "pandas": "calibration",
    "pyarrow": "calibration",
    "networkx": "analysis",
    "matplotlib": "analysis",
    "sklearn": "defender",
    "xgboost": "defender",
    "torch": "rl",
    "transformers": "generative",
    "accelerate": "generative",
}

EXEMPT_SUBPACKAGES = {
    "analysis",
    "calibration",
    "defender",
    "attacker",
    "generative",
    "orchestration",
}

RUNTIME_MODULES = [
    "fraudsim",
    "fraudsim.ids",
    "fraudsim.clock",
    "fraudsim.rng",
    "fraudsim.protocols",
    "fraudsim.config.base",
]


def _runtime_sources() -> list[Path]:
    out = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative.parts and relative.parts[0] in EXEMPT_SUBPACKAGES:
            continue
        out.append(path)
    return out


def _imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _runtime_sources(), ids=lambda p: p.name)
def test_runtime_modules_avoid_upper_tiers(path: Path) -> None:
    offenders = _imported_roots(path.read_text(encoding="utf-8")) & FORBIDDEN.keys()
    assert not offenders, (
        f"{path.relative_to(PACKAGE)} imports {sorted(offenders)}, which belong to the "
        f"{FORBIDDEN[sorted(offenders)[0]]} tier"
    )


@pytest.mark.parametrize("module", RUNTIME_MODULES)
def test_runtime_modules_import_cleanly(module: str) -> None:
    __import__(module)
    assert module in sys.modules
