"""Tier detection, so a clean install skips what it cannot run.

The package installs a numpy-only runtime tier plus five optional extras, and
`test_import_firewall.py` enforces that separation. The suite did not honour it:
a clone with only the core dependencies produced thirteen errors rather than
thirteen skips, because the test modules import their tier at module scope.

Two traps this had to work around, both found the hard way:

* xgboost imports successfully and then fails when it loads its shared library,
  so `find_spec` says yes and the first fit dies. Detection has to be a real
  import inside try/except.
* torch bundles its own libomp and Homebrew's xgboost links another. Both in one
  process segfaults on macOS with no traceback. dyld reads DYLD_LIBRARY_PATH at
  process start, so this restarts pytest once with it set rather than pretending
  a ctypes preload fixes it, which it does not.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------- libomp guard

_OPT_OUT = "GAUNTLET_NO_OPENMP_FIX"


def _openmp_collision() -> str | None:
    """torch's bundled libomp, when a second copy would also be loaded.

    torch ships its own libomp and Homebrew's xgboost links another. Both in one
    process segfaults on macOS, taking the run down with no traceback. dyld reads
    DYLD_LIBRARY_PATH at process start, so a ctypes preload cannot fix it from
    here (tried; it still segfaults) and the variable has to be set before the
    interpreter launches.
    """
    if sys.platform != "darwin" or os.environ.get(_OPT_OUT):
        return None
    try:
        import torch
    except Exception:
        return None
    bundled = Path(torch.__file__).parent / "lib" / "libomp.dylib"
    if not bundled.is_file():
        return None
    if str(bundled.parent) in os.environ.get("DYLD_LIBRARY_PATH", "").split(":"):
        return None
    try:
        import xgboost  # noqa: F401
    except Exception:
        return None  # only one of the two is present, so nothing can collide
    return str(bundled.parent)


def _fail_on_openmp_collision() -> None:
    """Turn a bare segmentation fault into an instruction.

    Re-execing pytest with the variable set was tried and lost the output, so
    this reports instead of pretending to fix it. Linux is unaffected, which is
    where CI runs the full suite.
    """
    lib = _openmp_collision()
    if lib is None:
        return
    raise pytest.UsageError(
        "torch and xgboost each bring their own libomp, and loading both in one "
        "process segfaults on macOS.\n"
        f"Re-run with a single copy pinned:\n\n"
        f"    DYLD_LIBRARY_PATH={lib} python -m pytest\n\n"
        "Or run one tier at a time, which is what CI does. "
        f"Set {_OPT_OUT}=1 to skip this check."
    )


_fail_on_openmp_collision()


# ------------------------------------------------------------ tier detection


def _usable(module: str) -> bool:
    """Whether a dependency both imports and initialises.

    Broad except on purpose: a missing package raises ImportError, but a package
    whose shared library will not load raises whatever that library chose, and
    both mean the same thing to a test.
    """
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


HAVE_TORCH = _usable("torch")
HAVE_XGBOOST = _usable("xgboost")
HAVE_SKLEARN = _usable("sklearn")
HAVE_PANDAS = _usable("pandas")
HAVE_NETWORKX = _usable("networkx")
HAVE_TRANSFORMERS = _usable("transformers")


def _extra(name: str, have: bool):
    return pytest.mark.skipif(have is False, reason=f'install the "{name}" extra')


requires_torch = _extra("rl", HAVE_TORCH)
requires_xgboost = _extra("defender", HAVE_XGBOOST)
requires_sklearn = _extra("defender", HAVE_SKLEARN)
requires_pandas = _extra("calibration", HAVE_PANDAS)
requires_networkx = _extra("analysis", HAVE_NETWORKX)
requires_transformers = _extra("generative", HAVE_TRANSFORMERS)


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Say which tiers are present, so a skip count is never a mystery."""
    tiers = {
        "rl (torch)": HAVE_TORCH,
        "defender (xgboost/sklearn)": HAVE_XGBOOST and HAVE_SKLEARN,
        "calibration (pandas)": HAVE_PANDAS,
        "analysis (networkx)": HAVE_NETWORKX,
        "generative (transformers)": HAVE_TRANSFORMERS,
    }
    present = ", ".join(name for name, ok in tiers.items() if ok) or "none"
    absent = ", ".join(name for name, ok in tiers.items() if not ok) or "none"
    lines = [f"tiers present: {present}", f"tiers absent : {absent}"]
    if sys.platform == "darwin" and os.environ.get("DYLD_LIBRARY_PATH"):
        lines.append("openmp: single copy pinned via DYLD_LIBRARY_PATH")
    return lines


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_tier(name): requires an optional dependency extra"
    )


# Runtime-tier tests must keep working with nothing optional installed, so the
# suite refuses to pass vacuously if a mistake ever skips everything.
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get("GAUNTLET_ALLOW_EMPTY_RUN"):
        return
    if items:
        return
    raise pytest.UsageError("collected no tests; the suite would pass vacuously")
