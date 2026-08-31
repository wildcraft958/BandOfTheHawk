"""The two dependency files agree, and every pin is installable.

They did not. pyproject.toml asked for numpy>=2.5 while requirements.txt pinned
numpy==2.0.2, and neither that nor scipy==1.14.1 publishes a wheel for the
interpreter this project requires, so the documented install failed at its first
command. Not one of the eleven pins matched what was actually installed and
working.

pyproject.toml is the contract (lower bounds). requirements.txt is one exact set
that reproduces a run. These tests keep the second from contradicting the first.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"

_PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)")
_BOUND = re.compile(r"^([A-Za-z0-9_.\-]+)>=([^\s,]+)")


def normalise(name: str) -> str:
    """PEP 503 name comparison, so PyYAML and pyyaml are one package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def pins() -> dict[str, str]:
    return {
        normalise(m.group(1)): m.group(2)
        for line in REQUIREMENTS.read_text().splitlines()
        if (m := _PIN.match(line.strip()))
    }


def lower_bounds() -> dict[str, str]:
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    specs = list(project["dependencies"])
    for group in project["optional-dependencies"].values():
        specs.extend(group)
    return {
        normalise(m.group(1)): m.group(2)
        for spec in specs
        if (m := _BOUND.match(spec))
    }


PINS = pins()
BOUNDS = lower_bounds()


def test_both_files_declare_something() -> None:
    """Guards the guard: empty parses would make every case below vacuous."""
    assert len(PINS) > 15
    assert len(BOUNDS) > 10


@pytest.mark.parametrize("package", sorted(set(PINS) & set(BOUNDS)))
def test_every_pin_satisfies_its_lower_bound(package: str) -> None:
    pinned, minimum = PINS[package], BOUNDS[package]
    assert version_tuple(pinned) >= version_tuple(minimum), (
        f"requirements.txt pins {package}=={pinned} but pyproject.toml requires "
        f">={minimum}. Installing both would conflict."
    )


def test_every_declared_dependency_is_pinned() -> None:
    """A dependency with no pin is a version the reproduction does not fix."""
    missing = sorted(set(BOUNDS) - set(PINS))
    assert not missing, f"declared in pyproject but not pinned: {missing}"


def test_the_core_tier_is_pinned() -> None:
    """These four are what the numpy-only runtime needs."""
    for package in ("numpy", "scipy", "pydantic", "pyyaml"):
        assert package in PINS, f"{package} is not pinned"


def test_requirements_says_which_file_is_the_contract() -> None:
    """The distinction is the point; losing the note loses the point."""
    text = REQUIREMENTS.read_text()
    assert "pyproject.toml is" in text
    assert "NOT the dependency contract" in text
