"""Every configured field is read by something.

The complaint this restructure started from was config that does nothing: a
value declared in the YAML, validated on load, and then ignored because the code
carried its own copy. Four fields were in that state
(`step_up_challenge_rate`, `step_up_abandon_rate`, `document_forensic_threshold`
and `check_invariants_every`), and one more was actively contradicted
(`manual_review_cost`, against a hardcoded `CostModel(review_cost=8.0)`).

Catching that by reading is unreliable, so it is checked. A field that nothing
outside `settings/` names is either dead and should go, or is a feature that was
described and never built and should be listed below as such.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from fraudsim.settings.simulation import SimulationConfig

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "fraudsim"

# Fields the config declares that the package does not name directly, each with
# the reason. Anything added here needs a reason a reviewer would accept, and the
# first two below are the only kind that is actually fine.
ALLOWED_UNREAD: dict[str, str] = {
    # Read through an accessor rather than directly: the field is optional and
    # derived from the fan-out target when unset, so resolved_fingerprint_count()
    # is the interface and reading the raw field is the bug (it was, once).
    "population.fingerprint_count": "read via resolved_fingerprint_count()",

    # Populated by the calibration artifact as a swept route
    # (SWEPT_ROUTES -> amount_by_category_spread) but not yet consumed by the
    # amount model. The sweep records a value nothing acts on. Left in place
    # because removing it would change the artifact contract; it should either
    # be wired into behavior.amount or dropped from SWEPT_ROUTES.
    "behavior.amount.category_spread": "swept into the artifact, not yet consumed",

    # Declared and validated, never implemented. Pre-existing; listed rather than
    # deleted so the gap is visible instead of silently carried.
    "behavior.circadian.min_history_days": "declared, never implemented",
    "population.households.single_occupant_share": "declared, never implemented",
}


def leaf_fields(model: type[BaseModel], prefix: str = "") -> dict[str, str]:
    """Every scalar field in the config tree, as dotted path -> attribute name."""
    found: dict[str, str] = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        nested = (
            annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel)
            else None
        )
        path = f"{prefix}{name}"
        if nested is not None:
            found.update(leaf_fields(nested, prefix=f"{path}."))
        else:
            found[path] = name
    return found


def names_used_outside_settings() -> set[str]:
    """Every identifier and string the package mentions, excluding settings/."""
    used: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(PACKAGE).parts
        if "__pycache__" in parts or parts[0] == "settings":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                used.add(node.value)
            elif isinstance(node, ast.keyword) and node.arg:
                used.add(node.arg)
    return used


ALL_FIELDS = leaf_fields(SimulationConfig)


def test_the_config_tree_is_not_empty() -> None:
    """Guards the guard: a broken walker would make every case below vacuous."""
    assert len(ALL_FIELDS) > 100


@pytest.mark.parametrize("dotted", sorted(ALL_FIELDS))
def test_every_configured_field_is_read_somewhere(dotted: str) -> None:
    attribute = ALL_FIELDS[dotted]
    if dotted in ALLOWED_UNREAD:
        pytest.skip(ALLOWED_UNREAD[dotted])
    assert attribute in names_used_outside_settings(), (
        f"{dotted} is declared and validated but nothing outside settings/ reads "
        f"it. Either wire it up, delete it, or record why in ALLOWED_UNREAD."
    )
