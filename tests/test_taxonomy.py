"""One registry defines the verticals, and every consumer agrees with it.

A vertical's name used to be a bare string in four unconnected places: the
scripted policy, the prompt template, the generator's tool map and skeleton, and
the benign counterpart the episode runner injects. Adding one meant editing all
four and nothing checked you had. These tests are that check.
"""

from __future__ import annotations

import pytest

from fraudsim import taxonomy
from fraudsim.attacker.scripted import VERTICALS, ZERO_SHOT_HOLDOUTS
from fraudsim.engine.actions import ACTION_INDEX, ActionName
from fraudsim.generative.pool import MockGenerator, PoolArtifactSource
from fraudsim.generative.prompts import DETAILS, PROMPT_FOR_VERTICAL
from fraudsim.orchestration.run import EpisodeRunner


def test_every_policy_declares_a_registered_vertical() -> None:
    """A policy naming an unregistered vertical would never get text or a holdout."""
    assert set(VERTICALS) == set(taxonomy.ALL_VERTICALS)


def test_every_registered_vertical_has_a_policy() -> None:
    """The reverse: a registry entry with no policy is an attack nothing runs."""
    for name in taxonomy.ALL_VERTICALS:
        assert name in VERTICALS, f"{name} is registered but has no scripted policy"


def test_holdouts_come_from_the_registry() -> None:
    assert ZERO_SHOT_HOLDOUTS == taxonomy.HELD_OUT
    assert taxonomy.HELD_OUT, "at least one vertical must be held out for zero-shot"
    assert not taxonomy.HELD_OUT - set(taxonomy.ALL_VERTICALS)


@pytest.mark.parametrize("vertical", taxonomy.TEXT_VERTICALS)
def test_every_text_vertical_has_its_full_set(vertical: str) -> None:
    """Prompt, details, tool, skeleton and a benign counterpart, or it breaks mid-run."""
    assert vertical in PROMPT_FOR_VERTICAL, f"{vertical} has no prompt template"
    assert vertical in DETAILS, f"{vertical} has no prompt details"
    assert vertical in taxonomy.SKELETONS, f"{vertical} has no fallback skeleton"
    assert vertical in taxonomy.TOOL_TO_VERTICAL.values(), f"{vertical} has no tool"
    assert vertical in dict(taxonomy.TEXT_ACTIONS), f"{vertical} has no benign action"


def test_consumers_use_the_derived_mappings() -> None:
    """The four copies are gone, not merely equal by coincidence."""
    assert PoolArtifactSource._TOOL_TO_VERTICAL is taxonomy.TOOL_TO_VERTICAL
    assert MockGenerator._SKELETON is taxonomy.SKELETONS
    assert EpisodeRunner._BENIGN_TEXT_ACTIONS is taxonomy.TEXT_ACTIONS


def test_text_actions_name_real_actions() -> None:
    """A benign counterpart the engine cannot resolve would fail at injection."""
    for vertical, action in taxonomy.TEXT_ACTIONS:
        assert ActionName(action) in ACTION_INDEX, f"{vertical} names unknown {action}"


def test_a_text_vertical_without_its_assets_is_rejected() -> None:
    with pytest.raises(ValueError, match="tool or skeleton"):
        taxonomy.VerticalSpec(
            taxonomy.Vertical.CARD_TESTING, text_action=ActionName.FILE_DISPUTE
        )


def test_a_non_text_vertical_declaring_text_assets_is_rejected() -> None:
    with pytest.raises(ValueError, match="no text action"):
        taxonomy.VerticalSpec(taxonomy.Vertical.CARD_TESTING, tool="write_dispute")


def test_vertical_values_compare_equal_to_their_names() -> None:
    """A str enum, so every call site that passed a plain string still works."""
    assert taxonomy.Vertical.REFUND_ABUSE == "refund_abuse"
    assert taxonomy.spec_for("refund_abuse").held_out is True
