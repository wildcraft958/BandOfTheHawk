"""The fraud verticals, and the facts each one carries.

A vertical's name was written as a bare string in up to four files that had no
link between them: the scripted policy that drives it, the prompt template that
writes its text, the generator's tool name and fallback skeleton, and the benign
counterpart the episode runner injects. `"refund_abuse"` appeared seven times
across four files. Adding a vertical meant editing all of them, and nothing
checked that you had.

One registry, and the four mappings are derived from it. A vertical that needs
text but declares no prompt key now fails at import rather than at run time.

Only the taxonomy lives here, never behaviour: the policy classes stay in
`attacker.scripted`, the templates in `generative.prompts`. This module sits on
the runtime side of the import firewall and imports nothing heavier than the
action enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .engine.actions import ActionName


class Vertical(StrEnum):
    """The attack families the simulation covers.

    A StrEnum so a value compares equal to the name it replaced; every existing
    call site that passed a plain string keeps working.
    """

    CARD_TESTING = "card_testing"
    VOICE_CLONE = "voice_clone"
    DEEPFAKE_ONBOARDING = "deepfake_onboarding"
    PHISHING_ATO = "phishing_ato"
    SIM_SWAP = "sim_swap"
    MULE_LAYERING = "mule_layering"
    FRIENDLY_FRAUD = "friendly_fraud"
    SUPPORT_SE = "support_se"
    REFUND_ABUSE = "refund_abuse"


@dataclass(frozen=True, slots=True)
class VerticalSpec:
    """Everything about one vertical that is not its policy or its prompt text.

    `text_action` marks the verticals whose attack is a piece of written
    language rather than a transaction. Those are the ones the generative tier
    produces text for, and the only ones with a tool, a skeleton and a benign
    counterpart.
    """

    vertical: Vertical
    held_out: bool = False
    text_action: ActionName | None = None
    tool: str | None = None
    skeleton: str | None = None

    @property
    def is_text(self) -> bool:
        return self.text_action is not None

    @property
    def text_assets(self) -> tuple[ActionName, str, str]:
        """The action, tool and skeleton of a text vertical.

        A checker cannot narrow the three optional fields through `is_text`, and
        `__post_init__` has already guaranteed they are present together, so the
        guarantee is stated here where it can be read.
        """
        if self.text_action is None or self.tool is None or self.skeleton is None:
            raise TypeError(f"{self.vertical.value} is not a text vertical")
        return self.text_action, self.tool, self.skeleton

    def __post_init__(self) -> None:
        # A text vertical without its full set would fail later, in the middle
        # of a run, as a KeyError on a dict built somewhere else.
        if self.is_text and not (self.tool and self.skeleton):
            raise ValueError(
                f"{self.vertical.value} carries text but is missing a tool or skeleton"
            )
        if not self.is_text and (self.tool or self.skeleton):
            raise ValueError(
                f"{self.vertical.value} has no text action but declares text assets"
            )


SPECS: dict[Vertical, VerticalSpec] = {
    spec.vertical: spec
    for spec in (
        VerticalSpec(Vertical.CARD_TESTING),
        VerticalSpec(Vertical.VOICE_CLONE),
        VerticalSpec(Vertical.DEEPFAKE_ONBOARDING),
        VerticalSpec(Vertical.PHISHING_ATO),
        # Held out of training, so recall on it measures generalisation to an
        # attack family the defender has never been fitted on.
        VerticalSpec(Vertical.SIM_SWAP, held_out=True),
        VerticalSpec(Vertical.MULE_LAYERING),
        VerticalSpec(
            Vertical.FRIENDLY_FRAUD,
            text_action=ActionName.FILE_DISPUTE,
            tool="write_dispute",
            skeleton="I am writing about a charge on my account.",
        ),
        VerticalSpec(
            Vertical.SUPPORT_SE,
            text_action=ActionName.OPEN_TICKET,
            tool="write_ticket",
            skeleton="I need help with my account as soon as possible.",
        ),
        VerticalSpec(
            Vertical.REFUND_ABUSE,
            held_out=True,
            text_action=ActionName.REQUEST_REFUND,
            tool="write_refund_claim",
            skeleton="I am requesting a refund for a recent order.",
        ),
    )
}

# The four mappings that used to be written out by hand, now derived.
ALL_VERTICALS: tuple[str, ...] = tuple(v.value for v in SPECS)
HELD_OUT: frozenset[str] = frozenset(
    spec.vertical.value for spec in SPECS.values() if spec.held_out
)
TEXT_VERTICALS: tuple[str, ...] = tuple(
    spec.vertical.value for spec in SPECS.values() if spec.is_text
)
_TEXT_SPECS: tuple[VerticalSpec, ...] = tuple(
    spec for spec in SPECS.values() if spec.is_text
)
TOOL_TO_VERTICAL: dict[str, str] = {
    spec.text_assets[1]: spec.vertical.value for spec in _TEXT_SPECS
}
SKELETONS: dict[str, str] = {
    spec.vertical.value: spec.text_assets[2] for spec in _TEXT_SPECS
}
TEXT_ACTIONS: tuple[tuple[str, str], ...] = tuple(
    (spec.vertical.value, spec.text_assets[0].value) for spec in _TEXT_SPECS
)


def spec_for(vertical: str) -> VerticalSpec:
    """The registry entry for a vertical, by its name."""
    return SPECS[Vertical(vertical)]


__all__ = [
    "ALL_VERTICALS",
    "HELD_OUT",
    "SKELETONS",
    "SPECS",
    "TEXT_ACTIONS",
    "TEXT_VERTICALS",
    "TOOL_TO_VERTICAL",
    "Vertical",
    "VerticalSpec",
    "spec_for",
]
