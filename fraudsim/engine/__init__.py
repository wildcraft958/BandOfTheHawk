"""Simulation loop, action resolution, and outcome reporting.

The simulator steps one action at a time in a fixed order: legality check,
artifact acquisition, world mutation, event construction, scoring, and
outcome reporting. Ordinary holders and attackers go through the same path
so the events carry no structural tell.
"""

from .actions import Action, ActionName, action_cost
from .outcome import Outcome, OutcomeCode
from .resolution import ActionResolver
from .simulator import Simulator

__all__ = [
    "Action",
    "ActionName",
    "ActionResolver",
    "Outcome",
    "OutcomeCode",
    "Simulator",
    "action_cost",
]
