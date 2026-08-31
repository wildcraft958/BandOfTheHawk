"""Velocity rules and naive detection baseline.

Eight rules from published fraud-detection practice. They serve as the
baseline any learned detector must beat, and as the instrument measuring
how much ordinary traffic looks suspicious.
"""

from .definitions import VelocityRule, build_rules
from .engine import RuleReport, VelocityRuleEngine

__all__ = [
    "RuleReport",
    "VelocityRule",
    "VelocityRuleEngine",
    "build_rules",
]
