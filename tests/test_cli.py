"""Shared CLI flags resolve into config overrides, and only where declared."""

from __future__ import annotations

import argparse

import pytest

from fraudsim.cli import add_scale_flags, base_parser, load_config, overrides_from
from fraudsim.paths import DEFAULT_ARTIFACT, DEFAULT_CONFIG


def parse(argv: list[str], **flags: bool) -> argparse.Namespace:
    parser = base_parser("test")
    add_scale_flags(parser, **flags)
    return parser.parse_args(argv)


def test_base_parser_defaults_come_from_paths() -> None:
    args = base_parser("test").parse_args([])
    assert args.config == DEFAULT_CONFIG
    assert args.artifact == DEFAULT_ARTIFACT


def test_no_flags_means_no_overrides() -> None:
    assert overrides_from(parse([])) is None


def test_holders_becomes_a_nested_override() -> None:
    assert overrides_from(parse(["--holders", "500"])) == {
        "population": {"n_holders": 500}
    }


def test_seed_zero_still_overrides() -> None:
    """Zero is the configured default, so it has to survive as an override."""
    assert overrides_from(parse(["--seed", "0"])) == {"seed": 0}


def test_holders_zero_is_treated_as_unset() -> None:
    """A population of zero is not a size; the call sites always ignored it."""
    assert overrides_from(parse(["--holders", "0"])) is None


def test_flags_combine_into_one_tree() -> None:
    args = parse(["--holders", "40", "--fraud-rate", "0.05", "--seed", "3"],
                 fraud_rate=True)
    assert overrides_from(args) == {
        "population": {"n_holders": 40},
        "engine": {"fraud_base_rate": 0.05},
        "seed": 3,
    }


def test_undeclared_flags_are_not_offered() -> None:
    parser = base_parser("test")
    add_scale_flags(parser, fraud_rate=False)
    with pytest.raises(SystemExit):
        parser.parse_args(["--fraud-rate", "0.1"])


def test_a_same_named_flag_is_not_hijacked_as_an_override() -> None:
    """`rules rate --seed` and `timing gate --seed` seed a local demo RNG.

    Those subcommands never called add_scale_flags, so their --seed must stay
    their own and never reach the config.
    """
    parser = base_parser("test")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(["--seed", "99"])
    assert args.seed == 99
    assert overrides_from(args) is None


def test_overrides_reach_the_resolved_config() -> None:
    assert load_config(parse(["--holders", "700"])).population.n_holders == 700
