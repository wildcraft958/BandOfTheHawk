"""The flags and the config loading every entry point shares.

Each of the eleven subpackage CLIs declared `--config` and `--artifact` for
itself and then repeated the same four lines to turn them into a
`SimulationConfig`. The flags drifted: two of them offered `--holders` and a
third did not, and only `orchestration` ever exposed `--seed`, so a run could
not be reseeded from anywhere else.

Subpackages keep owning their own subcommands. Only the shared part lives here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from .calibration.artifact import FittedParams
from .logs import configure
from .paths import DEFAULT_ARTIFACT, DEFAULT_CONFIG
from .settings.simulation import ResolvedConfig, resolve

if TYPE_CHECKING:
    from .settings.simulation import SimulationConfig


def base_parser(prog: str, description: str | None = None) -> argparse.ArgumentParser:
    """A parser carrying the flags every entry point accepts."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    add_config_flags(parser)
    parser.add_argument(
        "--log-level", default=None,
        help="diagnostic verbosity on stderr (DEBUG, INFO, WARNING, ERROR). "
             "Report output on stdout is unaffected",
    )
    return parser


def add_config_flags(parser: argparse.ArgumentParser) -> None:
    """Where to read the configuration and the calibration artifact."""
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="simulation config YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--artifact", type=Path, default=DEFAULT_ARTIFACT,
        help="calibration artifact; ignored when absent (default: %(default)s)",
    )


# Flag name -> the config path it overrides, and whether zero counts as a value.
# Seed 0 is the configured default and must override; a population of zero is
# not a size, so it is treated as unset, which is what the call sites did.
_OVERRIDE_PATHS = {
    "holders": (("population", "n_holders"), False),
    "fraud_rate": (("engine", "fraud_base_rate"), False),
    "seed": (("seed",), True),
}

# Marker recording which overrides a subparser actually declared. Reading the
# namespace with getattr instead would capture unrelated flags that merely share
# a name: `rules rate --seed` and `timing gate --seed` seed a local demo RNG,
# not the simulation config, and must never become a config override.
_DECLARED = "_declared_overrides"


def add_scale_flags(
    parser: argparse.ArgumentParser,
    *,
    holders: bool = True,
    fraud_rate: bool = False,
    seed: bool = True,
) -> None:
    """The overrides a subcommand may apply on top of the resolved config."""
    declared: list[str] = []
    if holders:
        parser.add_argument("--holders", type=int, default=None,
                            help="override population.n_holders")
        declared.append("holders")
    if fraud_rate:
        parser.add_argument("--fraud-rate", type=float, default=None,
                            help="override engine.fraud_base_rate")
        declared.append("fraud_rate")
    if seed:
        parser.add_argument(
            "--seed", type=int, default=None,
            help="override the config seed. A single run is one sample from a "
                 "heavy-tailed distribution, so any claim comparing two "
                 "configurations needs several seeds each",
        )
        declared.append("seed")
    parser.set_defaults(**{_DECLARED: tuple(declared)})


def overrides_from(args: argparse.Namespace) -> dict | None:
    """The override tree implied by the scale flags this subcommand declared."""
    overrides: dict = {}
    for name in getattr(args, _DECLARED, ()):
        value = getattr(args, name, None)
        path, zero_counts = _OVERRIDE_PATHS[name]
        if value is None or (not zero_counts and not value):
            continue
        if len(path) == 1:
            overrides[path[0]] = value
        else:
            overrides.setdefault(path[0], {})[path[1]] = value
    return overrides or None


def apply_log_level(args: argparse.Namespace) -> None:
    """Apply --log-level before a subcommand produces any output."""
    configure(level=getattr(args, "log_level", None))


def load_artifact(args: argparse.Namespace) -> FittedParams | None:
    """The calibration artifact, or None when it has not been generated yet."""
    return FittedParams.load(args.artifact) if args.artifact.exists() else None


def load_resolved(args: argparse.Namespace) -> ResolvedConfig:
    """Config plus its provenance ledger, for callers that report on origins."""
    apply_log_level(args)
    return resolve(args.config, artifact=load_artifact(args),
                   overrides=overrides_from(args))


def load_config(args: argparse.Namespace) -> SimulationConfig:
    """The validated config a subcommand should run against."""
    return load_resolved(args).config
