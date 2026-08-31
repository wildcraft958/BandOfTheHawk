"""World factory: config to warm simulator in one call.

Replaces the 8-line setup ritual repeated across 6 CLI files.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..settings.simulation import SimulationConfig
from ..engine.simulator import Simulator
from ..features.builder import EventBuilder
from ..features.state import FeatureStateStore
from ..protocols import AlwaysApproveScorer, RiskScorer
from ..timing.circadian import HolderClockModel
from ..world.graph import EntityGraph
from .builder import PopulationBuilder
from .warmstart import WarmStartRunner


@dataclass
class WarmWorld:
    """A populated, warmed simulator ready for use."""

    graph: EntityGraph
    simulator: Simulator
    config: SimulationConfig


def build_warm_world(
    config: SimulationConfig,
    *,
    scorer: RiskScorer | None = None,
    warm: bool = True,
    seed: int | None = None,
) -> WarmWorld:
    """Build population, wire the simulator, optionally warm-start."""
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(
        graph, states, config.engine.windows,
        HolderClockModel(config.behavior.circadian),
    )
    sim = Simulator(
        graph, config, builder,
        scorer=scorer or AlwaysApproveScorer(),
    )
    if warm:
        WarmStartRunner(sim, config, seed=seed or config.seed).run()
    return WarmWorld(graph=graph, simulator=sim, config=config)
