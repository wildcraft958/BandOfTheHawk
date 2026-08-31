"""Mitigation closes the loop at the level of the world.

A score alone changes nothing; these tests prove the mitigations mutate the
graph so the attacker's next action genuinely fails — a deleted binding is gone,
a frozen card refuses, a blocklisted device is unusable. This is the mitigation
half of the edge symmetry the design is built on.
"""

from __future__ import annotations

import pytest

from fraudsim.settings.simulation import SimulationConfig
from fraudsim.engine.actions import Action, ActionName
from fraudsim.engine.outcome import OutcomeCode
from fraudsim.engine.simulator import Actor, ActorKind, Simulator
from fraudsim.engine.stages import Stage
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.engine.mitigation import (
    BlocklistDevice,
    DetachPayee,
    FreezeCard,
    UnbindDevice,
    apply_all,
)
from fraudsim.population.builder import PopulationBuilder
from fraudsim.protocols import RiskAction, RiskAssessment
from fraudsim.world.entities import CardStatus


@pytest.fixture
def world():
    config = SimulationConfig.model_validate({"population": {"n_holders": 300}})
    graph, _ = PopulationBuilder(config).build()
    return graph, config


def _single_device_card(graph):
    for c in graph.cards:
        ds = graph.devices_of_card(c)
        if len(ds) == 1:
            return c, next(iter(ds))
    pytest.skip("no single-device card in this world")


def _sim_with(graph, config, scorer):
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    return Simulator(graph, config, builder, scorer=scorer)


def test_unbind_deletes_the_edge(world):
    graph, _ = world
    card_id, device_id = next(iter(graph.provisioned))
    assert device_id in graph.devices_of_card(card_id)
    assert UnbindDevice(int(card_id), int(device_id)).apply(graph, now=0)
    assert device_id not in graph.devices_of_card(card_id)


def test_freeze_makes_card_unusable(world):
    graph, _ = world
    card_id = next(iter(graph.cards))
    assert FreezeCard(int(card_id), hours=24).apply(graph, now=1000)
    card = graph.cards[card_id]
    assert card.status is CardStatus.FROZEN
    assert not card.is_usable(1000)
    assert card.is_usable(1000 + 24 * 60)  # usable again past the horizon


def test_blocklist_marks_device(world):
    graph, _ = world
    device_id = next(iter(graph.devices))
    assert BlocklistDevice(int(device_id)).apply(graph, now=0)
    assert graph.devices[device_id].blocklisted


def test_blocklist_fails_the_next_auth(world):
    graph, config = world
    card_id, device_id = _single_device_card(graph)
    holder = int(graph.cards[card_id].holder_id)

    class BlocklistOnce:
        fired = False

        def score(self, event):
            if not self.fired:
                self.fired = True
                return RiskAssessment(
                    0.99, RiskAction.APPROVE, mitigations=(BlocklistDevice(int(event.device_id)),)
                )
            return RiskAssessment(0.0, RiskAction.APPROVE)

    sim = _sim_with(graph, config, BlocklistOnce())
    sim.register_actor(
        Actor(actor_id=1, kind=ActorKind.ADVERSARIAL, holder_id=holder, cards=[card_id], stage=Stage.BOUND)
    )
    merchant = int(next(iter(graph.merchants)))
    first = sim.step(
        1, Action(name=ActionName.ATTEMPT_AUTH, target_id=int(card_id), secondary_id=merchant, amount=50.0, device_id=int(device_id))
    )
    second = sim.step(
        1, Action(name=ActionName.ATTEMPT_AUTH, target_id=int(card_id), secondary_id=merchant, amount=50.0)
    )
    assert first.code is OutcomeCode.APPROVED
    # The only device is blocklisted, so there is no usable binding left.
    assert second.code is OutcomeCode.FAILED


def test_apply_all_counts_real_mutations(world):
    graph, _ = world
    card_id = next(iter(graph.cards))
    # One real freeze, one no-op on a missing card.
    n = apply_all(
        [FreezeCard(int(card_id)), FreezeCard(-1)],
        graph,
        now=0,
    )
    assert n == 1
