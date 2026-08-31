"""The referee: legality, ordering, and what an actor is told."""

from __future__ import annotations

import pytest

from fraudsim.engine.actions import (
    ACTION_INDEX,
    ACTION_SPECS,
    N_ACTIONS,
    Action,
    ActionName,
    action_cost,
)
from fraudsim.engine.outcome import OutcomeCode
from fraudsim.engine.simulator import Actor, ActorKind, Simulator
from fraudsim.engine.stages import LEGAL_ACTIONS, Stage, StageGate
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.population.builder import PopulationBuilder
from fraudsim.rules.engine import VelocityRuleScorer
from fraudsim.settings.simulation import SimulationConfig


@pytest.fixture
def sim():
    config = SimulationConfig.model_validate({"population": {"n_holders": 300}})
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    simulator = Simulator(
        graph, config, builder, scorer=VelocityRuleScorer(config.engine.rules)
    )
    card_id, _ = next(iter(graph.provisioned))
    merchants = list(graph.merchants)
    actor = simulator.register_actor(
        Actor(
            actor_id=1, kind=ActorKind.BENIGN,
            holder_id=graph.cards[card_id].holder_id, cards=[card_id],
            stage=Stage.BOUND,
        )
    )
    return simulator, actor, card_id, merchants


def auth(sim_bundle, amount=50.0, delay=60, merchant_index=0):
    simulator, _, card_id, merchants = sim_bundle
    return simulator.step(
        1,
        Action(
            name=ActionName.ATTEMPT_AUTH, target_id=card_id,
            secondary_id=merchants[merchant_index % len(merchants)],
            amount=amount, delay_minutes=delay,
        ),
    )


# ------------------------------------------------------------------ actions


def test_the_action_space_is_fixed_at_twenty() -> None:
    """The width sets the legality mask and any policy head that later chooses
    among these, so changing it after those exist means rebuilding both."""
    assert N_ACTIONS == 20
    assert len(ACTION_INDEX) == 20


def test_merchant_collusion_actions_are_absent() -> None:
    """That vertical needs a settlement and clawback model the money layer does
    not implement, so it is described rather than run."""
    names = {name.value for name in ActionName}
    assert not names & {"onboard_merchant", "self_auth", "bust_out"}


def test_every_action_costs_something() -> None:
    """A free action invites a policy to spam it, and the result says more
    about the reward than about fraud."""
    assert all(action_cost(name) > 0 for name in ActionName)
    assert len(ACTION_SPECS) == N_ACTIONS


# ------------------------------------------------------------------- stages


def test_stage_gate_permits_only_its_own_actions() -> None:
    gate = StageGate()
    assert gate.is_legal(Stage.NONE, ActionName.BUY_CREDS)
    assert not gate.is_legal(Stage.NONE, ActionName.ATTEMPT_AUTH)
    assert gate.is_legal(Stage.BOUND, ActionName.ATTEMPT_AUTH)
    assert not gate.is_legal(Stage.BOUND, ActionName.BUY_CREDS)


def test_terminal_permits_nothing() -> None:
    assert StageGate().legal_actions(Stage.TERMINAL) == ()


def test_legal_mask_matches_the_table() -> None:
    gate = StageGate()
    for stage, allowed in LEGAL_ACTIONS.items():
        mask = gate.legal_mask(stage)
        assert mask.shape == (N_ACTIONS,)
        assert int(mask.sum()) == len(allowed)
        for name in allowed:
            assert mask[ACTION_INDEX[name]]


def test_legal_mask_is_a_copy() -> None:
    """A caller mutating a returned mask must not change the gate."""
    gate = StageGate()
    mask = gate.legal_mask(Stage.NONE)
    mask[:] = True
    assert not gate.legal_mask(Stage.NONE).all()


def test_failure_never_advances_a_stage() -> None:
    """An actor that could not provision a card still holds credentials it
    cannot spend, which is the state the attempt was trying to leave."""
    gate = StageGate()
    assert (
        gate.advance(Stage.ACQUIRED, ActionName.CALL_IVR_PROVISION, succeeded=False)
        is Stage.ACQUIRED
    )
    assert (
        gate.advance(Stage.ACQUIRED, ActionName.CALL_IVR_PROVISION, succeeded=True)
        is Stage.BOUND
    )


def test_success_without_a_transition_leaves_the_stage() -> None:
    gate = StageGate()
    assert gate.advance(Stage.BOUND, ActionName.ADD_PAYEE, succeeded=True) is Stage.BOUND


# ---------------------------------------------------------------- the step


def test_an_illegal_action_costs_nothing(sim) -> None:
    """An impossible action is not a failed attempt. Charging for one would
    teach a policy the world pushed back when it never saw the request."""
    simulator, actor, *_ = sim
    before = actor.cost_incurred
    outcome = simulator.step(1, Action(name=ActionName.BUY_CREDS))
    assert outcome.code is OutcomeCode.ILLEGAL
    assert outcome.cost == 0.0
    assert actor.cost_incurred == before
    assert len(simulator.log) == 0


def test_an_approved_authorisation_extracts_its_amount(sim) -> None:
    outcome = auth(sim, amount=80.0)
    if outcome.code is OutcomeCode.APPROVED:
        assert outcome.value_extracted == pytest.approx(80.0)
        assert outcome.reward == pytest.approx(80.0 - action_cost(ActionName.ATTEMPT_AUTH))


def test_a_refused_authorisation_extracts_nothing(sim) -> None:
    simulator, _, card_id, merchants = sim
    refused = []
    for index in range(15):
        outcome = auth(sim, amount=400.0, delay=0, merchant_index=index)
        if outcome.code is not OutcomeCode.APPROVED:
            refused.append(outcome)
    assert refused, "a rapid burst should trip something"
    assert all(o.value_extracted == 0.0 for o in refused)
    assert all(o.reward < 0 for o in refused)


def test_the_scorer_is_reached(sim) -> None:
    """A burst has to change outcomes, or the detector is not in the loop."""
    codes = [auth(sim, amount=300.0, delay=0, merchant_index=i).code for i in range(12)]
    assert any(code is not OutcomeCode.APPROVED for code in codes)


def test_every_action_emits_at_most_one_event(sim) -> None:
    simulator, *_ = sim
    for index in range(6):
        auth(sim, merchant_index=index)
    assert len(simulator.log) == 6


def test_events_are_labelled_only_when_the_episode_closes(sim) -> None:
    """Nothing knows the answer while the episode is running."""
    simulator, actor, *_ = sim
    actor.kind = ActorKind.ADVERSARIAL
    simulator.open_episode(1)
    for index in range(3):
        auth(sim, merchant_index=index)

    assert simulator.log.labelled() == []
    assert simulator.close_episode(1) == 3
    assert all(event.is_fraud for event in simulator.log.labelled())


def test_a_benign_episode_is_labelled_benign(sim) -> None:
    simulator, *_ = sim
    simulator.open_episode(1)
    auth(sim)
    simulator.close_episode(1)
    assert all(event.is_fraud is False for event in simulator.log.labelled())


def test_events_carry_no_actor_identity(sim) -> None:
    """Both kinds of actor pass through the same method, so nothing in an event
    may say which produced it."""
    simulator, actor, *_ = sim
    actor.kind = ActorKind.ADVERSARIAL
    auth(sim)
    fields = simulator.log.events[0].scoring_fields()
    assert not {"actor_id", "kind", "is_attacker"} & set(fields)


def test_the_clock_only_moves_forward(sim) -> None:
    simulator, *_ = sim
    before = simulator.clock.now
    auth(sim, delay=120)
    assert simulator.clock.now == before + 120


def test_authorising_without_a_binding_fails(sim) -> None:
    """No binding means nothing to authorise through, which is the constraint
    the stage machine exists to express."""
    simulator, _, _, merchants = sim
    unbound = next(
        card_id
        for card_id in simulator.graph.cards
        if not simulator.graph.devices_of_card(card_id)
    )
    outcome = simulator.step(
        1,
        Action(
            name=ActionName.ATTEMPT_AUTH, target_id=unbound,
            secondary_id=merchants[0], amount=50.0,
        ),
    )
    assert outcome.code is OutcomeCode.FAILED
    assert outcome.value_extracted == 0.0


def test_a_frozen_card_cannot_authorise(sim) -> None:
    simulator, _, card_id, _ = sim
    from fraudsim.world.entities import CardStatus

    card = simulator.graph.cards[card_id]
    card.status = CardStatus.FROZEN
    card.frozen_until = simulator.clock.now + 10_000
    assert auth(sim).code is not OutcomeCode.APPROVED


def test_binding_actions_need_their_prerequisite(sim) -> None:
    """An action that cannot do what it claims fails rather than reporting
    success, so a stage never advances on a capability nobody obtained."""
    simulator, actor, *_ = sim
    actor.stage = Stage.ACQUIRED

    bare = simulator.step(1, Action(name=ActionName.RESET_PASSWORD, target_id=1))
    assert bare.code is OutcomeCode.FAILED
    assert len(simulator.log) == 0

    # Credentials are only obtainable from the first stage, so the actor has to
    # go back and acquire them rather than being handed them mid-run.
    actor.stage = Stage.NONE
    assert simulator.step(1, Action(name=ActionName.BUY_CREDS)).succeeded

    armed = simulator.step(1, Action(name=ActionName.RESET_PASSWORD, target_id=1))
    assert armed.code is OutcomeCode.APPROVED
    assert len(simulator.log) == 1


def test_outcomes_do_not_reveal_the_risk_score(sim) -> None:
    """An actor learns whether it worked, not what the detector thought."""
    outcome = auth(sim)
    assert not hasattr(outcome, "risk_score")
    assert set(outcome.__slots__) == {
        "code", "stage", "reward", "value_extracted", "cost", "event_id",
    }
