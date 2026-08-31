"""Every action does what it claims, or fails.

Before these resolvers existed a single method stood in for nineteen actions
and reported success unconditionally. The stage advanced, the actor believed it
had gained a capability, and the world was untouched. A device binding emitted
an event saying a device had been bound without creating the edge, so the log
and the graph contradicted each other and the next authorisation failed for a
reason nothing in the log explained.
"""

from __future__ import annotations

import pytest

from fraudsim.settings.simulation import SimulationConfig
from fraudsim.engine.actions import Action, ActionName
from fraudsim.engine.outcome import OutcomeCode
from fraudsim.engine.resolution import registered_actions
from fraudsim.engine.simulator import Actor, ActorKind, Simulator
from fraudsim.engine.stages import Stage
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.population.builder import PopulationBuilder
from fraudsim.protocols import AlwaysApproveScorer

A = ActionName
HOUR = 60


@pytest.fixture
def world():
    config = SimulationConfig.model_validate({"population": {"n_holders": 300}})
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    simulator = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())

    card_id, _ = next(iter(graph.provisioned))
    holder_id = graph.cards[card_id].holder_id
    account_id = next(iter(graph.accounts_of_holder(holder_id)), None)
    if account_id is not None:
        graph.accounts[account_id].balance = 5000.0

    actor = simulator.register_actor(
        Actor(actor_id=1, kind=ActorKind.ADVERSARIAL, stage=Stage.NONE,
              holder_id=holder_id, cards=[card_id])
    )
    merchant_id = next(iter(graph.merchants))
    return simulator, graph, actor, card_id, merchant_id, account_id


def step(simulator, name, stage=None, actor=None, **kwargs):
    if stage is not None and actor is not None:
        actor.stage = stage
    return simulator.step(1, Action(name=name, **kwargs))


def test_every_action_has_a_resolver() -> None:
    """A missing one would fall through to whatever the dispatch did last,
    which is how the placeholder went unnoticed."""
    covered = registered_actions() | {A.ATTEMPT_AUTH}
    assert set(ActionName) - covered == set()


# ------------------------------------------------------- capability gating


@pytest.mark.parametrize(
    "label,stage,name",
    [
        ("bind without credentials", Stage.ACQUIRED, A.ADD_DEVICE_SELFSERVE),
        ("provision without a voice sample", Stage.ACQUIRED, A.CALL_IVR_PROVISION),
        ("submit identity papers without an identity", Stage.ACQUIRED, A.SUBMIT_KYC),
        ("answer a challenge without the number", Stage.BOUND, A.COMPLETE_3DS),
        ("transfer without a payee", Stage.BOUND, A.TRANSFER_P2P),
        ("cash out having moved nothing", Stage.MONETIZED, A.CASH_OUT),
        ("launder having moved nothing", Stage.MONETIZED, A.LAUNDER_CHAIN),
        ("dispute a transaction that never happened", Stage.MONETIZED, A.FILE_DISPUTE),
    ],
)
def test_an_action_without_its_prerequisite_fails(world, label, stage, name) -> None:
    simulator, _, actor, card_id, _, _ = world
    outcome = step(simulator, name, stage, actor, target_id=int(card_id), amount=100.0)
    assert outcome.code is OutcomeCode.FAILED, label


def test_a_failed_action_does_not_advance_the_stage(world) -> None:
    simulator, _, actor, card_id, _, _ = world
    actor.stage = Stage.ACQUIRED
    outcome = simulator.step(1, Action(name=A.ADD_DEVICE_SELFSERVE, target_id=int(card_id)))
    assert not outcome.succeeded
    assert actor.stage is Stage.ACQUIRED


def test_a_failed_action_emits_nothing(world) -> None:
    """Emitting regardless is what let the log claim a binding that the graph
    did not have."""
    simulator, _, actor, card_id, _, _ = world
    actor.stage = Stage.ACQUIRED
    simulator.step(1, Action(name=A.ADD_DEVICE_SELFSERVE, target_id=int(card_id)))
    assert len(simulator.log) == 0


# ------------------------------------------------------------- mutations


def test_binding_creates_the_edge(world) -> None:
    simulator, graph, actor, card_id, _, _ = world
    step(simulator, A.BUY_CREDS, Stage.NONE, actor)
    before = len(graph.devices_of_card(card_id))

    outcome = step(simulator, A.ADD_DEVICE_SELFSERVE, Stage.ACQUIRED, actor,
                   target_id=int(card_id))
    assert outcome.succeeded
    assert len(graph.devices_of_card(card_id)) == before + 1
    assert len(simulator.log) == 1
    graph.check_invariants()


def test_provisioning_by_phone_needs_a_good_enough_sample(world) -> None:
    simulator, graph, actor, card_id, _, _ = world
    step(simulator, A.BUY_CREDS, Stage.NONE, actor)

    step(simulator, A.HARVEST_VOICE, Stage.NONE, actor, params={"quality": 0.2})
    weak = step(simulator, A.CALL_IVR_PROVISION, Stage.ACQUIRED, actor,
                target_id=int(card_id))
    assert weak.code is OutcomeCode.FAILED

    step(simulator, A.HARVEST_VOICE, Stage.ACQUIRED, actor, params={"quality": 0.95})
    strong = step(simulator, A.CALL_IVR_PROVISION, Stage.ACQUIRED, actor,
                  target_id=int(card_id))
    assert strong.succeeded


def test_adding_a_payee_sets_a_cooling_off_period(world) -> None:
    """A payee that could receive money at once would make the control
    decorative."""
    simulator, graph, actor, card_id, _, account_id = world
    outcome = step(simulator, A.ADD_PAYEE, Stage.BOUND, actor, target_id=int(card_id))
    assert outcome.succeeded

    payee_id = actor.payees[-1]
    edge = graph.added[(account_id, payee_id)]
    assert edge.is_cooling_off(simulator.clock.now)


def test_transfer_waits_out_the_cooling_off(world) -> None:
    simulator, graph, actor, card_id, _, account_id = world
    step(simulator, A.ADD_PAYEE, Stage.BOUND, actor, target_id=int(card_id))

    early = step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor,
                 target_id=int(card_id), amount=500.0, delay_minutes=30)
    assert early.code is OutcomeCode.FAILED

    late = step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor,
                target_id=int(card_id), amount=500.0, delay_minutes=25 * HOUR)
    assert late.succeeded


def test_transfer_moves_the_balance(world) -> None:
    simulator, graph, actor, card_id, _, account_id = world
    step(simulator, A.ADD_PAYEE, Stage.BOUND, actor, target_id=int(card_id))
    before = graph.accounts[account_id].balance

    step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor, target_id=int(card_id),
         amount=800.0, delay_minutes=25 * HOUR)
    assert graph.accounts[account_id].balance == pytest.approx(before - 800.0)
    assert actor.laundered == pytest.approx(800.0)


def test_transfer_refuses_more_than_the_balance(world) -> None:
    simulator, graph, actor, card_id, _, account_id = world
    step(simulator, A.ADD_PAYEE, Stage.BOUND, actor, target_id=int(card_id))
    outcome = step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor,
                   target_id=int(card_id), amount=99_999.0, delay_minutes=25 * HOUR)
    assert outcome.code is OutcomeCode.FAILED
    assert graph.accounts[account_id].balance == pytest.approx(5000.0)


def test_a_transfer_moves_value_rather_than_realising_it(world) -> None:
    """Counting a transfer as extracted and then counting the cash-out as well
    credits the same money twice, which would let an actor inflate its take by
    adding hops."""
    simulator, _, actor, card_id, _, _ = world
    step(simulator, A.ADD_PAYEE, Stage.BOUND, actor, target_id=int(card_id))
    outcome = step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor,
                   target_id=int(card_id), amount=800.0, delay_minutes=25 * HOUR)
    assert outcome.succeeded
    assert outcome.value_extracted == 0.0


def test_cash_out_applies_a_haircut(world) -> None:
    """Converting stolen funds is never at par."""
    simulator, _, actor, card_id, _, _ = world
    step(simulator, A.ADD_PAYEE, Stage.BOUND, actor, target_id=int(card_id))
    step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor, target_id=int(card_id),
         amount=1000.0, delay_minutes=25 * HOUR)

    outcome = step(simulator, A.CASH_OUT, Stage.MONETIZED, actor,
                   target_id=int(card_id), amount=1000.0)
    assert outcome.succeeded
    assert 0 < outcome.value_extracted < 1000.0


def test_cash_out_cannot_exceed_what_was_moved(world) -> None:
    simulator, _, actor, card_id, _, _ = world
    step(simulator, A.ADD_PAYEE, Stage.BOUND, actor, target_id=int(card_id))
    step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor, target_id=int(card_id),
         amount=300.0, delay_minutes=25 * HOUR)

    outcome = step(simulator, A.CASH_OUT, Stage.MONETIZED, actor,
                   target_id=int(card_id), amount=99_999.0)
    assert outcome.value_extracted <= 300.0


def test_laundering_costs_something_per_hop(world) -> None:
    """The chain buys distance from the source and pays for it."""
    simulator, _, actor, card_id, _, _ = world
    step(simulator, A.ADD_PAYEE, Stage.BOUND, actor, target_id=int(card_id))
    step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor, target_id=int(card_id),
         amount=1000.0, delay_minutes=25 * HOUR)

    before = actor.laundered
    step(simulator, A.LAUNDER_CHAIN, Stage.MONETIZED, actor, target_id=int(card_id),
         params={"hops": 3})
    assert actor.laundered < before
    assert actor.launder_hops == 3


def test_raising_a_limit_changes_the_card(world) -> None:
    simulator, graph, actor, card_id, _, _ = world
    step(simulator, A.OPEN_TICKET, Stage.BOUND, actor, target_id=int(card_id))
    before = graph.cards[card_id].credit_line

    outcome = step(simulator, A.ESCALATE_LIMIT, Stage.BOUND, actor,
                   target_id=int(card_id), params={"factor": 2.0})
    assert outcome.succeeded
    assert graph.cards[card_id].credit_line > before


def test_disputing_needs_a_transaction_to_dispute(world) -> None:
    simulator, graph, actor, card_id, merchant_id, _ = world
    bare = step(simulator, A.FILE_DISPUTE, Stage.MONETIZED, actor, target_id=int(card_id))
    assert bare.code is OutcomeCode.FAILED

    graph.record_transaction(card_id, merchant_id, 250.0, simulator.clock.now)
    real = step(simulator, A.FILE_DISPUTE, Stage.MONETIZED, actor, target_id=int(card_id))
    assert real.succeeded
    assert real.value_extracted > 0


def test_credentials_accumulate(world) -> None:
    simulator, _, actor, _, _, _ = world
    step(simulator, A.BUY_CREDS, Stage.NONE, actor, params={"count": 5})
    assert len(actor.credentials) == 5


def test_a_full_walk_reaches_monetized_and_moves_money(world) -> None:
    """The whole point: a run that ends with value realised and a world that
    reflects every step taken."""
    simulator, graph, actor, card_id, _, account_id = world
    opening = graph.accounts[account_id].balance

    assert step(simulator, A.BUY_CREDS, Stage.NONE, actor).succeeded
    assert step(simulator, A.ADD_DEVICE_SELFSERVE, Stage.ACQUIRED, actor,
                target_id=int(card_id)).succeeded
    assert step(simulator, A.ADD_PAYEE, Stage.BOUND, actor,
                target_id=int(card_id)).succeeded
    assert step(simulator, A.TRANSFER_P2P, Stage.BOUND, actor, target_id=int(card_id),
                amount=1200.0, delay_minutes=25 * HOUR).succeeded
    assert step(simulator, A.CASH_OUT, Stage.MONETIZED, actor, target_id=int(card_id),
                amount=1200.0).succeeded

    assert actor.stage is Stage.MONETIZED
    assert actor.value_extracted > 0
    assert graph.accounts[account_id].balance == pytest.approx(opening - 1200.0)
    graph.check_invariants()
