"""The five controls that keep a policy honest, each asserted to actually bite.

The design names five: an episode action cap, a per-action cost, a per-merchant
value cap, an episode duration cap, and per-episode jitter on the decision
thresholds. Three of the five were declared in the configuration and enforced
nowhere.

The consequence was not subtle. With no ceiling on what one merchant would
absorb, every seed of a co-adaptation run converged on the same shape — rotate
cards to dodge per-card velocity, then hammer authorisations — and took eight to
twenty-eight thousand an episode against a stated cap of two thousand. That
produced a rising extraction curve that looked exactly like an arms race and was
a hole in the simulator.

A control that exists only in a config file is worse than one that does not
exist, because it gets believed. These tests assert each one changes what the
world permits.
"""

from __future__ import annotations

import pytest

from conftest import requires_torch

from fraudsim.settings.simulation import SimulationConfig
from fraudsim.engine.actions import Action, ActionName
from fraudsim.engine.outcome import OutcomeCode
from fraudsim.engine.simulator import Actor, ActorKind, Simulator
from fraudsim.engine.stages import Stage
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.population.builder import PopulationBuilder
from fraudsim.protocols import AlwaysApproveScorer, RiskAction, RiskAssessment


def _world(overrides: dict | None = None):
    base = {"population": {"n_holders": 200}}
    if overrides:
        base.update(overrides)
    config = SimulationConfig.model_validate(base)
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    sim = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())
    return config, graph, sim


def _bound_card(graph):
    for card_id in graph.cards:
        if graph.devices_of_card(card_id):
            return card_id
    pytest.skip("no bound card in this world")


def _spender(sim, graph, card_id, actor_id=7001):
    holder = int(graph.cards[card_id].holder_id)
    sim.register_actor(
        Actor(
            actor_id=actor_id,
            kind=ActorKind.ADVERSARIAL,
            holder_id=holder,
            cards=[card_id],
            stage=Stage.BOUND,
        )
    )
    sim.open_episode(actor_id)
    return actor_id


# ------------------------------------------------- 3. per-merchant value cap


def test_per_merchant_cap_stops_repeated_spend_at_one_merchant():
    """The control that was missing, and that a policy found within one run."""
    config, graph, sim = _world({"engine": {"episode": {"max_value_per_merchant": 500.0}}})
    card_id = _bound_card(graph)
    merchant = int(next(iter(graph.merchants)))
    actor_id = _spender(sim, graph, card_id)

    extracted = 0.0
    refusals = 0
    for _ in range(20):
        out = sim.step(
            actor_id,
            Action(
                name=ActionName.ATTEMPT_AUTH,
                target_id=int(card_id),
                secondary_id=merchant,
                amount=100.0,
            ),
        )
        extracted += out.value_extracted
        refusals += int(out.code is OutcomeCode.FAILED)

    assert extracted <= 500.0, "the cap must bound what one merchant absorbs"
    assert refusals > 0, "attempts past the cap must be refused, not silently allowed"


def test_the_cap_is_per_merchant_not_per_episode():
    """Spreading across merchants is a real strategy and must stay available.

    A cap that applied to the episode as a whole would forbid the behaviour it
    is meant to shape rather than bound it.
    """
    config, graph, sim = _world({"engine": {"episode": {"max_value_per_merchant": 300.0}}})
    card_id = _bound_card(graph)
    merchants = [int(m) for m in list(graph.merchants)[:3]]
    if len(merchants) < 3:
        pytest.skip("need three merchants")
    actor_id = _spender(sim, graph, card_id)

    total = 0.0
    for merchant in merchants:
        for _ in range(4):
            out = sim.step(
                actor_id,
                Action(
                    name=ActionName.ATTEMPT_AUTH,
                    target_id=int(card_id),
                    secondary_id=merchant,
                    amount=100.0,
                ),
            )
            total += out.value_extracted

    assert total > 300.0, "spreading across merchants must still be possible"


def test_the_cap_resets_between_episodes():
    """A tally carried across episodes would silently shrink every later one."""
    config, graph, sim = _world({"engine": {"episode": {"max_value_per_merchant": 300.0}}})
    card_id = _bound_card(graph)
    merchant = int(next(iter(graph.merchants)))

    takes = []
    for i in range(2):
        actor_id = _spender(sim, graph, card_id, actor_id=7100 + i)
        got = 0.0
        for _ in range(6):
            out = sim.step(
                actor_id,
                Action(
                    name=ActionName.ATTEMPT_AUTH,
                    target_id=int(card_id),
                    secondary_id=merchant,
                    amount=100.0,
                ),
            )
            got += out.value_extracted
        sim.close_episode(actor_id)
        takes.append(got)

    assert takes[1] > 0, "a fresh episode must start with its full headroom"


def test_an_over_cap_attempt_is_refused_not_trimmed():
    """Trimming to fit would reward the overreach with whatever was left.

    A policy learns from that to always ask for more than it can have, which is
    the opposite of what the cap is for.
    """
    config, graph, sim = _world({"engine": {"episode": {"max_value_per_merchant": 500.0}}})
    card_id = _bound_card(graph)
    merchant = int(next(iter(graph.merchants)))
    actor_id = _spender(sim, graph, card_id)

    out = sim.step(
        actor_id,
        Action(
            name=ActionName.ATTEMPT_AUTH,
            target_id=int(card_id),
            secondary_id=merchant,
            amount=5000.0,
        ),
    )
    assert out.code is OutcomeCode.FAILED
    assert out.value_extracted == 0.0


# ---------------------------------------------------- 4. episode duration cap


@requires_torch
def test_episode_duration_cap_ends_a_long_running_attack():
    """Stretching an episode across months steps outside the detector's window.

    A brand-new device ages into an ordinary one and the velocity windows forget
    everything before. That is not patience, it is leaving the problem.
    """
    from fraudsim.attacker.env import AttackEnv
    from fraudsim.attacker.nets import STEALTH_AGED_COOL
    from fraudsim.engine.actions import ACTION_INDEX
    from fraudsim.protocols import Target

    config, graph, sim = _world({"engine": {"episode": {"max_hours": 48}}})
    card_id = _bound_card(graph)
    target = Target(
        card_id=int(card_id),
        holder_id=int(graph.cards[card_id].holder_id),
        account_id=None,
        merchants=tuple(int(m) for m in list(graph.merchants)[:5]),
    )
    env = AttackEnv(sim, target)
    env.reset()

    # A large positive raw delay squashes to the top of the range, so each step
    # advances the clock a long way and the duration cap is what stops it.
    done = False
    steps = 0
    while not done and steps < 100:
        _, _, done, _ = env.step(
            ACTION_INDEX[ActionName.BUY_CREDS], 0.0, 20.0, STEALTH_AGED_COOL
        )
        steps += 1
    env.close()

    assert done, "the episode must end"
    assert steps < 40, "it must end on the clock, well before the action cap"


# ------------------------------------------------ 5. per-episode band jitter


def test_threshold_jitter_moves_the_decision_between_episodes():
    """A fixed boundary is a number a policy can find and sit underneath.

    The jitter is drawn once per episode, so it is stable while an attacker acts
    and different next time — the boundary cannot be binary-searched.
    """
    config, graph, sim = _world({"engine": {"episode": {"threshold_jitter": 0.05}}})

    offsets = set()
    for i in range(12):
        actor_id = 7200 + i
        sim.register_actor(
            Actor(actor_id=actor_id, kind=ActorKind.ADVERSARIAL, stage=Stage.BOUND)
        )
        sim.open_episode(actor_id)
        offsets.add(round(sim.actor(actor_id).threshold_offset, 6))
        sim.close_episode(actor_id)

    assert len(offsets) > 1, "the offset must differ between episodes"
    assert all(abs(o) <= 0.05 + 1e-9 for o in offsets), "and stay inside the spread"


def test_jitter_is_stable_within_one_episode():
    """An offset redrawn per action would be noise, not a moving boundary."""
    config, graph, sim = _world({"engine": {"episode": {"threshold_jitter": 0.05}}})
    card_id = _bound_card(graph)
    merchant = int(next(iter(graph.merchants)))
    actor_id = _spender(sim, graph, card_id)

    first = sim.actor(actor_id).threshold_offset
    for _ in range(3):
        sim.step(
            actor_id,
            Action(
                name=ActionName.ATTEMPT_AUTH,
                target_id=int(card_id),
                secondary_id=merchant,
                amount=25.0,
            ),
        )
    assert sim.actor(actor_id).threshold_offset == first


def test_zero_jitter_leaves_the_decision_exactly_as_scored():
    """The static benchmarks need a fixed operating point to be comparable."""
    config, graph, sim = _world({"engine": {"episode": {"threshold_jitter": 0.0}}})
    actor_id = 7300
    sim.register_actor(
        Actor(actor_id=actor_id, kind=ActorKind.ADVERSARIAL, stage=Stage.BOUND)
    )
    sim.open_episode(actor_id)
    assert sim.actor(actor_id).threshold_offset == 0.0


def test_jitter_can_change_the_action_taken():
    """The offset must reach the decision, not merely be recorded on the actor."""
    from fraudsim.engine.bands import RiskBands, shift_assessment

    class Scorer:
        bands = RiskBands()

    # A score just under the decline band: a negative offset pushes it over.
    scored = RiskAssessment(risk_score=0.78, action=RiskAction.HOLD, mitigations=())

    lenient = shift_assessment(scored, +0.05, None, Scorer())
    strict = shift_assessment(scored, -0.05, None, Scorer())

    assert lenient.action is not strict.action, "the offset must move the decision"
    # The belief itself is untouched; the jitter belongs to the decision.
    assert lenient.risk_score == 0.78 and strict.risk_score == 0.78
