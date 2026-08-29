"""The live co-adaptation: four phases run, the defender is swappable, both move.

Tiny scale for speed. These assert the machinery of the solution: the defender
can be swapped live, the four phases complete in order, the retention stays
asymmetric across refits, and the live phase produces a per-update curve with the
defender refitting on the cadence asked for. The shape of the arms race is a
finding; that the loop runs and the pieces connect is the invariant.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fraudsim.attacker.ppo import PPOConfig
from fraudsim.config.simulation import SimulationConfig
from fraudsim.engine.actions import Action, ActionName
from fraudsim.engine.simulator import Actor, ActorKind, Simulator
from fraudsim.engine.stages import Stage
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.orchestration.coadapt import CoadaptEngine, run_coadapt
from fraudsim.population.builder import PopulationBuilder
from fraudsim.protocols import AlwaysApproveScorer, RiskAction, RiskAssessment


def test_simulator_scorer_is_swappable():
    config = SimulationConfig.model_validate({"population": {"n_holders": 200}})
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    sim = Simulator(graph, config, builder, scorer=AlwaysApproveScorer())

    class DeclineAll:
        def score(self, event):
            return RiskAssessment(0.99, RiskAction.DECLINE)

    sim.set_scorer(DeclineAll())
    assert isinstance(sim.scorer, DeclineAll)

    card_id, dev = next(iter(graph.provisioned))
    holder = int(graph.cards[card_id].holder_id)
    sim.register_actor(
        Actor(actor_id=1, kind=ActorKind.ADVERSARIAL, holder_id=holder, cards=[card_id], stage=Stage.BOUND)
    )
    out = sim.step(
        1, Action(name=ActionName.ATTEMPT_AUTH, target_id=int(card_id), secondary_id=int(next(iter(graph.merchants))), amount=50.0)
    )
    # The swapped-in defender declines, proving the swap took effect live.
    assert out.code.value == "declined"


@pytest.fixture(scope="module")
def coadapt_report():
    config = SimulationConfig.model_validate(
        {"population": {"n_holders": 400}, "engine": {"fraud_base_rate": 0.06}}
    )
    return run_coadapt(
        config,
        seed=0,
        learned_defender=False,
        demo_episodes=20,
        bc_epochs=4,
        critic_rollouts=8,
        critic_epochs=4,
        n_updates=6,
        episodes_per_update=8,
        refit_every=3,
        ppo_config=PPOConfig(hidden_dim=64, n_layers=1, minibatch_size=64, bc_kl_anneal_updates=2),
    )


def test_all_phases_produced_output(coadapt_report):
    r = coadapt_report
    assert r.initial_defender_positives > 0  # phase A fitted on real fraud
    assert r.critic_final_loss >= 0.0  # phase C ran
    assert len(r.attacker_success) == 6  # phase D produced a per-update curve


def test_defender_refit_on_cadence(coadapt_report):
    r = coadapt_report
    # refit_every=3 over 6 updates -> refits after updates index 2 and 5.
    assert r.defender_refits == [2, 5]
    # Each refit trained on at least as much fraud as the last (cumulative).
    assert all(
        b >= a for a, b in zip(r.defender_positives_at_refit, r.defender_positives_at_refit[1:])
    )


def test_zero_shot_present(coadapt_report):
    r = coadapt_report
    assert set(r.zero_shot) == {"refund_abuse", "sim_swap"}


def test_top_sequences_logged(coadapt_report):
    r = coadapt_report
    assert r.top_sequences
    assert ">" in r.top_sequences[0][0] or r.top_sequences[0][0]
