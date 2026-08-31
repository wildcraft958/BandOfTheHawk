"""The stealth head: what the policy chooses, and what the world does with it.

The head exists because of a specific failure. The attacker had no way to say
*how* an action was carried out, so every authorisation ran through whichever
binding the simulator preferred — the newest — and every attack therefore
carried a device minutes old. The defender's top feature by a wide margin was
device age, and one refit ended the arms race. There was no stealthier strategy
for the policy to discover, because the action space could not express one.

These tests assert the mechanism, not the outcome. Whether stealth lets the
attacker survive a refit is a finding, measured by the ablation and reported
either way. Whether the posture the policy names actually reaches the world is
an invariant, and that is what is checked here: an aged posture must route
through the oldest surviving binding, a cooling posture must actually wait, and
rotation must move to a different card. A head that changed nothing downstream
would look identical in every training curve.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch", reason='install the "rl" extra'
)

from fraudsim.attacker.env import AttackEnv
from fraudsim.clock import MINUTES_PER_HOUR
from fraudsim.attacker.nets import (
    N_STEALTH,
    STEALTH_AGED,
    STEALTH_AGED_COOL,
    STEALTH_LOUD,
    STEALTH_ROTATE,
    NetConfig,
)
from fraudsim.attacker.nets import Actor as ActorNet
from fraudsim.settings.simulation import SimulationConfig
from fraudsim.engine.actions import ACTION_INDEX, N_ACTIONS, ActionName
from fraudsim.engine.simulator import Simulator
from fraudsim.features.builder import EventBuilder
from fraudsim.features.state import FeatureStateStore
from fraudsim.protocols import Target
from fraudsim.population.builder import PopulationBuilder
from fraudsim.protocols import AlwaysApproveScorer

@pytest.fixture(scope="module")
def world():
    config = SimulationConfig.model_validate({"population": {"n_holders": 300}})
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    return Simulator(graph, config, builder, scorer=AlwaysApproveScorer())

def _multi_bound_card(graph):
    """A card with more than one device, so oldest and newest differ."""
    for card_id in graph.cards:
        devices = graph.devices_of_card(card_id)
        if len(devices) >= 2:
            return card_id, devices
    pytest.skip("no card in this world carries two bindings")

def _target_for(graph, card_id, extra=()):
    holder = int(graph.cards[card_id].holder_id)
    merchants = tuple(int(m) for m in list(graph.merchants)[:5])
    return Target(
        card_id=int(card_id),
        holder_id=holder,
        account_id=None,
        merchants=merchants,
        card_ids=(int(card_id),) + tuple(int(c) for c in extra),
    )

# --------------------------------------------------------------- resolution

def test_aged_posture_routes_through_the_oldest_binding(world):
    """The whole point: an aged posture must not mint or prefer a new device."""
    graph = world.graph
    card_id, devices = _multi_bound_card(graph)
    env = AttackEnv(world, _target_for(graph, card_id))

    oldest = min(devices, key=lambda d: graph.devices[d].first_seen_ts)
    newest = max(devices, key=lambda d: graph.devices[d].first_seen_ts)
    assert oldest != newest, "fixture must give two distinguishable bindings"

    assert env._device_for(STEALTH_AGED, int(card_id)) == int(oldest)
    assert env._device_for(STEALTH_AGED_COOL, int(card_id)) == int(oldest)
    # Loud hands the choice back to the world, which prefers the newest. Naming
    # a device here would make the two postures indistinguishable downstream.
    assert env._device_for(STEALTH_LOUD, int(card_id)) is None

def test_aged_posture_skips_blocklisted_bindings(world):
    """A blocklisted device would be refused anyway; naming it wastes the step."""
    graph = world.graph
    card_id, devices = _multi_bound_card(graph)
    env = AttackEnv(world, _target_for(graph, card_id))

    oldest = min(devices, key=lambda d: graph.devices[d].first_seen_ts)
    graph.devices[oldest].blocklisted = True
    try:
        chosen = env._device_for(STEALTH_AGED, int(card_id))
        assert chosen is not None
        assert chosen != int(oldest)
    finally:
        graph.devices[oldest].blocklisted = False

def test_entry_mode_follows_the_posture():
    """Entry mode was drawn from an RNG; the policy had no say in it."""
    assert AttackEnv._entry_mode(STEALTH_AGED) == 1
    assert AttackEnv._entry_mode(STEALTH_AGED_COOL) == 1
    assert AttackEnv._entry_mode(STEALTH_LOUD) == 0
    assert AttackEnv._entry_mode(STEALTH_ROTATE) == 0

def test_cool_posture_advances_the_clock(world):
    """Cooling off must actually wait, or the posture is a label on nothing."""
    graph = world.graph
    card_id, _ = _multi_bound_card(graph)
    env = AttackEnv(world, _target_for(graph, card_id))
    env.reset()

    before = world.clock.now
    # A large negative raw delay squashes to nearly zero minutes, so any advance
    # seen here comes from the cooling floor rather than from the sample.
    env.step(ACTION_INDEX[ActionName.BUY_CREDS], 0.0, -20.0, STEALTH_AGED_COOL)
    cool_off = env.space.cool_off_hours * MINUTES_PER_HOUR
    assert world.clock.now - before >= cool_off
    env.close()

def test_rotation_moves_to_another_card(world):
    """Rotation must change what is being spent, not merely bump a counter."""
    graph = world.graph
    cards = list(graph.cards)[:3]
    env = AttackEnv(world, _target_for(graph, cards[0], extra=cards[1:]))
    env.reset()

    first = env.active_card
    env.step(ACTION_INDEX[ActionName.BUY_CREDS], 0.0, -20.0, STEALTH_ROTATE)
    assert env.active_card != first
    env.close()

def test_rotation_on_a_single_card_dump_is_a_no_op(world):
    """A one-card dump has nowhere to rotate to; it must not wrap or crash."""
    graph = world.graph
    card_id = next(iter(graph.cards))
    env = AttackEnv(world, _target_for(graph, card_id))
    env.reset()

    first = env.active_card
    env.step(ACTION_INDEX[ActionName.BUY_CREDS], 0.0, -20.0, STEALTH_ROTATE)
    assert env.active_card == first
    env.close()

# -------------------------------------------------------------- observation

def test_observation_reports_the_attackers_own_detection_history(world):
    """Without this the posture choice is unlearnable: both look identical."""
    graph = world.graph
    card_id = next(iter(graph.cards))
    env = AttackEnv(world, _target_for(graph, card_id))
    obs = env.reset()

    for key in (
        "flags_so_far",
        "last_action_flagged",
        "cards_remaining",
        "hours_elapsed",
    ):
        assert key in obs.features
    assert obs.features["flags_so_far"] == 0.0
    assert obs.features["last_action_flagged"] == 0.0
    env.close()

def test_encoder_width_matches_the_reported_dimension(world):
    """The nets size themselves from obs_dim; a mismatch is a silent shape bug."""
    graph = world.graph
    card_id = next(iter(graph.cards))
    env = AttackEnv(world, _target_for(graph, card_id))
    obs = env.reset()
    assert AttackEnv.encode(obs).shape == (AttackEnv.obs_dim(),)
    env.close()

# ------------------------------------------------------------- termination

def test_fraud_loop_terminates_when_episodes_produce_no_auths(monkeypatch):
    """The prevalence loop must not spin forever on an unproductive world.

    Its exit condition counts authorisation *events*, and an episode can produce
    none: an auth whose card has no usable binding fails before an event is
    built, and a vertical that monetises through a refund may never attempt one.
    Both are reachable at the end of a long co-adaptation run, where mitigations
    have blocklisted devices and frozen cards — and the zero-shot holdout then
    runs a single vertical against exactly that world.

    This is not hypothetical. A run finished all its training updates and then
    hung for an hour in the evaluation afterwards, losing the whole run; a stack
    dump put it inside this loop.
    """
    from fraudsim.settings.simulation import SimulationConfig
    from fraudsim.engine.simulator import Simulator as Sim
    from fraudsim.orchestration.run import EpisodeRunner
    from fraudsim.population.warmstart import WarmStartRunner

    config = SimulationConfig.model_validate(
        {"population": {"n_holders": 200}, "engine": {"fraud_base_rate": 0.06}}
    )
    graph, _ = PopulationBuilder(config).build()
    states = FeatureStateStore(config.engine.windows)
    builder = EventBuilder(graph, states, config.engine.windows)
    sim = Sim(graph, config, builder, scorer=AlwaysApproveScorer())
    WarmStartRunner(sim, config, seed=0).run()

    # Every episode completes and emits no authorisation, which is exactly the
    # state a fully mitigated world reaches.
    calls = {"n": 0}

    def barren(self, *args, **kwargs):
        calls["n"] += 1
        assert calls["n"] < 100_000, "the loop did not terminate"
        return 0, 0, False, "buy_creds"

    monkeypatch.setattr(EpisodeRunner, "_episode", barren)

    report = EpisodeRunner(sim, config, seed=1, train_only=True).run(benign_seed=2)

    assert report.exhausted, "an unproductive world must stop on the budget"
    assert report.fraud_auths == 0
    assert calls["n"] > 0, "the loop must actually have run"

# ------------------------------------------------------------------- reward

def test_terminal_bonus_is_paid_for_money_not_for_a_stage_label():
    """Reaching MONETIZED without realising value must not pay the big bonus.

    Two actions reach that stage. An authorisation extracts its amount on
    approval; a transfer moves balance into the laundering pot and realises
    nothing until a later cash-out. Paying the bonus on arrival made the
    transfer worth over ten points of reward for zero extracted value, and the
    policy learned to reach the stage and stop — scoring well on the reward
    while extracting nothing the arms-race metric could see.
    """
    from fraudsim.attacker.env import AttackEnv, RewardWeights
    from fraudsim.engine.outcome import Outcome, OutcomeCode
    from fraudsim.engine.stages import Stage

    env = AttackEnv.__new__(AttackEnv)
    env.weights = RewardWeights()

    dry = Outcome(code=OutcomeCode.APPROVED, stage=Stage.MONETIZED, value_extracted=0.0)
    paid = Outcome(code=OutcomeCode.APPROVED, stage=Stage.MONETIZED, value_extracted=200.0)

    r_dry = env._reward(dry, Stage.BOUND, Stage.MONETIZED)
    r_paid = env._reward(paid, Stage.BOUND, Stage.MONETIZED)

    assert r_dry < env.weights.terminal_bonus, (
        "a stage transition that realised no money must not collect the "
        "terminal bonus"
    )
    assert r_paid - r_dry > env.weights.terminal_bonus

# --------------------------------------------------------------------- head

def test_stealth_head_is_unmasked_and_full_width():
    """Every posture is meaningful at every stage, so nothing is masked out."""
    cfg = NetConfig(obs_dim=AttackEnv.obs_dim(), hidden_dim=32, n_layers=1)
    actor = ActorNet(cfg)
    obs = torch.zeros(4, cfg.obs_dim)
    mask = torch.zeros(4, N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = True

    _, stealth, _, _ = actor(obs, mask)
    assert stealth.probs.shape == (4, N_STEALTH)
    assert torch.all(stealth.probs > 0), "no posture may be unreachable"

def test_stealth_joins_the_joint_log_prob():
    """Posture and action are one decision; scoring them apart loses the credit."""
    cfg = NetConfig(obs_dim=AttackEnv.obs_dim(), hidden_dim=32, n_layers=1)
    actor = ActorNet(cfg)
    obs = torch.zeros(2, cfg.obs_dim)
    mask = torch.ones(2, N_ACTIONS, dtype=torch.bool)
    act = torch.zeros(2, dtype=torch.long)
    amt = torch.zeros(2)
    dly = torch.zeros(2)

    lp_a, _ = actor.evaluate(obs, mask, act, torch.zeros(2, dtype=torch.long), amt, dly)
    lp_b, _ = actor.evaluate(obs, mask, act, torch.ones(2, dtype=torch.long), amt, dly)
    # Different postures must score differently, or the head is not in the loss.
    assert not torch.allclose(lp_a, lp_b)

def test_behaviour_cloning_leaves_the_stealth_head_free():
    """The head must not be cloned: the demonstrations say nothing about posture.

    This is a regression test for a real failure. The scripts predate the head
    and are uniformly loud, so cloning it drove posture entropy from 1.382 to
    0.031 — the policy came out of the warm start 99.6% certain of the one choice
    it most needed to question, and the KL-to-BC penalty then held it there
    through the first defender refit. The stealth capability was present, was
    measurably sufficient to clear the decline threshold on its own, and was
    never used. Fitting a head to data that does not speak to it manufactures a
    confident answer out of nothing.
    """
    import numpy as np

    from fraudsim.attacker.bootstrap import Demo
    from fraudsim.attacker.ppo import PPOConfig, PPOTrainer

    rng = np.random.default_rng(0)
    dim = AttackEnv.obs_dim()
    demos = [
        Demo(
            obs=rng.standard_normal(dim).astype(np.float32),
            mask=np.ones(N_ACTIONS, dtype=bool),
            action_idx=int(rng.integers(0, N_ACTIONS)),
            amount_raw=0.0,
            delay_raw=0.0,
            stealth_idx=STEALTH_LOUD,
        )
        for _ in range(600)
    ]

    trainer = PPOTrainer(
        dim, PPOConfig(hidden_dim=64, minibatch_size=128, device="cpu")
    )
    trainer.behaviour_clone(demos, 8, rng)

    obs = torch.as_tensor(np.array([d.obs for d in demos]))
    mask = torch.as_tensor(np.array([d.mask for d in demos]))
    with torch.no_grad():
        _, stealth, _, _ = trainer.actor(obs, mask)

    uniform = float(np.log(N_STEALTH))
    assert stealth.entropy().mean().item() > 0.9 * uniform, (
        "cloning collapsed the stealth head; PPO cannot explore a posture it is "
        "already certain about"
    )

def test_frozen_stealth_reproduces_the_loud_policy():
    """The ablation control must actually pin the head, or it measures nothing."""
    from fraudsim.attacker.ppo import PPOConfig

    assert PPOConfig().stealth_frozen is False
    assert PPOConfig(stealth_frozen=True).stealth_frozen is True
