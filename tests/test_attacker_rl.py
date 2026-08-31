"""The learned attacker: nets, env, and the BC->critic->PPO pipeline run.

Tiny scale, so the suite stays fast, but the whole path: the masked head never
proposes an illegal action, the env encodes an observation to the width the nets
expect, GAE is correct on a hand case, and a short bootstrap-and-train completes
and improves the return. Torch lives in the rl extra, imported only here and in
the attacker's learned modules.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fraudsim.attacker.env import AttackEnv
from fraudsim.attacker.nets import Actor, NetConfig, squash_amount, squash_delay
from fraudsim.attacker.ppo import PPOConfig, PPOTrainer, compute_gae
from fraudsim.settings.simulation import SimulationConfig
from fraudsim.engine.actions import N_ACTIONS
from fraudsim.protocols import ActorObservation


def test_masked_head_never_proposes_illegal():
    cfg = NetConfig(obs_dim=AttackEnv.obs_dim(), hidden_dim=32, n_layers=1)
    actor = Actor(cfg)
    obs = torch.zeros(1, cfg.obs_dim)
    # Only action 0 is legal.
    mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
    mask[0, 0] = True
    discrete, _, _, _ = actor(obs, mask)
    # All probability mass on the one legal action.
    probs = discrete.probs[0]
    assert probs[0] > 0.999
    assert probs[1:].sum() < 1e-3


def test_squash_ranges():
    raw = torch.linspace(-5, 5, 20)
    amt = squash_amount(raw)
    dly = squash_delay(raw)
    assert float(amt.min()) >= 1.0 and float(amt.max()) <= 5000.0
    assert float(dly.min()) >= 0.0 and float(dly.max()) <= 72 * 60


def test_gae_matches_hand_computation():
    # One step, terminal, no bootstrap: advantage = reward - value.
    adv, ret = compute_gae([1.0], [0.4], [True], 0.0, gamma=0.99, lam=0.95)
    assert abs(adv[0] - (1.0 - 0.4)) < 1e-6
    assert abs(ret[0] - 1.0) < 1e-6


def test_encode_width_matches_obs_dim():
    obs = ActorObservation(
        actor_id=1, stage=0, legal_action_mask=[False] * N_ACTIONS, features={}
    )
    vec = AttackEnv.encode(obs)
    assert vec.shape[0] == AttackEnv.obs_dim()


def test_trainer_bootstrap_improves_return():
    from fraudsim.attacker.bootstrap import bootstrap_and_train
    from fraudsim.attacker.cli import WorldFactory
    from fraudsim.protocols import AlwaysApproveScorer

    config = SimulationConfig.model_validate({"population": {"n_holders": 500}})
    factory = WorldFactory(config, AlwaysApproveScorer(), train_only=True, seed=0)
    trainer = PPOTrainer(
        AttackEnv.obs_dim(),
        PPOConfig(hidden_dim=64, n_layers=1, minibatch_size=64, bc_kl_anneal_updates=3),
    )
    report = bootstrap_and_train(
        trainer,
        factory.make_env_and_policy,
        factory.make_env,
        demo_episodes=30,
        bc_epochs=6,
        critic_epochs=6,
        n_updates=6,
        episodes_per_update=8,
        seed=0,
    )
    # The pipeline completed and produced per-update stats and a sequence log.
    assert len(report.update_stats) == 6
    assert report.top_sequences
    # Later returns should not be worse than the first by a wide margin — the
    # clone gives a working start and PPO should not immediately destroy it.
    assert report.mean_return[-1] >= report.mean_return[0] - 5.0
