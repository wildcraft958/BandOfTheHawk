"""The same seed produces the same run.

The repo claimed reproducibility and did not have it. `RngHub` seeds named numpy
streams properly, but torch was never seeded anywhere: network initialisation
drew from the unseeded global generator, and so did every action, amount and
delay sample, because they go through `torch.distributions`. Two runs at the
same seed produced different attackers.

These tests fail on the code as it was.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch", reason='install the "rl" extra'
)

import numpy as np

from fraudsim.attacker.env import AttackEnv
from fraudsim.attacker.ppo import PPOTrainer
from fraudsim.rng import RngHub, set_seed
from fraudsim.settings.training import PPOConfig


def trainer(seed: int) -> PPOTrainer:
    return PPOTrainer(
        AttackEnv.obs_dim(),
        PPOConfig(hidden_dim=16, n_layers=1, minibatch_size=8, device="cpu", seed=seed),
    )

def weights(model: PPOTrainer) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.actor.parameters()]

def test_the_same_seed_initialises_the_same_network() -> None:
    """Network init drew from the unseeded global generator."""
    a, b = weights(trainer(7)), weights(trainer(7))
    for left, right in zip(a, b, strict=True):
        assert np.array_equal(left, right)

def test_a_different_seed_initialises_a_different_network() -> None:
    """Guards the guard: equal weights everywhere would pass the test above."""
    a, b = weights(trainer(7)), weights(trainer(8))
    assert any(not np.array_equal(x, y) for x, y in zip(a, b, strict=True))

def test_the_same_seed_samples_the_same_actions() -> None:
    """Action, amount and delay all go through torch.distributions."""

    def sample(seed: int) -> list[float]:
        model = trainer(seed)
        obs = torch.zeros(1, AttackEnv.obs_dim())
        mask = torch.ones(1, model.actor.discrete.out_features, dtype=torch.bool)
        drawn: list[float] = []
        with torch.no_grad():
            for _ in range(8):
                discrete, stealth, amount, delay = model.actor(obs, mask)
                drawn += [
                    float(discrete.sample().item()),
                    float(stealth.sample().item()),
                    float(amount.sample().item()),
                    float(delay.sample().item()),
                ]
        return drawn

    assert sample(11) == sample(11)

def test_an_unseeded_config_still_works() -> None:
    """seed=None keeps the old behaviour rather than failing."""
    model = PPOTrainer(
        AttackEnv.obs_dim(),
        PPOConfig(hidden_dim=16, n_layers=1, minibatch_size=8, device="cpu"),
    )
    assert model.config.seed is None

def test_set_seed_covers_python_and_numpy() -> None:
    import random

    set_seed(3)
    first = (random.random(), float(np.random.random()))
    set_seed(3)
    assert (random.random(), float(np.random.random())) == first

def test_set_seed_returns_the_value_it_applied() -> None:
    assert set_seed(42) == 42

def test_named_streams_are_independent_of_declaration_order() -> None:
    """Adding a stream must not shift an existing one."""
    a = RngHub(5)
    b = RngHub(5)
    b.stream("unrelated")
    assert np.array_equal(a.stream("victims").random(4), b.stream("victims").random(4))

# The derived seeds a run uses, by purpose. These are offsets from the root seed
# rather than named RngHub streams: they are distinct and deterministic, and
# renaming them would change every number in the repository's results for no
# correctness gain. The test is here so a new one cannot silently collide with
# an existing one and make two independent draws identical.
SEED_OFFSETS = {
    "episode_runner": 1,
    "benign_sweep": 2,
    "circadian_marginal": 17,
    "zero_shot_runner": 700,
    "zero_shot_benign": 800,
    "arrival_simulation": 4242,
    "live_benign_base": 5000,
}

def test_derived_seeds_do_not_collide() -> None:
    """Two purposes sharing an offset would draw the same numbers."""
    assert len(set(SEED_OFFSETS.values())) == len(SEED_OFFSETS)

def test_live_benign_rounds_stay_clear_of_other_offsets() -> None:
    """It is base + round number, so it must not walk into a neighbour."""
    base = SEED_OFFSETS["live_benign_base"]
    others = {v for k, v in SEED_OFFSETS.items() if k != "live_benign_base"}
    assert not {base + n for n in range(1000)} & others
