"""The policy and value networks.

The reference this is ported from operates on a graph state; ours is flat — an
observation is a stage, a legal-action mask and a feature mapping — so the
encoder is a multilayer perceptron rather than a graph network. Everything else
follows the reference: a masked categorical over the discrete actions, and the
same clipped-surrogate training the PPO module drives.

The action space is mixed. Which of the twenty actions to take is discrete and
masked by legality before the softmax, so the policy never places probability on
an action the stage forbids. How much to spend and how long to wait are
continuous, drawn from squashed Gaussians and rescaled into their ranges, since
those are the decisions a stage does not force.

Sizes are configuration, not constants baked in. The defaults are set for a real
run; a smoke test passes smaller ones. Nothing here assumes a toy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from ..engine.actions import N_ACTIONS
from ..engine.stages import Stage

N_STAGES = len(Stage)

# The continuous action ranges. Amount is log-scaled because spend spans orders
# of magnitude; delay is in minutes.
AMOUNT_MIN, AMOUNT_MAX = 1.0, 5000.0
DELAY_MIN, DELAY_MAX = 0.0, 72.0 * 60.0

# A masked logit is pushed here rather than to -inf, which would produce NaNs in
# the log-prob of a fully-masked row; no row is ever fully masked in practice,
# but the finite floor keeps the maths safe.
MASK_FILL = -1e9


@dataclass(slots=True)
class NetConfig:
    """Network sizes and the observation width, resolved once.

    `obs_dim` is set by the environment from the observation it produces, so the
    encoder matches the features actually supplied rather than a guessed width.
    """

    obs_dim: int
    hidden_dim: int = 256
    n_layers: int = 2


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(max(0, layers - 1)):
        mods += [nn.Linear(hidden, hidden), nn.Tanh()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class Actor(nn.Module):
    """Masked discrete head plus two continuous heads.

    Returns a distribution the caller samples, and can evaluate the log-prob and
    entropy of stored actions for the PPO update. The continuous heads share the
    trunk with the discrete one, since amount and delay are decisions made in the
    same context as which action to take.
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.trunk = _mlp(config.obs_dim, config.hidden_dim, config.hidden_dim, config.n_layers)
        self.act_relu = nn.Tanh()
        self.discrete = nn.Linear(config.hidden_dim, N_ACTIONS)
        # Continuous heads output a mean; the log-std is a free parameter, the
        # standard choice for a stable continuous policy.
        self.amount_mean = nn.Linear(config.hidden_dim, 1)
        self.delay_mean = nn.Linear(config.hidden_dim, 1)
        self.amount_log_std = nn.Parameter(torch.zeros(1))
        self.delay_log_std = nn.Parameter(torch.zeros(1))

    def _trunk(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act_relu(self.trunk(obs))

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """Distributions for one batch of observations.

        The mask is applied to the discrete logits before the softmax, so the
        categorical never proposes an illegal action. The continuous heads are
        unconditioned on the mask, since amount and delay are legal wherever an
        acting action is.
        """
        h = self._trunk(obs)
        logits = self.discrete(h)
        logits = torch.where(mask, logits, torch.full_like(logits, MASK_FILL))
        discrete = Categorical(logits=logits)

        amount = Normal(self.amount_mean(h).squeeze(-1), self.amount_log_std.exp())
        delay = Normal(self.delay_mean(h).squeeze(-1), self.delay_log_std.exp())
        return discrete, amount, delay

    def evaluate(self, obs, mask, action_idx, amount_raw, delay_raw):
        """Log-prob and entropy of stored actions, for the surrogate loss.

        The continuous log-probs are of the raw pre-squash samples, matching how
        they were stored at collection time, so the ratio is over the same
        quantity on both sides.
        """
        discrete, amount, delay = self.forward(obs, mask)
        log_prob = (
            discrete.log_prob(action_idx)
            + amount.log_prob(amount_raw)
            + delay.log_prob(delay_raw)
        )
        entropy = discrete.entropy() + amount.entropy() + delay.entropy()
        return log_prob, entropy


class Critic(nn.Module):
    """A value head over the observation."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.net = _mlp(config.obs_dim, config.hidden_dim, 1, config.n_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def squash_amount(raw: torch.Tensor) -> torch.Tensor:
    """A raw Gaussian sample to an amount in range, monotonically.

    A sigmoid maps the real line to (0, 1), then a log-scale interpolation lands
    it in [AMOUNT_MIN, AMOUNT_MAX], so the policy explores across orders of
    magnitude rather than linearly.
    """
    import math

    unit = torch.sigmoid(raw)
    lo, hi = math.log(AMOUNT_MIN), math.log(AMOUNT_MAX)
    return torch.exp(lo + unit * (hi - lo))


def squash_delay(raw: torch.Tensor) -> torch.Tensor:
    unit = torch.sigmoid(raw)
    return DELAY_MIN + unit * (DELAY_MAX - DELAY_MIN)
