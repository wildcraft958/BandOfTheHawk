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

A second discrete head chooses *how* the action is carried out rather than which
action it is. The first version of this policy had no such head, and the
consequence was not a worse attacker but an attacker with no strategy space at
all: every authorisation went through whichever device the world picked, which is
the newest, so every attack carried a device barely minutes old and the detector
needed one feature to end the arms race. The modifier head is what makes stealth
representable. It is unmasked, because every modifier is meaningful at every
stage; where one has no effect on a given action the environment ignores it.

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

# The stealth-modifier space. Four ways to carry out whatever action was chosen,
# ordered so that index 0 is the behaviour the policy had before this head
# existed — which makes it the natural behaviour-cloning target and keeps the
# clone's meaning unchanged.
#
#   0 LOUD        whatever binding the world prefers (the newest), online entry
#   1 AGED        route through the card's oldest surviving binding
#   2 AGED_COOL   as AGED, and wait out a floor before acting
#   3 ROTATE      move to another card in the dump before acting
#
# These are intents, not mechanisms: the environment turns each into concrete
# action fields. The policy names a posture and never sees the graph that
# resolves it.
N_STEALTH = 4
STEALTH_LOUD, STEALTH_AGED, STEALTH_AGED_COOL, STEALTH_ROTATE = range(N_STEALTH)

STEALTH_NAMES = ("loud", "aged", "aged_cool", "rotate")

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
        # How to carry the action out, as opposed to which action it is. Shares
        # the trunk because posture and choice are decided in one context.
        self.stealth = nn.Linear(config.hidden_dim, N_STEALTH)
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
        stealth = Categorical(logits=self.stealth(h))

        amount = Normal(self.amount_mean(h).squeeze(-1), self.amount_log_std.exp())
        delay = Normal(self.delay_mean(h).squeeze(-1), self.delay_log_std.exp())
        return discrete, stealth, amount, delay

    def evaluate(
        self,
        obs,
        mask,
        action_idx,
        stealth_idx,
        amount_raw,
        delay_raw,
        include_stealth: bool = True,
    ):
        """Log-prob and entropy of stored actions, for the surrogate loss.

        The continuous log-probs are of the raw pre-squash samples, matching how
        they were stored at collection time, so the ratio is over the same
        quantity on both sides. The stealth head joins the joint log-prob rather
        than being trained separately: posture and action are one decision, and
        splitting them would let the surrogate credit a good return to the action
        while leaving the posture that made it possible unreinforced.

        `include_stealth=False` drops the head from both terms, which is what the
        ablation's control arm needs. Pinning the sampled posture while leaving
        the head in the loss would not reproduce the earlier policy: the entropy
        bonus would still push a head whose output is discarded, and its gradient
        would still reach the shared trunk. The control would then differ from
        the treatment by more than the one thing under measurement.
        """
        discrete, stealth, amount, delay = self.forward(obs, mask)
        log_prob = (
            discrete.log_prob(action_idx)
            + amount.log_prob(amount_raw)
            + delay.log_prob(delay_raw)
        )
        entropy = discrete.entropy() + amount.entropy() + delay.entropy()
        if include_stealth:
            log_prob = log_prob + stealth.log_prob(stealth_idx)
            entropy = entropy + stealth.entropy()
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
