"""PPO over the masked mixed-action policy.

The mechanics are the reference's, on our flat state: generalised advantage
estimation, a clipped surrogate with separate actor and critic optimisers, an
entropy bonus, and gradient clipping. Two things are added that the attacker
needs and the reference did not have.

The policy is behaviour-cloned from scripted demonstrations before any PPO step,
because a cold policy in this world reaches a cash-out perhaps once in tens of
thousands of episodes and, seeing only negative reward, learns to do nothing. The
clone starts it from a working strategy so PPO refines rather than discovers.

And for the first updates the surrogate carries a penalty on the divergence from
the cloned policy, annealed to zero. Without it the first noisy advantage
estimate can wipe out the clone, and the run is back to a cold start having paid
for the demonstrations.

Sizes and counts are configuration. The defaults are for a real run; a smoke
test passes small ones. Nothing assumes a toy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import AttackEnv
from .nets import Actor, Critic, NetConfig


@dataclass(slots=True)
class PPOConfig:
    """Training hyperparameters, all with real-run defaults."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    # No value coefficient: the critic has its own optimiser and its loss is
    # backpropagated separately, so there is no combined loss to weight it in.
    max_grad_norm: float = 0.5
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    epochs_per_update: int = 4
    minibatch_size: int = 256
    hidden_dim: int = 256
    n_layers: int = 2
    # Behaviour-cloning regularisation, annealed over this many updates.
    bc_kl_coef: float = 0.5
    bc_kl_anneal_updates: int = 20
    # "auto" takes the GPU when one is visible and falls back to the CPU when it
    # is not, so CUDA_VISIBLE_DEVICES alone decides where this runs. An explicit
    # "cpu" or "cuda" overrides that.
    device: str = "auto"


@dataclass
class RolloutBatch:
    """A collected set of transitions as tensors, ready for the update."""

    obs: torch.Tensor
    mask: torch.Tensor
    action_idx: torch.Tensor
    amount_raw: torch.Tensor
    delay_raw: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    advantage: torch.Tensor
    ret: torch.Tensor

    def __len__(self) -> int:
        return self.obs.shape[0]

    def minibatches(self, size: int, rng: np.random.Generator):
        idx = rng.permutation(len(self))
        for start in range(0, len(self), size):
            sl = idx[start : start + size]
            yield RolloutBatch(
                obs=self.obs[sl],
                mask=self.mask[sl],
                action_idx=self.action_idx[sl],
                amount_raw=self.amount_raw[sl],
                delay_raw=self.delay_raw[sl],
                log_prob=self.log_prob[sl],
                value=self.value[sl],
                advantage=self.advantage[sl],
                ret=self.ret[sl],
            )


def _resolve_device(spec: str) -> str:
    """Turn a device spec into a concrete device.

    "auto" means take the GPU if one is visible. Nothing else in the system
    chooses a device, so setting CUDA_VISIBLE_DEVICES is enough to place the
    training where it is wanted.
    """
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation, in reverse as the reference does.

    Advantages are the discounted sum of temporal-difference errors; returns are
    advantages plus the baseline. Bootstrapping off `last_value` at a truncation
    keeps a capped episode from being treated as a terminal one.
    """
    advantages = np.zeros(len(rewards), dtype=np.float32)
    future = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        future = delta + gamma * lam * non_terminal * future
        advantages[t] = future
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


class PPOTrainer:
    """Holds the networks and runs collection, BC, critic fit, and PPO."""

    def __init__(self, obs_dim: int, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()
        net_cfg = NetConfig(
            obs_dim=obs_dim, hidden_dim=self.config.hidden_dim, n_layers=self.config.n_layers
        )
        self.device = torch.device(_resolve_device(self.config.device))
        self.actor = Actor(net_cfg).to(self.device)
        self.critic = Critic(net_cfg).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        self._bc_actor: Actor | None = None
        self._updates = 0

    # --------------------------------------------------------------- collect

    def collect(self, make_env, n_episodes: int, rng: np.random.Generator) -> RolloutBatch:
        """Run episodes with the current policy, returning a batch with GAE.

        `make_env` is a thunk producing a fresh `AttackEnv` per episode, so the
        trainer stays agnostic to how targets and the world are set up.
        """
        obs_b, mask_b, act_b, amt_b, dly_b, lp_b, val_b = [], [], [], [], [], [], []
        adv_b, ret_b = [], []

        for _ in range(n_episodes):
            env = make_env()
            obs = env.reset()
            ep_obs, ep_mask, ep_act, ep_amt, ep_dly = [], [], [], [], []
            ep_lp, ep_val, ep_rew, ep_done = [], [], [], []

            done = False
            while not done:
                vec = AttackEnv.encode(obs)
                mask = AttackEnv.mask_vector(obs)
                t_obs = torch.as_tensor(vec, device=self.device).unsqueeze(0)
                t_mask = torch.as_tensor(mask, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    discrete, amount, delay = self.actor(t_obs, t_mask)
                    a_idx = discrete.sample()
                    a_amt = amount.sample()
                    a_dly = delay.sample()
                    log_prob = (
                        discrete.log_prob(a_idx)
                        + amount.log_prob(a_amt)
                        + delay.log_prob(a_dly)
                    )
                    value = self.critic(t_obs)

                nxt, reward, done, _ = env.step(
                    int(a_idx.item()), float(a_amt.item()), float(a_dly.item())
                )

                ep_obs.append(vec)
                ep_mask.append(mask)
                ep_act.append(int(a_idx.item()))
                ep_amt.append(float(a_amt.item()))
                ep_dly.append(float(a_dly.item()))
                ep_lp.append(float(log_prob.item()))
                ep_val.append(float(value.item()))
                ep_rew.append(float(reward))
                ep_done.append(bool(done))
                obs = nxt

            env.close()

            # Bootstrap value at truncation is zero here: an episode ends either
            # terminal or at the action cap, and the cap's continuation value is
            # not modelled, so it is treated as terminal for the advantage.
            adv, ret = compute_gae(
                ep_rew, ep_val, ep_done, 0.0, self.config.gamma, self.config.gae_lambda
            )
            obs_b += ep_obs
            mask_b += ep_mask
            act_b += ep_act
            amt_b += ep_amt
            dly_b += ep_dly
            lp_b += ep_lp
            val_b += ep_val
            adv_b += adv.tolist()
            ret_b += ret.tolist()

        return self._to_batch(obs_b, mask_b, act_b, amt_b, dly_b, lp_b, val_b, adv_b, ret_b)

    def _to_batch(self, obs, mask, act, amt, dly, lp, val, adv, ret) -> RolloutBatch:
        adv_arr = np.asarray(adv, dtype=np.float32)
        # Normalise advantages, the standard PPO stabiliser.
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=self.device)
        return RolloutBatch(
            obs=t(obs),
            mask=t(mask, torch.bool),
            action_idx=t(act, torch.long),
            amount_raw=t(amt),
            delay_raw=t(dly),
            log_prob=t(lp),
            value=t(val),
            advantage=t(adv_arr),
            ret=t(ret),
        )

    # --------------------------------------------------- behaviour cloning

    def behaviour_clone(self, demos, epochs: int, rng: np.random.Generator) -> list[float]:
        """Supervised fit of the actor to scripted (obs, action) demonstrations.

        Cross-entropy on the discrete head, and a regression on the continuous
        heads where a demonstration carried an amount or a delay. Trains the
        actor only; the critic is fit separately, since the demonstrations have
        no value function.
        """
        obs = torch.as_tensor(
            np.asarray([d.obs for d in demos], dtype=np.float32), device=self.device
        )
        mask = torch.as_tensor(
            np.asarray([d.mask for d in demos]), dtype=torch.bool, device=self.device
        )
        act = torch.as_tensor([d.action_idx for d in demos], dtype=torch.long, device=self.device)
        amt = torch.as_tensor([d.amount_raw for d in demos], dtype=torch.float32, device=self.device)
        dly = torch.as_tensor([d.delay_raw for d in demos], dtype=torch.float32, device=self.device)

        losses = []
        for _ in range(epochs):
            idx = rng.permutation(len(demos))
            for start in range(0, len(demos), self.config.minibatch_size):
                sl = idx[start : start + self.config.minibatch_size]
                discrete, amount, delay = self.actor(obs[sl], mask[sl])
                loss = F.cross_entropy(discrete.logits, act[sl])
                loss = loss - amount.log_prob(amt[sl]).mean() * 0.1
                loss = loss - delay.log_prob(dly[sl]).mean() * 0.1
                self.opt_actor.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.opt_actor.step()
                losses.append(float(loss.item()))
        # Freeze a copy of the cloned policy for the KL penalty.
        self._bc_actor = self._snapshot_actor()
        return losses

    def fit_critic(self, batch: RolloutBatch, epochs: int) -> list[float]:
        """Regress the critic onto realised returns.

        Run after cloning and before PPO, so the first advantage estimates rest
        on a critic that has seen the demonstrations' returns rather than noise.
        """
        losses = []
        for _ in range(epochs):
            pred = self.critic(batch.obs)
            loss = F.mse_loss(pred, batch.ret)
            self.opt_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.opt_critic.step()
            losses.append(float(loss.item()))
        return losses

    # --------------------------------------------------------------- update

    def update(self, batch: RolloutBatch, rng: np.random.Generator) -> dict:
        """One PPO update: clipped surrogate, value loss, entropy, BC penalty."""
        cfg = self.config
        bc_coef = self._bc_coef()
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "bc_kl": []}

        for _ in range(cfg.epochs_per_update):
            for mb in batch.minibatches(cfg.minibatch_size, rng):
                log_prob, entropy = self.actor.evaluate(
                    mb.obs, mb.mask, mb.action_idx, mb.amount_raw, mb.delay_raw
                )
                ratio = torch.exp(log_prob - mb.log_prob)
                surr1 = mb.advantage * ratio
                surr2 = mb.advantage * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                policy_loss = -torch.min(surr1, surr2).mean()
                ent = entropy.mean()

                bc_kl = torch.tensor(0.0, device=self.device)
                if bc_coef > 0 and self._bc_actor is not None:
                    bc_kl = self._bc_divergence(mb)

                loss = policy_loss - cfg.entropy_coef * ent + bc_coef * bc_kl
                self.opt_actor.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.opt_actor.step()

                value = self.critic(mb.obs)
                value_loss = F.mse_loss(value, mb.ret)
                self.opt_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt_critic.step()

                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(ent.item()))
                stats["bc_kl"].append(float(bc_kl.item()))

        self._updates += 1
        return {k: float(np.mean(v)) for k, v in stats.items()} | {"bc_coef": bc_coef}

    # -------------------------------------------------------------- internals

    def _bc_coef(self) -> float:
        """The BC-KL coefficient, annealed linearly to zero."""
        if self._bc_actor is None or self._updates >= self.config.bc_kl_anneal_updates:
            return 0.0
        frac = 1.0 - self._updates / max(1, self.config.bc_kl_anneal_updates)
        return self.config.bc_kl_coef * frac

    def _bc_divergence(self, mb: RolloutBatch) -> torch.Tensor:
        """KL from the current discrete policy to the frozen cloned one.

        Keeps the early updates from walking away from the demonstrations before
        the advantages are trustworthy. Only the discrete head is regularised,
        since it carries the strategy the clone captured.
        """
        with torch.no_grad():
            bc_dist, _, _ = self._bc_actor(mb.obs, mb.mask)
        cur_dist, _, _ = self.actor(mb.obs, mb.mask)
        return torch.distributions.kl_divergence(bc_dist, cur_dist).mean()

    def _snapshot_actor(self) -> Actor:
        net_cfg = NetConfig(
            obs_dim=self.actor.trunk[0].in_features,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
        )
        clone = Actor(net_cfg).to(self.device)
        clone.load_state_dict(self.actor.state_dict())
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    # ------------------------------------------------------------- checkpoint

    def save(self, path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "updates": self._updates,
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self._updates = ckpt.get("updates", 0)
