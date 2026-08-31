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

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PPOConfig lives in settings.training: it is configuration, and keeping a
# second copy here is how the CLI defaults came to disagree with it.
from ..settings.training import PPOConfig
from .env import AttackEnv
from .nets import Actor, Critic, NetConfig


@dataclass(frozen=True, slots=True)
class PPOStats:
    """What one update reports. Every other metrics record here is a dataclass.

    This one was a bare dict, so `stats["entropy"]` at two call sites in the
    co-adaptation loop was unchecked and a rename here would have failed there
    at runtime rather than at import.
    """

    policy_loss: float
    value_loss: float
    entropy: float
    bc_kl: float
    bc_coef: float


@dataclass
class RolloutBatch:
    """A collected set of transitions as tensors, ready for the update."""

    obs: torch.Tensor
    mask: torch.Tensor
    action_idx: torch.Tensor
    stealth_idx: torch.Tensor
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
                stealth_idx=self.stealth_idx[sl],
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
        # Seed before the networks exist. Initialisation draws from torch's
        # global generator, and so does every action, amount and delay sample,
        # because they go through torch.distributions. Unseeded, two runs at the
        # same config seed produced different attackers, which is what the
        # repository's reproducibility claim rested on.
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
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

    def collect(
        self,
        make_env,
        n_episodes: int,
        rng: np.random.Generator,
        on_episode_end=None,
    ) -> RolloutBatch:
        """Run episodes with the current policy, returning a batch with GAE.

        `make_env` is a thunk producing a fresh `AttackEnv` per episode, so the
        trainer stays agnostic to how targets and the world are set up.

        `on_episode_end`, where given, is called with the episode's env and its
        total reward as each episode closes. The victim-selection bandit uses it
        to credit the return to the features that chose that victim; the trainer
        itself stays unaware of what the caller does with it.
        """
        obs_b, mask_b, act_b, amt_b, dly_b, lp_b, val_b = [], [], [], [], [], [], []
        stl_b: list[int] = []
        adv_b, ret_b = [], []

        for _ in range(n_episodes):
            env = make_env()
            obs = env.reset()
            ep_obs, ep_mask, ep_act, ep_amt, ep_dly = [], [], [], [], []
            ep_stl: list[int] = []
            ep_lp, ep_val, ep_rew, ep_done = [], [], [], []

            done = False
            while not done:
                vec = AttackEnv.encode(obs)
                mask = AttackEnv.mask_vector(obs)
                t_obs = torch.as_tensor(vec, device=self.device).unsqueeze(0)
                t_mask = torch.as_tensor(mask, device=self.device).unsqueeze(0)

                frozen = self.config.stealth_frozen
                with torch.no_grad():
                    discrete, stealth, amount, delay = self.actor(t_obs, t_mask)
                    a_idx = discrete.sample()
                    a_stl = torch.zeros_like(a_idx) if frozen else stealth.sample()
                    a_amt = amount.sample()
                    a_dly = delay.sample()
                    log_prob = (
                        discrete.log_prob(a_idx)
                        + amount.log_prob(a_amt)
                        + delay.log_prob(a_dly)
                    )
                    # The stored log-prob must contain exactly the terms the
                    # update recomputes, or the importance ratio is over two
                    # different quantities and the clipping means nothing.
                    if not frozen:
                        log_prob = log_prob + stealth.log_prob(a_stl)
                    value = self.critic(t_obs)

                nxt, reward, done, _ = env.step(
                    int(a_idx.item()),
                    float(a_amt.item()),
                    float(a_dly.item()),
                    int(a_stl.item()),
                )

                ep_obs.append(vec)
                ep_mask.append(mask)
                ep_act.append(int(a_idx.item()))
                ep_stl.append(int(a_stl.item()))
                ep_amt.append(float(a_amt.item()))
                ep_dly.append(float(a_dly.item()))
                ep_lp.append(float(log_prob.item()))
                ep_val.append(float(value.item()))
                ep_rew.append(float(reward))
                ep_done.append(bool(done))
                obs = nxt

            env.close()
            if on_episode_end is not None:
                on_episode_end(env, float(sum(ep_rew)))

            # Bootstrap value at truncation is zero here: an episode ends either
            # terminal or at the action cap, and the cap's continuation value is
            # not modelled, so it is treated as terminal for the advantage.
            adv, ret = compute_gae(
                ep_rew, ep_val, ep_done, 0.0, self.config.gamma, self.config.gae_lambda
            )
            obs_b += ep_obs
            mask_b += ep_mask
            act_b += ep_act
            stl_b += ep_stl
            amt_b += ep_amt
            dly_b += ep_dly
            lp_b += ep_lp
            val_b += ep_val
            adv_b += adv.tolist()
            ret_b += ret.tolist()

        return self._to_batch(
            obs_b, mask_b, act_b, stl_b, amt_b, dly_b, lp_b, val_b, adv_b, ret_b
        )

    def _to_batch(self, obs, mask, act, stl, amt, dly, lp, val, adv, ret) -> RolloutBatch:
        adv_arr = np.asarray(adv, dtype=np.float32)
        # Normalise advantages, the standard PPO stabiliser.
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
        def t(x, dt=torch.float32):
            """One field of the batch as a tensor on the trainer's device."""
            return torch.as_tensor(np.asarray(x), dtype=dt, device=self.device)

        return RolloutBatch(
            obs=t(obs),
            mask=t(mask, torch.bool),
            action_idx=t(act, torch.long),
            stealth_idx=t(stl, torch.long),
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

        The stealth head is deliberately *not* cloned, and that is a correction
        of an earlier mistake rather than an omission.

        The scripts predate the head and are uniformly loud, so cloning it fit
        the head to a constant: measured, the posture entropy fell from 1.382 to
        0.031, leaving the policy 99.6% certain of the one posture it most needed
        to question. The KL-to-BC penalty then held it there through the first
        several updates — which is exactly when the defender first refits and
        when an alternative posture would have to be discovered. The head existed,
        worked, and was never used.

        The demonstrations carry no information about posture, because posture
        did not exist when they were written. Fitting a head to data that does
        not speak to it manufactures a confident answer out of nothing. Left
        alone, the head starts near uniform and PPO chooses, which is the honest
        division: clone what the scripts actually demonstrate, learn what they
        do not.
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
                discrete, _stealth, amount, delay = self.actor(obs[sl], mask[sl])
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

    def critic_relative_error(self, batch: RolloutBatch) -> float:
        """Root-mean-square critic error as a fraction of the return scale.

        Absolute mean-squared error says nothing on its own: it scales with the
        square of the reward, so changing a reward weight moves it by orders of
        magnitude while the fit is exactly as good. Dividing the root error by
        the spread of the returns gives a number that means the same thing in
        every run.
        """
        with torch.no_grad():
            pred = self.critic(batch.obs)
            rmse = float(torch.sqrt(F.mse_loss(pred, batch.ret)).item())
        spread = float(batch.ret.std().item())
        return rmse / spread if spread > 1e-9 else float("nan")

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

    def update(self, batch: RolloutBatch, rng: np.random.Generator) -> PPOStats:
        """One PPO update: clipped surrogate, value loss, entropy, BC penalty."""
        cfg = self.config
        bc_coef = self._bc_coef()
        stats: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "bc_kl": []
        }

        for _ in range(cfg.epochs_per_update):
            for mb in batch.minibatches(cfg.minibatch_size, rng):
                log_prob, entropy = self.actor.evaluate(
                    mb.obs,
                    mb.mask,
                    mb.action_idx,
                    mb.stealth_idx,
                    mb.amount_raw,
                    mb.delay_raw,
                    include_stealth=not cfg.stealth_frozen,
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
        means = {name: float(np.mean(values)) for name, values in stats.items()}
        return PPOStats(**means, bc_coef=bc_coef)

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
        the advantages are trustworthy. Only the action head is regularised,
        since it carries the strategy the clone captured. The stealth head is
        excluded for the same reason it is not cloned: the demonstrations say
        nothing about posture, so anchoring to them would hold the policy at
        whatever the untrained head happened to prefer, through precisely the
        updates in which it needs to explore.
        """
        with torch.no_grad():
            bc_dist, _, _, _ = self._bc_actor(mb.obs, mb.mask)
        cur_dist, _, _, _ = self.actor(mb.obs, mb.mask)
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
                # The architecture the weights belong to. Without it a checkpoint
                # can be loaded into a differently shaped network and fail with a
                # key error that says nothing about the cause.
                "obs_dim": self.actor.trunk[0].in_features,
                "hidden_dim": self.config.hidden_dim,
                "n_layers": self.config.n_layers,
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        saved_obs = ckpt.get("obs_dim")
        current_obs = self.actor.trunk[0].in_features
        if saved_obs is not None and saved_obs != current_obs:
            raise ValueError(
                f"checkpoint was trained with obs_dim={saved_obs}; "
                f"this trainer has obs_dim={current_obs}"
            )
        saved = (ckpt.get("hidden_dim"), ckpt.get("n_layers"))
        current = (self.config.hidden_dim, self.config.n_layers)
        if saved != (None, None) and saved != current:
            raise ValueError(
                f"checkpoint was trained with hidden_dim={saved[0]}, n_layers={saved[1]}; "
                f"this trainer has hidden_dim={current[0]}, n_layers={current[1]}"
            )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self._updates = ckpt.get("updates", 0)
