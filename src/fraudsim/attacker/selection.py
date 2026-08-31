"""Choosing whom to attack.

The attack itself is sequential -- phishing unlocks a reset, which unlocks a
binding, which unlocks a spend -- and that is why it is learned with policy
gradient. Choosing the victim is not sequential at all: a card is picked, a
return is observed, and the next episode starts from an unrelated card. Nothing
chosen now changes what is available later. That is a contextual bandit, and
solving it with reinforcement learning would be spending a sequential method on a
problem with no sequence in it.

**What the attacker is allowed to know.** Only what a bought card dump carries:
the BIN tier and roughly how old the card is. Not the credit line, not the
balance, not the tenure, and never the graph -- those are facts the bank holds,
and letting the attacker select on them would quietly answer a different and much
easier question than the one being asked.

**Thompson sampling over a Bayesian linear model.** The reward is a continuous
episode return, so a Beta-Bernoulli bandit is the wrong likelihood; and treating
each of thousands of cards as its own arm would learn nothing, since a card is
rarely seen twice. Modelling the return as linear in the dump features with a
Normal-Inverse-Gamma posterior generalises across cards and learns the noise
scale rather than assuming it.

**The posterior is discounted.** The attacker is improving while the bandit
learns, and the defender refits underneath both, so a return from fifty episodes
ago describes a world that no longer exists. Down-weighting old observations lets
the estimate follow the change instead of averaging across it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Card ages are banded rather than passed raw: a dump tells a buyer roughly how
# old an account is, not the issue timestamp, and a band is what generalises.
AGE_BAND_EDGES_DAYS = (180, 365, 1095)  # under 6m, 6m-1y, 1-3y, over 3y
N_AGE_BANDS = len(AGE_BAND_EDGES_DAYS) + 1
N_BIN_TIERS = 4


def card_context(bin_tier: int, card_age_days: float) -> np.ndarray:
    """The dump-knowable features of one card, as a vector.

    A one-hot BIN tier, a one-hot age band, and an intercept. Everything here is
    on a card dump; nothing here is on a bank statement.
    """
    # Drop-first encoding. A full one-hot for each factor plus an intercept is
    # rank deficient -- each one-hot sums to the intercept -- so the ridge spreads
    # weight arbitrarily across collinear columns and every level comes out with
    # the same coefficient. Dropping the first level of each factor makes the
    # design full rank, and the coefficients read as differences from that
    # reference level.
    bin_hot = np.zeros(N_BIN_TIERS - 1, dtype=np.float64)
    tier = min(max(int(bin_tier), 0), N_BIN_TIERS - 1)
    if tier > 0:
        bin_hot[tier - 1] = 1.0

    age_hot = np.zeros(N_AGE_BANDS - 1, dtype=np.float64)
    band = min(int(np.searchsorted(AGE_BAND_EDGES_DAYS, card_age_days)), N_AGE_BANDS - 1)
    if band > 0:
        age_hot[band - 1] = 1.0

    return np.concatenate([bin_hot, age_hot, [1.0]])


CONTEXT_DIM = (N_BIN_TIERS - 1) + (N_AGE_BANDS - 1) + 1


@dataclass
class ThompsonSelector:
    """Discounted Bayesian linear regression, sampled for selection.

    Keeps the sufficient statistics of a Normal-Inverse-Gamma posterior over the
    weight vector. Selection draws a weight vector from that posterior and takes
    the best-scoring candidate, so exploration is proportional to how uncertain
    the model still is rather than governed by a tuned exploration constant.

    `warmup_updates` holds selection back at the start. The bandit still records
    everything it sees, so when it takes over it has a real posterior rather than
    a cold one; it simply does not act while the attacker is still learning the
    basic attack and a poor return says more about the policy than the victim.
    """

    dim: int = CONTEXT_DIM
    ridge: float = 1.0
    discount: float = 0.99
    warmup_updates: int = 10
    seed: int = 0

    _A: np.ndarray = field(init=False)
    _b: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False)
    _n: int = field(init=False, default=0)
    _updates: int = field(init=False, default=0)
    # Noise-scale statistics, so the variance is learned rather than assumed.
    _sum_sq: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._A = np.eye(self.dim) * self.ridge
        self._b = np.zeros(self.dim)
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------- lifecycle

    @property
    def active(self) -> bool:
        """Whether selection is in force, as opposed to uniform sampling."""
        return self._updates >= self.warmup_updates

    def end_update(self) -> None:
        """Called once per training update, to advance the warm-up counter."""
        self._updates += 1

    # -------------------------------------------------------------- learning

    def record(self, context: np.ndarray, reward: float) -> None:
        """Fold one observed (victim, return) into the posterior.

        Discounted, so the estimate follows the attacker's improvement and the
        defender's refits rather than averaging over a world that has changed.
        """
        self._A = self.discount * self._A + np.outer(context, context)
        self._b = self.discount * self._b + context * reward
        self._sum_sq = self.discount * self._sum_sq + reward * reward
        self._n += 1

    # ------------------------------------------------------------- selecting

    def select(self, contexts: list[np.ndarray]) -> int:
        """Index of the candidate to attack.

        Before the warm-up ends this is a uniform draw, which is the honest
        thing to do while the reward says more about the policy than the victim.
        After it, a weight vector is drawn from the posterior and the best
        candidate under that draw is taken.
        """
        if not contexts:
            raise ValueError("no candidates to select from")
        if not self.active:
            return int(self._rng.integers(len(contexts)))

        weights = self._sample_weights()
        scores = [float(weights @ c) for c in contexts]
        return int(np.argmax(scores))

    def _sample_weights(self) -> np.ndarray:
        """One draw from the posterior over the weight vector.

        Sampled through a Cholesky factor of the precision matrix rather than by
        forming the covariance and handing it to a multivariate normal. Two
        reasons: the discounted precision can pick up tiny negative eigenvalues
        from rounding, which makes a covariance that is not quite
        positive-definite and produces a warning on every draw; and solving
        against the factor avoids inverting the matrix twice.

        If `w ~ N(mu, sigma^2 A^-1)` and `A = L L^T`, then `w = mu + sigma * L^-T z`
        for a standard normal `z`, which is a triangular solve.
        """
        # A jitter on the diagonal keeps the factorisation well conditioned when
        # the discount has shrunk the precision toward singular.
        A = self._A + np.eye(self.dim) * 1e-8
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # Degenerate only in pathological cases; fall back to the mean, which
            # is exploitation without exploration for this one draw.
            return np.linalg.pinv(A) @ self._b

        # mean = A^-1 b, obtained by two triangular solves rather than an inverse.
        y = np.linalg.solve(L, self._b)
        mean = np.linalg.solve(L.T, y)

        dof = max(1.0, self._n - self.dim)
        residual = max(self._sum_sq - float(mean @ self._b), 0.0)
        sigma = np.sqrt(max(residual / dof, 1e-3))

        z = self._rng.standard_normal(self.dim)
        return mean + sigma * np.linalg.solve(L.T, z)

    # ---------------------------------------------------------------- report

    def weights(self) -> np.ndarray:
        """The posterior mean, for reading what the bandit learned to prefer."""
        A = self._A + np.eye(self.dim) * 1e-8
        return np.linalg.solve(A, self._b)

    def describe(self) -> str:
        """What the selector prefers, in terms a reader can check."""
        w = self.weights()
        lines = ["  victim selection (posterior mean by feature)"]
        lines.append("    (coefficients are differences from the reference level)")
        lines.append("    bin tier 0       reference")
        for i in range(1, N_BIN_TIERS):
            lines.append(f"    bin tier {i}      {w[i - 1]:>+8.3f}")
        names = ("under 6m", "6m to 1y", "1y to 3y", "over 3y")
        lines.append(f"    age {names[0]:<12} reference")
        off = N_BIN_TIERS - 1
        for i in range(1, N_AGE_BANDS):
            lines.append(f"    age {names[i]:<12}{w[off + i - 1]:>+8.3f}")
        lines.append(f"    observations   {self._n:>8,}")
        lines.append(f"    selecting      {'yes' if self.active else 'not yet (warm-up)'}")
        return "\n".join(lines)
