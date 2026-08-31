"""Co-adaptation measurement: success, zero-shot recall, action sequences."""

from __future__ import annotations

from .coadapt_report import _REFUSING_ACTIONS
from .run import EpisodeRunner


def measure_success(sim, trainer, make_env_fn, stealth_frozen: bool, episodes: int) -> float:
    """What the attacker extracts per episode under the defender in force.

    Value extracted per episode covers every monetisation channel, so it falls
    when the defender closes one and recovers when the attacker finds another.
    """
    import torch

    from ..attacker.env import AttackEnv

    snapshot = len(sim.log)
    total_value = 0.0
    for _ in range(episodes):
        env = make_env_fn()
        obs = env.reset()
        done = False
        while not done:
            vec = torch.as_tensor(
                AttackEnv.encode(obs), device=trainer.device
            ).unsqueeze(0)
            mask = torch.as_tensor(
                AttackEnv.mask_vector(obs), device=trainer.device
            ).unsqueeze(0)
            with torch.no_grad():
                discrete, stealth, amount, delay = trainer.actor(vec, mask)
                a_idx = int(discrete.sample().item())
                a_stl = 0 if stealth_frozen else int(stealth.sample().item())
                a_amt = float(amount.sample().item())
                a_dly = float(delay.sample().item())
            obs, _, done, outcome = env.step(a_idx, a_amt, a_dly, a_stl)
            total_value += float(outcome.value_extracted)
        env.close()
    sim.log.truncate(snapshot)
    return total_value / max(episodes, 1)


def refusal_rate(defender, events, sample_size: int) -> float:
    """Share of events the defender in force would refuse."""
    sample = events[:sample_size]
    if not sample:
        return 0.0
    refused = sum(
        1
        for event in sample
        if defender.score(event).action in _REFUSING_ACTIONS
    )
    return refused / len(sample)


def zero_shot_recall(sim, config, defender, seed: int, holdouts) -> dict[str, float]:
    """Recall on verticals held out of training."""
    recalls: dict[str, float] = {}
    for vertical in holdouts:
        recalls[vertical] = _recall_on_vertical(sim, config, defender, vertical, seed)
    return recalls


def _recall_on_vertical(sim, config, defender, vertical: str, seed: int) -> float:
    runner = _SingleVerticalRunner(sim, config, vertical, seed=seed + 700)
    before = len(sim.log)
    runner.run(benign_seed=seed + 800)
    new_events = sim.log.events[before:]

    from ..features.schema import AuthAttemptEvent

    fraud = [e for e in new_events if isinstance(e, AuthAttemptEvent) and e.is_fraud]
    if not fraud:
        return 0.0
    caught = sum(1 for e in fraud if defender.score(e).risk_score >= 0.5)
    return caught / len(fraud)


def log_sequences(sim, trainer, make_env_fn, stealth_frozen: bool, episodes: int):
    """Top action sequences the trained attacker produces."""
    from collections import Counter

    import torch

    from ..attacker.env import AttackEnv
    from ..attacker.nets import STEALTH_LOUD, STEALTH_NAMES
    from ..engine.actions import ACTION_ORDER
    from ..engine.outcome import OutcomeCode

    snapshot = len(sim.log)
    counter: Counter = Counter()
    for _ in range(episodes):
        env = make_env_fn()
        obs = env.reset()
        names: list[str] = []
        done = False
        while not done:
            vec = torch.as_tensor(
                AttackEnv.encode(obs), device=trainer.device
            ).unsqueeze(0)
            mask = torch.as_tensor(
                AttackEnv.mask_vector(obs), device=trainer.device
            ).unsqueeze(0)
            with torch.no_grad():
                discrete, stealth, amount, delay = trainer.actor(vec, mask)
                a_idx = int(discrete.probs.argmax().item())
                a_stl = 0 if stealth_frozen else int(stealth.probs.argmax().item())
                a_amt = float(amount.mean.item())
                a_dly = float(delay.mean.item())
            suffix = "" if a_stl == STEALTH_LOUD else f"[{STEALTH_NAMES[a_stl]}]"
            obs, _, done, outcome = env.step(a_idx, a_amt, a_dly, a_stl)
            if outcome.code is OutcomeCode.ILLEGAL:
                suffix += "!illegal"
            elif outcome.code is OutcomeCode.FAILED:
                suffix += "!failed"
            names.append(ACTION_ORDER[a_idx].value + suffix)
        env.close()
        counter[">".join(names)] += 1
    sim.log.truncate(snapshot)
    return counter.most_common(8)


def mask_table(table, mask):
    """A row subset of a feature table, keeping every aligned column."""
    from ..defender.table import FeatureTable

    return FeatureTable(
        X=table.X[mask], y=table.y[mask], columns=table.columns,
        event_type=table.event_type[mask], is_warm_start=table.is_warm_start[mask],
        episode_id=table.episode_id[mask], group=table.group[mask],
        events=table.events[mask],
    )


class _SingleVerticalRunner(EpisodeRunner):
    """An episode runner restricted to one vertical, for zero-shot evaluation."""

    def __init__(self, simulator, config, vertical: str, seed: int = 0):
        super().__init__(simulator, config, seed=seed, train_only=False)
        self._verticals = [vertical]
