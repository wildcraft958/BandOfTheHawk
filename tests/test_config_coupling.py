"""A configured value must reach every side that depends on it.

The scripted attacker has to act against the same channel rules the engine
enforces. Where it hardcoded its own copy, the two agreed at the shipped
configuration and would silently diverge the moment anyone changed it: the
attacker would wait out a cooling-off that was no longer the real one, or
present a face below a liveness bar that had been raised.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudsim.attacker.scripted import build_policy
from fraudsim.protocols import Target
from fraudsim.settings.simulation import SimulationConfig


def config(**channel: float | int) -> SimulationConfig:
    return SimulationConfig.model_validate({"engine": {"channel": channel}})


def target() -> Target:
    return Target(card_id=1, holder_id=1, account_id=1, merchants=(1, 2, 3))


def actions_of(vertical: str, cfg: SimulationConfig) -> list:
    """Every action the policy would take, walked far enough to see the delays."""
    return build_policy(vertical, target(), np.random.default_rng(0), config=cfg)


def test_voice_quality_clears_the_configured_threshold() -> None:
    """Raise the bar and the attacker must raise its artifact to match."""
    raised = config(voice_similarity_threshold=0.97)
    policy = build_policy("voice_clone", target(), np.random.default_rng(0),
                          config=raised)
    quality = policy.voice_quality
    assert quality > 0.97, (
        f"voice quality {quality} no longer clears the configured threshold 0.97"
    )


def test_face_quality_clears_the_configured_threshold() -> None:
    raised = config(liveness_threshold=0.97)
    policy = build_policy("deepfake_onboarding", target(), np.random.default_rng(0),
                          config=raised)
    assert policy.face_quality > 0.97


def test_quality_defaults_match_the_shipped_configuration() -> None:
    """The margins that were hardcoded, now derived, must reproduce them."""
    shipped = SimulationConfig()
    voice = build_policy("voice_clone", target(), np.random.default_rng(0),
                         config=shipped)
    face = build_policy("deepfake_onboarding", target(), np.random.default_rng(0),
                        config=shipped)
    assert voice.voice_quality == pytest.approx(0.90)
    assert face.face_quality == pytest.approx(0.95)


def test_payee_cooling_follows_the_configured_period() -> None:
    """The engine enforces the configured cooling-off; the attacker must wait it."""
    longer = config(payee_cooling_off_hours=48)
    policy = build_policy("mule_layering", target(), np.random.default_rng(0),
                          config=longer)
    assert policy.cooling_minutes == 48 * 60


def test_payee_cooling_default_matches_the_shipped_configuration() -> None:
    policy = build_policy("mule_layering", target(), np.random.default_rng(0),
                          config=SimulationConfig())
    assert policy.cooling_minutes == 24 * 60


def test_engine_and_attacker_agree_on_the_cooling_period() -> None:
    """The number the engine enforces and the number the attacker waits."""
    from fraudsim.clock import MINUTES_PER_HOUR

    cfg = config(payee_cooling_off_hours=36)
    engine_side = cfg.engine.channel.payee_cooling_off_hours * MINUTES_PER_HOUR
    policy = build_policy("mule_layering", target(), np.random.default_rng(0), config=cfg)
    assert policy.cooling_minutes == engine_side


def test_warm_start_clock_has_no_default_contradicting_the_config() -> None:
    """It is unused, but its default said 90 days where the config says 180."""
    import inspect

    from fraudsim.clock import WarmStartClock

    lookback = inspect.signature(WarmStartClock.__init__).parameters["lookback_minutes"]
    assert lookback.default is inspect.Parameter.empty, (
        "a default here restates warm_start.lookback_days and will drift from it"
    )
