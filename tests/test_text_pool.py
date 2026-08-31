"""The text pool builds without a model, and the tier ladder is monotone.

The default generator is the mock, so these tests never load Qwen. They check
what the capability-tier claim rests on: as the tier rises, generated text
carries more checkable detail and reuses less of a fixed skeleton. They also
check the pool is deterministic under a seed and round-trips through disk, since
a run records which corpus it used.
"""

from __future__ import annotations

import numpy as np

from fraudsim.generative.pool import (
    MockGenerator,
    PoolArtifactSource,
    TextPool,
    build_pool,
)
from fraudsim.generative.prompts import PROMPT_FOR_VERTICAL
from fraudsim.generative.scoring import TextScorer
from fraudsim.protocols import ArtifactRequest


def _scorer_over(pool):
    return TextScorer(reference=[e.text for e in pool.entries])


def test_pool_covers_every_key():
    pool = build_pool(per_key=3, seed=0)
    verticals = set(PROMPT_FOR_VERTICAL)
    for v in verticals:
        for tier in (0, 1, 2, 3):
            assert pool.key(v, tier, fraudulent=True)
            assert pool.key(v, tier, fraudulent=False)


def test_entity_consistency_rises_with_tier():
    pool = build_pool(per_key=8, seed=1)
    scorer = _scorer_over(pool)
    by_tier = {t: [] for t in (0, 1, 2, 3)}
    for e in pool.entries:
        by_tier[e.tier].append(scorer.entity_consistency(e.text, e.facts))
    means = [np.mean(by_tier[t]) for t in (0, 1, 2, 3)]
    # Monotone non-decreasing: higher tiers state more of the facts.
    assert means[0] <= means[1] <= means[2] <= means[3]
    assert means[3] > means[0]


def test_template_similarity_falls_with_tier():
    """Higher tiers reuse less of the shared skeleton.

    Each tier is scored against the OTHER tiers' text, never against itself. An
    item compared to a reference containing that same item scores exactly 1.0
    whatever it says, which measures nothing — the point is how close a tier's
    text is to text it did not come from.
    """
    pool = build_pool(per_key=8, seed=2)
    by_tier = {t: [e.text for e in pool.entries if e.tier == t] for t in (0, 1, 2, 3)}

    means = []
    for tier in (0, 1, 2, 3):
        others = [t for k, texts in by_tier.items() if k != tier for t in texts]
        scorer = TextScorer(reference=others)
        scores = [scorer.template_similarity(t) for t in by_tier[tier]]
        means.append(np.mean(scores))

    # The lowest tier is the most templated, the highest the least.
    assert means[0] > means[3]


def test_pool_is_deterministic_under_seed():
    a = build_pool(per_key=4, seed=5)
    b = build_pool(per_key=4, seed=5)
    assert a.fingerprint == b.fingerprint


def test_pool_round_trips_through_disk(tmp_path):
    pool = build_pool(per_key=3, seed=0)
    path = tmp_path / "pool.json"
    pool.save(path)
    loaded = TextPool.load(path)
    assert loaded.fingerprint == pool.fingerprint
    assert loaded.key("friendly_fraud", 3, fraudulent=True)


def test_artifact_source_serves_text_and_scores():
    pool = build_pool(per_key=4, seed=0)
    scorer = _scorer_over(pool)
    source = PoolArtifactSource(pool, scorer=scorer, seed=0)
    artifact = source.generate(ArtifactRequest(tool_name="write_dispute", capability_tier=3))
    assert artifact.content
    assert "template_similarity" in artifact.scores
    assert "entity_consistency" in artifact.scores
    # An unknown tool yields an empty artifact, not an error.
    assert source.generate(ArtifactRequest(tool_name="unknown")).content is None


def test_mock_generator_needs_no_model():
    # Constructing and running the mock must not import torch or transformers.

    gen = MockGenerator()
    from fraudsim.generative.prompts import PromptFacts

    facts = PromptFacts(
        amount=123.45,
        merchant_name="Northgate Electronics",
        bank_name="First National",
        date="2024-03-04",
        persona="a retiree",
        tone="polite but firm",
        detail="a text alert from the bank",
    )
    text = gen.generate("friendly_fraud", 3, True, facts)
    assert "123.45" in text
    # No assertion on sys.modules here. The mock generator must not import
    # torch, but torch may already be loaded by another test in the same
    # process, so its presence proves nothing either way. The previous line
    # asserted `... or True`, which can never fail.
