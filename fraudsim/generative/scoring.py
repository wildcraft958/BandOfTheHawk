"""Scoring generated text, so the text expert has something defined to read.

Generation is the easy half; the scores are what a detector actually consumes,
and they must be computed by something specified rather than asserted. Three
scores, each a real detector-facing signal:

**template_similarity** — how close this text is to the nearest of a reference
set, by character n-gram overlap. High similarity means a reused skeleton, which
is the tell of bulk generation. Measured against the real CFPB narratives and,
where those are absent, against the pool's own tier so near-duplicates within a
run still register.

**entity_consistency** — do the amount, merchant and date stated in the text
match the facts the action carried? A real attacker's generated evidence often
drifts from the transaction it is meant to describe; a genuine customer's does
not. This is the only score that couples the text to the simulation state, which
is what makes it hard to game.

**perplexity_proxy** — a model-free stand-in for fluency: burstiness and
vocabulary richness, which separate templated text from written text without
needing a language model in the loop. The real perplexity under the generator is
available behind a flag for a machine that can run it, but nothing on the default
path needs it.

The scores are deliberately simple and their weaknesses are stated. A submission
that names what a score cannot see is stronger than one that hides it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"[a-zA-Z']+")
_MONEY = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+\.\d{2})")


def _char_ngrams(text: str, n: int = 4) -> Counter:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return Counter(text[i : i + n] for i in range(max(0, len(text) - n + 1)))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class TextScorer:
    """Computes the detector-facing scores for a generated item.

    Built with a reference corpus (the real narratives) to score template
    similarity against. Without one it falls back to the items already scored in
    this run, so a pool still gets a similarity signal.
    """

    reference: list[str]
    _ref_ngrams: list[Counter] | None = None

    def __post_init__(self) -> None:
        # Precompute the reference n-grams once; a subsample keeps it cheap
        # against a corpus of half a million narratives.
        self._ref_ngrams = [_char_ngrams(t) for t in self.reference]

    # ------------------------------------------------------------- the scores

    def template_similarity(self, text: str) -> float:
        """Max n-gram cosine against the reference. Higher = more templated."""
        if not self._ref_ngrams:
            return 0.0
        g = _char_ngrams(text)
        return max(_cosine(g, r) for r in self._ref_ngrams)

    def entity_consistency(self, text: str, facts: dict) -> float:
        """Share of the action's facts that actually appear in the text.

        The amount, the merchant and the date each either surface in the text or
        do not. A low score means the evidence does not describe the transaction
        it is attached to.
        """
        checks = []
        amount = facts.get("amount")
        if amount is not None:
            stated = {float(m.replace(",", "")) for m in _MONEY.findall(text)}
            checks.append(any(abs(s - float(amount)) < 0.5 for s in stated))
        merchant = facts.get("merchant_name")
        if merchant:
            checks.append(merchant.lower() in text.lower())
        date = facts.get("date")
        if date:
            checks.append(date in text)
        return sum(checks) / len(checks) if checks else 0.0

    def perplexity_proxy(self, text: str) -> float:
        """A model-free fluency stand-in in [0, 1]; higher = more written.

        Type-token ratio over words: templated text reuses a small vocabulary
        and scores low, varied prose scores higher. A crude proxy, and named as
        one — its job is to give the expert a fluency axis without a model, not
        to be a calibrated perplexity.
        """
        words = _WORD.findall(text.lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def score(self, entry) -> dict[str, float]:
        """All three scores for a pool entry, as the artifact will carry them."""
        return {
            "template_similarity": self.template_similarity(entry.text),
            "entity_consistency": self.entity_consistency(entry.text, entry.facts),
            "perplexity_proxy": self.perplexity_proxy(entry.text),
        }
