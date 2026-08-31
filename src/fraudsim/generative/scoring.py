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

import hashlib
import re
from dataclasses import dataclass


def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")

_WORD = re.compile(r"[a-zA-Z']+")
_MONEY = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+\.\d{2})")


@dataclass
class TextScorer:
    """Computes the detector-facing scores for a generated item.

    Built with a reference corpus (the real narratives) to score template
    similarity against. Without one it falls back to the items already scored in
    this run, so a pool still gets a similarity signal.

    The reference is hashed into a fixed-width matrix once, so scoring a pool is
    a single matrix product rather than a comparison per (item, reference) pair.
    The pairwise form was quadratic and became the slowest thing in the pipeline
    at a realistic pool size — twenty-four million Counter comparisons for a
    twelve-thousand-item pool against two thousand narratives.
    """

    reference: list[str]
    hash_dim: int = 4096
    _ref_matrix: object = None

    def __post_init__(self) -> None:

        if not self.reference:
            self._ref_matrix = None
            return
        self._ref_matrix = self._hash_matrix(self.reference)

    def _hash_matrix(self, texts: list[str]):
        """Character n-grams hashed into a fixed width, row-normalised.

        Normalising the rows means a dot product is the cosine, so the maximum
        similarity over the whole reference is one matrix product and a row
        maximum.
        """
        import numpy as np

        out = np.zeros((len(texts), self.hash_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = re.sub(r"\s+", " ", text.lower()).strip()
            for j in range(len(t) - 3):
                out[i, _stable_hash(t[j : j + 4]) % self.hash_dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out

    # ------------------------------------------------------------- the scores

    def template_similarity(self, text: str) -> float:
        """Max cosine against the reference. Higher = more templated."""
        import numpy as np

        if self._ref_matrix is None:
            return 0.0
        vec = self._hash_matrix([text])[0]
        return float(np.max(self._ref_matrix @ vec))

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
