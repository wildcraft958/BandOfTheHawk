"""Embedding the generated text.

The text expert reads meaning, not only the surface statistics. A generated
dispute and a real one differ in ways a template-overlap score cannot see — tone,
coherence, the shape of the argument — and a dense sentence embedding is what
carries those. This wraps a sentence-transformer so every pool item gets a
vector, computed once, offline, and stored alongside the text.

The model is small (0.6B), so it runs where the pipeline runs; it is loaded only
when embedding is asked for, and the import of sentence-transformers is lazy so
the runtime path never touches it.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")

# A compact embedding model: strong sentence vectors, small enough to run beside
# the rest rather than needing its own machine.
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# The model's full width. Qwen3-Embedding supports Matryoshka truncation to any
# width between 32 and this, and a shorter vector is often the better choice
# here: the text expert fits on the few hundred text events a run produces, and
# a thousand columns against that many rows is mostly noise. The default
# truncation below is chosen on that ratio, not on cost.
from ..settings.generation import EmbeddingConfig
EMBED_FULL_DIM = EmbeddingConfig().full_dim
DEFAULT_TRUNCATE_DIM = 256


class Embedder:
    """Turns text into a dense vector with a sentence-transformer.

    Constructed only when real embedding is wanted. Loading the model is the one
    heavy thing here, and it happens on construction, not import.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        truncate_dim: int | None = DEFAULT_TRUNCATE_DIM,
    ) -> None:
        """Load the model, optionally truncating its output width.

        `truncate_dim` uses the model's Matryoshka property: the first k
        dimensions are themselves a usable embedding, so a shorter vector loses
        far less than a naive slice of an ordinary embedding would. Passing None
        keeps the model's full width.
        """
        from sentence_transformers import SentenceTransformer  # lazy; generative extra

        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, truncate_dim=truncate_dim
        )
        self.name = f"{model_name}@{truncate_dim}" if truncate_dim else model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """A batch of texts to an (n, dim) array, normalised.

        Normalised so cosine and dot product agree and the expert sees vectors on
        a common scale. Batched, since embedding one at a time wastes the model.
        """
        vecs = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """A deterministic stand-in that needs no model.

    Not a semantic embedding — a hashed bag of character n-grams projected to a
    fixed width. It exists so the pool and the text expert have vectors to work
    with when the real model is not being loaded, and so a run is reproducible
    without a download. The real model replaces it wherever semantics matter.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = "hash"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = text.lower()
            for j in range(len(t) - 2):
                gram = t[j : j + 3]
                out[i, _stable_hash(gram) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out
