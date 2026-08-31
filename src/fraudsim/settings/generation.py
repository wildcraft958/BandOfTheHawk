"""Text-generation parameters: which models, how much they produce, how it is embedded.

Model names and sampling settings were literals in `generative/loader.py` and
`generative/embed.py`, so swapping the model or shortening a run meant editing
Python. Every default here is the value the code used before it moved.

Nothing in this file may import torch or transformers. It is plain data on the
runtime side of the import firewall; the generative tier reads it and loads.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .base import PositiveFloat, StrictModel, UnitInterval


class GenerationConfig(StrictModel):
    """The causal model that writes dispute, ticket and refund text (Mode A)."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: Annotated[int, Field(ge=1, le=8192)] = 400
    temperature: PositiveFloat = 0.9
    top_p: UnitInterval = 0.95
    batch_size: Annotated[int, Field(ge=1, le=1024)] = 16


class EmbeddingConfig(StrictModel):
    """The embedder whose vectors the text expert scores against."""

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    # The model's native width, before truncation.
    full_dim: Annotated[int, Field(ge=1)] = 1024
    # Matryoshka truncation: the first N dimensions are trained to stand alone,
    # so a shorter vector costs little and keeps the feature table narrow.
    truncate_dim: Annotated[int, Field(ge=1)] | None = 256
    batch_size: Annotated[int, Field(ge=1, le=4096)] = 64


class GenerativeConfig(StrictModel):
    """Everything the generative tier reads."""

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
