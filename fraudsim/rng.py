"""Named random streams.

Every generator draws from its own stream, seeded by a stable hash of its name.
Adding a new stream therefore never shifts the draws of an existing one, which
is what makes a parameter sweep isolate the parameter being swept.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _name_to_int(name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class RngHub:
    """Issues and caches independent generators keyed by name."""

    __slots__ = ("_root_seed", "_streams")

    def __init__(self, seed: int) -> None:
        self._root_seed = seed
        self._streams: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        return self._root_seed

    def stream(self, name: str) -> np.random.Generator:
        cached = self._streams.get(name)
        if cached is None:
            entropy = [self._root_seed, _name_to_int(name)]
            cached = np.random.default_rng(np.random.SeedSequence(entropy))
            self._streams[name] = cached
        return cached

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._streams))

    def reset(self) -> None:
        self._streams.clear()
