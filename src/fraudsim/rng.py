"""Named random streams, and the process-wide seeding a run starts from.

Every generator draws from its own stream, seeded by a stable hash of its name.
Adding a new stream therefore never shifts the draws of an existing one, which
is what makes a parameter sweep isolate the parameter being swept.

`set_seed` covers the two global generators the package can reach from here.
Torch is deliberately not seeded in this module: it sits on the runtime side of
the import firewall, and the firewall's AST check walks into function bodies, so
even a lazy `import torch` would fail it. `PPOTrainer` seeds torch from
`training.ppo.seed` instead, in the tier that already carries the dependency.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

from .logs import get_logger

_log = get_logger(__name__)


def set_seed(seed: int) -> int:
    """Seed the global generators and say so, returning the seed applied.

    Named streams from `RngHub` are the preferred source and are unaffected by
    this. It exists for the libraries that reach for a global generator anyway,
    and so a run records the seed it used.
    """
    random.seed(seed)
    np.random.seed(seed)
    _log.info("seed %d", seed)
    return seed


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
