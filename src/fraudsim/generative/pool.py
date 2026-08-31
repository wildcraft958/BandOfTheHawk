"""The text pool.

Generation is slow and, for a large model, heavy. Calling it inside the
simulation loop would put a model on the hot path and make the run
non-deterministic. Instead text is generated once, offline, into a versioned
pool keyed by (vertical, tier, persona, class), and the artifact source that
feeds the simulator becomes a lookup. The run stays deterministic under a seed
and never waits on a model.

Two generators satisfy the same interface.

`MockGenerator` is the default. It composes text from the facts and the tier
deterministically, with the tier controlling how much checkable detail lands and
how much of a fixed skeleton is reused. It exists so the whole pipeline — pool,
scoring, the text expert — runs on any machine, and so the tier ladder is
exercised without a GPU. It is not trying to fool a human; it is trying to give
the scores something with the right structure to read.

`QwenGenerator` is the real one, and it is opt-in. It is constructed only when a
caller asks for it, loads the model only then, and is never on the default path.
The rest of the system cannot tell which produced the pool.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logs import get_logger
from ..protocols import Artifact, ArtifactRequest
from ..taxonomy import SKELETONS, TOOL_TO_VERTICAL
from .prompts import (
    DETAILS,
    PERSONAS,
    PROMPT_FOR_VERTICAL,
    TONES,
    PromptFacts,
)

_log = get_logger(__name__)

TEXT_VERTICALS = tuple(PROMPT_FOR_VERTICAL)
TIERS = (0, 1, 2, 3)


@dataclass(slots=True)
class PoolEntry:
    """One generated item and the facts it was built from."""

    vertical: str
    tier: int
    fraudulent: bool
    persona: str
    text: str
    facts: dict
    embedding: tuple = ()


# ------------------------------------------------------------------ generators


class MockGenerator:
    """Deterministic text from facts and tier. The default; no model.

    The tier is the whole point. A low tier emits a short line off a fixed
    skeleton, so many low-tier items look alike — high template similarity, thin
    detail. A high tier adds the amount, the merchant, the date and a timeline,
    each phrased around the facts, so items diverge and carry checkable content.
    That is the ordinal ladder the capability claim rests on, produced without a
    model so it runs anywhere.
    """

    name = "mock"

    _SKELETON = SKELETONS

    def generate(self, vertical: str, tier: int, fraudulent: bool, facts: PromptFacts) -> str:
        parts = [self._SKELETON[vertical]]
        # Detail is added as the tier rises. The amount and date are the
        # checkable facts the entity-consistency score reads; withholding them
        # at low tiers is what makes low-tier text score as less consistent.
        if tier >= 1:
            parts.append(f"The amount was ${facts.amount:.2f}.")
        if tier >= 2:
            parts.append(f"This was at {facts.merchant_name} on {facts.date}.")
            parts.append(f"I noticed because {facts.detail}.")
        if tier >= 3:
            parts.append(
                f"I have banked with {facts.bank_name} for years and this has not "
                f"happened before. As a {facts.persona}, I would like a written "
                f"reply. To be clear, the {facts.merchant_name} charge of "
                f"${facts.amount:.2f} on {facts.date} is the one in question."
            )
        return " ".join(parts)


class QwenGenerator:
    """Real generation. Opt-in, loads the model on construction, never default.

    Constructing this is the only thing in the system that loads a model. The
    default pool build never constructs it, so a machine that cannot hold the
    model is never asked to.
    """

    name = "qwen"

    def __init__(self, model_name: str | None = None) -> None:
        from .loader import DEFAULT_MODEL, load_checkpoint  # lazy

        self._checkpoint = load_checkpoint(model_name or DEFAULT_MODEL)

    def generate(self, vertical: str, tier: int, fraudulent: bool, facts: PromptFacts) -> str:
        from .loader import generate_one  # lazy

        builder = PROMPT_FOR_VERTICAL[vertical]
        system, user = builder(facts, tier, fraudulent)
        return generate_one(self._checkpoint, system, user)

    # A batch method, which is what `build_pool` uses. One prompt per forward
    # pass leaves the model idle between items; a corpus of thousands is the
    # difference between minutes and hours.
    supports_batch = True

    def generate_many(self, specs: list, batch_size: int = 16, progress: bool = True) -> list[str]:
        from .loader import generate_batch  # lazy

        prompts = [
            PROMPT_FOR_VERTICAL[vertical](facts, tier, fraudulent)
            for vertical, tier, fraudulent, facts in specs
        ]
        return generate_batch(
            self._checkpoint, prompts, batch_size=batch_size, progress=progress
        )


# ----------------------------------------------------------------- the pool


@dataclass
class TextPool:
    """A generated corpus plus the index the artifact source reads.

    Keyed by (vertical, tier, class), each key holding a list of items. The
    source draws from the matching key, so an action tagged with a vertical and
    tier gets text of that kind without generating anything at run time.
    """

    entries: list[PoolEntry] = field(default_factory=list)
    generator_name: str = "mock"
    seed: int = 0
    embed_model: str = "hash"
    embed_dim: int = 0
    _index: dict[tuple[str, int, bool], list[int]] = field(default_factory=dict)

    def _rebuild_index(self) -> None:
        self._index = {}
        for i, e in enumerate(self.entries):
            self._index.setdefault((e.vertical, e.tier, e.fraudulent), []).append(i)

    def key(self, vertical: str, tier: int, fraudulent: bool) -> list[PoolEntry]:
        if not self._index:
            self._rebuild_index()
        return [self.entries[i] for i in self._index.get((vertical, tier, fraudulent), [])]

    def key_any_class(self, vertical: str, tier: int) -> list[PoolEntry]:
        """Both classes of a vertical/tier.

        The source draws from here, because at generation time nothing knows the
        ground truth — a benign dispute and a fraudulent one request the same
        tool, and the text differs by facts, not by a label the generator sees.
        The episode's outcome sets the label; the text is just text.
        """
        return self.key(vertical, tier, True) + self.key(vertical, tier, False)

    @property
    def fingerprint(self) -> str:
        """A stable hash of what the pool contains, for versioning.

        Two pools with the same fingerprint hold the same text, so a run can
        record which corpus it used the way it records the fitted-params split.
        """
        h = hashlib.sha256()
        h.update(f"{self.generator_name}:{self.seed}".encode())
        for e in self.entries:
            h.update(e.text.encode("utf-8"))
        return h.hexdigest()

    # ------------------------------------------------------------- persistence

    def save(self, path: Path | str) -> None:
        path = Path(path)
        payload = {
            "generator": self.generator_name,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "entries": [
                {
                    "vertical": e.vertical,
                    "tier": e.tier,
                    "fraudulent": e.fraudulent,
                    "persona": e.persona,
                    "text": e.text,
                    "facts": e.facts,
                    "embedding": list(e.embedding),
                }
                for e in self.entries
            ],
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> TextPool:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        pool = cls(
            entries=[
                PoolEntry(
                    vertical=e["vertical"],
                    tier=e["tier"],
                    fraudulent=e["fraudulent"],
                    persona=e["persona"],
                    text=e["text"],
                    facts=e["facts"],
                    embedding=tuple(e.get("embedding", ())),
                )
                for e in payload["entries"]
            ],
            generator_name=payload["generator"],
            seed=payload["seed"],
        )
        pool.embed_model = payload.get("embed_model", "hash")
        pool.embed_dim = payload.get("embed_dim", 0)
        pool._rebuild_index()
        return pool


# ------------------------------------------------------------------- building


_MERCHANT_NAMES = (
    "Northgate Electronics",
    "Riverside Grocery",
    "BlueLine Transit",
    "Summit Outdoor",
    "CityFuel",
    "Harbor Books",
)
_BANKS = ("First National", "Meridian Bank", "Coastal Credit Union")


def _draw_facts(rng: np.random.Generator, vertical: str) -> PromptFacts:
    """Facts drawn from shared pools, so nothing in them marks the class."""
    day = int(rng.integers(1, 28))
    month = int(rng.integers(1, 13))
    return PromptFacts(
        amount=float(round(rng.uniform(30, 900), 2)),
        merchant_name=str(rng.choice(_MERCHANT_NAMES)),
        bank_name=str(rng.choice(_BANKS)),
        date=f"2024-{month:02d}-{day:02d}",
        persona=str(rng.choice(PERSONAS)),
        tone=str(rng.choice(TONES)),
        detail=str(rng.choice(DETAILS[vertical])),
    )


def build_pool(
    generator=None,
    per_key: int = 8,
    seed: int = 0,
    embedder=None,
    batch_size: int = 16,
    progress: bool = True,
) -> TextPool:
    """Generate the full pool: every vertical, tier and class.

    Defaults to the mock generator, so calling this loads no model. Pass a
    `QwenGenerator` to build the real corpus on a capable machine instead.
    """
    generator = generator or MockGenerator()
    rng = np.random.default_rng(seed)

    # Every item to generate, drawn first. Collecting the specifications before
    # generating any of them is what lets a model-backed generator run them as
    # batches rather than one at a time.
    specs = []
    for vertical in TEXT_VERTICALS:
        for tier in TIERS:
            for fraudulent in (True, False):
                for _ in range(per_key):
                    specs.append((vertical, tier, fraudulent, _draw_facts(rng, vertical)))

    if getattr(generator, "supports_batch", False):
        _log.info("generating %s texts in batches of %d ...", f"{len(specs):,}", batch_size)
        texts = generator.generate_many(specs, batch_size=batch_size, progress=progress)
    else:
        texts = []
        step = max(1, len(specs) // 10)
        for i, (vertical, tier, fraudulent, facts) in enumerate(specs):
            texts.append(generator.generate(vertical, tier, fraudulent, facts))
            if progress and (i + 1) % step == 0:
                _log.info("  generated %s/%s", f"{i + 1:,}", f"{len(specs):,}")

    entries = [
        PoolEntry(
            vertical=vertical,
            tier=tier,
            fraudulent=fraudulent,
            persona=facts.persona,
            text=text,
            facts={
                "amount": facts.amount,
                "merchant_name": facts.merchant_name,
                "bank_name": facts.bank_name,
                "date": facts.date,
                "detail": facts.detail,
            },
        )
        for (vertical, tier, fraudulent, facts), text in zip(specs, texts, strict=False)
    ]

    # Embed every item once, in one batch, so the vectors are computed here and
    # stored rather than recomputed at run time. Defaults to the hash stand-in so
    # the pool builds with no model; pass a real Embedder to store semantic
    # vectors instead.
    if embedder is None:
        from .embed import HashEmbedder
        embedder = HashEmbedder()
    if progress:
        _log.info("embedding %s texts ...", f"{len(entries):,}")
    vectors = embedder.encode([e.text for e in entries])
    for entry, vec in zip(entries, vectors, strict=False):
        entry.embedding = tuple(float(x) for x in vec)

    pool = TextPool(
        entries=entries, generator_name=generator.name, seed=seed
    )
    pool.embed_model = embedder.name
    pool.embed_dim = int(vectors.shape[1]) if len(vectors) else 0
    pool._rebuild_index()
    return pool


# ------------------------------------------------------- the artifact source


class PoolArtifactSource:
    """Serves text from a built pool as an ArtifactSource.

    The simulator asks for an artifact by tool name and target; here that maps
    to a vertical, a tier and a class, and the source returns a drawn item with
    its scores already attached. No generation happens at run time.
    """

    _TOOL_TO_VERTICAL = TOOL_TO_VERTICAL

    def __init__(self, pool: TextPool, scorer=None, seed: int = 0) -> None:
        self._pool = pool
        self._scorer = scorer
        self._rng = np.random.default_rng(seed)
        # A shuffled cursor per key. Sampling with replacement reuses a few items
        # heavily and leaves most of the pool untouched, and a text expert
        # reading embeddings would memorise the repeats rather than learn what
        # generated text looks like. Cycling a shuffled order spreads the draws
        # evenly and only repeats once a key is exhausted.
        self._order: dict = {}
        self._cursor: dict = {}

    def _next_index(self, vertical: str, tier: int, n: int) -> int:
        """The next item for this key, cycling a shuffled order.

        Reshuffles when the key is exhausted, so a long run keeps drawing in a
        different order rather than repeating the same cycle.
        """
        key = (vertical, tier)
        order = self._order.get(key)
        cursor = self._cursor.get(key, 0)
        if order is None or cursor >= len(order) or len(order) != n:
            order = self._rng.permutation(n)
            cursor = 0
        index = int(order[cursor])
        self._order[key] = order
        self._cursor[key] = cursor + 1
        return index

    def generate(self, request: ArtifactRequest) -> Artifact:
        vertical = self._TOOL_TO_VERTICAL.get(request.tool_name)
        if vertical is None:
            return Artifact()
        tier = request.capability_tier
        # The source does not know the ground-truth class, and must not. It draws
        # across both classes of the vertical and tier, since a benign and a
        # fraudulent request for the same tool are indistinguishable at
        # generation time; the label comes from how the episode ends, not here.
        items = self._pool.key_any_class(vertical, tier)
        if not items:
            return Artifact()
        entry = items[self._next_index(vertical, tier, len(items))]
        scores = {}
        if self._scorer is not None:
            scores = self._scorer.score(entry)
        return Artifact(scores=scores, content=entry.text, embedding=entry.embedding)


def load_artifact_source(pool_path, cfpb_path=None, seed: int = 0):
    """Load a saved pool as an ArtifactSource, scorer attached.

    The convenience the simulators use: read the versioned pool, build a text
    scorer against the real narratives where available, and return the source
    ready to hand to a Simulator. Returns None if the pool is absent, so a run
    without a pool falls back to the null source rather than failing.
    """
    from pathlib import Path

    from .scoring import TextScorer

    pool_path = Path(pool_path)
    if not pool_path.exists():
        return None
    pool = TextPool.load(pool_path)
    reference = []
    if cfpb_path is not None and Path(cfpb_path).exists():
        from .cfpb import load_reference

        reference = load_reference(cfpb_path, limit=1500, seed=seed)
    scorer = TextScorer(reference=reference or [e.text for e in pool.entries])
    return PoolArtifactSource(pool, scorer=scorer, seed=seed)
