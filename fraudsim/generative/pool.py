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

from ..protocols import Artifact, ArtifactRequest, ArtifactSource
from .prompts import (
    DETAILS,
    PERSONAS,
    PROMPT_FOR_VERTICAL,
    TONES,
    PromptFacts,
)

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

    _SKELETON = {
        "friendly_fraud": "I am writing about a charge on my account.",
        "support_se": "I need help with my account as soon as possible.",
        "refund_abuse": "I am requesting a refund for a recent order.",
    }

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
    _index: dict[tuple[str, int, bool], list[int]] = field(default_factory=dict)

    def _rebuild_index(self) -> None:
        self._index = {}
        for i, e in enumerate(self.entries):
            self._index.setdefault((e.vertical, e.tier, e.fraudulent), []).append(i)

    def key(self, vertical: str, tier: int, fraudulent: bool) -> list[PoolEntry]:
        if not self._index:
            self._rebuild_index()
        return [self.entries[i] for i in self._index.get((vertical, tier, fraudulent), [])]

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
                }
                for e in self.entries
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "TextPool":
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
                )
                for e in payload["entries"]
            ],
            generator_name=payload["generator"],
            seed=payload["seed"],
        )
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
) -> TextPool:
    """Generate the full pool: every vertical, tier and class.

    Defaults to the mock generator, so calling this loads no model. Pass a
    `QwenGenerator` to build the real corpus on a capable machine instead.
    """
    generator = generator or MockGenerator()
    rng = np.random.default_rng(seed)
    entries: list[PoolEntry] = []
    for vertical in TEXT_VERTICALS:
        for tier in TIERS:
            for fraudulent in (True, False):
                for _ in range(per_key):
                    facts = _draw_facts(rng, vertical)
                    text = generator.generate(vertical, tier, fraudulent, facts)
                    entries.append(
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
                    )
    pool = TextPool(entries=entries, generator_name=generator.name, seed=seed)
    pool._rebuild_index()
    return pool


# ------------------------------------------------------- the artifact source


class PoolArtifactSource:
    """Serves text from a built pool as an ArtifactSource.

    The simulator asks for an artifact by tool name and target; here that maps
    to a vertical, a tier and a class, and the source returns a drawn item with
    its scores already attached. No generation happens at run time.
    """

    _TOOL_TO_VERTICAL = {
        "write_dispute": "friendly_fraud",
        "write_ticket": "support_se",
        "write_refund_claim": "refund_abuse",
    }

    def __init__(self, pool: TextPool, scorer=None, seed: int = 0) -> None:
        self._pool = pool
        self._scorer = scorer
        self._rng = np.random.default_rng(seed)

    def generate(self, request: ArtifactRequest) -> Artifact:
        vertical = self._TOOL_TO_VERTICAL.get(request.tool_name)
        if vertical is None:
            return Artifact()
        tier = request.capability_tier
        # The source does not know the ground-truth class, and must not — it is
        # asked for an artifact, not for a label. It draws from the fraudulent
        # side, since these tools are only invoked by an attacking action; the
        # benign twins exist for training the text expert, drawn elsewhere.
        items = self._pool.key(vertical, tier, fraudulent=True)
        if not items:
            return Artifact()
        entry = items[int(self._rng.integers(len(items)))]
        scores = {}
        if self._scorer is not None:
            scores = self._scorer.score(entry)
        return Artifact(scores=scores, content=entry.text)
