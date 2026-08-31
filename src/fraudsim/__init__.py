"""Entity-graph world model and simulator for payment fraud research.

One runtime base, five optional extras, deliberately separated:

    runtime      numpy, scipy, pydantic, pyyaml       the whole simulation
    analysis     + networkx, matplotlib               graph metrics, plotting
    calibration  + pandas, pyarrow                    fitting from real data
    defender     + scikit-learn, xgboost              the detectors
    rl           + torch                              the learned attacker
    generative   + transformers, sentence-transformers  text and embeddings

Nothing reachable from this module imports any extra, so the simulation path
stays installable and testable on the runtime tier alone. That is not a
convention but an enforced property: `tests/test_import_firewall.py` walks the
AST of every runtime module, function bodies included, and CI runs the suite
once with no extra installed.
"""

from .clock import DAY, HOUR, MINUTE, WEEK, SimClock, WarmStartClock
from .ids import (
    AccountId,
    ActorId,
    BucketId,
    CardId,
    DeviceId,
    EntityKind,
    HolderId,
    IdMinter,
    MerchantId,
    PayeeId,
)
from .rng import RngHub

__version__ = "1.0.0"

__all__ = [
    "DAY",
    "HOUR",
    "MINUTE",
    "WEEK",
    "AccountId",
    "ActorId",
    "BucketId",
    "CardId",
    "DeviceId",
    "EntityKind",
    "HolderId",
    "IdMinter",
    "MerchantId",
    "PayeeId",
    "RngHub",
    "SimClock",
    "WarmStartClock",
]
