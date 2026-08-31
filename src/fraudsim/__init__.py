"""Entity-graph world model and simulator for payment fraud research.

Three dependency tiers, deliberately separated:

    runtime      numpy, scipy, pydantic, pyyaml
    analysis     + networkx, matplotlib      (graph metrics, plotting)
    calibration  + pandas, pyarrow           (parameter fitting from real data)

Nothing reachable from this module imports the analysis or calibration tiers,
so the simulation path stays installable and testable on the runtime tier alone.
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
