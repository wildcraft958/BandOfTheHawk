"""Per-entity feature extraction (numpy only, no ML imports).

Two-call protocol: build reads state before the event, commit folds
the event in afterwards. Collapsing them would make every count include
the event it describes.
"""

from .builder import EventBuilder
from .schema import AuthAttemptEvent, BindingEvent, EventLog, EventType
from .state import FeatureStateStore

__all__ = [
    "AuthAttemptEvent",
    "BindingEvent",
    "EventBuilder",
    "EventLog",
    "EventType",
    "FeatureStateStore",
]
