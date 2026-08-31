"""Simulation clock.

Time is an integer count of minutes since an arbitrary origin. Every timestamp
in the system uses this unit; conversion to wall-clock happens only at export.
Integer minutes keep window arithmetic exact and hashable.
"""

from __future__ import annotations

MINUTE = 1
HOUR = 60
DAY = 24 * HOUR
WEEK = 7 * DAY

MINUTES_PER_HOUR = HOUR
MINUTES_PER_DAY = DAY
SECONDS_PER_MINUTE = 60

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 604_800


class SimClock:
    """Monotonic minute-resolution clock."""

    __slots__ = ("_now", "_origin")

    def __init__(self, origin: int = 0) -> None:
        self._origin = origin
        self._now = origin

    @property
    def now(self) -> int:
        return self._now

    @property
    def origin(self) -> int:
        return self._origin

    def advance(self, minutes: int) -> int:
        if minutes < 0:
            raise ValueError("clock cannot move backwards")
        self._now += minutes
        return self._now

    def advance_to(self, ts: int) -> int:
        if ts < self._now:
            raise ValueError(f"cannot rewind clock from {self._now} to {ts}")
        self._now = ts
        return self._now

    def elapsed(self) -> int:
        return self._now - self._origin

    def rewind_to_origin(self) -> None:
        """Reset for a fresh run. Only valid between runs, never mid-episode."""
        self._now = self._origin

    def rewind_to(self, ts: int) -> int:
        """Move the clock back to start a backdated phase.

        Only for setting up history before the observation window opens.
        Nothing during a run may call this: the rolling windows evict from the
        front assuming it holds the oldest entry, and time moving backwards
        underneath them silently discards history an event still needs.
        """
        self._now = ts
        return self._now


class WarmStartClock(SimClock):
    """Clock for backdated history generation.

    History is produced before the observation window opens, so it runs from a
    negative offset up to the origin. Everything downstream still sees a
    monotonic clock; only the starting point differs.
    """

    def __init__(self, origin: int = 0, lookback_minutes: int = 90 * DAY) -> None:
        super().__init__(origin=origin - lookback_minutes)
        self._target = origin

    @property
    def target(self) -> int:
        return self._target

    def is_complete(self) -> bool:
        return self.now >= self._target
