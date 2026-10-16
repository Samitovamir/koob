"""Clock injection point: every module takes time from a `Clock`, so races and schedules
are deterministic in tests (spec §19.1).

v0.4 §16.4: one process clock. `use(clock)` sets it (`api.wire` and `forget.wire` call
it), `current()` returns it; `store.now()`, `forget`, `policy`, the scheduler, freezes and
`trash.until` all read it, so a `ManualClock` drives every timestamp alike. Unset, it is the
system clock."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class ManualClock:
    """A clock tests move by hand."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> datetime:
        self._now += timedelta(seconds=seconds)
        return self._now


_current: Clock = SystemClock()
# A clock scoped to one call tree (`scoped`): a scheduler tick in a background thread reads
# its own clock without swapping the process clock under a turn running in another thread.
_scoped: ContextVar[Clock | None] = ContextVar("memory_clock", default=None)


def use(clock: Clock | None) -> None:
    """Set the process clock; None restores the system clock."""
    global _current
    _current = clock if clock is not None else SystemClock()


def current() -> Clock:
    """The clock every memory module reads: the scoped one inside `scoped`, else the process
    clock."""
    return _scoped.get() or _current


def now() -> datetime:
    """`current().now()`."""
    return current().now()


@contextmanager
def scoped(clock: Clock) -> Iterator[Clock]:
    """Within the block (this thread or task only) `current()` and `now()` read `clock`; the
    process clock is left as it is."""
    token = _scoped.set(clock)
    try:
        yield clock
    finally:
        _scoped.reset(token)


def iso(dt: datetime) -> str:
    """UTC ISO-8601 as stored in every memory table (§3.2)."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
