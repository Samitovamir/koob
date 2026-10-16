"""Scenario time: local `tz` times and `+10m` offsets onto the injected UTC clock (README: clock)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from .schema import REL_RE

_UNIT = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def parse_offset(spec: str) -> timedelta:
    m = REL_RE.match(spec)
    if not m:
        raise ValueError(f"bad offset {spec!r}")
    return timedelta(seconds=int(m.group(1)) * _UNIT[m.group(2)])


def to_utc(local: datetime, tz: str) -> datetime:
    """A naive local datetime in `tz` to aware UTC (aware input is converted as is)."""
    if local.tzinfo is None:
        local = local.replace(tzinfo=ZoneInfo(tz))
    return local.astimezone(UTC)


def resolve_at(
    at: str | datetime | None, prev: datetime, tz: str, default: timedelta = timedelta(minutes=1)
) -> datetime:
    """Absolute local time, or `prev + offset`; no `at` means one minute after the previous event."""
    if at is None:
        return prev + default
    if isinstance(at, str) and REL_RE.match(at):
        return prev + parse_offset(at)
    if isinstance(at, str):
        at = datetime.fromisoformat(at)
    return to_utc(at, tz)
