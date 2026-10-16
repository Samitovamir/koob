"""Recall tests run on the day the ledger helpers stamp (`_ledger.ts()`: 2026-10-12).

The always-on core keeps only the recently active people, so a test that adds a person with a
fixed 2026-10-12 stamp and reads the core on the real date would fail a week later. The clock
still moves (leases, waits and timing behave as before); only the day is pinned."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from koob.memory import clock as _clock

TEST_DAY = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


class _DayClock:
    """Real time, shifted so that the run starts at `TEST_DAY`."""

    def __init__(self) -> None:
        self._shift = TEST_DAY - datetime.now(UTC)

    def now(self) -> datetime:
        return datetime.now(UTC) + self._shift


@pytest.fixture(autouse=True)
def _test_day():
    _clock.use(_DayClock())
    yield
    _clock.use(None)
