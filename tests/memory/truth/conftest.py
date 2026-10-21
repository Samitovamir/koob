"""Truth tests run on the day the ledger helpers stamp (`_ledger.ts()`: 2026-10-12), so the
always-on core sees fixed-date decisions and facts as current whatever the real date is."""

from __future__ import annotations

import pytest

from koob.memory import clock as _clock

from .._ledger import DayClock


@pytest.fixture(autouse=True)
def _test_day():
    _clock.use(DayClock())
    yield
    _clock.use(None)
