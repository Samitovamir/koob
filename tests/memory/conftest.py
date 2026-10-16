"""Fixtures for the memory suite: offline, ":memory:" or tmp_path only."""

from __future__ import annotations

import pytest

from koob.memory import policy, store
from koob.memory.model import TurnCtx, TurnOrigin


@pytest.fixture(autouse=True)
def _restore_label_classifier():
    """`api.startup` installs the labeler process-wide; without a reset it outlives the test and a
    later test that labels text could start a real background model load."""
    before = policy.classifier()
    yield
    policy.set_classifier(before)


@pytest.fixture
def turn() -> TurnCtx:
    """A clean user turn."""
    return TurnCtx(
        turn_id="turn-1",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words="запомни: у Саши день рождения 12 мая",
        message_id="msg-1",
        clock="2026-10-12T09:00:00.000000Z",
    )


@pytest.fixture
def mem(tmp_path):
    """A migrated memory.db in tmp_path."""
    conn = store.open(tmp_path / "memory" / "memory.db", home=tmp_path / "home")
    yield conn
    conn.close()
