"""the memory spec §6.2 step 3, §9.7: identical text re-ingested after a forget is
refused by its HMAC; short texts are exempt. Spec marker: `[T forget/test_reingest::<case>]`."""

from __future__ import annotations

from koob.memory import api, forget
from koob.memory.model import ForgetScope

TS = "2026-10-12T09:00:00.000000Z"


def _ingest(mem, turn_id: str, text: str) -> int:
    rows = [{"role": "owner", "ts": TS, "text": text}]
    return api.ingest_turn(
        turn_id=turn_id, thread_id="thread-1", origin="owner", status="done", rows=rows, store=mem
    )


def _raw(mem, turn_id: str) -> str:
    return mem.execute("SELECT id FROM raw_chunks WHERE turn_id = ?", (turn_id,)).fetchone()[0]


def _active(mem) -> int:
    return mem.execute("SELECT count(*) FROM raw_chunks WHERE status = 'active'").fetchone()[0]


def test_rejected_by_hmac(mem):
    """§6.2: identical text re-ingested after a forget is rejected by the journal HMAC."""
    text = "Мой новый адрес: улица Лесная, дом 5, квартира 12"
    assert _ingest(mem, "turn-1", text) == 1
    forget.stop_use(mem, [_raw(mem, "turn-1")], scope=ForgetScope.ITEM, forever=False)
    assert _ingest(mem, "turn-2", text) == 0
    assert _ingest(mem, "turn-3", "  мой НОВЫЙ адрес: улица Лесная,  дом 5, квартира 12") == 0
    assert _active(mem) == 0
    events = [e for (e,) in mem.execute("SELECT event FROM audit")]
    assert "ingest.reject.forgotten" in events
    # the journal alone still refuses it (a store that lost the row, e.g. an old backup)
    mem.execute("PRAGMA foreign_keys = OFF")
    mem.execute("DELETE FROM raw_chunks")
    assert _ingest(mem, "turn-4", text) == 0
    # a different text is admitted
    assert _ingest(mem, "turn-5", "Мой новый адрес: улица Садовая, дом 7, квартира 3") == 1


def test_short_text_not_blocked(mem):
    """§6.2: texts under 20 normalized characters are exempt from the journal match."""
    assert _ingest(mem, "turn-1", "буду в субботу") == 1
    forget.stop_use(mem, [_raw(mem, "turn-1")], scope=ForgetScope.ITEM, forever=False)
    assert _ingest(mem, "turn-2", "буду в субботу") == 1
    assert _active(mem) == 1
