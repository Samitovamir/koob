"""Race: a re-ingest of a text racing a forget of the same text (§6.2 step 3, §9.1, G5).

Whatever the interleaving, once both finish no active raw chunk, FTS row or queued job
carries the forgotten text: arrival re-checks the journal inside its write transaction, and
keeps one copy per content HMAC so the forget reaches every copy."""

from __future__ import annotations

import json

from koob.memory import api, forget, ingest, store
from koob.memory.model import ForgetScope

from ._kit import open_db, run_threads

TEXT = "Домофон во втором подъезде не работает, заходи через двор после семи"
TS = "2026-10-12T09:00:00.000000Z"


def _row(msg_id: int) -> dict:
    return {
        "role": "other",
        "ts": TS,
        "source": "telegram",
        "chat_id": "tg:777",
        "msg_id": msg_id,
        "sender_tg_id": 777,
        "text": TEXT,
    }


def _h() -> str:
    return ingest.content_hmac(ingest.normalize(TEXT))


def _leaks(conn) -> list[str]:
    out = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM raw_chunks WHERE content_hmac = ? AND status = 'active'", (_h(),)
        )
    ]
    if conn.execute("SELECT rowid FROM fts_raw WHERE fts_raw MATCH 'домофон*'").fetchall():
        out.append("fts_raw")
    # a queued job may name the forgotten raw by id (no content); never a second copy of it
    known = {r[0] for r in conn.execute("SELECT id FROM raw_chunks")}
    for (payload,) in conn.execute("SELECT payload FROM jobs WHERE jkind = 'extract'"):
        out += [f"job:{i}" for i in json.loads(payload)["raw_ids"] if i not in known]
    return out


def _first(conn) -> str:
    api.sweep(rows=[_row(1)], store=conn)
    return conn.execute("SELECT id FROM raw_chunks WHERE content_hmac = ?", (_h(),)).fetchone()[0]


def test_forget_commits_between_plan_and_write(db_path, monkeypatch):
    """Another process stores the text and forgets it after this ingest planned (it saw no
    copy and no journal match) but before its write: the in-transaction re-check refuses."""
    a, b = open_db(db_path), open_db(db_path)
    real = ingest._junk_row
    fired: list[str] = []

    def hook(row, text):
        if not fired:
            fired.append("")  # the other process's own arrival runs the real gate
            monkeypatch.setattr(ingest, "_junk_row", real)
            fired[0] = _first(b)
            forget.stop_use(b, [fired[0]], scope=ForgetScope.ITEM, forever=False)
        return real(row, text)

    monkeypatch.setattr(ingest, "_junk_row", hook)
    assert api.sweep(rows=[_row(2)], store=a) == 0
    assert fired and _leaks(a) == []
    events = [e for (e,) in a.execute("SELECT event FROM audit")]
    assert "ingest.reject.forgotten" in events


def test_copy_stored_between_plan_and_write_is_not_duplicated(db_path, monkeypatch):
    """The other process only stores the text: one copy stays, so a later forget is total."""
    a, b = open_db(db_path), open_db(db_path)
    real = ingest._junk_row
    first: list[str] = []

    def hook(row, text):
        if not first:
            monkeypatch.setattr(ingest, "_junk_row", real)
            first.append(_first(b))
        return real(row, text)

    monkeypatch.setattr(ingest, "_junk_row", hook)
    api.sweep(rows=[_row(2)], store=a)
    assert a.execute("SELECT count(*) FROM raw_chunks").fetchone()[0] == 1
    forget.stop_use(a, first, scope=ForgetScope.ITEM, forever=False)
    assert _leaks(a) == []
    api.sweep(rows=[_row(3)], store=a)  # identical text after the forget: refused
    assert _leaks(a) == []


def test_threads_reingest_vs_forget(db_path):
    """Free-running threads, both orders many times: never a surviving copy."""
    for i in range(12):
        setup = open_db(db_path)
        with store.write_tx(setup):
            setup.execute("DELETE FROM jobs WHERE jkind = 'extract'")
        rid = _first(setup) if i == 0 else None
        if rid is None:  # after the first round the text is forgotten: plant a fresh copy
            setup.execute("PRAGMA foreign_keys = OFF")
            with store.write_tx(setup):
                setup.execute("DELETE FROM raw_chunks WHERE content_hmac = ?", (_h(),))
            forget.journal_path(setup).unlink(missing_ok=True)
            ingest._JOURNAL_CACHE.clear()
            rid = _first(setup)
        setup.close()
        a, b = open_db(db_path), open_db(db_path)
        fns = [
            lambda a=a, rid=rid: forget.stop_use(a, [rid], scope=ForgetScope.ITEM, forever=False),
            lambda b=b, i=i: api.sweep(rows=[_row(100 + i)], store=b),
        ]
        run_threads(*(fns if i % 2 else fns[::-1]))
        assert _leaks(b) == [], f"round {i}"
        a.close()
        b.close()
