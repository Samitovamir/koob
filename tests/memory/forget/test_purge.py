"""Acceptance for the memory spec §9.1: stop-use, trash, undo and the verified purge.
Spec marker: `[T forget/test_purge::<case>]`."""

from __future__ import annotations

import sqlite3

import pytest

from koob.memory import forget, store
from koob.memory.model import ForgetScope, Label

from . import _seed as s

clock = s.clock  # fixture

TOKENS = ("QZXPURGE42", "ПУРГАКАНАРЕЙКА")


def _seed(mem, *, labels: int = 0) -> None:
    s.relation(mem, "likes", "preference", "multi")
    s.entity(mem, "ent-ilya", "Илья", [("Илья", "name")])
    s.raw(mem, "raw-1", f"Илья обожает {TOKENS[1]} и {TOKENS[0]}", turn="7", labels=labels)
    s.fact(
        mem,
        "f-1",
        "ent-ilya",
        "likes",
        value=f"{TOKENS[0]} {TOKENS[1]}",
        text_en=f"Ilya loves {TOKENS[0]}",
        quote=f"Илья обожает {TOKENS[1]} и {TOKENS[0]}",
        raw="raw-1",
        labels=labels,
    )
    s.derived(mem, "d-1", f"Today: Ilya loves {TOKENS[0]}", [("f-1", 1)])


def _vectors(mem) -> sqlite3.Connection:
    v = store.connect(forget.memory_dir(mem) / "vectors.db")
    v.execute("CREATE TABLE vec (record_id TEXT, embedder_id TEXT, v BLOB)")
    v.executemany("INSERT INTO vec VALUES (?, 'e5', x'00ff')", [("f-1",), ("ent-ilya",)])
    return v


def test_verified_or_retrying(mem, clock):
    """§9.1: «удалил полностью» only after the checkpoint, orphan sweep, token scan and
    backup rewrite; else the status stays «удаляю» and the purge retries with backoff."""
    _seed(mem)
    backups = forget.memory_dir(mem) / "backups" / "2026-10-11"
    backups.mkdir(parents=True)
    mem.execute(f"VACUUM INTO '{backups / 'memory.db'}'")  # a set older than the forget

    fid = forget.stop_use(mem, ["f-1"], scope=ForgetScope.ITEM, forever=True)
    assert forget.due(mem) == [fid]  # «насовсем»: no 24 h wait

    # 1. An old backup still holds the text: not verified, retry later.
    assert forget.purge(mem, fid) is False
    assert forget.status(mem, fid) == "purging"
    job = forget._job(mem, fid)
    assert job["attempts"] == 1 and job["state"] == "queued"
    assert forget.due(mem) == []  # backoff
    with pytest.raises(forget.ForgetError):
        forget.undo(mem, fid)  # «Вернуть» only from the trash

    # 2. The nightly rewrite replays the journal into the set; a reader now holds a snapshot.
    old = store.connect(backups / "memory.db")
    forget.replay_journal(old, forget.journal_path(mem), purge=forget.purge_started(mem))
    old.close()
    (backups / "memory.db-wal").unlink(missing_ok=True)
    (backups / "memory.db-shm").unlink(missing_ok=True)
    reader = sqlite3.connect(forget.memory_dir(mem) / "memory.db")
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM records").fetchone()
    clock.advance(120)
    assert forget.due(mem) == [fid]
    assert forget.purge(mem, fid) is False  # checkpoint busy
    assert forget.status(mem, fid) == "purging"
    reader.execute("ROLLBACK")
    reader.close()

    # 3. Everything clean: purged.
    clock.advance(400)
    assert forget.run_due(mem) == {fid: True}
    assert forget.status(mem, fid) == "purged"
    files = s.files_under(forget.memory_dir(mem))
    assert s.hits(files, TOKENS) == []
    assert mem.execute("SELECT status FROM records WHERE id = 'f-1'").fetchone()[0] == "purged"
    assert mem.execute("SELECT count(*) FROM trash WHERE payload IS NOT NULL").fetchone()[0] == 0
    assert mem.execute("SELECT count(*) FROM derived WHERE id = 'd-1'").fetchone()[0] == 0


def test_wired_step_must_pass(mem, clock):
    """A store outside memory/ (SDK files, logs, career backups) that is not clean yet keeps
    the purge retrying; the step gets the ids and the scan strings."""
    _seed(mem)
    seen = {}

    def sdk_files(ctx: forget.PurgeCtx) -> bool:
        seen["ids"] = ctx.record_ids
        seen["keys"] = ctx.keys
        return seen.setdefault("ok", False)

    forget.wire(purge_steps={"sdk": sdk_files})
    fid = forget.stop_use(mem, ["f-1"], scope=ForgetScope.ITEM, forever=True)
    assert forget.purge(mem, fid) is False
    assert "f-1" in seen["ids"] and any(TOKENS[0] in k for k in seen["keys"])
    seen["ok"] = True
    clock.advance(120)
    assert forget.purge(mem, fid) is True


def test_trash_window_24h(mem, clock):
    """«забудь» waits 24 h in the trash; sensitive or «насовсем» at once."""
    _seed(mem)
    fid = forget.stop_use(mem, ["f-1"], scope=ForgetScope.ITEM, forever=False)
    assert forget.due(mem) == []
    assert forget.due(mem, now=clock.advance(24 * 3600 - 1)) == []
    assert forget.due(mem, now=clock.advance(2)) == [fid]


def test_sensitive_purges_at_once(mem, clock):
    _seed(mem, labels=int(Label.HEALTH))
    fid = forget.stop_use(mem, ["f-1"], scope=ForgetScope.ITEM, forever=False)
    assert forget.due(mem) == [fid]


def test_stop_use_is_immediate_and_undo_restores(mem, clock):
    """Stage 1: nothing usable, searchable or embedded remains; stage 2 brings it all back
    at new versions (versions never go backwards)."""
    _seed(mem)
    v = _vectors(mem)
    before = store.memory_version(mem)
    fid = forget.stop_use(mem, ["f-1"], scope=ForgetScope.ITEM, forever=False)

    status_, version = mem.execute(
        "SELECT status, version FROM records WHERE id = 'f-1'"
    ).fetchone()
    assert status_ == "forgotten" and version == before + 1
    assert tuple(
        mem.execute("SELECT text_en, quote_orig FROM facts WHERE id = 'f-1'").fetchone()
    ) == ("[забыто]", None)
    assert tuple(
        mem.execute("SELECT status, text FROM raw_chunks WHERE id = 'raw-1'").fetchone()
    ) == ("forgotten", None)
    assert mem.execute("SELECT valid FROM derived WHERE id = 'd-1'").fetchone()[0] == 0
    assert (
        mem.execute(
            "SELECT count(*) FROM fts_records WHERE fts_records MATCH ?", (TOKENS[0].casefold(),)
        ).fetchone()[0]
        == 0
    )
    assert (
        mem.execute(
            "SELECT count(*) FROM fts_raw WHERE fts_raw MATCH ?", (TOKENS[0].casefold(),)
        ).fetchone()[0]
        == 0
    )
    assert [r[0] for r in v.execute("SELECT record_id FROM vec")] == ["ent-ilya"]
    assert s.hits([forget.memory_dir(mem) / "vectors.db"], ["f-1"]) == []
    events = mem.execute("SELECT kind FROM events ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert events == "forget"
    assert mem.execute("SELECT count(*) FROM suppressions").fetchone()[0] >= 1

    forget.undo(mem, fid)
    status_, version2 = mem.execute(
        "SELECT status, version FROM records WHERE id = 'f-1'"
    ).fetchone()
    assert status_ == "active" and version2 > version
    assert TOKENS[0] in mem.execute("SELECT text_en FROM facts WHERE id = 'f-1'").fetchone()[0]
    assert mem.execute("SELECT status FROM raw_chunks WHERE id = 'raw-1'").fetchone()[0] == "active"
    assert (
        mem.execute(
            "SELECT count(*) FROM fts_records WHERE fts_records MATCH ?", (TOKENS[0].casefold(),)
        ).fetchone()[0]
        == 1
    )
    assert mem.execute("SELECT count(*) FROM suppressions").fetchone()[0] == 0
    assert mem.execute("SELECT count(*) FROM trash").fetchone()[0] == 0
    assert forget.status(mem, fid) == "undone"
    assert forget.purge(mem, fid) is False
    v.close()


def test_orphan_vectors_swept_and_hotlog_scrubbed(mem, clock):
    """§9.2: the purge sweeps vectors of records that are not usable (a crash between the
    ledger commit and the vector delete) and deletes hot-log ops naming the ids."""
    _seed(mem)
    fid = forget.stop_use(mem, ["f-1"], scope=ForgetScope.ITEM, forever=True)
    v = _vectors(mem)  # vectors written back as if the delete had crashed
    hot = store.connect(forget.memory_dir(mem) / "hotlog.db", journal_mode="DELETE")
    hot.execute("CREATE TABLE ops (id INTEGER PRIMARY KEY, op TEXT, payload TEXT)")
    hot.execute(
        "INSERT INTO ops (op, payload) VALUES ('remember', ?)",
        (f'{{"id": "f-1", "text": "{TOKENS[0]}"}}',),
    )
    hot.execute("INSERT INTO ops (op, payload) VALUES ('remember', '{\"id\": \"f-9\"}')")
    hot.close()
    assert forget.purge(mem, fid) is True
    assert [r[0] for r in v.execute("SELECT record_id FROM vec")] == ["ent-ilya"]
    hot = store.connect(forget.memory_dir(mem) / "hotlog.db", journal_mode="DELETE")
    assert [r[0] for r in hot.execute("SELECT payload FROM ops")] == ['{"id": "f-9"}']
    hot.close()
    v.close()


def test_raw_target_takes_its_records(mem, clock):
    """Forgetting a message forgets what was learned from it."""
    _seed(mem)
    forget.stop_use(mem, ["raw-1"], scope=ForgetScope.ITEM, forever=False)
    assert mem.execute("SELECT status FROM records WHERE id = 'f-1'").fetchone()[0] == "forgotten"


def test_not_found(mem, clock):
    with pytest.raises(forget.ForgetNotFound):
        forget.stop_use(mem, ["nope"], scope=ForgetScope.ITEM, forever=False)
    assert store.memory_version(mem) == 0
