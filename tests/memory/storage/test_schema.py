"""The core schema (the memory spec §3.2, migrations/memory/0001_core.sql)."""

from __future__ import annotations

import sqlite3

import pytest

from koob.memory import store

TABLES = {
    "meta",
    "schema_migrations",
    "records",
    "record_stats",
    "relations",
    "entities",
    "aliases",
    "facts",
    "impressions",
    "decisions",
    "justifications",
    "work_states",
    "intentions",
    "intention_events",
    "loops",
    "lessons",
    "criteria",
    "episodes",
    "episode_links",
    "raw_chunks",
    "turn_sources",
    "edges",
    "derived",
    "read_deps",
    "served",
    "pending_updates",
    "suppressions",
    "raw_retractions",
    "retired_values",
    "trash",
    "jobs",
    "actions",
    "chat_policy",
    "admit_windows",
    "events",
    "audit",
}
FTS = {"fts_records", "fts_names", "fts_raw"}


def test_schema_applies_on_empty_db(mem):
    names = {r[0] for r in mem.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
    assert names >= TABLES
    assert names >= FTS
    assert [r[0] for r in mem.execute("SELECT name FROM schema_migrations")] == [
        "0001_core.sql",
        "0002_ledger_fixes.sql",
    ]
    meta = mem.execute("SELECT * FROM meta").fetchone()
    assert (meta["memory_version"], meta["policy_version"], meta["schema_version"]) == (0, 0, 2)
    assert mem.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert mem.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "memory.db"
    store.open(path, home=tmp_path).close()
    conn = store.open(path, home=tmp_path)
    assert store.migrate(conn) == []
    assert conn.execute("SELECT count(*) FROM meta").fetchone()[0] == 1
    conn.close()


def test_in_memory_db_works():
    conn = store.open(":memory:")
    assert store.memory_version(conn) == 0
    conn.close()


@pytest.mark.parametrize("table", sorted(FTS))
def test_fts_secure_delete_on(mem, table):
    """Every FTS table keeps secure-delete on."""
    row = mem.execute(f"SELECT v FROM {table}_config WHERE k = 'secure-delete'").fetchone()
    assert row is not None and row[0] == 1


def test_fts_raw_is_contentless_and_deletable(mem):
    mem.execute("INSERT INTO fts_raw (rowid, text) VALUES (7, 'саш переех берлин')")
    hit = mem.execute("SELECT rowid FROM fts_raw WHERE fts_raw MATCH 'берлин'").fetchall()
    assert [tuple(r) for r in hit] == [(7,)]
    assert mem.execute("SELECT text FROM fts_raw").fetchone()[0] is None  # no stored content
    mem.execute("DELETE FROM fts_raw WHERE rowid = 7")
    assert (
        mem.execute("SELECT count(*) FROM fts_raw WHERE fts_raw MATCH 'берлин'").fetchone()[0] == 0
    )


def test_trigram_names_match_cyrillic_substring(mem):
    mem.execute("INSERT INTO fts_names (rowid, name) VALUES (1001, 'Александр Кузнецов')")
    hit = mem.execute("SELECT rowid FROM fts_names WHERE fts_names MATCH 'ксандр'").fetchall()
    assert [tuple(r) for r in hit] == [(1001,)]


def _record(conn, rid, *, kind="entity", version=1, status="active"):
    conn.execute(
        "INSERT INTO records (id, kind, version, status, author_tier, recorded_at, created_at,"
        " updated_at) VALUES (?, ?, ?, ?, 'user', 't', 't', 't')",
        (rid, kind, version, status),
    )


def test_fts_rowids_are_assigned_and_stable(mem):
    _record(mem, "01A")
    _record(mem, "01B")
    rows = mem.execute(
        "SELECT id, fts_rowid FROM records WHERE id <> 'self' ORDER BY id"  # §4.5 seed
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("01A", 2), ("01B", 3)]
    mem.execute("VACUUM")
    assert mem.execute("SELECT fts_rowid FROM records WHERE id = '01B'").fetchone()[0] == 3


def test_checks_refuse_bad_values(mem):
    with pytest.raises(sqlite3.IntegrityError):
        _record(mem, "01C", status="deleted")
    with pytest.raises(sqlite3.IntegrityError):
        mem.execute(
            "INSERT INTO admit_windows (id, chat_id, after_ts, reason, created_at)"
            " VALUES ('w', 'c', 't', 'reply', 't')"
        )


def test_kind_rows_need_their_header(mem):
    with pytest.raises(sqlite3.IntegrityError):
        mem.execute(
            "INSERT INTO entities (id, etype, display_name) VALUES ('nohdr', 'person', 'X')"
        )
