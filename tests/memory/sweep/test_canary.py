"""Acceptance for the memory spec §19.5: row 1 (a banned chat, never read: the
admission side), row 4 (an item the user forgets, with «Вернуть» and «насовсем»), row 6 (a
chat later banned: memory side, and end to end through the mode change) and row 7 (4 and 6,
then a restore of a backup taken before them). Rows 2, 3 and 5, and the worker and nightly
parts of row 1, depend on the ingest, hot-path and worker components and are added here later.
Spec marker: `[T sweep/test_canary::<case>]`."""

from __future__ import annotations

import shutil
import sqlite3

from koob.memory import api, backup, forget, policy, store
from koob.memory.model import AuthorTier, ChatMode, ForgetScope, TurnCtx

from ..forget import _seed as s

clock = s.clock  # fixture
career = s.career  # fixture
conv = s.conv  # fixture

# Latin, Cyrillic and digit spellings of one planted token.
TOKENS = ("QZXCANARYALPHA", "ЖЩКАНАРЕЙКААЛЬФА", "8675309421")


def _plant(mem, career) -> str:
    """The canary in every memory store and in the user's island chat; returns the turn id."""
    th = s.thread(career)
    t = s.turn(
        career,
        th,
        ("user", f"запомни: код от шкафчика Ильи {TOKENS[1]} {TOKENS[2]}"),
        ("assistant", f"Запомнил: {TOKENS[0]}."),
    )
    s.relation(mem, "locker_code", "secret_like")
    s.entity(mem, "ent-ilya", "Илья", [("Илья", "name")])
    s.raw(mem, "raw-c", f"запомни: код от шкафчика Ильи {TOKENS[1]} {TOKENS[2]}", turn=t, chat=th)
    s.fact(
        mem,
        "f-c",
        "ent-ilya",
        "locker_code",
        value=f"{TOKENS[2]}",
        text_en=f"Ilya's locker code is {TOKENS[0]} {TOKENS[2]}",
        quote=f"код от шкафчика Ильи {TOKENS[1]} {TOKENS[2]}",
        raw="raw-c",
    )
    s.derived(mem, "d-c", f"Today: locker code {TOKENS[0]}", [("f-c", 1)])
    s.served(mem, th, "f-c", t)
    root = forget.memory_dir(mem)
    v = store.connect(root / "vectors.db")
    v.execute("CREATE TABLE vec (record_id TEXT, embedder_id TEXT, v BLOB)")
    v.execute("INSERT INTO vec VALUES ('f-c', 'e5', x'00ff')")
    v.close()
    hot = backup.open_hotlog(root / "hotlog.db")  # the real hot log
    backup.hotlog_append(
        hot,
        "remember",
        {"id": "f-c", "text": f"{TOKENS[0]} {TOKENS[1]}"},
        memory_version=store.memory_version(mem),
    )
    hot.close()
    return t


def _memory_hits(mem) -> list[tuple[str, str]]:
    return s.hits(s.files_under(forget.memory_dir(mem)), TOKENS)


def _rows_outside_trash(mem) -> bool:
    """Any token in memory.db rows other than `trash` (stop-use leaves it only there)."""
    for (name,) in mem.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "AND name <> 'trash' AND name NOT LIKE 'fts_%'"
    ).fetchall():
        for row in mem.execute(f"SELECT * FROM {name}"):
            blob = " ".join(str(v) for v in tuple(row) if isinstance(v, str))
            if any(t.casefold() in blob.casefold() for t in TOKENS):
                return True
    return False


def _career_files(career) -> list:
    career.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    path = [r[2] for r in career.execute("PRAGMA database_list") if r[1] == "main"][0]
    return s.files_under(__import__("pathlib").Path(path).parent)


def test_canary(mem, career, conv, clock):
    """Row 4: stop-use → «Вернуть» → «забудь» → purge after the trash window. Allowed hits:
    `trash` until the purge; then none in memory files or the user's island chat."""
    t = _plant(mem, career)
    assert _memory_hits(mem)  # planted

    fid = forget.stop_use(mem, ["f-c"], scope=ForgetScope.ITEM, forever=False)
    assert not _rows_outside_trash(mem)
    assert s.texts(career, t) == ["[забыто]", "[забыто]"]

    forget.undo(mem, fid)
    assert s.texts(career, t)[1] == f"Запомнил: {TOKENS[0]}."
    assert _rows_outside_trash(mem)

    fid = forget.stop_use(mem, ["f-c"], scope=ForgetScope.ITEM, forever=False)
    assert forget.run_due(mem) == {}  # 24 h in the trash
    clock.advance(24 * 3600 + 1)
    assert forget.run_due(mem) == {fid: True}
    assert _memory_hits(mem) == []
    assert s.hits(_career_files(career), TOKENS) == []


def test_canary_forever(mem, career, conv, clock):
    """Row 4, «насовсем»: the purge runs at once and leaves no hit."""
    _plant(mem, career)
    fid = forget.stop_use(mem, ["f-c"], scope=ForgetScope.ITEM, forever=True)
    assert forget.run_due(mem) == {fid: True}
    assert _memory_hits(mem) == []
    assert s.hits(_career_files(career), TOKENS) == []


def test_canary_chat_banned_memory_side(mem, career, conv, clock):
    """Row 6 (memory side; the mode change itself is policy's): a normal chat's raw and
    what was learned from it are purged from memory files; the user's island history is kept."""
    t = _plant(mem, career)
    th = mem.execute("SELECT chat_id FROM raw_chunks WHERE id = 'raw-c'").fetchone()[0]
    fid = forget.stop_use(mem, [th], scope=ForgetScope.CHAT, forever=True)
    assert forget.purge(mem, fid) is True
    assert _memory_hits(mem) == []
    assert s.texts(career, t)[0].startswith("запомни")


def test_scan_reads_every_encoding(tmp_path):
    """The purge's own scanner finds UTF-8, JSON-escaped and UTF-16LE spellings."""
    pattern = forget.scan_keys([TOKENS[1]])
    for data in (
        TOKENS[1].encode("utf-8"),
        TOKENS[1].encode("utf-16-le"),
        b"\x00" + TOKENS[1].encode("utf-16-le"),
        __import__("json").dumps(TOKENS[1]).encode("ascii"),
    ):
        assert forget.scan_bytes(b"xx" + data + b"yy", pattern)
    db = sqlite3.connect(tmp_path / "x.db")
    db.execute("CREATE TABLE t (v TEXT)")
    db.execute("INSERT INTO t VALUES (?)", (TOKENS[1].lower(),))
    db.commit()
    db.close()
    assert forget.scan_bytes((tmp_path / "x.db").read_bytes(), pattern)


def _owner(words: str) -> TurnCtx:
    return TurnCtx(turn_id="t-owner", thread_id="th-owner", owner_words=words)


def test_canary_row1_banned_chat_is_never_admitted(mem, clock):
    """Row 1 (admission side): a banned chat's background message is refused before any
    write; the mode change itself stores no content."""
    chat = "tg:canary"
    api.set_chat_mode(chat, ChatMode.BANNED, turn=_owner("забань этот чат"), store=mem)
    ok, _reason = policy.admit(
        mem,
        source_kind="telegram",
        chat_id=chat,
        text=f"Дима: {TOKENS[1]} {TOKENS[0]} {TOKENS[2]}",
        author_tier=AuthorTier.OTHER,
        turn=None,
    )
    assert ok is False
    mem.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert _memory_hits(mem) == []


def test_canary_row6_chat_banned_end_to_end(mem, career, conv, clock):
    """Row 6 through the mode change (policy + forget): ingest, ban, purge leave no
    hit in memory files; the user's island history is not memory's to blank (§5.2)."""
    t = _plant(mem, career)
    th = mem.execute("SELECT chat_id FROM raw_chunks WHERE id = 'raw-c'").fetchone()[0]
    change = api.set_chat_mode(th, ChatMode.BANNED, turn=_owner("забань этот чат"), store=mem)
    assert change.forget_id is not None
    assert forget.run_due(mem) == {change.forget_id: True}
    assert _memory_hits(mem) == []
    assert s.texts(career, t)[0].startswith("запомни")


def test_canary_row7_restore_of_an_older_backup(mem, career, conv, clock, tmp_path):
    """Row 7: rows 4 («насовсем») and 6, then a restore of a backup taken before them (a
    copy kept outside memory/): hits stay as before the restore, none."""
    _plant(mem, career)
    th = mem.execute("SELECT chat_id FROM raw_chunks WHERE id = 'raw-c'").fetchone()[0]
    root = forget.memory_dir(mem)
    paths = store.MemoryPaths(root)
    old_set = backup.make_set(mem, clock=clock)
    offsite = tmp_path / "offsite" / old_set.name
    shutil.copytree(old_set, offsite)
    fid = forget.stop_use(mem, ["f-c"], scope=ForgetScope.ITEM, forever=True)
    ban = api.set_chat_mode(th, ChatMode.BANNED, turn=_owner("забань этот чат"), store=mem)
    assert forget.purge(mem, fid) is False  # the managed set still holds the tokens
    assert backup.rewrite_after_forgets(paths.backups) == 1
    assert forget.purge(mem, fid) is True
    if ban.forget_id is not None:
        assert forget.purge(mem, ban.forget_id) is True
    assert _memory_hits(mem) == []
    mem.close()

    backup.restore(offsite, paths.db, home=tmp_path / "home")

    conn = store.open(paths.db, home=tmp_path / "home")
    try:
        assert _memory_hits(conn) == []
        assert conn.execute("SELECT status FROM records WHERE id = 'f-c'").fetchone()[0] == (
            "purged"
        )
    finally:
        conn.close()
    assert s.hits(_career_files(career), TOKENS) == []
