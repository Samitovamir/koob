"""Restore after forgets: the memory spec §9.4, §9.7, G5 (store + forget).

The forget is the real one (`forget.stop_use`, `forget.purge`, `forget.undo`); restore and
the nightly rewrite replay the journal through `forget.replay_journal`, the one replay.
Spec marker: `[T forget/test_restore::<case>]`."""

from __future__ import annotations

import shutil

import pytest

from koob.memory import backup, forget, store
from koob.memory.model import ForgetScope, ReadRef

from .._store_seed import (
    add_derived,
    add_fact,
    add_op,
    add_raw,
    fact_text,
    files_containing,
    fix_op,
    hmac_of,
    ref,
    ref_apply,
    status,
)
from . import _seed as s

clock = s.clock  # fixture: a manual clock wired into forget

CANARY_RAW = "у Саши день рождения 12 мая zebracanary"
CANARY_FACT = "Sasha birthday is 12 May quokkacanary"
CANARIES = ("quokkacanary", "zebracanary")


@pytest.fixture
def mem_paths(tmp_path):
    return store.MemoryPaths(tmp_path / "data" / "memory")


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def _seed(conn) -> None:
    with store.write_tx(conn):
        add_raw(conn, "w1", CANARY_RAW)
        add_fact(conn, "r1", CANARY_FACT, raw="w1", version=1, quote="12 мая")
        add_raw(conn, "w2", "Дима живёт в Твери")
        add_fact(conn, "r2", "Dima lives in Tver", raw="w2", version=1)
        add_derived(conn, "d1", [ref(conn, "r1"), ref(conn, "r2")], f"summary: {CANARY_FACT}")


def _forget(conn, rid: str = "r1", *, forever: bool = False) -> str:
    return forget.stop_use(conn, [rid], scope=ForgetScope.ITEM, forever=forever)


def _assert_forgotten(conn, *, purged: bool) -> None:
    assert status(conn, "r1") == ("purged" if purged else "forgotten")
    assert CANARY_FACT not in (fact_text(conn, "r1") or "")  # content left the kind row
    fts = conn.execute("SELECT fts_rowid FROM records WHERE id = 'r1'").fetchone()[0]
    assert (
        conn.execute("SELECT count(*) FROM fts_records WHERE rowid = ?", (fts,)).fetchone()[0] == 0
    )
    raw = conn.execute("SELECT status, text FROM raw_chunks WHERE id = 'w1'").fetchone()
    assert raw["status"] != "active" and CANARY_RAW not in (raw["text"] or "")
    d1 = conn.execute("SELECT valid FROM derived WHERE id = 'd1'").fetchone()
    assert d1 is None or d1[0] == 0  # invalid at stop-use, gone once a purge ran
    sent = store.pre_send_check(conn, thread_id="t", read_set=[ReadRef(id="r1", version=1)])
    assert not sent.ok
    # the unrelated fact is untouched
    assert status(conn, "r2") == "active" and fact_text(conn, "r2") == "Dima lives in Tver"
    assert store.check_invariants(conn) == []


def test_forget_then_restore_old_backup(mem_paths, home, clock):
    """§9.4: restoring a backup older than a forget never brings the content back, and the
    purge it was waiting on finishes once the nightly rewrite cleans the old set."""
    conn = store.open(mem_paths.db, home=home)
    _seed(conn)
    old_set = backup.make_set(conn, clock=clock)
    clock.advance(3600)
    fid = _forget(conn, forever=True)
    # §9.1 stage 3: not «удалил полностью» while the old set still holds the text
    assert forget.purge(conn, fid) is False
    _assert_forgotten(conn, purged=False)
    version_before = store.memory_version(conn)
    conn.close()

    new_version = backup.restore(old_set, mem_paths.db, home=home)

    conn = store.open(mem_paths.db, home=home)
    try:
        _assert_forgotten(conn, purged=False)
        assert new_version > version_before
        assert store.memory_version(conn) == new_version
        # «насовсем» survives the restore: the purge is due now, not in 24 h
        assert fid in forget.due(conn)
        # the nightly rewrite takes the text out of the old set, then the purge verifies
        assert backup.rewrite_after_forgets(mem_paths.backups) == 1
        assert forget.purge(conn, fid) is True
        _assert_forgotten(conn, purged=True)
    finally:
        conn.close()
    for canary in CANARIES:
        assert files_containing(mem_paths.root, canary) == []
    # the purged line reaches the set's journal copy; after that a night has nothing to do
    backup.rewrite_after_forgets(mem_paths.backups)
    assert backup.rewrite_after_forgets(mem_paths.backups) == 0


def test_purged_then_restore_of_an_older_copy_leaves_no_bytes(mem_paths, home, clock, tmp_path):
    """§9.4: a copy of a set taken before a completed purge (kept outside memory/) is
    restored: the replay purges it again and the live files carry no trace."""
    conn = store.open(mem_paths.db, home=home)
    _seed(conn)
    old_set = backup.make_set(conn, clock=clock)
    offsite = tmp_path / "offsite" / old_set.name
    shutil.copytree(old_set, offsite)
    clock.advance(3600)
    fid = _forget(conn, forever=True)
    assert forget.purge(conn, fid) is False  # the managed old set still holds it
    assert backup.rewrite_after_forgets(mem_paths.backups) == 1
    assert forget.purge(conn, fid) is True
    conn.close()

    report = backup.restore_report(offsite, mem_paths.db, home=home)

    assert report.forgets_replayed == 1
    conn = store.open(mem_paths.db, home=home)
    try:
        _assert_forgotten(conn, purged=True)
    finally:
        conn.close()
    for canary in CANARIES:
        assert files_containing(mem_paths.root, canary) == []


def test_corruption_after_forget_stays_forgotten(mem_paths, home, clock):
    """§9.4: corruption at 15:00 after a forget at 10:00: still forgotten after recovery."""
    conn = store.open(mem_paths.db, home=home)
    _seed(conn)
    backup.make_set(conn, clock=clock)  # the nightly set, before the forget
    clock.advance(3600)  # 10:00
    fid = _forget(conn)
    _assert_forgotten(conn, purged=False)
    conn.close()
    clock.advance(5 * 3600)  # 15:00: the disk eats a run of pages
    data = bytearray(mem_paths.db.read_bytes())
    for i in range(4096, min(len(data), 4096 * 12)):
        data[i] = (i * 31) & 0xFF
    mem_paths.db.write_bytes(bytes(data))
    assert not backup.check_live(mem_paths.db)

    report = backup.recover(mem_paths.db, home=home)

    assert report is not None and report.forgets_replayed == 1
    assert backup.check_live(mem_paths.db)
    conn = store.open(mem_paths.db, home=home)
    try:
        _assert_forgotten(conn, purged=False)
        # still in the trash under the same id, so «Вернуть» keeps working after recovery
        forget.undo(conn, fid)
        assert status(conn, "r1") == "active" and fact_text(conn, "r1") == CANARY_FACT
    finally:
        conn.close()
    assert backup.recover(mem_paths.db, home=home) is None  # sound now: nothing to do


def test_fix_survives_restore(mem_paths, home, clock):
    """§9.4: a hot-path fix after the last backup survives a restore via the hot log."""
    conn = store.open(mem_paths.db, home=home)
    _seed(conn)
    hot = backup.open_hotlog(mem_paths.hotlog)
    the_set = backup.make_set(conn, clock=clock)
    proposal = fix_op(conn, "r2", "Dima lives in Samara")
    result = store.commit(conn, proposal, ref_apply)
    assert result.ok
    backup.hotlog_append(
        hot, "proposal", proposal.model_dump(mode="json"), memory_version=result.versions["r2"]
    )
    version_before = store.memory_version(conn)
    conn.close()
    hot.close()

    report = backup.restore_report(the_set, mem_paths.db, home=home, apply=ref_apply)

    assert report.hot_replayed == 1 and report.unreplayed == []
    conn = store.open(mem_paths.db, home=home)
    try:
        assert fact_text(conn, "r2") == "Dima lives in Samara"
        assert report.memory_version > version_before
        assert store.memory_version(conn) == report.memory_version
    finally:
        conn.close()


def _hot_add(conn, hot, rid: str, text: str) -> None:
    proposal = add_op(rid, text)
    result = store.commit(conn, proposal, ref_apply)
    backup.hotlog_append(
        hot, "proposal", proposal.model_dump(mode="json"), memory_version=result.versions[rid]
    )


def _hot_payloads(path) -> list[object]:
    h = backup.open_hotlog(path)
    try:
        return [op.payload for op in backup.hotlog_ops(h)]
    finally:
        h.close()


def test_hot_op_before_a_forget_is_not_replayed(mem_paths, home, clock):
    """§9.4: a hot write the journal later forgot never comes back through the hot log, even
    while its op still holds the payload (a 24 h forget scrubs the hot log at purge)."""
    conn = store.open(mem_paths.db, home=home)
    hot = backup.open_hotlog(mem_paths.hotlog)
    the_set = backup.make_set(conn, clock=clock)
    _hot_add(conn, hot, "r9", f"secret {CANARY_FACT}")
    clock.advance(60)
    _forget(conn, "r9")
    assert _hot_payloads(mem_paths.hotlog)[0] is not None  # kept for «Вернуть»
    conn.close()
    hot.close()

    report = backup.restore_report(the_set, mem_paths.db, home=home, apply=ref_apply)

    assert report.hot_replayed == 0 and report.hot_skipped == 1
    conn = store.open(mem_paths.db, home=home)
    try:
        assert conn.execute("SELECT count(*) FROM records WHERE id = 'r9'").fetchone()[0] == 0
    finally:
        conn.close()


def test_forever_forget_scrubs_the_hot_log_at_stop_use(mem_paths, home, clock):
    """§9.4: «насовсем» has no «Вернуть», so its hot ops lose their payload at stop-use."""
    conn = store.open(mem_paths.db, home=home)
    hot = backup.open_hotlog(mem_paths.hotlog)
    backup.make_set(conn, clock=clock)
    _hot_add(conn, hot, "r9", f"secret {CANARY_FACT}")
    _hot_add(conn, hot, "r8", "Dima likes tea")
    hot.close()
    _forget(conn, "r9", forever=True)
    conn.close()
    payloads = _hot_payloads(mem_paths.hotlog)
    assert payloads[0] is None and payloads[1] is not None
    assert files_containing(mem_paths.root, "quokkacanary") == ["memory.db"]  # the trash


def test_undo_then_restore_keeps_a_hot_only_item(mem_paths, home, clock):
    """«Вернуть» then a restore from a set older than the item: the hot log still holds it,
    so the item comes back (why a 24 h forget does not scrub the hot log at stop-use)."""
    conn = store.open(mem_paths.db, home=home)
    hot = backup.open_hotlog(mem_paths.hotlog)
    the_set = backup.make_set(conn, clock=clock)
    _hot_add(conn, hot, "r9", "Dima likes tea")
    hot.close()
    clock.advance(60)
    fid = _forget(conn, "r9")
    clock.advance(60)
    forget.undo(conn, fid)
    conn.close()

    report = backup.restore_report(the_set, mem_paths.db, home=home, apply=ref_apply)

    assert report.hot_replayed == 1
    conn = store.open(mem_paths.db, home=home)
    try:
        assert status(conn, "r9") == "active" and fact_text(conn, "r9") == "Dima likes tea"
    finally:
        conn.close()


def test_undo_after_the_backup_is_replayed(mem_paths, home, clock):
    """§9.1 stage 2: a set taken between «забудь» and «Вернуть» comes back un-forgotten."""
    conn = store.open(mem_paths.db, home=home)
    _seed(conn)
    fid = _forget(conn)
    the_set = backup.make_set(conn, clock=clock)
    clock.advance(600)
    forget.undo(conn, fid)
    assert status(conn, "r1") == "active"
    conn.close()

    backup.restore(the_set, mem_paths.db, home=home)

    conn = store.open(mem_paths.db, home=home)
    try:
        assert status(conn, "r1") == "active"
        assert fact_text(conn, "r1") == CANARY_FACT
        raw = conn.execute("SELECT status, text FROM raw_chunks WHERE id = 'w1'").fetchone()
        assert raw["status"] == "active" and raw["text"] == CANARY_RAW
    finally:
        conn.close()


def test_raw_newer_than_the_set_leaves_a_tombstone(mem_paths, home, clock):
    """§9.4 step (4): raw that arrived after the set and was then forgotten is absent from
    the restored store; a content-free tombstone keeps its HMAC so the sweep cannot
    re-ingest the same text."""
    conn = store.open(mem_paths.db, home=home)
    _seed(conn)
    the_set = backup.make_set(conn, clock=clock)
    text = "Лена переехала в Ригу ibiscanary"
    with store.write_tx(conn):
        add_raw(conn, "w5", text)
        add_fact(conn, "r5", "Lena moved to Riga ibiscanary", raw="w5", version=1)
    clock.advance(60)
    _forget(conn, "r5")
    conn.close()

    backup.restore(the_set, mem_paths.db, home=home)

    conn = store.open(mem_paths.db, home=home)
    try:
        row = conn.execute(
            "SELECT id, status, text FROM raw_chunks WHERE content_hmac = ?", (hmac_of(text),)
        ).fetchone()
        assert row is not None and row["id"].startswith("tomb:")
        assert row["status"] == "purged" and row["text"] is None
        assert store.check_invariants(conn) == []
    finally:
        conn.close()
    assert [h for h in files_containing(mem_paths.root, "ibiscanary") if "backups" not in h] == []
