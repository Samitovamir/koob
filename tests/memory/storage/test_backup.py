"""Backup sets, hot log, rotation and restore bookkeeping: MEMORY.md §3.1, §7.7, §9.3, §9.4,
§18 (store). The G5 acceptance cases live in tests/memory/forget/test_restore.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from koob.memory import backup, store
from koob.memory.clock import ManualClock
from koob.memory.model import ForgetStage

from .._store_seed import add_fact, add_op, append, journal_line, ref_apply


@pytest.fixture
def paths(tmp_path):
    return store.MemoryPaths(tmp_path / "data" / "memory")


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def _clock(day: int = 12, hour: int = 3) -> ManualClock:
    return ManualClock(datetime(2026, 10, day, hour, 0, tzinfo=UTC))


def test_set_layout_no_vectors_manifest_last(paths, home):
    conn = store.open(paths.db, home=home)
    paths.vectors.write_bytes(b"vectors")  # must never enter a set
    append(
        paths.journal, journal_line("f-1", ForgetStage.STOPPED, record_ids=["x"], memory_version=0)
    )
    backup.open_hotlog(paths.hotlog).close()
    s = backup.make_set(conn, clock=_clock())
    assert sorted(p.name for p in s.iterdir()) == [
        "forget.jsonl",
        "hotlog.db",
        "manifest.json",
        "memory.db",
    ]
    manifest = json.loads((s / "manifest.json").read_text())
    assert manifest["verified"] and manifest["journal_lines"] == 1
    assert manifest["memory_version"] == store.memory_version(conn)
    assert [x.path for x in backup.list_sets(paths.backups)] == [s]
    conn.close()


def test_vacuum_into_refused_inside_a_transaction(paths, home):
    """§3.1: `VACUUM INTO` runs outside a transaction."""
    conn = store.open(paths.db, home=home)
    conn.execute("BEGIN")
    with pytest.raises(store.MemoryStoreError):
        backup.make_set(conn, clock=_clock())
    conn.execute("ROLLBACK")
    conn.close()


def test_hotlog_truncated_to_ops_the_set_lacks(paths, home):
    """§9.4: the hot log is truncated after each verified backup; later ops stay."""
    conn = store.open(paths.db, home=home)
    hot = backup.open_hotlog(paths.hotlog)
    r = store.commit(conn, add_op("r1", "a"), ref_apply)
    backup.hotlog_append(
        hot, "proposal", add_op("r1", "a").model_dump(mode="json"), memory_version=r.versions["r1"]
    )
    backup.make_set(conn, clock=_clock())
    assert backup.hotlog_ops(hot) == []
    r = store.commit(conn, add_op("r2", "b"), ref_apply)
    backup.hotlog_append(
        hot, "proposal", add_op("r2", "b").model_dump(mode="json"), memory_version=r.versions["r2"]
    )
    ops = backup.hotlog_ops(hot)
    assert [o.item_ids for o in ops] == [["r2"]]
    assert backup.hotlog_scrub(hot, ["r2"]) == 1
    assert backup.hotlog_ops(hot)[0].payload is None
    hot.close()
    conn.close()


def test_hotlog_uses_delete_journal_and_pragmas(paths):
    hot = backup.open_hotlog(paths.hotlog)
    assert hot.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert hot.execute("PRAGMA secure_delete").fetchone()[0] == 1
    hot.close()


def test_rotation_keeps_three_daily_and_two_weekly(paths, home):
    """§9.4, §18: 3 daily + 2 weekly sets; leftovers of crashed writes are removed."""
    conn = store.open(paths.db, home=home)
    clock = _clock(day=1)  # Thu 2026-10-01
    made = []
    for _ in range(21):  # one set a day, Oct 1 .. Oct 21
        made.append(backup.make_set(conn, clock=clock))
        clock.advance(86400)
    (paths.backups / "20261005T030000000000Z.old").mkdir()
    removed = backup.rotate(paths.backups, clock=clock)
    kept = sorted(p.name[:8] for p in (s.path for s in backup.list_sets(paths.backups)))
    # dailies Oct 19-21 (Mon-Wed of ISO week 43); weeklies: the newest set of weeks 42, 41
    assert kept == ["20261011", "20261018", "20261019", "20261020", "20261021"]
    assert len(removed) == 21 - 5 + 1
    assert not (paths.backups / "20261005T030000000000Z.old").exists()
    conn.close()


def test_rotation_never_drops_the_newest_set(paths, home):
    conn = store.open(paths.db, home=home)
    s = backup.make_set(conn, clock=_clock())
    assert backup.rotate(paths.backups, daily=0, weekly=0) == []
    assert s.exists()
    conn.close()


def test_torn_journal_line_is_skipped(paths):
    append(
        paths.journal, journal_line("f-1", ForgetStage.STOPPED, record_ids=["a"], memory_version=3)
    )
    with paths.journal.open("a") as fh:
        fh.write('{"ts": "2026-10-12T09:00:00Z", "stage": "stop')  # crash mid-append
    assert [x.forget_id for x in backup.read_journal(paths.journal)] == ["f-1"]


def test_fold_journal_last_stage_wins():
    lines = [
        journal_line(
            "f-1",
            ForgetStage.STOPPED,
            record_ids=["a"],
            memory_version=3,
            ts="2026-10-12T09:00:00Z",
        ),
        journal_line(
            "f-1", ForgetStage.UNDONE, record_ids=["a"], memory_version=4, ts="2026-10-12T09:05:00Z"
        ),
        journal_line(
            "f-2",
            ForgetStage.STOPPED,
            record_ids=["b"],
            memory_version=5,
            ts="2026-10-12T09:06:00Z",
        ),
        journal_line(
            "f-2", ForgetStage.PURGED, record_ids=["b"], memory_version=9, ts="2026-10-13T09:06:00Z"
        ),
    ]
    states = {s.forget_id: s for s in backup.fold_journal(lines)}
    assert states["f-1"].stage is ForgetStage.UNDONE
    assert states["f-2"].stage is ForgetStage.PURGED and states["f-2"].memory_version == 9


def test_restore_continues_versions_and_event_ids(paths, home):
    """§7.1: versions never go backwards; a restore continues from max(set, journal, hot
    log, old live) + 1; event ids keep rising so §7.7 hosts see the restore."""
    conn = store.open(paths.db, home=home)
    add_fact(conn, "r1", "a")
    s = backup.make_set(conn, clock=_clock())
    for i in range(5):
        store.commit(conn, add_op(f"n{i}", "x"), ref_apply)
    live_version = store.memory_version(conn)
    last_event = conn.execute("SELECT max(id) FROM events").fetchone()[0]
    append(
        paths.journal,
        journal_line("f-1", ForgetStage.STOPPED, record_ids=["gone"], memory_version=500),
    )
    conn.close()

    reserved = store.read_hwm(paths.db.parent / store.HWM_FILE)
    assert reserved >= live_version  # every served version was reserved first (§7.1, A5)

    v = backup.restore(s, paths.db, home=home)

    assert v == max(500, reserved) + 1 and v > live_version
    assert store.read_hwm(paths.db.parent / store.HWM_FILE) >= v
    conn = store.open(paths.db, home=home)
    changes = store.changes_since(conn, memory_version=live_version, last_event_id=last_event)
    assert changes.memory_version == v
    assert changes.last_event_id > last_event
    kinds = [
        conn.execute("SELECT kind FROM events WHERE id = ?", (i,)).fetchone()[0]
        for i in changes.forget_events
    ]
    assert "restore" in kinds
    conn.close()


def test_restore_drops_vectors_and_refuses_a_partial_set(paths, home):
    conn = store.open(paths.db, home=home)
    s = backup.make_set(conn, clock=_clock())
    conn.close()
    paths.vectors.write_bytes(b"stale vectors")
    report = backup.restore_report(s, paths.db, home=home)
    assert report.vectors_dropped and not paths.vectors.exists()
    (s / "manifest.json").unlink()
    with pytest.raises(store.MemoryStoreError):
        backup.restore(s, paths.db, home=home)


def test_unknown_hot_op_is_reported_not_dropped_silently(paths, home):
    conn = store.open(paths.db, home=home)
    hot = backup.open_hotlog(paths.hotlog)
    s = backup.make_set(conn, clock=_clock())
    backup.hotlog_append(hot, "mode", {"chat_id": "c1", "mode": "no_dig"}, memory_version=99)
    hot.close()
    conn.close()
    seen = []
    report = backup.restore_report(s, paths.db, home=home)
    assert report.unreplayed == ["mode"]
    report = backup.restore_report(
        s, paths.db, home=home, replayers={"mode": lambda c, p: seen.append(p)}
    )
    assert seen == [{"chat_id": "c1", "mode": "no_dig"}] and report.unreplayed == []


def test_changes_since_quiet_when_nothing_moved(mem):
    v = store.memory_version(mem)
    assert store.changes_since(mem, memory_version=v, last_event_id=0).forget_events == []
    mem.execute("INSERT INTO events (ts, kind, memory_version) VALUES ('t', 'forget', ?)", (v + 1,))
    mem.execute("UPDATE meta SET memory_version = ?", (v + 1,))
    ch = store.changes_since(mem, memory_version=v, last_event_id=0)
    assert ch.memory_version == v + 1 and len(ch.forget_events) == 1


def test_stamp_order_matches_time():
    a = datetime(2026, 10, 12, 3, tzinfo=UTC)
    names = [backup._stamp(a + timedelta(hours=h)) for h in (0, 1, 25)]
    assert names == sorted(names)
