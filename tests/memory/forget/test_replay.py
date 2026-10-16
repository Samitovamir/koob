"""the memory spec §9.3-§9.4, §9.8 (v0.4): the write-ahead journal line, replay of the
same forget (scope, masks, trash window) and the version floor of a restore.
Spec marker: `[T forget/test_replay::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import backup, forget, store
from koob.memory.adapters.conversation import FORGOTTEN
from koob.memory.model import ForgetScope, ForgetStage

from . import _seed as s
from .test_about_entity import _world as _ilya_world

clock = s.clock  # fixture


class _Crash(RuntimeError):
    pass


@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "memory" / "memory.db"


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def _status(conn, rid: str) -> str:
    return conn.execute("SELECT status FROM records WHERE id = ?", (rid,)).fetchone()[0]


def _stages(conn, fid: str) -> list[ForgetStage]:
    return [x.stage for x in forget.read_journal(forget.journal_path(conn)) if x.forget_id == fid]


def _gleb(conn) -> None:
    s.relation(conn, "lives_in", "residence")
    s.entity(conn, "e-gleb", "Глеб", [("Глеб", "name")])
    s.raw(conn, "raw-1", "Глеб живёт в Твери", turn="1")
    s.fact(
        conn, "f-home", "e-gleb", "lives_in", value="Tver", text_en="Gleb lives in Tver",
        quote="Глеб живёт в Твери", raw="raw-1",
    )  # fmt: skip
    conn.commit()


def test_stopping_line_before_commit(mem, clock, monkeypatch):
    """A `stopping` line (forget_id, scope, target_ids) is on disk before the stop-use commit."""
    _gleb(mem)
    seen: list = []
    real = store.commit

    def commit(conn, proposal, apply, **kw):
        seen.extend(forget.read_journal(forget.journal_path(conn)))  # read from disk
        assert _status(conn, "f-home") == "active"  # the commit has not happened yet
        return real(conn, proposal, apply, **kw)

    monkeypatch.setattr(forget.store, "commit", commit)
    fid = forget.stop_use(mem, ["f-home"], scope=ForgetScope.ITEM, forever=False)
    monkeypatch.undo()

    assert [(x.stage, x.forget_id) for x in seen] == [(ForgetStage.STOPPING, fid)]
    ahead = seen[0]
    assert ahead.scope is ForgetScope.ITEM and ahead.target_ids == ["f-home"]
    assert ahead.record_ids == [] and ahead.raw_hmacs == []  # nothing planned yet, no content
    lines = forget.read_journal(forget.journal_path(mem))
    assert [x.stage for x in lines] == [ForgetStage.STOPPING, ForgetStage.STOPPED]
    done = lines[1]
    assert done.scope is ForgetScope.ITEM and done.target_ids == ["f-home"]
    assert done.record_ids == ["f-home"] and done.until == forget._job(mem, fid)["not_before"]

    # nothing to forget: no line at all
    with pytest.raises(forget.ForgetNotFound):
        forget.stop_use(mem, ["f-nope"], scope=ForgetScope.ITEM, forever=False)
    assert len(forget.read_journal(forget.journal_path(mem))) == 2


def test_stopping_only_finished_at_startup(db, home, clock, monkeypatch):
    """A crash after `stopping`: startup finishes the forget and writes `stopped`; a restore
    of an older backup replays it from target_ids and scope."""
    conn = store.open(db, home=home)
    _gleb(conn)
    clock.advance(60)

    def crash(*_a, **_k):
        raise _Crash("power cut between the write-ahead line and the commit")

    with monkeypatch.context() as m:
        m.setattr(forget.store, "commit", crash)
        with pytest.raises(_Crash):
            forget.stop_use(conn, ["f-home"], scope=ForgetScope.ITEM, forever=True)
    (line,) = forget.read_journal(forget.journal_path(conn))
    fid = line.forget_id
    assert line.stage is ForgetStage.STOPPING and _status(conn, "f-home") == "active"

    # startup (§18.1 step 2): the user's request stands, the forget is finished now
    assert forget.heal_journal(conn) == 1
    assert _status(conn, "f-home") == "forgotten"
    assert _stages(conn, fid) == [ForgetStage.STOPPING, ForgetStage.STOPPED]
    assert fid in forget.due(conn)  # «насовсем» (forever) survived the crash
    assert forget.heal_journal(conn) == 0  # idempotent

    # a crash after the commit, before the `stopped` line: the line comes from the job
    s.entity(conn, "e-ira", "Ира", [("Ира", "name")])
    s.fact(conn, "f-ira", "e-ira", "lives_in", value="Omsk", text_en="Ira lives in Omsk")
    conn.commit()
    real = forget.append_journal

    def no_stopped(path, ln):
        if ln.stage is ForgetStage.STOPPED:
            raise _Crash("power cut before the stopped line")
        real(path, ln)

    with monkeypatch.context() as m:
        m.setattr(forget, "append_journal", no_stopped)
        with pytest.raises(_Crash):
            forget.stop_use(conn, ["f-ira"], scope=ForgetScope.ITEM, forever=False)
    fid2 = next(x.forget_id for x in forget.read_journal(forget.journal_path(conn))[::-1])
    assert _status(conn, "f-ira") == "forgotten" and _stages(conn, fid2) == [ForgetStage.STOPPING]
    assert forget.heal_journal(conn) == 1
    (stopped,) = [x for x in forget.read_journal(forget.journal_path(conn)) if x.forget_id == fid2][
        1:
    ]
    assert stopped.stage is ForgetStage.STOPPED and stopped.record_ids == ["f-ira"]
    assert stopped.suppression_hmacs and stopped.target_ids == ["f-ira"]
    conn.close()


def test_stopping_only_replayed_after_restore(db, home, clock, monkeypatch):
    """§9.1 A4: a forget whose last line is `stopping` is re-planned from target_ids and
    scope when an older backup is restored (no heal ran in between)."""
    conn = store.open(db, home=home)
    _gleb(conn)
    old_set = backup.make_set(conn, clock=clock)
    clock.advance(60)

    def crash(*_a, **_k):
        raise _Crash("power cut")

    with monkeypatch.context() as m:
        m.setattr(forget.store, "commit", crash)
        with pytest.raises(_Crash):
            forget.stop_use(conn, ["e-gleb"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    conn.close()
    backup.restore(old_set, db, home=home)
    conn = store.open(db, home=home)
    try:
        assert _status(conn, "e-gleb") == "forgotten" and _status(conn, "f-home") == "forgotten"
        raw = conn.execute("SELECT status FROM raw_chunks WHERE id = 'raw-1'").fetchone()[0]
        assert raw == "forgotten"
    finally:
        conn.close()


def test_restore_masks_not_forgets(db, home, clock):
    """An «всё про X» (everything about X) forget replayed after a restore masks X in
    `masked_ids`; those records stay usable."""
    conn = store.open(db, home=home)
    _ilya_world(conn)
    conn.commit()
    old_set = backup.make_set(conn, clock=clock)
    clock.advance(3600)
    fid = forget.stop_use(conn, ["ent-x"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    line = [x for x in forget.read_journal(forget.journal_path(conn)) if x.forget_id == fid][-1]
    assert line.scope is ForgetScope.ABOUT_ENTITY and line.target_ids == ["ent-x"]
    assert "f-masha" in line.masked_ids and "raw-m" in line.masked_ids
    assert "f-masha" not in line.record_ids and "f-subject" in line.record_ids
    conn.close()

    backup.restore(old_set, db, home=home)
    conn = store.open(db, home=home)
    try:
        assert _status(conn, "f-subject") == "forgotten" and _status(conn, "ent-x") == "forgotten"
        assert _status(conn, "f-masha") == "active", "a masked mention was forgotten"
        text_en, speaker = conn.execute(
            "SELECT text_en, speaker_entity_id FROM facts WHERE id = 'f-masha'"
        ).fetchone()
        assert text_en == f"{FORGOTTEN} says Masha moved to Paris" and speaker is None
        raw = conn.execute("SELECT status, text FROM raw_chunks WHERE id = 'raw-m'").fetchone()
        assert raw[0] == "active" and raw[1].startswith(FORGOTTEN)
    finally:
        conn.close()


def test_until_kept_on_restore(db, home, clock):
    """A restored 24 h forget keeps its `until`; the trash window does not restart."""
    conn = store.open(db, home=home)
    _gleb(conn)
    old_set = backup.make_set(conn, clock=clock)
    clock.advance(60)
    fid = forget.stop_use(conn, ["f-home"], scope=ForgetScope.ITEM, forever=False)
    until = forget._job(conn, fid)["not_before"]
    conn.close()

    clock.advance(20 * 3600)
    backup.restore(old_set, db, home=home)
    conn = store.open(db, home=home)
    try:
        assert _status(conn, "f-home") == "forgotten"
        assert forget._job(conn, fid)["not_before"] == until
        assert forget.status(conn, fid) == "in_trash"
        assert fid not in forget.due(conn)
        clock.advance(4 * 3600 + 60)  # 24 h after the forget, not after the restore
        assert fid in forget.due(conn)
    finally:
        conn.close()


def test_version_floor_with_corrupt_live(db, home, clock):
    """With the live file unreadable, the new version is above every version ever served
    (`version.hwm`)."""
    conn = store.open(db, home=home)
    _gleb(conn)
    old_set = backup.make_set(conn, clock=clock)
    # background commits after the set: in neither the journal nor the hot log, and past
    # the reservation the set carries
    served = store.read_hwm(db.parent / store.HWM_FILE) + 10
    store.reserve_version(conn, served)
    conn.execute("UPDATE meta SET memory_version = ?", (served,))
    conn.commit()
    conn.close()
    assert store.read_hwm(old_set / backup.SET_HWM) < served
    for f in db.parent.glob(db.name + "*"):
        f.write_bytes(b"not a database " * 64)  # the live file is unreadable

    new_version = backup.restore(old_set, db, home=home)

    conn = store.open(db, home=home)
    try:
        assert new_version > served
        assert store.memory_version(conn) > served
        assert _status(conn, "f-home") == "active"  # the set's content is back
    finally:
        conn.close()


def test_restore_restarts_every_session(tmp_path):
    """A `restore` event restarts every live session from a seed and drops cached core."""
    from ..adapters._turnkit import Restarts, host, memory_db, task_turn

    db_path = memory_db(tmp_path)
    restarts = Restarts()
    with host(db_path, restarts=restarts) as h:
        conn = store.open(db_path)
        try:
            turn = task_turn("thread-a", "s-a")
            assert h.before(turn, first_question=False).restart_session is False
            with store.write_tx(conn):
                conn.execute(
                    "INSERT INTO events (ts, kind, memory_version, item_ids)"
                    " VALUES (?, 'restore', ?, '[]')",
                    (store.now(), store.memory_version(conn)),
                )
            inj = h.before(turn, first_question=False)
            assert inj.restart_session is True and restarts.all == 1
            # the next first question rebuilds the core: nothing is cached in the host
            assert h.before(turn, first_question=False).restart_session is False
        finally:
            conn.close()
