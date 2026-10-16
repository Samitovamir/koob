"""Restore of an older backup (spec §9.3, §9.4, §9.7, §7.1): forgotten items stay forgotten
after journal replay, also when the live file is corrupted; an undo in the journal wins over
the stop line; a hot-path fix survives; versions never repeat."""

from __future__ import annotations

import json
import shutil

import pytest

from koob.memory import api, backup, store
from koob.memory.model import ForgetScope, OpKind, Proposal, ProposalOp, ReadRef

from .._contract import not_built
from ._kit import (
    add_raw,
    bare_fact,
    fact,
    handle,
    journal_path,
    lineage,
    open_db,
    owner_turn,
    person,
    ref,
    scan,
    seed,
    status_of,
    usable_slot,
    usable_texts,
)

TOKEN = "Quokka5523"


def _world(conn):
    sasha = person("Саша")
    raw = add_raw(
        conn, "flat", f"Саша снял квартиру, код домофона {TOKEN}", chat_id=None, turn_id="turn-0"
    )
    flat = fact(
        "flat", sasha, "other", TOKEN, f"Sasha's intercom code is {TOKEN}",
        f"код домофона {TOKEN}", source_raw_id=raw,
    )  # fmt: skip
    bday = fact(
        "bday", sasha, "birthday", "12 May", "Sasha's birthday is 12 May",
        "у Саши день рождения 12 мая", value_norm="05-12",
    )  # fmt: skip
    seed(conn, sasha, flat, bday, extra=lambda sql: lineage(sql, flat.id, raw))
    return sasha, flat, bday, raw


def _forget(conn, record, *, forever=False, n=2):
    return api.forget(
        handle(record.id),
        turn=owner_turn("забудь код домофона Саши", n=n),
        because_quote="код домофона Саши",
        scope=ForgetScope.ITEM,
        forever=forever,
        store=conn,
    )


def _corrupt(db_path):
    for side in ("-wal", "-shm"):
        db_path.with_name(db_path.name + side).unlink(missing_ok=True)
    data = bytearray(db_path.read_bytes())
    data[4096:12288] = b"\xde\xad\xbe\xef" * 2048
    db_path.write_bytes(bytes(data))


def _journal_max_version(db_path) -> int:
    lines = journal_path(db_path).read_text().splitlines()
    return max(json.loads(x)["memory_version"] for x in lines if x.strip())


@pytest.mark.parametrize("live", ["intact", "corrupted"])
def test_forget_then_restore_old_backup_stays_forgotten(db_path, conn, live):
    """§9.4, §9.7: a backup taken before the forget is restored (the live file intact, or
    corrupted at 15:00 after a forget at 10:00); journal replay through the full pipeline
    keeps the item forgotten, its raw too; the version continues past everything served."""
    _sasha, flat, bday, raw = _world(conn)
    held = [ref(conn, flat.id)]
    backup_set = backup.make_set(conn, db_path.parent / "backups")
    result = _forget(conn, flat)
    assert result.outcome == "stopped"
    v_live = store.memory_version(conn)
    conn.close()
    if live == "corrupted":
        _corrupt(db_path)

    v_new = backup.restore(backup_set, db_path)
    assert v_new > max(v_live if live == "intact" else 0, _journal_max_version(db_path))

    c = open_db(db_path)
    try:
        assert status_of(c, flat.id) in ("forgotten", "purged")
        assert status_of(c, bday.id) == "active"  # the rest of the backup is back
        raw_st = c.execute("SELECT status FROM raw_chunks WHERE id = ?", (raw,)).fetchone()[0]
        assert raw_st in ("forgotten", "purged")
        assert not any(TOKEN in t for t in usable_texts(c))
        pre = api.pre_send_check(thread_id="thread-owner", read_set=held, store=c)
        assert not pre.ok
        assert store.memory_version(c) >= v_new
        kinds = [r[0] for r in c.execute("SELECT kind FROM events ORDER BY id").fetchall()]
        assert "restore" in kinds
    finally:
        c.close()


@not_built("forget")
def test_purged_then_restore_of_an_offsite_copy_leaves_no_bytes(db_path, conn, tmp_path):
    """The item was purged (permanently); an older copy of a backup set (outside the managed
    folder, so the nightly rewrite never touched it) is restored; after replay and the purge
    the live files hold no byte of the content."""
    _sasha, flat, _bday, _raw = _world(conn)
    managed = backup.make_set(conn, db_path.parent / "backups")
    offsite = tmp_path / "offsite" / managed.name
    shutil.copytree(managed, offsite)
    result = _forget(conn, flat, forever=True)
    api.run_purges(store=conn)
    assert api.purge_status(result.forget_id, store=conn).state == "purged"
    conn.close()

    backup.restore(offsite, db_path)
    c = open_db(db_path)
    try:
        api.run_purges(store=c)
        assert status_of(c, flat.id) in ("forgotten", "purged")
        assert api.purge_status(result.forget_id, store=c).state == "purged"
    finally:
        c.close()
    assert scan(db_path.parent, TOKEN, skip=("backups",)) == []


def test_undo_in_the_journal_wins_over_its_stop_line_on_restore(db_path, conn):
    """§9.1 stage 2, §9.3: forget then «Вернуть» after the backup; restoring the backup replays
    both lines and the item is in use again, not forgotten."""
    _sasha, flat, _bday, _raw = _world(conn)
    backup_set = backup.make_set(conn, db_path.parent / "backups")
    result = _forget(conn, flat)
    undone = api.undo_forget(
        result.forget_id, turn=owner_turn("верни код домофона", n=3), store=conn
    )
    assert undone.outcome == "undone"
    conn.close()
    backup.restore(backup_set, db_path)
    c = open_db(db_path)
    try:
        assert status_of(c, flat.id) == "active"
    finally:
        c.close()


def test_fix_after_backup_survives_restore(db_path, conn):
    """§9.4 (3): the hot log replays a fix made after the backup; the old value does not come
    back in use."""
    sasha, _flat, bday, _raw = _world(conn)
    backup_set = backup.make_set(conn, db_path.parent / "backups")
    words = "нет, у Саши день рождения 14 мая"
    fixed = api.fix(
        handle(bday.id),
        "у Саши день рождения 14 мая",
        turn=owner_turn(words, n=2),
        because_quote="у Саши день рождения 14 мая",
        store=conn,
    )
    assert fixed.outcome == "committed"
    conn.close()
    backup.restore(backup_set, db_path)
    c = open_db(db_path)
    try:
        slot = usable_slot(c, sasha.id, "birthday")
        assert len(slot) == 1 and "14" in (slot[0][1] or "")
        assert status_of(c, bday.id) != "active"
    finally:
        c.close()


def test_restore_never_reuses_a_version_already_served(db_path, conn):
    """§7.1 «versions never go backwards»: background commits after the backup are in
    neither the journal nor the hot log, yet a session was served F at their version; after
    the restore no record may come back to that version with other content."""
    bare_fact(conn, "F", "12 May")
    backup_set = backup.make_set(conn, db_path.parent / "backups")

    def bump(ctx, op):
        ctx.conn.execute(
            "UPDATE facts SET value_text = ? WHERE id = ?", (op.payload["v"], op.target)
        )
        ctx.conn.execute("UPDATE records SET version = ? WHERE id = ?", (ctx.version, op.target))
        return [op.target]

    for i in range(3):
        v = conn.execute("SELECT version FROM records WHERE id = 'F'").fetchone()[0]
        op = ProposalOp(
            op=OpKind.UPDATE, target="F", group="g", read_set=[ReadRef(id="F", version=v)],
            payload={"v": f"worker {i}"},
        )  # fmt: skip
        assert store.commit(conn, Proposal(ops=[op]), bump).ok
    served_version = conn.execute("SELECT version FROM records WHERE id = 'F'").fetchone()[0]
    v_live = store.memory_version(conn)
    conn.close()

    v_new = backup.restore(backup_set, db_path)
    assert v_new > v_live
    c = open_db(db_path)
    try:
        op = ProposalOp(
            op=OpKind.UPDATE, target="F", group="g", read_set=[ReadRef(id="F", version=0)],
            payload={"v": "after restore"},
        )  # fmt: skip
        assert store.commit(c, Proposal(ops=[op]), bump).ok
        assert (
            c.execute("SELECT version FROM records WHERE id = 'F'").fetchone()[0] != served_version
        )
        stale = api.pre_send_check(
            thread_id="thread-owner", read_set=[ReadRef(id="F", version=served_version)], store=c
        )
        assert not stale.ok
    finally:
        c.close()
