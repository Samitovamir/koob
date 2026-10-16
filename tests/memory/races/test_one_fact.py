"""Several writers on one fact at once (spec §6.1, §6.3, §8.1, §8.2, §9.1, §9.6): two
corrections, a correction and a forget, a worker restating a fact while it is forgotten.
Whatever the interleaving, the slot ends with one usable value, the forgotten content is never
usable again, and no outcome is an exception."""

from __future__ import annotations

import json
import threading

from koob.memory import api, store
from koob.memory.model import OpKind, Proposal, ProposalOp, ReadRef, ReadSrc

from ._kit import (
    WAIT,
    add_raw,
    fact,
    handle,
    journal_path,
    lineage,
    open_db,
    owner_turn,
    person,
    ref,
    run_threads,
    seed,
    status_of,
    usable_slot,
    usable_texts,
)

OLD = ("12 May", "12 мая", "05-12")


def _world(conn):
    sasha = person("Саша")
    raw = add_raw(conn, "bday", "у Саши день рождения 12 мая", chat_id=None, turn_id="turn-0")
    bday = fact(
        "bday", sasha, "birthday", "12 May", "Sasha's birthday is 12 May",
        "у Саши день рождения 12 мая", source_raw_id=raw, value_norm="05-12",
    )  # fmt: skip
    seed(conn, sasha, bday, extra=lambda sql: lineage(sql, bday.id, raw))
    return sasha, bday, raw


def _fix(db_path, bday, day: str, n: int, gate: threading.Barrier):
    def run():
        c = open_db(db_path)
        try:
            gate.wait()
            words = f"нет, у Саши день рождения {day} мая"
            return api.fix(
                handle(bday.id),
                f"у Саши день рождения {day} мая",
                turn=owner_turn(words, n=n, thread=f"thread-{n}", session=f"session-{n}"),
                because_quote=f"у Саши день рождения {day} мая",
                store=c,
            )
        finally:
            c.close()

    return run


def test_two_fixes_on_one_fact_at_once_leave_one_value(db_path, conn):
    """§6.1, §8.1 «the latest wins»: two corrections of one single-cardinality slot commit at
    the same instant; the slot ends with exactly one usable value, the newest committed one,
    and the original is retired."""
    sasha, bday, _raw = _world(conn)
    gate = threading.Barrier(2, timeout=WAIT)
    a, b = run_threads(_fix(db_path, bday, "14", 2, gate), _fix(db_path, bday, "15", 3, gate))
    outcomes = {a.outcome, b.outcome}
    assert outcomes <= {"committed", "rejected"} and "committed" in outcomes
    assert status_of(conn, bday.id) in ("superseded", "retracted")
    slot = usable_slot(conn, sasha.id, "birthday")
    assert len(slot) == 1
    value = slot[0][1] or ""
    assert ("14" in value) != ("15" in value)
    newest = conn.execute(
        "SELECT f.id FROM facts f JOIN records r ON r.id = f.id"
        " WHERE f.subject_id = ? AND f.relation = 'birthday' AND r.id <> ?"
        " ORDER BY r.version DESC LIMIT 1",
        (sasha.id, bday.id),
    ).fetchone()[0]
    assert slot[0][0] == newest
    fixes = conn.execute("SELECT count(*) FROM events WHERE kind = 'fix'").fetchone()[0]
    assert fixes == [a.outcome, b.outcome].count("committed")


def test_fix_and_forget_on_one_fact_at_once(db_path, conn):
    """«исправь» (fix) and «забудь» (forget) on the same fact land together. In either order the
    forgotten content is never usable again, the target is out of use, prepared read sets
    holding it are refused, and the forget is in the journal."""
    _sasha, bday, raw = _world(conn)
    held = [ref(conn, bday.id)]
    gate = threading.Barrier(2, timeout=WAIT)

    def forget():
        c = open_db(db_path)
        try:
            gate.wait()
            return api.forget(
                handle(bday.id),
                turn=owner_turn("забудь день рождения Саши", n=3, thread="thread-3"),
                because_quote="день рождения Саши",
                store=c,
            )
        finally:
            c.close()

    fixed, forgot = run_threads(_fix(db_path, bday, "14", 2, gate), forget)
    assert fixed.outcome in ("committed", "rejected", "ask_first")
    assert forgot.outcome in ("stopped", "not_found")
    assert status_of(conn, bday.id) not in ("active", "archived")
    for text in usable_texts(conn):
        assert not any(o in text for o in OLD), text
    assert not api.pre_send_check(thread_id="thread-owner", read_set=held, store=conn).ok
    if forgot.outcome == "stopped":
        assert status_of(conn, bday.id) in ("forgotten", "purged")
        raw_st = conn.execute("SELECT status FROM raw_chunks WHERE id = ?", (raw,)).fetchone()[0]
        assert raw_st != "active"
        lines = [json.loads(x) for x in journal_path(db_path).read_text().splitlines() if x]
        assert any(bday.id in x["record_ids"] for x in lines if x["stage"] == "stopped")


def test_worker_holding_a_forgotten_fact_and_its_raw_cannot_commit(db_path, conn):
    """§6.3, §9.1, §9.6: a worker snapshotted the fact and its raw chunk; the forget commits;
    the worker's group is stale on both refs and nothing it proposed lands."""
    sasha, bday, raw = _world(conn)
    rev = conn.execute("SELECT rev FROM raw_chunks WHERE id = ?", (raw,)).fetchone()[0]
    snap = [ref(conn, bday.id), ReadRef(id=raw, version=rev, src=ReadSrc.RAW)]
    applied: list[str] = []

    def touch(ctx, op):
        applied.append(op.target)
        ctx.conn.execute("UPDATE records SET importance = 1.0 WHERE id = ?", (op.target,))
        return [op.target]

    api.forget(
        handle(bday.id),
        turn=owner_turn("забудь день рождения Саши", n=2),
        because_quote="день рождения Саши",
        store=conn,
    )
    res = store.commit(
        conn,
        Proposal(ops=[ProposalOp(op=OpKind.FLAG, target=bday.id, group=sasha.id, read_set=snap)]),
        touch,
    )
    assert not res.ok and applied == []
    assert {r.id for r in res.stale[sasha.id]} == {bday.id, raw}
