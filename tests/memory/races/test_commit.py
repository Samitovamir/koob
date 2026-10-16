"""Races on `store.commit` (spec §6.3, §7.1, §7.2, G2): stale groups, concurrent writers on one
fact, a version bump while a read set is validated, a writer that fails or dies mid-commit.

Store level: these run against the contract-built `store.commit` with a test applier, so
they hold today. Every thread and process has its own connection. Spec marker:
`[T races/test_commit::stale_group_rejected_others_commit]`."""

from __future__ import annotations

import contextlib
import shutil
import threading
import time
from pathlib import Path

import pytest

from koob.memory import store
from koob.memory.model import EventKind, OpKind, Proposal, ProposalOp, ReadRef

from ._kit import WAIT, kill_points, open_db, run_child, run_threads


def _bare_fact(c, rid: str, value: str) -> None:
    c.execute(
        "INSERT INTO records (id, kind, version, author_tier, recorded_at, created_at, updated_at)"
        " VALUES (?, 'fact', 0, 'user', 't', 't', 't')",
        (rid,),
    )
    c.execute(
        "INSERT INTO facts (id, subject_id, relation, value_text, value_norm, text_en, fkind)"
        " VALUES (?, 'subj', 'birthday', ?, ?, ?, 'state')",
        (rid, value, value, f"value {value}"),
    )


def _value(c, rid: str) -> str:
    return c.execute("SELECT value_text FROM facts WHERE id = ?", (rid,)).fetchone()[0]


def _version(c, rid: str) -> int:
    return c.execute("SELECT version FROM records WHERE id = ?", (rid,)).fetchone()[0]


def _op(rid: str, version: int, group: str, value: str, *extra_reads: ReadRef) -> ProposalOp:
    return ProposalOp(
        op=OpKind.UPDATE,
        target=rid,
        group=group,
        read_set=[ReadRef(id=rid, version=version), *extra_reads],
        payload={"value": value},
    )


def _set_value(ctx: store.CommitCtx, op: ProposalOp) -> list[str]:
    ctx.conn.execute(
        "UPDATE facts SET value_text = ?, value_norm = ? WHERE id = ?",
        (op.payload["value"], op.payload["value"], op.target),
    )
    ctx.conn.execute("UPDATE records SET version = ? WHERE id = ?", (ctx.version, op.target))
    return [op.target]


def _events(c) -> list[str]:
    return [r[0] for r in c.execute("SELECT kind FROM events ORDER BY id").fetchall()]


@pytest.fixture
def three(db_path: Path):
    c = open_db(db_path)
    _bare_fact(c, "F", "12 May")  # a contact's birthday, the fact being corrected
    _bare_fact(c, "H", "Moscow")  # same contact's city: same group as F, untouched by the fix
    _bare_fact(c, "G", "Tver")  # a contact's city: another group
    yield c
    c.close()


def test_stale_group_rejected_others_commit(db_path, three):
    """§6.3: a worker snapshots, a fix lands, the worker commits: its group that read the fixed
    fact rolls back whole and is reported; its other group commits; the requeued group, from a
    fresh snapshot, commits without undoing the fix."""
    snapshot_taken, fixed = threading.Event(), threading.Event()
    applied: list[str] = []

    def spy(ctx, op):
        applied.append(op.target)
        return _set_value(ctx, op)

    def worker():
        w = open_db(db_path)
        try:
            v = {rid: _version(w, rid) for rid in ("F", "H", "G")}
            old_birthday = _value(w, "F")
            proposal = Proposal(
                ops=[
                    _op("H", v["H"], "sasha", "Moscow (since 2024)"),
                    _op("F", v["F"], "sasha", f"{old_birthday} (restated)"),
                    _op("G", v["G"], "dima", "Tver (since 2025)"),
                ],
                model_id="gold-extractor",
                prompt_id="extract-v1",
            )
            snapshot_taken.set()
            assert fixed.wait(WAIT)
            return store.commit(w, proposal, spy)
        finally:
            w.close()

    def hot_fix():
        assert snapshot_taken.wait(WAIT)
        h = open_db(db_path)
        try:
            res = store.commit(
                h,
                Proposal(ops=[_op("F", _version(h, "F"), "sasha", "14 May")]),
                _set_value,
                event_kind=EventKind.FIX,
            )
            assert res.ok
        finally:
            h.close()
            fixed.set()

    res, _ = run_threads(worker, hot_fix)

    assert res.committed == ["dima"]
    assert list(res.stale) == ["sasha"]
    assert ReadRef(id="F", version=0) in res.stale["sasha"]
    assert all(r.id != "H" for r in res.stale["sasha"])  # only what changed is reported
    assert applied == ["G"]  # a stale group is never handed to the applier
    c = three
    assert _value(c, "F") == "14 May"  # the fix stands
    assert _value(c, "H") == "Moscow"  # the stale group rolled back whole
    assert _value(c, "G") == "Tver (since 2025)"
    assert store.memory_version(c) == 2
    assert _events(c) == ["fix", "write"]
    assert not c.in_transaction

    # The stale group requeues alone with a fresh snapshot (§6.3) and now commits.
    fresh = {rid: _version(c, rid) for rid in ("F", "H")}
    requeued = Proposal(
        ops=[
            _op("H", fresh["H"], "sasha", "Moscow (since 2024)"),
            _op("F", fresh["F"], "sasha", f"{_value(c, 'F')} (restated)"),
        ]
    )
    again = store.commit(c, requeued, _set_value)
    assert again.ok and again.committed == ["sasha"]
    assert _value(c, "F") == "14 May (restated)"  # built on the corrected value, not the old
    assert store.memory_version(c) == 3


@pytest.mark.parametrize("round_no", range(8))
def test_concurrent_fixes_on_one_fact_exactly_one_wins(db_path, three, round_no):
    """G2: two writers read F at the same version and commit at the same instant: exactly one
    commits, the other is stale; the version moves once; no lost update."""
    gate = threading.Barrier(2, timeout=WAIT)

    def writer(value: str):
        def run():
            w = open_db(db_path)
            try:
                read = _version(w, "F")
                gate.wait()
                return value, store.commit(
                    w, Proposal(ops=[_op("F", read, "sasha", value)]), _set_value
                )
            finally:
                w.close()

        return run

    outcomes = run_threads(writer(f"14 May #{round_no}"), writer(f"15 May #{round_no}"))
    winners = [v for v, r in outcomes if r.ok]
    losers = [r for _, r in outcomes if not r.ok]
    assert len(winners) == 1 and len(losers) == 1
    assert list(losers[0].stale) == ["sasha"]
    c = three
    assert _value(c, "F") == winners[0]
    assert _version(c, "F") == 1 and store.memory_version(c) == 1
    assert _events(c) == ["write"]


def test_version_bump_while_read_set_is_validated(db_path, three):
    """§6.3, §7.2: a commit that starts while another writer holds the write lock waits, then
    validates against what that writer committed (the check runs inside BEGIN IMMEDIATE)."""
    inside, release, b_waiting = threading.Event(), threading.Event(), threading.Event()
    read_v = _version(three, "F")

    def slow(ctx, op):
        out = _set_value(ctx, op)
        inside.set()
        assert release.wait(WAIT)
        return out

    def a():
        w = open_db(db_path)
        try:
            return store.commit(w, Proposal(ops=[_op("F", read_v, "sasha", "14 May")]), slow)
        finally:
            w.close()

    def b():
        assert inside.wait(WAIT)
        w = open_db(db_path)
        try:
            b_waiting.set()
            return store.commit(w, Proposal(ops=[_op("F", read_v, "sasha", "15 May")]), _set_value)
        finally:
            w.close()

    def releaser():
        assert b_waiting.wait(WAIT)
        time.sleep(0.05)  # let b reach BEGIN IMMEDIATE and block on the lock
        release.set()

    ra, rb, _ = run_threads(a, b, releaser)
    assert ra.ok
    assert not rb.ok and list(rb.stale) == ["sasha"]
    assert _value(three, "F") == "14 May"
    assert store.memory_version(three) == 1


def test_heat_and_importance_do_not_make_a_read_set_stale(db_path, three):
    """§7.1: `importance`, `confidence`, `review` and `record_stats` are not version-bearing; a
    worker's read set survives a concurrent use-count bump and still commits."""
    read_v = _version(three, "F")
    w = open_db(db_path)
    try:
        w.execute(
            "UPDATE records SET importance = 0.9, confidence = 0.4, review = 1 WHERE id = 'F'"
        )
        w.execute("INSERT INTO record_stats (record_id, uses, heat) VALUES ('F', 7, 3.5)")
    finally:
        w.close()
    res = store.commit(three, Proposal(ops=[_op("F", read_v, "sasha", "14 May")]), _set_value)
    assert res.ok


class _Boom(Exception):
    pass


def test_applier_failure_rolls_back_its_group_only(three):
    """A writer that fails mid-group leaves that group unapplied, earlier groups committed, the
    version counting only what committed, and no open transaction."""

    def apply(ctx, op):
        if op.target == "G":
            raise _Boom("worker crashed mid-group")
        return _set_value(ctx, op)

    proposal = Proposal(
        ops=[
            _op("F", 0, "sasha", "14 May"),
            _op("H", 0, "dima", "Moscow (moved)"),
            _op("G", 0, "dima", "Tver (moved)"),
        ]
    )
    with contextlib.suppress(_Boom):  # raising or reporting are both fine; the state is not
        store.commit(three, proposal, apply)
    c = three
    assert not c.in_transaction
    assert _value(c, "F") == "14 May"
    assert _value(c, "H") == "Moscow" and _value(c, "G") == "Tver"
    assert store.memory_version(c) == 1
    assert _events(c) == ["write"]


def test_process_dies_mid_commit_nothing_half_applied(db_path, three, tmp_path):
    """§19.1 restart mid-commit: the process dies at points spread over a two-group commit; on
    reopen every group is all-or-nothing, in order, and the version counts committed groups."""
    three.close()
    proposal = Proposal(
        ops=[
            _op("F", 0, "sasha", "14 May"),
            _op("G", 0, "dima", "Tver (moved)"),
            _op("H", 0, "dima", "Moscow (moved)"),
        ]
    )
    pristine = tmp_path / "pristine"
    shutil.copytree(db_path.parent.parent, pristine)

    def attempt(kill_at: int | None, name: str):
        root = tmp_path / name
        shutil.copytree(pristine, root)
        db = root / "memory" / "memory.db"
        spec = {
            "db": str(db),
            "home": str(root / "home"),
            "kill_at": kill_at,
            "target": "commit",
            "proposal": proposal.model_dump(mode="json"),
        }
        code, out = run_child(spec)
        return db, code, out

    _, code, out = attempt(None, "dry")
    assert code == 0 and out is not None
    total = out["steps"]
    assert total > 10

    killed = 0
    for k in kill_points(total):
        db, code, _ = attempt(k, f"kill-{k}")
        killed += code == 77
        c = open_db(db)
        try:
            assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            g1 = _value(c, "F") == "14 May"
            g2 = _value(c, "G") == "Tver (moved)"
            assert g2 == (_value(c, "H") == "Moscow (moved)"), f"group torn at step {k}"
            assert g1 or not g2, f"groups out of order at step {k}"
            if not g1:
                assert _value(c, "F") == "12 May"
            if not g2:
                assert (_value(c, "G"), _value(c, "H")) == ("Tver", "Moscow")
            assert store.memory_version(c) == int(g1) + int(g2), f"version off at step {k}"
            assert len(_events(c)) == int(g1) + int(g2)
            assert code == 77 or (g1 and g2)
        finally:
            c.close()
    assert killed >= 3  # the process really died mid-call at several points
