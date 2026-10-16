"""`store.commit`: BEGIN IMMEDIATE, read-set validation, version bump (§6.3, §7.1, §7.2).

The race suite (`tests/memory/races/`) covers interleavings; this is the unit."""

from __future__ import annotations

import json

import pytest

from koob.memory import store
from koob.memory.model import EventKind, OpKind, Proposal, ProposalOp, ReadRef, ReadSrc


def _record(conn, rid, version):
    conn.execute(
        "INSERT INTO records (id, kind, version, author_tier, recorded_at, created_at,"
        " updated_at) VALUES (?, 'fact', ?, 'user', 't', 't', 't')",
        (rid, version),
    )


def _bump(ctx: store.CommitCtx, op: ProposalOp):
    ctx.conn.execute("UPDATE records SET version = ? WHERE id = ?", (ctx.version, op.target))
    return [op.target]


def _op(target, version, group="g1"):
    return ProposalOp(
        op=OpKind.UPDATE, target=target, group=group, read_set=[ReadRef(id=target, version=version)]
    )


def test_commit_bumps_version_and_writes_event(mem):
    _record(mem, "r1", 0)
    res = store.commit(mem, Proposal(ops=[_op("r1", 0)]), _bump, event_kind=EventKind.FIX)
    assert res.ok and res.committed == ["g1"] and res.versions == {"g1": 1}
    assert store.memory_version(mem) == 1
    assert mem.execute("SELECT version FROM records WHERE id = 'r1'").fetchone()[0] == 1
    ev = mem.execute("SELECT kind, memory_version, item_ids FROM events").fetchall()
    assert [(e[0], e[1], json.loads(e[2])) for e in ev] == [("fix", 1, ["r1"])]
    assert not mem.in_transaction


def test_commit_rejects_stale_read_set(mem):
    _record(mem, "r1", 0)
    store.commit(mem, Proposal(ops=[_op("r1", 0)]), _bump)  # r1 is now at version 1
    calls = []

    def spy(ctx, op):
        calls.append(op.target)
        return [op.target]

    res = store.commit(mem, Proposal(ops=[_op("r1", 0)]), spy)  # read at the old version
    assert not res.ok and res.committed == []
    assert res.stale == {"g1": [ReadRef(id="r1", version=0)]}
    assert calls == []
    assert store.memory_version(mem) == 1
    assert mem.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    assert not mem.in_transaction


def test_stale_group_alone_is_rejected(mem):
    """§6.3: a stale group requeues alone; the others commit."""
    _record(mem, "a", 0)
    _record(mem, "b", 5)
    res = store.commit(mem, Proposal(ops=[_op("a", 0, "ga"), _op("b", 4, "gb")]), _bump)
    assert res.committed == ["ga"] and list(res.stale) == ["gb"]
    assert store.memory_version(mem) == 1


def test_missing_item_and_raw_and_derived_refs_are_stale(mem):
    mem.execute(
        "INSERT INTO raw_chunks (id, rev, source_kind, author_tier, ts, text, content_hmac,"
        " policy_version, created_at, updated_at)"
        " VALUES ('w1', 2, 'owner_message', 'user', 't', 'x', 'h', 0, 't', 't')"
    )
    mem.execute(
        "INSERT INTO derived (id, dkind, content, read_set, built_at, valid, created_at,"
        " updated_at) VALUES ('d1', 'summary', 'x', '[]', 't', 0, 't', 't')"
    )
    refs = [
        ReadRef(id="gone", version=1),
        ReadRef(id="w1", version=1, src=ReadSrc.RAW),
        ReadRef(id="w1", version=2, src=ReadSrc.RAW),
        ReadRef(id="d1", version=0, src=ReadSrc.DERIVED),
    ]
    assert store.stale_refs(mem, refs) == [refs[0], refs[1], refs[3]]


def test_pre_send_view_requires_usable(mem):
    """§7.4: the pre-send form also refuses non-usable records at the same version."""
    _record(mem, "r1", 3)
    mem.execute("UPDATE records SET status = 'needs_review' WHERE id = 'r1'")
    ref = ReadRef(id="r1", version=3)
    assert store.stale_refs(mem, [ref]) == []
    assert store.stale_refs(mem, [ref], require_usable=True) == [ref]


def test_applier_error_rolls_back(mem):
    _record(mem, "r1", 0)

    def boom(ctx, op):
        ctx.conn.execute("UPDATE records SET version = 99 WHERE id = 'r1'")
        raise RuntimeError("applier failed")

    with pytest.raises(RuntimeError):
        store.commit(mem, Proposal(ops=[_op("r1", 0)]), boom)
    assert store.memory_version(mem) == 0
    assert mem.execute("SELECT version FROM records WHERE id = 'r1'").fetchone()[0] == 0
    assert not mem.in_transaction
