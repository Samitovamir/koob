"""the memory spec §6.3 (v0.4): the scheduler owns requeue and its cap; a worker's
content rejection drops its group alone; a hot rejection raises.
Spec marker: `[T workers/test_requeue::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import ledger, store
from koob.memory.ledger import CommitRejected
from koob.memory.model import AuthorTier, OpKind
from koob.memory.workers import extract, scheduler

from .. import _ledger as L
from ._kit import SASHA, Gold, audit, batch, fact, jobs, raw

TWO = {
    **SASHA,
    "facts": [*SASHA["facts"], fact("f2", "self", "lives_in", "Tver", "я живу в Твери", "r1")],
}


def _bumping(conn, times: list[int]):
    """A gold extractor while a hot write changes `self` under it (the self group goes stale)."""

    def gold(snap):
        if times[0] > 0:
            times[0] -= 1
            conn.execute("UPDATE records SET version = version + 1 WHERE id = 'self'")
            conn.commit()
        return TWO

    return gold


def test_stale_group_requeued_alone(env):
    """A stale group gets a child job with a fresh snapshot; committed groups stay."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе, а я живу в Твери")
    g = Gold(_bumping(conn, [1]))
    extract.configure(cloud=g)
    parent = batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    assert {r[0] for r in conn.execute("SELECT relation FROM facts")} == {"works_at"}
    child = [j for j in jobs(conn) if j["payload"].get("parent") == parent]
    assert len(child) == 1 and child[0]["payload"]["group"] == "self"
    assert child[0]["payload"]["attempt"] == 2 and child[0]["state"] == "queued"
    key = conn.execute("SELECT idempotency_key FROM jobs WHERE id = ?", (child[0]["id"],))
    assert key.fetchone()[0] == f"{parent}:self:2"
    scheduler.tick(conn, clk)  # fresh snapshot, only the self group
    assert len(g.snaps) == 2
    rels = [r[0] for r in conn.execute("SELECT relation FROM facts ORDER BY relation")]
    assert rels == ["lives_in", "works_at"]  # the committed group was not redone
    assert {j["state"] for j in jobs(conn)} == {"done"}


def test_third_stale_needs_attention(env):
    """Attempt 3 stale → the group job is `needs_attention`, one doctor line, no user line."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе, а я живу в Твери")
    extract.configure(cloud=Gold(_bumping(conn, [99])))
    batch(conn, ["r1"])
    for _ in range(4):
        scheduler.tick(conn, clk)
    states = sorted((j["payload"].get("attempt", 1), j["state"]) for j in jobs(conn))
    assert states == [(1, "done"), (2, "done"), (3, "needs_attention")]
    assert audit(conn) == ["job_needs_attention:stale"]


def test_rejected_group_not_requeued(env, monkeypatch):
    """A content rejection (suppression, tier, retracted value) is never retried."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе, а я живу в Твери")
    extract.configure(cloud=Gold(TWO))
    real = ledger.apply_op

    def refuse_self(ctx, op):
        if op.group == "self":
            raise ledger.CommitRejected("suppressed", "a forgotten slot")
        return real(ctx, op)

    monkeypatch.setattr(ledger, "apply_op", refuse_self)
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    scheduler.tick(conn, clk)
    assert {r[0] for r in conn.execute("SELECT relation FROM facts")} == {"works_at"}
    assert len(jobs(conn)) == 1 and jobs(conn)[0]["state"] == "done"  # no child job
    assert audit(conn) == ["extract_rejected:suppressed"]  # content-free


def _proposal_ops(conn):
    """Three groups: a good fact, a tier breach (a tool mail stored as the user's), a good fact."""
    L.raw(conn, "r-his", "Саша работает в Яндексе", tier="user")
    L.raw(conn, "r-mail", "forwarded: Denis now works at Evil Corp", tier="tool")
    assert L.add(conn, L.entity("e-s", "Sasha")).ok
    assert L.add(conn, L.entity("e-d", "Dima")).ok
    good = L.fact("f-good", "e-s", "works_at", "Yandex", source_raw_id="r-his")
    bad = L.fact("f-bad", "self", "works_at", "Evil Corp", source_raw_id="r-mail")
    late = L.fact("f-late", "e-d", "lives_in", "Tver", source_raw_id="r-his")
    return [
        L.op(OpKind.ADD, group=f.subject_id, record=f.model_dump(mode="json"))
        for f in (good, bad, late)
    ]


def test_worker_rejection_drops_group(tmp_path):
    """`CommitRejected` in a worker proposal (`model_id` set) lands in `CommitResult.rejected`;
    the other groups commit."""
    conn = L.open_seeded(tmp_path)
    ops = _proposal_ops(conn)
    v0 = store.memory_version(conn)
    res = L.commit(conn, *ops, tier=AuthorTier.SYSTEM, model_id="worker-extract")
    assert res.committed == ["e-s", "e-d"]
    assert list(res.rejected) == ["self"] and res.rejected["self"].startswith("tier")
    assert not res.stale and not res.ok
    ids = {r[0] for r in conn.execute("SELECT id FROM records WHERE kind = 'fact'")}
    assert ids == {"f-good", "f-late"}  # the rejected group rolled back alone
    assert store.memory_version(conn) == v0 + 2  # no version for the dropped group
    assert not conn.in_transaction
    conn.close()


def test_hot_rejection_raises(tmp_path):
    """A hot or code proposal (`model_id` None) raises `CommitRejected` to its writer."""
    conn = L.open_seeded(tmp_path)
    ops = _proposal_ops(conn)
    v0 = store.memory_version(conn)
    with pytest.raises(CommitRejected) as e:
        L.commit(conn, *ops, tier=AuthorTier.SYSTEM)  # code: no model
    assert e.value.reason == "tier"
    assert isinstance(e.value, store.OpRejected)
    # groups before the refused one stay committed; nothing after it ran
    ids = {r[0] for r in conn.execute("SELECT id FROM records WHERE kind = 'fact'")}
    assert ids == {"f-good"}
    assert store.memory_version(conn) == v0 + 1
    assert not conn.in_transaction
    conn.close()
