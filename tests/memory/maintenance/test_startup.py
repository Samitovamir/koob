"""the memory spec §18.1 (v0.4): `api.startup` repairs what a crash left before the host
serves a turn. Spec marker: `[T maintenance/test_startup::<case>]`."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from koob.memory import api, forget, policy, store
from koob.memory.clock import ManualClock, iso
from koob.memory.model import ChatMode, ForgetScope

from .. import _ledger as L
from ..policy.conftest import Mem

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


@pytest.fixture
def clock():
    clk = ManualClock(T0)
    forget.reset_wiring()
    forget.use_key(None)
    api.wire(clock=clk)
    yield clk
    forget.reset_wiring()
    forget.use_key(None)


@pytest.fixture
def conn(tmp_path, clock):
    c = L.open_seeded(tmp_path)
    yield c
    c.close()


def _job(conn, job_id: str, *, state: str, attempts: int = 0, lease: str | None = None) -> None:
    conn.execute(
        "INSERT INTO jobs (id, jkind, payload, state, attempts, not_before, idempotency_key,"
        " created_at, updated_at) VALUES (?, 'rebuild_derived', '{}', ?, ?, ?, ?, ?, ?)",
        (job_id, state, attempts, lease, f"k:{job_id}", iso(T0), iso(T0)),
    )


def _state(conn, job_id: str) -> tuple[str, int]:
    row = conn.execute("SELECT state, attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row[0], row[1]


def _purge_job(conn, fid: str) -> tuple[str, int, str]:
    row = conn.execute(
        "SELECT state, attempts, payload FROM jobs WHERE idempotency_key LIKE ?", (f"%{fid}%",)
    ).fetchone()
    return row[0], row[1], json.loads(row[2]).get("stage")


def test_running_jobs_requeued(conn, clock):
    """A `running` job whose lease ended → `queued`, attempts + 1 (purge jobs included)."""
    assert L.add(conn, L.entity("e-s", "Sasha")).ok
    assert L.add(conn, L.fact("f-1", "e-s", "moving_to", "Berlin", text="Sasha moves to Berlin")).ok
    fid = forget.stop_use(conn, ["f-1"], scope=ForgetScope.ITEM, forever=True)

    def crash(_ctx):  # the process dies inside the purge, after the claim
        raise KeyboardInterrupt

    forget.wire(purge_steps={"crash": crash})
    with pytest.raises(KeyboardInterrupt):
        forget.purge(conn, fid)
    forget.wire(purge_steps={})
    assert _purge_job(conn, fid)[0] == "running"  # stranded without startup

    _job(conn, "j-old", state="running", lease=iso(T0))  # lease ended at T0
    _job(conn, "j-none", state="running")  # claimed before leases existed
    _job(conn, "j-tired", state="running", attempts=3, lease=iso(T0))  # third crash
    clock.advance(store.LEASE_S + 1)

    report = api.startup(store=conn)

    assert _purge_job(conn, fid)[:2] == ("queued", 1)
    assert _state(conn, "j-old") == ("queued", 1)
    assert _state(conn, "j-none") == ("queued", 1)
    assert _state(conn, "j-tired") == ("needs_attention", 4)
    assert (report.jobs_requeued, report.jobs_needing_attention) == (3, 1)
    # the requeued purge is due at once and finishes on the normal run
    assert forget.run_due(conn) == {fid: True}
    assert _purge_job(conn, fid)[2] == "purged"
    # idempotent: a second startup finds nothing to repair
    again = api.startup(store=conn)
    assert (again.jobs_requeued, again.jobs_needing_attention, again.actions_unknown) == (0, 0, 0)


def test_live_lease_not_taken(conn, clock):
    """A `running` job with a live lease (another process works on it) is left alone."""
    _job(conn, "j-live", state="running", lease=store.lease_until())
    clock.advance(store.LEASE_S - 5)
    store.renew_lease(conn, "j-live")  # renewed per group
    clock.advance(store.LEASE_S - 5)  # past the first lease, inside the renewed one

    report = api.startup(store=conn)

    assert _state(conn, "j-live") == ("running", 0)
    assert report.jobs_requeued == 0
    clock.advance(10)  # the other process died: its lease runs out
    assert api.startup(store=conn).jobs_requeued == 1
    assert _state(conn, "j-live") == ("queued", 1)


def test_pending_actions_unknown(conn, clock):
    """§16.2 step 4 runs inside startup."""
    for i, state in enumerate(("pending", "pending", "ok", "failed")):
        conn.execute(
            "INSERT INTO actions (id, action_key, origin_key, seq, tool, stable_target, state,"
            " reserved_at) VALUES (?, ?, 'msg-1', ?, 'tg_send', 'tg:1', ?, ?)",
            (f"a-{i}", f"k-{i}", i, state, iso(T0)),
        )
    report = api.startup(store=conn)
    states = dict(conn.execute("SELECT id, state FROM actions").fetchall())
    assert states == {"a-0": "unknown", "a-1": "unknown", "a-2": "ok", "a-3": "failed"}
    assert report.actions_unknown == 2
    closed = conn.execute("SELECT closed_at FROM actions WHERE id = 'a-0'").fetchone()[0]
    assert closed == iso(T0)  # the wired clock
    assert api.startup(store=conn).actions_unknown == 0


def test_startup_runs_registered_startup_kinds(conn, clock):
    """§18.1 step 5, generic by kind: queued jobs of a kind registered `at_startup` run before
    any turn; a runner that raises leaves its job queued with attempts + 1."""
    ran: list[str] = []

    def runner(_conn, job):
        ran.append(job["id"])
        if job["payload"].get("fail"):
            raise RuntimeError("boom")

    store.register_job_runner("test_startup_kind", runner, at_startup=True)
    try:
        for jid, payload in (("j-a", {}), ("j-b", {"fail": True})):
            conn.execute(
                "INSERT INTO jobs (id, jkind, payload, state, attempts, idempotency_key,"
                " created_at, updated_at) VALUES (?, 'test_startup_kind', ?, 'queued', 0, ?, ?, ?)",
                (jid, json.dumps(payload), f"k:{jid}", iso(T0), iso(T0)),
            )
        report = api.startup(store=conn)
    finally:
        store._job_runners.pop("test_startup_kind", None)
    assert sorted(ran) == ["j-a", "j-b"]
    assert _state(conn, "j-a") == ("done", 0)
    assert _state(conn, "j-b") == ("queued", 1)
    assert report.mode_changes_finished == 1


# Green once `api.startup` runs step 5 (`policy.finish_mode_forgets`); then drop the marker.
# The job side is checked without startup in review/test_policy.py.
def test_interrupted_mode_change_finished(mem):
    """§5.2: a mode row without its finished `mode_forget` job is completed before any turn."""
    db = Mem(mem)
    db.raw("r-dima", text="Дима: живём в отеле Морской", chat="tg:dima")
    db.fact("f-hotel", subject="e-dima", text_en="Dima stays at the Morskoy hotel",
            tier="other", source_raw_id="r-dima")  # fmt: skip
    # the mode row, its event and its job committed; the process died before stop-use
    policy.set_mode(mem, "tg:dima", ChatMode.BANNED)
    job = "SELECT state FROM jobs WHERE jkind = ?"
    assert mem.execute(job, (policy.MODE_FORGET_JOB,)).fetchone()[0] == "queued"
    status = "SELECT status FROM {} WHERE id = ?"
    assert mem.execute(status.format("records"), ("f-hotel",)).fetchone()[0] == "active"

    api.startup(store=mem)

    assert mem.execute(job, (policy.MODE_FORGET_JOB,)).fetchone()[0] == "done"
    assert mem.execute(status.format("records"), ("f-hotel",)).fetchone()[0] != "active"
    assert mem.execute(status.format("raw_chunks"), ("r-dima",)).fetchone()[0] != "active"
