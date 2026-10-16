"""Write-lock budget: the memory spec §3.1, G9 (store).

A nightly pass commits group after group; SQLite's lock is not fair, so without the store's
priority rule a hot write starves until its busy timeout (measured: 5 s, then "database is
locked"). Spec marker: `[T perf/test_locks::hot_write_under_60ms_during_nightly_commit]`."""

from __future__ import annotations

import threading
import time

from koob.memory import store
from koob.memory.model import OpKind, Proposal, ProposalOp

GROUPS = 40
GROUP_HOLD_S = 0.030  # a worker group inside the 50 ms budget, FTS writes included


def _nightly_proposal() -> Proposal:
    ops = [
        ProposalOp(op=OpKind.FLAG, target=None, group=f"entity-{i}", payload={"i": i})
        for i in range(GROUPS)
    ]
    return Proposal(ops=ops, model_id="worker-model", prompt_id="nightly-v1")


def _slow_apply(ctx: store.CommitCtx, op: ProposalOp):
    ctx.conn.execute(
        "INSERT INTO audit (ts, event, record_id, actor) VALUES (?, 'nightly', ?, 'worker')",
        (ctx.ts, op.group),
    )
    time.sleep(GROUP_HOLD_S)
    return []


def _hot_apply(ctx: store.CommitCtx, op: ProposalOp):
    ctx.conn.execute(
        "INSERT INTO audit (ts, event, record_id, actor) VALUES (?, 'hot', ?, 'owner')",
        (ctx.ts, op.group),
    )
    return []


def _scenario(root) -> tuple[list[float], store.CommitResult]:
    db = root / "memory" / "memory.db"
    night = store.open(db, home=root / "home")
    hot = store.connect(db)
    results: dict[str, store.CommitResult] = {}

    def run_nightly():
        results["night"] = store.commit(night, _nightly_proposal(), _slow_apply)

    worker = threading.Thread(target=run_nightly)
    worker.start()
    time.sleep(GROUP_HOLD_S * 2)
    waits = []
    try:
        for i in range(12):
            t0 = time.perf_counter()
            r = store.commit(
                hot,
                Proposal(ops=[ProposalOp(op=OpKind.ADD, group=f"hot-{i}")]),
                _hot_apply,
            )
            waits.append((time.perf_counter() - t0) * 1000)
            assert r.ok
            time.sleep(0.017)  # land at different points of the nightly's groups
    finally:
        worker.join(timeout=30)
        night.close()
        hot.close()
    assert not worker.is_alive()
    return waits, results["night"]


def test_hot_write_under_60ms_during_nightly_commit(tmp_path):
    """§3.1: a hot write waits < 60 ms while a nightly commit runs.

    Starvation (what the priority rule prevents) costs the full 5 s busy timeout every
    time; a loaded machine can add a one-off scheduling hiccup, so one fresh rerun is
    allowed before the 60 ms bound fails."""
    waits, night = _scenario(tmp_path / "a")
    if max(waits) >= 60 or night.over_budget:
        waits, night = _scenario(tmp_path / "b")
    assert len(night.committed) == GROUPS  # the worker is slowed, never starved
    assert night.over_budget == [], {k: round(v, 1) for k, v in night.lock_ms.items()}
    assert max(waits) < 60, f"hot write waits {sorted(round(w, 1) for w in waits)} ms"


def test_commit_reports_lock_hold_and_over_budget(mem):
    """§3.1: each group's lock hold is measured so a worker can chunk smaller."""

    def slow(ctx, op):
        time.sleep(0.06 if op.group == "big" else 0)
        return []

    ops = [
        ProposalOp(op=OpKind.FLAG, group="small"),
        ProposalOp(op=OpKind.FLAG, group="big"),
    ]
    result = store.commit(mem, Proposal(ops=ops), slow)
    assert set(result.lock_ms) == {"small", "big"}
    assert result.over_budget == ["big"]


def test_worker_proposals_default_to_background(tmp_path, monkeypatch):
    """§3.1: a worker proposal (model_id set) yields; hot-path and code writes do not."""
    seen = []
    real = store.begin_write

    def spy(conn, *, priority="hot"):
        seen.append(priority)
        return real(conn, priority=priority)

    monkeypatch.setattr(store, "begin_write", spy)
    conn = store.open(tmp_path / "m" / "memory.db", home=tmp_path / "home")
    store.commit(conn, Proposal(ops=[ProposalOp(op=OpKind.FLAG, group="a")], model_id="m"))
    store.commit(conn, Proposal(ops=[ProposalOp(op=OpKind.FLAG, group="a")]))
    store.commit(conn, Proposal(ops=[ProposalOp(op=OpKind.FLAG, group="a")]), priority="background")
    assert seen == ["background", "hot", "background"]
    conn.close()


def test_stale_hot_intent_is_ignored(tmp_path):
    """A crashed hot writer's intent file must not stall workers forever."""
    import os

    db = tmp_path / "memory.db"
    d = db.with_name(db.name + "-hot")
    d.mkdir()
    marker = d / "123-456-0"
    marker.touch()
    assert store.hot_waiting(db)
    old = time.time() - 3600
    os.utime(marker, (old, old))
    assert not store.hot_waiting(db)
    assert not marker.exists()
    assert not store.hot_waiting(None)  # ":memory:" has no intents


def test_write_tx_rolls_back_on_error(mem):
    try:
        with store.write_tx(mem):
            mem.execute("UPDATE meta SET memory_version = 77")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert store.memory_version(mem) == 0
    assert not mem.in_transaction
