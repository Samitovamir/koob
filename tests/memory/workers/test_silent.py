"""the memory spec §6.3, §6.1: background learning is silent; the urgent safety-net job
runs its turn's extraction now. Spec marker: `[T workers/test_silent::<case>]`."""

from __future__ import annotations

from koob.memory.model import Label
from koob.memory.workers import extract, scheduler
from koob.memory.writers import hot

from ._kit import SASHA, Gold, batch, jobs, raw


def _safety(conn, turn: str) -> str:
    return scheduler.enqueue(
        conn,
        scheduler.SAFETY,
        {"turn_id": turn, "thread_id": "th", "entities": []},
        idempotency_key=f"safety_net:{turn}",
    )


def test_no_owner_message_from_worker(env):
    """§6.3: background learning sends the user no lines."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе")
    raw(conn, "r2", "Саша работает в Яндексе", tier="other", labels=int(Label.HEALTH))
    extract.configure(cloud=Gold(SASHA))
    batch(conn, ["r1"])
    batch(conn, ["r2"], turn="t2")  # waits: no local processor
    assert isinstance(scheduler.tick(conn, clk), int)
    assert hot.failure_line(conn, "th") is None
    assert conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0
    events = [r[0] for r in conn.execute("SELECT DISTINCT actor FROM audit")]
    assert set(events) <= {"worker", "ingest", "memory", "system"}


def test_safety_net_runs_the_turn_now(env):
    """§6.1 step 2: the urgent job runs its turn's queued batch; with no allowed processor it
    fails (the hot path says its one line) and the batch keeps waiting."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе", turn="t1")
    extract.configure(cloud=Gold(SASHA))
    batch(conn, ["r1"], turn="t1")
    net = _safety(conn, "t1")
    scheduler.tick(conn, clk)
    assert {j["state"] for j in jobs(conn)} == {"done"} and net
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1

    raw(conn, "r2", "Саша работает в Яндексе", tier="other", labels=int(Label.HEALTH), turn="t2")
    batch(conn, ["r2"], turn="t2")
    _safety(conn, "t2")
    scheduler.tick(conn, clk)
    state = {j["jkind"]: j["state"] for j in jobs(conn) if j["payload"].get("turn_id") == "t2"}
    assert state == {"safety_net": "failed", "extract": "queued"}
    assert hot.failure_line(conn, "th") == "Не смог применить исправление про это — уточни."
