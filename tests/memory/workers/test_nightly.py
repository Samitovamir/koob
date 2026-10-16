"""night: the night pass's gates, hooks and caps (MEMORY.md §6.3, §6.4, §9.4, §18)."""

from __future__ import annotations

import json

import pytest

from koob.memory import api, store
from koob.memory.recall import core_view
from koob.memory.workers import nightly

from .. import _ledger as L


@pytest.fixture(autouse=True)
def _on_ac():
    nightly.use_power(lambda: True)
    yield
    nightly.use_power(None)


def test_battery_skips_and_keeps_the_watermark(mem):
    nightly.use_power(lambda: False)
    assert api.run_nightly(store=mem) == {"skipped_power": 1}
    assert nightly.last_mark(mem) is None


def test_register_step_and_a_failing_step(mem):
    seen = []
    nightly.register_step("assoc", lambda night: seen.append(night.since_version) or 3)

    def boom(night):
        raise RuntimeError("x")

    nightly.register_step("boom", boom)
    try:
        counts = api.run_nightly(store=mem)
    finally:
        nightly.unregister_step("assoc")
        nightly.unregister_step("boom")
    assert seen == [0] and counts["assoc"] == 3 and counts["failed_boom"] == 1
    assert counts["backups"] == 1
    events = [r[0] for r in mem.execute("SELECT event FROM audit")]
    assert "nightly_failed:boom" in events


def test_watermark_moves_and_holds_counts_only(mem):
    L.add(mem, L.entity("sasha", "Sasha"))
    L.add(mem, L.fact("f1", "sasha", "nationality", "German"))
    assert api.run_nightly(store=mem)["delta"] == 1
    mark = nightly.last_mark(mem)
    assert mark["memory_version"] == store.memory_version(mem) or mark["memory_version"] > 0
    assert "German" not in json.dumps(mark)
    L.add(mem, L.fact("f2", "sasha", "nationality", "Swiss"), rule_match=False)
    assert api.run_nightly(store=mem)["delta"] == 1


def test_rebuilds_capped_newest_first(mem, monkeypatch):
    ran = []
    monkeypatch.setitem(core_view._REBUILDERS, "summary", lambda conn, p: ran.append(p["dkey"]))
    for i in range(3):
        ts = f"2026-10-0{i + 1}T00:00:00.000000Z"
        mem.execute(
            "INSERT INTO jobs (id, jkind, payload, idempotency_key, created_at, updated_at)"
            " VALUES (?, 'rebuild_derived', ?, ?, ?, ?)",
            (f"j{i}", json.dumps({"dkind": "summary", "dkey": f"d{i}"}), f"k{i}", ts, ts),
        )
    counts = nightly.run(mem, cap=2)
    assert counts["rebuilt"] == 2 and ran == ["d2", "d1"]
    left = mem.execute("SELECT state FROM jobs WHERE id = 'j0'").fetchone()[0]
    assert left == "queued"  # the rest waits for the next night
