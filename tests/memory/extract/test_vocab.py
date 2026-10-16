"""the memory spec §4.1: the closed relation vocabulary.
Spec marker: `[T extract/test_vocab::<case>]`."""

from __future__ import annotations

from koob.memory.workers import extract, scheduler

from ..workers._kit import Gold, batch, fact, raw


def test_unknown_relation_is_other(env):
    """§4.1: a statement outside the closed vocabulary is stored as relation 'other'."""
    conn, clk = env
    raw(conn, "r1", "Я собираю марки с детства")
    gold = {
        "facts": [
            fact(
                "f1", "self", "collects_stamps", None, "Я собираю марки", "r1", stability="lasting"
            )
        ]
    }
    snap, rs = extract.snapshot(conn, ["r1"])
    assert "collects_stamps" not in {r["relation"] for r in snap["relations"]}
    assert extract.propose(snap, rs, Gold(gold)).ops[0].payload["record"]["relation"] == "other"
    extract.configure(cloud=Gold(gold))
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    assert conn.execute("SELECT relation FROM facts").fetchone()[0] == "other"
