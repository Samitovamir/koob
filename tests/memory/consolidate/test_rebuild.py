"""Acceptance for the memory spec §6.4 (G10). Spec marker:
`[T consolidate/test_rebuild::<case>]`."""

from __future__ import annotations

import json
from datetime import UTC

import pytest

from koob.memory import api, episodes
from koob.memory.model import OWNER_ENTITY_ID, AuthorTier, FKind, OpKind
from koob.memory.workers import nightly

from .._ledger import add, commit, entity, fact, op, open_seeded, raw

DAY = "2026-09-05"


@pytest.fixture
def conn(tmp_path):
    nightly.use_power(lambda: True)
    episodes.DAY_TZ[0] = UTC
    c = open_seeded(tmp_path)
    yield c
    c.close()
    episodes.DAY_TZ[0] = None
    nightly.use_power(None)


def test_summary_from_records_only(conn):
    """§6.4: summaries and episodes are built from usable records, never from the previous
    summary. A tampered summary and a tampered episode do not survive a correction: the
    rebuild reads the ledger (the corrected fact) and raw quotes only."""
    assert add(conn, entity("e-ilya", "Илья")).ok
    raw(conn, "r1", "Ужин с Ильёй в субботу", ts_=f"{DAY}T12:00:00Z")
    assert add(
        conn,
        fact(
            "f-sat",
            OWNER_ENTITY_ID,
            "other",
            None,
            fkind=FKind.PLAN,
            value_entity_id="e-ilya",
            text="Dinner with Ilya on Saturday",
            source_raw_id="r1",
        ),  # fmt: skip
    ).ok
    api.run_nightly(store=conn)
    sid = episodes.summary_id(DAY)
    assert (
        "Dinner with Ilya on Saturday"
        in conn.execute("SELECT content FROM derived WHERE id = ?", (sid,)).fetchone()[0]
    )
    (ep_id,) = [r[0] for r in conn.execute("SELECT id FROM episodes")]

    # a "previous summary" and episode text that exist nowhere in the ledger
    conn.execute("UPDATE derived SET content = content || ' POISON-SUM' WHERE id = ?", (sid,))
    conn.execute(
        "UPDATE episodes SET summary_en = summary_en || ' POISON-EP' WHERE id = ?", (ep_id,)
    )

    # the user's correction: Sunday, not Saturday
    fixed = fact("f-sun", OWNER_ENTITY_ID, "other", None, fkind=FKind.PLAN,
                 value_entity_id="e-ilya", text="Dinner with Ilya on Sunday", source_raw_id="r1")  # fmt: skip
    body = {"record": fixed.model_dump(mode="json"), "fix": True}
    assert commit(conn, op(OpKind.SUPERSEDE, target="f-sat", **body), tier=AuthorTier.USER).ok
    assert conn.execute("SELECT valid FROM derived WHERE id = ?", (sid,)).fetchone()[0] == 0
    queued = conn.execute(
        "SELECT count(*) FROM jobs WHERE jkind = 'rebuild_derived' AND state = 'queued'"
    ).fetchone()[0]
    assert queued >= 1

    api.run_nightly(store=conn)
    row = conn.execute(
        "SELECT content, read_set, valid FROM derived WHERE id = ?", (sid,)
    ).fetchone()
    assert row["valid"] == 1
    assert "Sunday" in row["content"] and "Saturday" not in row["content"]
    assert "POISON" not in row["content"]
    assert all(len(r) < 3 or r[2] != "derived" for r in json.loads(row["read_set"]))
    assert {r[0] for r in json.loads(row["read_set"])} == {"f-sun", "r1"}
    assert (
        conn.execute(
            "SELECT count(*) FROM jobs WHERE jkind = 'rebuild_derived' AND state = 'queued'"
        ).fetchone()[0]
        == 0
    )

    live = conn.execute(
        "SELECT e.id, e.summary_en FROM episodes e JOIN records r ON r.id = e.id"
        " WHERE r.status = 'active'"
    ).fetchall()
    assert len(live) == 1 and live[0]["id"] != ep_id
    assert "Sunday" in live[0]["summary_en"] and "POISON" not in live[0]["summary_en"]
    assert "Saturday" not in live[0]["summary_en"]
