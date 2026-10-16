"""Episodes and daily summaries (episodes): the memory spec §4.3, §6.4, §6.5, §15.3,
G1 (labels inherited), G10 (never from the previous summary)."""

from __future__ import annotations

import json
from datetime import UTC

import pytest

from koob.memory import api, episodes
from koob.memory.model import (
    OWNER_ENTITY_ID,
    AuthorTier,
    FKind,
    Label,
    OpKind,
    TurnCtx,
    TurnOrigin,
    validate_payload,
)
from koob.memory.recall import vectors
from koob.memory.workers import nightly

from .._ledger import add, commit, entity, fact, op, open_seeded, raw

DAY = "2026-08-20"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    nightly.use_power(lambda: True)
    episodes.DAY_TZ[0] = UTC
    c = open_seeded(tmp_path)
    yield c
    c.close()
    episodes.DAY_TZ[0] = None
    nightly.use_power(None)


def _turn(words: str) -> TurnCtx:
    return TurnCtx(
        turn_id="turn-1",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock="2026-10-12T09:00:00.000000Z",
    )


def _episodes(conn, status: str = "active") -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT e.*, r.status, r.labels FROM episodes e JOIN records r ON r.id = e.id"
            " WHERE r.status = ? ORDER BY e.span_start, e.id",
            (status,),
        )
    ]


def _links(conn, eid: str) -> set[tuple[str, str]]:
    return {
        (r[0], r[1])
        for r in conn.execute(
            "SELECT role, target_id FROM episode_links WHERE episode_id = ?", (eid,)
        )
    }


def _derived_from(conn, rid: str) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT to_id FROM edges WHERE from_id = ? AND etype = 'derived_from'", (rid,)
        )
    }


def _trip_and_meeting(conn) -> None:
    assert add(conn, entity("e-ilya", "Илья")).ok
    assert add(conn, entity("e-berlin", "Берлин", etype="place")).ok
    raw(conn, "r1", "Хотели встретиться с Ильёй, но сорвалось из-за поездки в Берлин",
        ts_=f"{DAY}T12:00:00Z")  # fmt: skip
    assert add(
        conn,
        fact(
            "f-trip",
            OWNER_ENTITY_ID,
            "travels_to",
            "Berlin",
            fkind=FKind.PLAN,
            value_entity_id="e-berlin",
            text="Denis's trip to Berlin",
            source_raw_id="r1",
            quote_orig="поездки в Берлин",
            valid_from="2026-08-21",
        ),  # fmt: skip
    ).ok
    assert add(
        conn,
        fact(
            "f-meet",
            OWNER_ENTITY_ID,
            "other",
            None,
            fkind=FKind.EVENT,
            value_entity_id="e-ilya",
            text="Meeting with Ilya fell through",
            source_raw_id="r1",
            quote_orig="Хотели встретиться с Ильёй, но сорвалось из-за поездки",
        ),  # fmt: skip
        justifications=[{"member": "f-trip", "provenance": "stated", "group_no": 0}],
    ).ok


def test_episode_has_participants_place_time_outcome_and_cause(conn):
    """§4.3, §15.3: a bounded happening with participants, place, time, outcome and cause
    links; a cause is its own episode, linked, not merged."""
    _trip_and_meeting(conn)
    counts = api.run_nightly(store=conn)
    assert counts["episodes"] == 2 and not counts.get("failed_episodes")
    trip, meet = sorted(_episodes(conn), key=lambda e: e["event_type"])[::-1]
    assert meet["event_type"] == "meeting" and meet["outcome"] == "cancelled"
    assert meet["span_start"] is None  # unknown stays NULL (§4.1): no invented time
    assert "Because of: Denis's trip to Berlin" in meet["summary_en"]
    assert "сорвалось" in meet["summary_en"]  # the user's quote, for cues in the user's words
    assert _links(conn, meet["id"]) == {
        ("participant", "e-ilya"),
        ("cause", "f-trip"),
        ("outcome", "f-meet"),
    }
    assert _derived_from(conn, meet["id"]) == {"f-meet", "f-trip"}
    assert trip["event_type"] == "trip" and trip["place"] == "Berlin"
    assert trip["span_start"] == "2026-08-21" and trip["outcome"] is None  # a plan, no word yet
    assert ("place", "e-berlin") in _links(conn, trip["id"])
    assert all(("participant", OWNER_ENTITY_ID) not in _links(conn, e["id"]) for e in (trip, meet))


def test_c3_finds_the_built_episode_from_a_vague_hint(conn):
    """§15.3: the night's episode is what the C3 finder answers with."""
    _trip_and_meeting(conn)
    api.run_nightly(store=conn)
    meet = next(e for e in _episodes(conn) if e["event_type"] == "meeting")
    b = api.recall(_turn("помнишь, мы хотели встретиться, но сорвалось из-за поездки?"), store=conn)
    eps = [i for i in b.items if i.kind == "episode"]
    assert eps and eps[0].id == meet["id"] and eps[0].ask is None


def test_plan_then_outcome_supersedes_the_episode_and_reruns_are_quiet(conn):
    """§6.4 delta ops: a later word on the same plan supersedes the episode (history kept);
    a night with nothing new writes nothing."""
    assert add(conn, entity("e-ilya", "Илья")).ok
    raw(conn, "r1", "В пятницу ужин с Ильёй", ts_="2026-10-01T12:00:00Z")
    assert add(
        conn,
        fact(
            "f-plan",
            OWNER_ENTITY_ID,
            "other",
            None,
            fkind=FKind.PLAN,
            value_entity_id="e-ilya",
            text="Dinner with Ilya on Friday",
            source_raw_id="r1",
        ),  # fmt: skip
    ).ok
    api.run_nightly(store=conn)
    (first,) = _episodes(conn)
    assert first["event_type"] == "dinner" and first["outcome"] is None

    raw(conn, "r2", "Ужин с Ильёй отменили", ts_="2026-10-03T12:00:00Z")
    assert add(
        conn,
        fact(
            "f-off",
            OWNER_ENTITY_ID,
            "other",
            None,
            fkind=FKind.EVENT,
            value_entity_id="e-ilya",
            text="Dinner with Ilya cancelled",
            source_raw_id="r2",
        ),  # fmt: skip
    ).ok
    api.run_nightly(store=conn)
    (second,) = _episodes(conn)
    assert second["outcome"] == "cancelled" and second["id"] != first["id"]
    assert _derived_from(conn, second["id"]) == {"f-plan", "f-off"}
    assert ("outcome", "f-off") in _links(conn, second["id"])
    assert [e["id"] for e in _episodes(conn, "superseded")] == [first["id"]]
    edge = conn.execute(
        "SELECT 1 FROM edges WHERE from_id = ? AND to_id = ? AND etype = 'supersedes'",
        (second["id"], first["id"]),
    ).fetchone()
    assert edge is not None

    before = conn.execute("SELECT count(*) FROM records").fetchone()[0]
    again = api.run_nightly(store=conn)
    assert again["episodes"] == 0 and again["summaries"] == 0
    assert conn.execute("SELECT count(*) FROM records").fetchone()[0] == before


def test_episode_loses_support_when_its_facts_leave_use(conn):
    """§4.2: an episode whose every seed was retracted goes to needs_review (excluded)."""
    assert add(conn, entity("e-mark", "Марк")).ok
    assert add(
        conn,
        fact(
            "f-call",
            OWNER_ENTITY_ID,
            "other",
            None,
            fkind=FKind.EVENT,
            value_entity_id="e-mark",
            text="Call with Mark went well",
        ),  # fmt: skip
    ).ok
    api.run_nightly(store=conn)
    (ep,) = _episodes(conn)
    assert ep["event_type"] == "call" and ep["outcome"] == "done"
    assert commit(conn, op(OpKind.RETRACT, target="f-call"), tier=AuthorTier.USER).ok
    api.run_nightly(store=conn)
    assert [e["id"] for e in _episodes(conn, "needs_review")] == [ep["id"]]


def test_summary_inherits_labels_and_keeps_lineage(conn):
    """G1, §5.3: a health fact stays labelled inside the daily summary; the read set (and
    `read_deps`) names every record and raw chunk it used, never a derived item."""
    raw(conn, "r-h", "У меня мигрень с утра", ts_=f"{DAY}T09:00:00Z", labels=int(Label.HEALTH))
    assert add(
        conn,
        fact(
            "f-h",
            OWNER_ENTITY_ID,
            "has_condition",
            "migraine",
            fkind=FKind.STATE,
            text="Denis has a migraine",
            source_raw_id="r-h",
            quote_orig="У меня мигрень",
        ),  # fmt: skip
    ).ok
    raw(conn, "r-x", "Купил новый ноутбук", ts_="2026-08-21T09:00:00Z")
    assert add(
        conn,
        fact(
            "f-x",
            OWNER_ENTITY_ID,
            "owns",
            "laptop",
            fkind=FKind.EVENT,
            text="Denis bought a new laptop",
            source_raw_id="r-x",
        ),  # fmt: skip
    ).ok
    api.run_nightly(store=conn)
    row = conn.execute(
        "SELECT content, labels, read_set, valid FROM derived WHERE id = ?",
        (episodes.summary_id(DAY),),
    ).fetchone()
    assert row["valid"] == 1 and Label(row["labels"]) & Label.HEALTH
    assert "Denis has a migraine («У меня мигрень»)" in row["content"]
    assert "laptop" not in row["content"]  # another day, its own summary
    refs = json.loads(row["read_set"])
    assert {r[0] for r in refs} == {"f-h", "r-h"} and all(
        len(r) < 3 or r[2] != "derived" for r in refs
    )
    deps = {
        r[0]
        for r in conn.execute(
            "SELECT item_id FROM read_deps WHERE holder_kind = 'derived' AND holder_id = ?",
            (episodes.summary_id(DAY),),
        )
    }
    assert deps == {"f-h", "r-h"}
    nxt = conn.execute(
        "SELECT labels FROM derived WHERE id = ?", (episodes.summary_id("2026-08-21"),)
    ).fetchone()
    assert nxt is not None and not Label(nxt["labels"]) & Label.HEALTH


def test_proposals_use_typed_payloads(conn):
    """§6.5: every op the builders propose validates against its payload model."""
    _trip_and_meeting(conn)
    props = [episodes.propose_episodes(conn, since=""), episodes.propose_daily_summary(conn, DAY)]
    ops = [o for p in props for o in p.ops]
    assert {o.op for o in ops} == {OpKind.ADD}
    for o in ops:
        validate_payload(o)
    summary = props[1].ops[0].payload["derived"]
    assert summary["dkind"] == "summary" and summary["dkey"] == DAY
    assert conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 0  # pure
