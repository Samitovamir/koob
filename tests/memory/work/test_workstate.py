"""§14 work memory (work): decisions with reasons, rejected alternatives and revision
conditions; the work state as JSON; resume re-validates premises, changes first."""

from __future__ import annotations

import json

import pytest

from koob.memory import api, workstate
from koob.memory.model import (
    Alternative,
    AuthorTier,
    Decision,
    Domain,
    Label,
    OpKind,
    ReadRef,
    TurnCtx,
    TurnOrigin,
    WorkStateBody,
)

from .._ledger import add, commit, entity, fact, op, open_seeded, row, status, ts


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    for rec in (
        entity("m-trip", "Trip to Boston", etype="matter"),
        entity("p-sasha", "Sasha"),
    ):
        assert add(c, rec).ok
    yield c
    c.close()


@pytest.fixture
def turn() -> TurnCtx:
    return TurnCtx(
        turn_id="t-1",
        thread_id="th-1",
        session_id="s-1",
        origin=TurnOrigin.OWNER,
        owner_words="что там с поездкой в Бостон",
        message_id="msg-1",
        clock="2026-10-12T09:00:00.000000Z",
    )


Q = "план поездки в Бостон"


def _fact(c, fid: str, text: str, **kw) -> str:
    assert add(c, fact(fid, "self", "plans", text, text=text, **kw)).ok
    return fid


def _ref(c, rid: str) -> ReadRef:
    return ReadRef(id=rid, version=row(c, rid)["version"])


def _decision(did: str = "d-1", **kw) -> Decision:
    base = dict(
        id=did,
        matter_id="m-trip",
        domain=Domain.TRAVEL,
        chosen_text="Take the Saturday bus",
        alternatives=[Alternative(text="Fly", reasons=["too expensive"])],
        author_tier=AuthorTier.USER,
        recorded_at=ts(),
        origin_quote="поеду автобусом в субботу: дёшево и успеваю",
    )
    base.update(kw)
    return Decision(**base)


def _supersede(c, old: str, new_id: str, text: str) -> None:
    new = fact(new_id, "self", "plans", text, text=text)
    res = commit(c, op(OpKind.SUPERSEDE, target=old, group=old, record=new.model_dump(mode="json")))
    assert res.ok


# ---------------------------------------------------------------- decisions


def test_decision_keeps_reasons_alternatives_and_revision_conditions(conn):
    """§14: reasons as justification groups (AND «дёшево и успевает»), rejected alternatives
    with reasons, a domain, and each revision condition armed as an intention."""
    _fact(conn, "f-cheap", "The bus costs $20")
    _fact(conn, "f-time", "The bus arrives by 6 pm")
    cond = {
        "text": "if the bus is cancelled",
        "trigger": {"kind": "status_change", "subject_id": "self", "relation": "plans"},
    }
    did = workstate.record_decision(
        conn, _decision(revision_conditions=[cond]), reasons=[["f-cheap", "f-time"]]
    )
    assert did == "d-1" and status(conn, did) == "active"
    d = conn.execute("SELECT * FROM decisions WHERE id = ?", (did,)).fetchone()
    assert d["domain"] == "travel" and d["decided_at"]
    assert json.loads(d["alternatives"]) == [{"text": "Fly", "reasons": ["too expensive"]}]
    js = conn.execute(
        "SELECT group_no, member_record_id, provenance FROM justifications"
        " WHERE target_record_id = ? ORDER BY member_record_id",
        (did,),
    ).fetchall()
    assert [tuple(j) for j in js] == [(1, "f-cheap", "stated"), (1, "f-time", "stated")]
    about = conn.execute(
        "SELECT 1 FROM edges WHERE from_id = ? AND to_id = 'm-trip' AND etype = 'about'", (did,)
    ).fetchone()
    assert about is not None
    (rc,) = json.loads(d["revision_conditions"])
    assert rc["armed"] is True and rc["v"] == workstate.SHAPE_VERSION
    assert rc["text"] == "if the bus is cancelled"
    i = conn.execute(
        "SELECT i.state, i.action, i.trigger FROM intentions i WHERE i.id = ?",
        (rc["intention_id"],),
    ).fetchone()
    assert i["state"] == "armed"
    assert json.loads(i["action"]) == {
        "kind": "revisit_decision",
        "decision_id": did,
        "matter_id": "m-trip",
    }
    assert json.loads(i["trigger"])["kind"] == "status_change"


def test_vague_revision_condition_becomes_a_semantic_trigger(conn):
    workstate.record_decision(
        conn, _decision(revision_conditions=[{"text": "if prices drop a lot"}])
    )
    (rc,) = json.loads(
        conn.execute("SELECT revision_conditions FROM decisions WHERE id = 'd-1'").fetchone()[0]
    )
    trig = json.loads(
        conn.execute(
            "SELECT trigger FROM intentions WHERE id = ?", (rc["intention_id"],)
        ).fetchone()[0]
    )
    assert trig["kind"] == "soft_opportunity" and trig["semantic"] == "if prices drop a lot"


@pytest.mark.parametrize(
    ("kw", "err"),
    [
        ({"matter_id": "nope"}, "unknown matter"),
        ({"matter_id": "p-sasha"}, "not a matter"),
        ({"revision_conditions": [{"text": "x", "trigger": {"kind": "bogus"}}]}, "bad revision"),
        ({"revision_conditions": [{"trigger": {}}]}, "needs its text"),
    ],
)
def test_bad_decision_is_refused_before_any_write(conn, kw, err):
    with pytest.raises(workstate.WorkError, match=err):
        workstate.record_decision(conn, _decision(**kw))
    assert conn.execute("SELECT count(*) FROM decisions").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM intentions").fetchone()[0] == 0


def test_stated_reason_needs_his_tier(conn):
    """§8.1: a `stated` link is the user's; a tool-tier member cannot be one."""
    _fact(conn, "f-web", "A site says buses are late", tier=AuthorTier.TOOL)
    with pytest.raises(workstate.WorkError, match="stated_not_user"):
        workstate.record_decision(conn, _decision(), reasons=[["f-web"]])


def test_revised_decision_supersedes_and_cancels_old_revisits(conn):
    workstate.record_decision(conn, _decision(revision_conditions=[{"text": "if it rains"}]))
    old_iid = json.loads(
        conn.execute("SELECT revision_conditions FROM decisions WHERE id = 'd-1'").fetchone()[0]
    )[0]["intention_id"]
    workstate.record_decision(
        conn, _decision("d-2", chosen_text="Take the Friday train"), replaces="d-1"
    )
    assert status(conn, "d-1") == "superseded" and status(conn, "d-2") == "active"
    st = conn.execute("SELECT state FROM intentions WHERE id = ?", (old_iid,)).fetchone()[0]
    assert st == "cancelled"
    with pytest.raises(workstate.WorkError, match="not a decision"):
        workstate.record_decision(conn, _decision("d-3"), replaces="f-missing")


# ---------------------------------------------------------------- work state


def _state(**kw) -> WorkStateBody:
    base = dict(
        goal="Get to Boston for the interview",
        direction="Cheapest option that arrives the day before",
        rejected=[Alternative(text="Fly", reasons=["$400"])],
        dead_ends=[Alternative(text="Rideshare", reasons=["nobody goes that day"])],
        open_questions=["Does the hotel allow late check-in?"],
        materials=["~/Documents/40_Personal/Travel/boston.pdf"],
        next_step="Buy the bus ticket",
    )
    base.update(kw)
    return WorkStateBody(**base)


def test_work_state_is_json_updated_in_place(conn):
    sid = workstate.save_state(
        conn, "m-trip", _state(), author_tier=AuthorTier.USER, origin_quote=Q
    )
    v1 = row(conn, sid)["version"]
    body = json.loads(
        conn.execute("SELECT state FROM work_states WHERE id = ?", (sid,)).fetchone()[0]
    )
    assert body["dead_ends"] == [{"text": "Rideshare", "reasons": ["nobody goes that day"]}]
    assert body["next_step"] == "Buy the bus ticket"
    sid2 = workstate.save_state(
        conn, "m-trip", _state(next_step="Pack"), author_tier=AuthorTier.USER, origin_quote=Q
    )
    assert sid2 == sid and row(conn, sid)["version"] > v1
    assert conn.execute("SELECT count(*) FROM work_states").fetchone()[0] == 1
    _, state = workstate.resume(conn, "m-trip")
    assert state.next_step == "Pack" and state.rejected[0].reasons == ["$400"]


_PROJECT = {"nodes": [{"id": "n1", "artefact": "sha256:ab"}]}


def _newer_writer(conn) -> str:
    """A state as a newer writer (the hands) would leave it: an unknown key, a broken field."""
    sid = workstate.save_state(
        conn, "m-trip", _state(), author_tier=AuthorTier.USER, origin_quote=Q
    )
    raw = json.loads(
        conn.execute("SELECT state FROM work_states WHERE id = ?", (sid,)).fetchone()[0]
    )
    raw["project"] = _PROJECT
    raw["open_questions"] = "not a list"
    conn.execute("UPDATE work_states SET state = ? WHERE id = ?", (json.dumps(raw), sid))
    return sid


def test_newer_keys_never_break_a_read(conn):
    """HANDS §3: the hands keep projects in this same work state; an older reader ignores
    what it does not know, and a broken field costs only itself."""
    sid = _newer_writer(conn)
    body, extra = workstate.load_state(
        conn.execute("SELECT state FROM work_states WHERE id = ?", (sid,)).fetchone()[0]
    )
    assert body.goal == "Get to Boston for the interview" and body.open_questions == []
    assert extra == {"project": _PROJECT}
    assert workstate.load_state("not json")[0] == WorkStateBody()
    _, state = workstate.resume(conn, "m-trip")
    assert state.goal == "Get to Boston for the interview"


@pytest.mark.xfail(
    strict=True,
    reason="contract: model.WorkStateBody is extra='forbid', so ledger.get_record refuses a "
    "state with a newer key; needs extra='allow' (model.py)",
)
def test_newer_keys_survive_a_resave(conn):
    sid = _newer_writer(conn)
    workstate.save_state(
        conn, "m-trip", _state(next_step="Pack"), author_tier=AuthorTier.USER, origin_quote=Q
    )
    after = json.loads(
        conn.execute("SELECT state FROM work_states WHERE id = ?", (sid,)).fetchone()[0]
    )
    assert after["project"] == _PROJECT and after["next_step"] == "Pack"


def test_save_on_a_moved_premise_is_refused(conn):
    _fact(conn, "f-day", "Interview is on Monday")
    old = _ref(conn, "f-day")
    _supersede(conn, "f-day", "f-day2", "Interview is on Tuesday")
    with pytest.raises(workstate.StalePremises) as e:
        workstate.save_state(
            conn, "m-trip", _state(premises=[old]), author_tier=AuthorTier.USER, origin_quote=Q
        )
    assert [r.id for r in e.value.refs] == ["f-day"]
    assert conn.execute("SELECT count(*) FROM work_states").fetchone()[0] == 0


def test_state_labels_come_from_the_writer_not_its_premises(conn):
    _fact(conn, "f-doc", "Doctor visit in Boston on Monday", labels=Label.HEALTH)
    premises = [_ref(conn, "f-doc")]
    sid = workstate.save_state(
        conn, "m-trip", _state(premises=premises), author_tier=AuthorTier.USER, origin_quote=Q
    )
    assert Label(row(conn, sid)["labels"]) == Label.NONE
    workstate.save_state(
        conn,
        "m-trip",
        _state(premises=premises),
        author_tier=AuthorTier.USER,
        origin_quote=Q,
        labels=Label.MONEY,
    )
    assert Label(row(conn, sid)["labels"]) == Label.MONEY


def test_save_needs_a_matter(conn):
    with pytest.raises(workstate.WorkError, match="unknown matter"):
        workstate.save_state(conn, "m-none", _state())


# ---------------------------------------------------------------- resume


def test_resume_tells_changes_first_then_repins(conn):
    """§14 `[eval group work.resume]`: a premise that moved comes first, with old and new;
    the premise follows the new value, so the same change is told once."""
    _fact(conn, "f-day", "Interview is on Monday")
    _fact(conn, "f-hotel", "Hotel booked near Back Bay")
    sid = workstate.save_state(
        conn,
        "m-trip",
        _state(premises=[_ref(conn, "f-day"), _ref(conn, "f-hotel")]),
        author_tier=AuthorTier.USER,
        origin_quote=Q,
    )
    v_before = row(conn, sid)["version"]
    _supersede(conn, "f-day", "f-day2", "Interview is on Tuesday")
    changes, state = workstate.resume(conn, "m-trip")
    assert changes == ['[m:f-day2] "Interview is on Monday" is now "Interview is on Tuesday"']
    assert [p.id for p in state.premises] == ["f-day2", "f-hotel"]
    assert row(conn, sid)["version"] > v_before  # read sets on the old state go stale
    again, _ = workstate.resume(conn, "m-trip")
    assert again == []
    assert conn.execute("SELECT last_resumed_at FROM work_states").fetchone()[0]


def test_resume_never_shows_forgotten_content(conn):
    _fact(conn, "f-secret", "Staying at Dima's flat")
    workstate.save_state(
        conn,
        "m-trip",
        _state(premises=[_ref(conn, "f-secret")]),
        author_tier=AuthorTier.USER,
        origin_quote=Q,
    )
    conn.execute("UPDATE records SET status = 'forgotten' WHERE id = 'f-secret'")
    changes, state = workstate.resume(conn, "m-trip")
    assert len(changes) == 1 and "forgotten" in changes[0]
    assert "Dima" not in changes[0] and state.premises == []


def test_resume_reports_retracted_premise(conn):
    _fact(conn, "f-bus", "The bus has Wi-Fi")
    workstate.save_state(
        conn,
        "m-trip",
        _state(premises=[_ref(conn, "f-bus")]),
        author_tier=AuthorTier.USER,
        origin_quote=Q,
    )
    assert commit(conn, op(OpKind.RETRACT, target="f-bus", group="f-bus")).ok
    changes, state = workstate.resume(conn, "m-trip")
    assert changes == ['[m:f-bus] "The bus has Wi-Fi" turned out untrue']
    assert state.premises == []


def test_resume_reports_broken_decision_and_fired_condition(conn):
    _fact(conn, "f-cheap", "The bus costs $20")
    workstate.save_state(conn, "m-trip", _state(), author_tier=AuthorTier.USER, origin_quote=Q)
    workstate.record_decision(
        conn,
        _decision(revision_conditions=[{"text": "if the bus is cancelled"}]),
        reasons=[["f-cheap"]],
    )
    assert workstate.resume(conn, "m-trip")[0] == []
    iid = json.loads(
        conn.execute("SELECT revision_conditions FROM decisions WHERE id = 'd-1'").fetchone()[0]
    )[0]["intention_id"]
    later = "2099-01-01T00:00:00.000000Z"
    conn.execute("UPDATE intentions SET state = 'fired', fired_at = ? WHERE id = ?", (later, iid))
    assert commit(conn, op(OpKind.FLAG, target="d-1", group="d-1", status="needs_review")).ok
    conn.execute("UPDATE records SET updated_at = ? WHERE id = 'd-1'", (later,))
    changes, _ = workstate.resume(conn, "m-trip")
    assert changes == [
        '[m:d-1] decision "Take the Saturday bus" lost its reasons; revisit it',
        '[m:d-1] revision condition met for "Take the Saturday bus": "if the bus is cancelled"',
    ]


def test_api_resume_view(conn, turn):
    """The facade: changes first, then the state with its decisions and their reasons."""
    _fact(conn, "f-cheap", "The bus costs $20")
    _fact(conn, "f-day", "Interview is on Monday")
    workstate.save_state(
        conn,
        "m-trip",
        _state(premises=[_ref(conn, "f-day")]),
        author_tier=AuthorTier.USER,
        origin_quote=Q,
    )
    workstate.record_decision(conn, _decision(), reasons=[["f-cheap"]])
    _supersede(conn, "f-day", "f-day2", "Interview is on Tuesday")
    v = api.resume("m-trip", turn=turn, store=conn)
    assert v.matter_id == "m-trip" and len(v.changed) == 1 and "Tuesday" in v.changed[0]
    assert v.state["v"] == workstate.SHAPE_VERSION and v.state["goal"].startswith("Get to")
    (d,) = v.state["decisions"]
    assert d["chosen"] == "Take the Saturday bus" and d["domain"] == "travel"
    assert d["reasons"] == [
        [{"id": "f-cheap", "text": "The bus costs $20", "provenance": "stated"}]
    ]
    assert d["rejected"] == [{"text": "Fly", "reasons": ["too expensive"]}]
    assert api.resume("m-unknown", turn=turn, store=conn).state == {}


def test_api_resume_fences_each_item_and_keeps_a_withheld_change(conn, turn):
    """§5.5: a sensitive premise's change is left out on a turn it does not apply to, the
    plan still shows, and the change is not consumed: a neutral turn tells it later."""
    _fact(conn, "f-doc", "Doctor visit in Boston on Monday", labels=Label.HEALTH)
    workstate.save_state(
        conn,
        "m-trip",
        _state(premises=[_ref(conn, "f-doc")]),
        author_tier=AuthorTier.USER,
        origin_quote=Q,
    )
    _supersede(conn, "f-doc", "f-doc2", "Doctor visit in Boston on Tuesday")
    v = api.resume("m-trip", turn=turn, store=conn)  # «что там с поездкой в Бостон»
    assert v.changed == [] and v.state["goal"].startswith("Get to")
    assert [p["id"] for p in v.state["premises"]] == ["f-doc"]
    neutral = turn.model_copy(update={"owner_words": None})
    later = api.resume("m-trip", turn=neutral, store=conn)
    assert len(later.changed) == 1 and "Tuesday" in later.changed[0]
    assert [p["id"] for p in later.state["premises"]] == ["f-doc2"]


def test_api_resume_fences_a_labelled_state_by_use_now(conn, turn):
    """A labelled state follows §5.5 like any item: shown on a turn about it (the policy
    matches «Бостон»), hidden on an unrelated one, shown on a neutral turn."""
    workstate.save_state(
        conn, "m-trip", _state(), author_tier=AuthorTier.USER, origin_quote=Q, labels=Label.HEALTH
    )
    assert api.resume("m-trip", turn=turn, store=conn).state["goal"]
    unrelated = turn.model_copy(update={"owner_words": "купи молоко и хлеб"})
    assert api.resume("m-trip", turn=unrelated, store=conn).state == {}
    neutral = turn.model_copy(update={"owner_words": None})
    assert api.resume("m-trip", turn=neutral, store=conn).state["goal"]
