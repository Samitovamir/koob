"""The SQL inspector reads every record kind, not only `facts`.

Rows are written straight into a migrated memory.db, the way the engine's ledger lays them
out, so `records:` checks on decisions, intentions, criteria, impressions, work states and
justifications are graded on what is stored rather than never matching."""

from __future__ import annotations

import json
from types import SimpleNamespace

from koob.memory import store
from koob.memory.eval.graders import check_records
from koob.memory.eval.schema import RecordCheck
from koob.memory.eval.sut import SqlInspector

T = "2026-10-01T10:00:00"


def _rec(conn, rid: str, kind: str, *, status: str = "active", tier: str = "user", review=0):
    conn.execute(
        "INSERT INTO records (id, kind, version, status, review, author_tier, recorded_at,"
        " created_at, updated_at) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)",
        (rid, kind, status, review, tier, T, T, T),
    )


def _db():
    conn = store.open(":memory:")
    _rec(conn, "e_nina", "entity")
    conn.execute(
        "INSERT INTO entities (id, etype, display_name) VALUES ('e_nina', 'person', 'Nina Petrova')"
    )
    _rec(conn, "d1", "decision", review=1)
    conn.execute(
        "INSERT INTO decisions (id, domain, chosen_text, alternatives) VALUES (?, ?, ?, ?)",
        ("d1", "work", "Use NATS for the job queue", json.dumps([{"text": "Kafka"}])),
    )
    _rec(conn, "d2", "decision")
    conn.execute(
        "INSERT INTO decisions (id, domain, chosen_text) VALUES ('d2', 'tools', 'Try the setup first')"
    )
    _rec(conn, "d3", "decision", status="superseded")
    conn.execute(
        "INSERT INTO decisions (id, domain, chosen_text) VALUES ('d3', 'study', 'Take the evening slot')"
    )
    _rec(conn, "i1", "intention")
    conn.execute(
        "INSERT INTO intentions (id, trigger, action, state) VALUES (?, ?, ?, 'fired')",
        (
            "i1",
            json.dumps({"kind": "date_reached", "ts": "2026-10-03T09:00", "goal": "train trip"}),
            json.dumps({"text": "buy tickets"}),
        ),
    )
    _rec(conn, "c1", "criterion", tier="system")
    conn.execute(
        "INSERT INTO criteria (id, text_en, evidence, status) VALUES (?, ?, ?, 'hypothesis')",
        ("c1", "He tries a tool hands-on before reading its docs", json.dumps(["d1", "d2", "d3"])),
    )
    _rec(conn, "im1", "impression")
    conn.execute(
        "INSERT INTO impressions (id, about_entity_id, text_en, quote_orig, observed_at)"
        " VALUES ('im1', 'e_nina', 'Promises a lot', 'много обещает', ?)",
        (T,),
    )
    _rec(conn, "w1", "work_state")
    conn.execute(
        "INSERT INTO work_states (id, matter_id, state) VALUES ('w1', 'm1', ?)",
        (
            json.dumps(
                {
                    "goal": "Postgres 17 upgrade",
                    "dead_ends": ["pglogical"],
                    "next_step": "pg_upgrade",
                }
            ),
        ),
    )
    for n, (target, member, pv) in enumerate(
        [
            ("d1", "raw1", "stated"),
            ("d1", "raw2", "stated"),
            ("c1", "d1", "inferred"),
            ("c1", "d2", "inferred"),
        ]
    ):
        conn.execute(
            "INSERT INTO justifications (id, target_record_id, group_no, member_record_id, provenance,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (f"j{n}", target, n, member, pv, T),
        )
    return SqlInspector(conn, {"nina": ["Nina Petrova", "Нина"]})


def _grade(insp, *specs):
    obs = SimpleNamespace(inspector=insp, unavailable={})
    checks = check_records(obs, [RecordCheck(**s) for s in specs])
    return {c.name: (c.status, c.reason) for c in checks}


def _ok(result):
    bad = {k: v for k, v in result.items() if v[0] != "pass"}
    assert not bad, bad


def test_decision_records_match_on_their_text_domain_and_review():
    insp = _db()
    _ok(
        _grade(
            insp,
            {
                "where": {"kind": "decision", "value": ["NATS"]},
                "expect": {"domain": "work", "review": True},
            },
        )
    )
    # a superseded decision is not usable unless the check asks for its status
    r = _grade(insp, {"where": {"kind": "decision", "value": ["evening slot"]}, "count": 0})
    _ok(r)
    _ok(_grade(insp, {"where": {"kind": "decision", "value": ["evening"], "status": "superseded"}}))
    # a rejected alternative is not the decision's text
    _ok(_grade(insp, {"where": {"kind": "decision", "value": ["Kafka"]}, "count": 0}))


def test_intention_state_list_names_alternatives_and_text_reaches_the_payload():
    insp = _db()
    _ok(
        _grade(
            insp,
            {
                "where": {"kind": "intention", "value": ["tickets", "билет"]},
                "expect": {"state": ["fired", "done"]},
            },
        )
    )
    r = _grade(
        insp, {"where": {"kind": "intention", "value": ["tickets"]}, "expect": {"state": ["armed"]}}
    )
    assert any(v[0] == "fail" for v in r.values()), r
    # a trigger's kind is not text; its goal is
    _ok(_grade(insp, {"where": {"kind": "intention", "value": ["date_reached"]}, "count": 0}))
    _ok(_grade(insp, {"where": {"kind": "intention", "value": ["train trip"]}}))


def test_criterion_reads_evidence_domains_and_provenance():
    insp = _db()
    _ok(
        _grade(
            insp,
            {
                "where": {"kind": "criterion", "value": ["hands-on", "попробовать"]},
                "expect": {
                    "provenance": "inferred",
                    "evidence_count_min": 2,
                    "domains": ["tools", "work"],
                },
            },
        )
    )
    r = _grade(
        insp,
        {"where": {"kind": "criterion", "value": ["hands-on"]}, "expect": {"provenance": "stated"}},
    )
    assert any(v[0] == "fail" for v in r.values()), r


def test_impression_work_state_and_justification_kinds():
    insp = _db()
    _ok(
        _grade(
            insp,
            {
                "where": {"kind": "impression", "about": "nina"},
                "expect": {"quote_orig": "много обещает"},
            },
        )
    )
    _ok(
        _grade(
            insp, {"where": {"kind": "impression", "about": "nina", "value": ["burn"]}, "count": 0}
        )
    )
    _ok(
        _grade(
            insp,
            {
                "where": {"kind": "work_state", "value": ["Postgres 17"]},
                "expect": {"state.dead_ends": ["pglogical"], "state.next_step": ["pg_upgrade"]},
            },
        )
    )
    _ok(
        _grade(
            insp,
            {
                "where": {"kind": "justification", "target": ["NATS"]},
                "expect": {"provenance": "stated"},
                "count": 2,
            },
        )
    )
    _ok(_grade(insp, {"where": {"kind": "justification", "target": ["Redis"]}, "count": 0}))


def test_fact_applicability_is_read_as_json():
    insp = _db()
    conn = insp.conn
    _rec(conn, "f1", "preference")
    conn.execute(
        "INSERT INTO facts (id, subject_id, relation, value_text, text_en, fkind, applicability) "
        "SELECT 'f1', 'self', relation, 'voice messages', 'prefers voice messages', 'state', ? "
        "FROM relations LIMIT 1",
        (json.dumps({"for_whom": "dima"}),),
    )
    rows = [r for r in insp.records({"kind": "preference"}) if r["id"] == "f1"]
    assert rows and rows[0]["applicability"] == {"for_whom": "dima"}
