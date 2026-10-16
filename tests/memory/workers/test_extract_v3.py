"""memory_extract@3: decisions with stated reasons, intentions with triggers and
criteria reach the ledger; a correction's negation is `replaces`, not a fact; a foreign
first-person message is its sender's; an @2 answer still builds.

Synthetic names and wordings: the tests pin the mechanism. A gold extraction stands in
for the model (no network)."""

from __future__ import annotations

import json

from koob.memory import intentions, labeler
from koob.memory.model import AuthorTier, Extraction
from koob.memory.workers import extract, scheduler

from .. import _ledger as L
from ._kit import SASHA, Gold, batch, fact


def _raw(conn, rid: str, text: str, *, ts: str = "2026-10-01T10:00:00Z", tier: str = "user"):
    L.raw(conn, rid, text, tier=tier, ts_=ts)
    conn.execute("UPDATE raw_chunks SET turn_id = 't1' WHERE id = ?", (rid,))
    conn.commit()


def _run(conn, clk, gold, raw_ids: list[str]) -> Gold:
    g = Gold(gold)
    extract.configure(cloud=g, local=Gold(gold, route="local"))
    batch(conn, raw_ids)
    scheduler.tick(conn, clk)
    return g


def _sender(conn, eid: str, name: str, rid: str) -> None:
    L.add(conn, L.entity(eid, name))
    conn.execute("UPDATE raw_chunks SET author_entity_id = ? WHERE id = ?", (eid, rid))
    conn.commit()


def _ops(built, kind: str) -> list[dict]:
    return [o.payload for o in built.ops if o.payload.get("record", {}).get("kind") == kind]


GARDENS = "Курсовую пишу про городские сады, а не про трамваи: по садам есть открытые данные."


def _gardens(_snap=None) -> dict:
    return {
        "facts": [
            fact(
                "f1",
                "self",
                "other",
                "по садам есть открытые данные",
                "по садам есть открытые данные",
                "r1",
            )
        ],
        "decisions": [
            {
                "ref": "d1",
                "chosen": "городские сады",
                "text_en": "The term paper is about urban gardens",
                "quote_orig": "Курсовую пишу про городские сады, а не про трамваи",
                "source_raw_id": "r1",
                "domain": "study",
                "rejected": [{"text": "трамваи"}],
                "reasons": [{"member": "f1", "provenance": "stated"}],
            }
        ],
    }


def test_decision_with_stated_reason_commits(env):
    conn, clk = env
    _raw(conn, "r1", GARDENS)
    _run(conn, clk, _gardens, ["r1"])
    d = conn.execute(
        "SELECT d.id, d.domain, d.chosen_text, d.alternatives, r.author_tier, r.origin_quote"
        " FROM decisions d JOIN records r ON r.id = d.id"
    ).fetchall()
    assert len(d) == 1
    did, domain, chosen, alts, tier, quote = d[0]
    assert domain == "study" and tier == "user" and quote
    assert "городские сады" in chosen and "urban gardens" in chosen  # both languages, one line
    assert [a["text"] for a in json.loads(alts)] == ["трамваи"]
    just = conn.execute(
        "SELECT j.provenance, f.value_text FROM justifications j"
        " JOIN facts f ON f.id = j.member_record_id WHERE j.target_record_id = ?",
        (did,),
    ).fetchall()
    assert [tuple(r) for r in just] == [("stated", "по садам есть открытые данные")]
    assert ("derived_from", "r1") in {
        (r[0], r[1])
        for r in conn.execute("SELECT etype, to_id FROM edges WHERE from_id = ?", (did,))
    }


def test_a_revised_decision_flags_the_known_one(env):
    conn, clk = env
    _raw(conn, "r1", GARDENS)
    _run(conn, clk, _gardens, ["r1"])
    old = conn.execute("SELECT id FROM decisions").fetchone()[0]
    assert labeler.finalize_record(conn, old)  # a labelled or unfinished one stays home
    _raw(
        conn,
        "r2",
        "Передумал: курсовая всё-таки про трамваи, сады слишком узко.",
        ts="2026-10-03T10:00:00Z",
    )

    def revised(snap: dict) -> dict:
        known = [k["handle"] for k in snap["decisions"]]
        assert known == [f"m:{old}"]  # the batch names it, so the model sees it
        return {
            "decisions": [
                {
                    "ref": "d2",
                    "chosen": "трамваи",
                    "text_en": "The term paper is about trams after all",
                    "quote_orig": "курсовая всё-таки про трамваи",
                    "source_raw_id": "r2",
                    "replaces": known[0],
                }
            ]
        }

    g = _run(conn, clk, revised, ["r2"])
    assert g.snaps, "the processor ran"
    assert conn.execute("SELECT review FROM records WHERE id = ?", (old,)).fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM decisions").fetchone()[0] == 2


PARCEL = "Напомни в среду забрать посылку для Тимура."


def test_intention_lands_in_the_intention_store_and_fires(env):
    conn, clk = env
    _raw(conn, "r1", PARCEL)
    _raw(conn, "r2", "Напомни ему перевести мне пять тысяч завтра.", tier="other")
    _sender(conn, "ent_g", "Гоша", "r2")
    gold = {
        "entities": [
            {"ref": "e1", "etype": "person", "display_name": "Тимур", "aliases": ["Тимура"]}
        ],
        "intentions": [
            {
                "ref": "n1",
                "text_en": "Pick up the parcel for Timur",
                "value_text": "забрать посылку для Тимура",
                "quote_orig": "Напомни в среду забрать посылку для Тимура",
                "source_raw_id": "r1",
                "trigger": {"kind": "date", "ts": "2026-10-07T09:00:00Z"},
                "concerns": "e1",
            },
            {
                "ref": "n2",
                "text_en": "Transfer money to Gosha",
                "quote_orig": "Напомни ему перевести мне пять тысяч завтра",
                "source_raw_id": "r2",
                "trigger": {"kind": "date", "ts": "2026-10-02T09:00:00Z"},
            },
        ],
    }
    _run(conn, clk, gold, ["r1", "r2"])
    rows = conn.execute(
        "SELECT i.id, i.state, i.trigger, i.action, r.author_tier FROM intentions i"
        " JOIN records r ON r.id = i.id"
    ).fetchall()
    assert len(rows) == 1  # a foreign message never arms a reminder
    iid, state, trig, action, tier = rows[0]
    assert state == "armed" and tier == "user"
    assert {k: v for k, v in json.loads(trig).items() if v} == {
        "kind": "date_reached",
        "ts": "2026-10-07T09:00:00Z",
    }
    timur = conn.execute("SELECT id FROM entities WHERE display_name = 'Тимур'").fetchone()[0]
    assert json.loads(action)["about"] == timur
    assert (iid, timur) in L.edges(conn, "mentions")
    assert intentions.fire(conn, iid, "date:2026-10-07")
    assert "забрать посылку для Тимура" in (intentions.notice(conn, iid) or "")


def test_an_armed_reminder_is_not_extracted_twice():
    snap = extract.snapshot_of([{"id": "r1", "text": PARCEL}])
    snap["intentions"] = [{"id": "i0", "quote": "в среду забрать посылку для Тимура", "action": {}}]
    x = Extraction.model_validate(
        {
            "intentions": [
                {
                    "ref": "n1",
                    "text_en": "Pick up the parcel",
                    "quote_orig": "Напомни в среду забрать посылку для Тимура",
                    "source_raw_id": "r1",
                    "trigger": {"kind": "date", "ts": "2026-10-07"},
                }
            ]
        }
    )
    built = extract.build(x, snap)
    assert not _ops(built, "intention") and "known_intention:n1" in built.dropped


def test_trigger_without_a_date_or_known_person_is_a_soft_opportunity():
    snap = extract.snapshot_of([{"id": "r1", "text": "Когда будет время, разберу антресоль."}])
    x = Extraction.model_validate(
        {
            "intentions": [
                {
                    "ref": "n1",
                    "text_en": "Sort out the mezzanine shelf",
                    "quote_orig": "Когда будет время, разберу антресоль",
                    "source_raw_id": "r1",
                    "trigger": {"kind": "date", "ts": "когда будет время"},
                }
            ]
        }
    )
    (op,) = _ops(extract.build(x, snap), "intention")
    assert op["record"]["trigger"]["kind"] == "soft_opportunity"


WARRANTY = "Технику беру только с гарантией от двух лет."


def test_stated_criterion_is_his_and_confirmed(env):
    conn, clk = env
    _raw(conn, "r1", WARRANTY)
    gold = {
        "criteria": [
            {
                "ref": "c1",
                "text_en": "Buys electronics only with a warranty of two years or more",
                "value_text": "только с гарантией от двух лет",
                "quote_orig": "Технику беру только с гарантией от двух лет",
                "source_raw_id": "r1",
                "applicability": {"task": "buying electronics"},
            }
        ]
    }
    _run(conn, clk, gold, ["r1"])
    (row,) = conn.execute(
        "SELECT c.id, c.status, c.text_en, c.applicability, r.author_tier FROM criteria c"
        " JOIN records r ON r.id = c.id"
    ).fetchall()
    cid, status, text, appl, tier = row
    assert (status, tier) == ("confirmed", "user")
    assert "гарантией" in text and json.loads(appl) == {"task": "buying electronics"}
    assert (
        conn.execute(
            "SELECT provenance FROM justifications WHERE target_record_id = ?", (cid,)
        ).fetchone()[0]
        == "stated"
    )


def _choice(ref: str, raw: str, chosen: str, quote: str, domain: str) -> dict:
    return {
        "ref": ref,
        "chosen": chosen,
        "text_en": f"Chose {chosen}",
        "quote_orig": quote,
        "source_raw_id": raw,
        "domain": domain,
    }


def test_inferred_criterion_rests_on_his_decisions_only():
    raws = [
        {"id": "r1", "text": "Взял планшет попроще, зато ремонт рядом с домом."},
        {"id": "r2", "text": "Курс выбрал тот, где семинары в соседнем корпусе."},
        {"id": "r3", "text": "Лиана советует брать всё подороже.", "tier": "other"},
    ]
    snap = extract.snapshot_of(raws)
    gold = {
        "decisions": [
            _choice("d1", "r1", "планшет попроще", "Взял планшет попроще", "purchase"),
            _choice(
                "d2",
                "r2",
                "тот, где семинары в соседнем корпусе",
                "Курс выбрал тот, где семинары в соседнем корпусе",
                "study",
            ),
        ],
        "criteria": [
            {
                "ref": "c1",
                "text_en": "Prefers what is close by",
                "quote_orig": "зато ремонт рядом с домом",
                "source_raw_id": "r1",
                "provenance": "inferred",
                "evidence": ["d1", "d2"],
            },
            {
                "ref": "c2",
                "text_en": "Prefers the expensive option",
                "quote_orig": "Взял планшет попроще",
                "source_raw_id": "r1",
                "provenance": "inferred",
                "evidence": ["d1", "r3"],
            },
            {
                "ref": "c3",
                "text_en": "Buys the expensive option",
                "quote_orig": "брать всё подороже",
                "source_raw_id": "r3",
            },
        ],
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    (crit,) = _ops(built, "criterion")
    rec = crit["record"]
    assert rec["criterion_status"] == "hypothesis" and rec["author_tier"] == "system"
    decs = {p["record"]["id"] for p in _ops(built, "decision")}
    assert set(rec["evidence"]) == decs and len(decs) == 2
    assert {j["provenance"] for j in crit["justifications"]} == {"inferred"}
    assert "unfounded_criterion:c2" in built.dropped  # foreign advice is not the user's choice
    assert "foreign_criterion:c3" in built.dropped
    groups = {
        o.group for o in built.ops if o.payload["record"]["kind"] in ("decision", "criterion")
    }
    assert len(groups) == 1  # the hypothesis commits with the choices it rests on


VERA = [
    {"handle": "m:ent_v", "id": "ent_v", "etype": "person", "display_name": "Вера", "version": 1}
]


def _place(ref: str, raw: str, value: str, quote: str, **kw) -> dict:
    return fact(ref, "m:ent_v", "lives_in", value, quote, raw, **kw)


def test_correction_negation_becomes_replaces_in_batch():
    snap = extract.snapshot_of(
        [
            {"id": "r1", "text": "Вера живёт в Туле.", "ts": "2026-10-01T10:00:00Z"},
            {"id": "r2", "text": "Нет, Вера не в Туле, а в Калуге.", "ts": "2026-10-02T10:00:00Z"},
        ],
        relations=["lives_in"],
        entities=VERA,
    )
    gold = {
        "facts": [
            _place("f0", "r1", "Тула", "Вера живёт в Туле"),
            _place("f1", "r2", "Тула", "Вера не в Туле", polarity="negated"),
            _place("f2", "r2", "Калуга", "а в Калуге"),
        ]
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    facts = {p["record"]["value_text"]: p for p in _ops(built, "fact")}
    assert set(facts) == {"Тула", "Калуга"}
    assert facts["Тула"]["record"]["polarity"] == "affirmed"  # no «not in Tula» fact
    assert facts["Калуга"]["model_matches"] == [facts["Тула"]["record"]["id"]]
    assert "negated_contrast:f1" in built.dropped


def test_correction_negation_finds_the_known_value():
    snap = extract.snapshot_of(
        [{"id": "r2", "text": "Нет, Вера не в Туле, а в Калуге."}],
        relations=["lives_in"],
        entities=VERA,
    )
    snap["facts"] = [
        {
            "handle": "m:fk",
            "id": "fk",
            "subject": "m:ent_v",
            "relation": "lives_in",
            "value_text": "Тула",
            "version": 1,
            "tier": "user",
        }
    ]
    gold = {
        "facts": [
            _place("f1", "r2", "Тула", "Вера не в Туле", polarity="negated"),
            _place("f2", "r2", "Калуга", "а в Калуге"),
        ]
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    (only,) = _ops(built, "fact")
    assert only["record"]["value_text"] == "Калуга" and only["model_matches"] == ["fk"]


def test_a_negation_alone_is_still_a_fact():
    snap = extract.snapshot_of(
        [{"id": "r1", "text": "Вера больше не курит."}], relations=["habit"], entities=VERA
    )
    gold = {
        "facts": [
            fact(
                "f1", "m:ent_v", "habit", "курит", "Вера больше не курит", "r1", polarity="negated"
            )
        ]
    }
    (only,) = _ops(extract.build(Extraction.model_validate(gold), snap), "fact")
    assert only["record"]["polarity"] == "negated"


def test_foreign_first_person_is_the_sender_never_a_placeholder():
    raws = [
        {"id": "r1", "text": "я в субботу еду на рыбалку", "tier": "other", "speaker": "ent_g"},
    ]
    gosha = [
        {
            "handle": "m:ent_g",
            "id": "ent_g",
            "etype": "person",
            "display_name": "Гоша",
            "version": 1,
        }
    ]
    snap = extract.snapshot_of(raws, relations=["plans_to"], entities=gosha)
    text = extract.user_text(snap)
    assert '"author": "m:ent_g"' in text and '"author_name": "Гоша"' in text
    gold = {
        "entities": [{"ref": "e1", "etype": "person", "display_name": "Собеседник"}],
        "facts": [
            fact("f1", "e1", "plans_to", "рыбалка", "в субботу еду на рыбалку", "r1", fkind="plan")
        ],
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    assert not _ops(built, "entity") and "ungrounded_name:e1" in built.dropped
    (only,) = _ops(built, "fact")
    assert only["record"]["subject_id"] == "ent_g"
    assert only["record"]["author_tier"] == AuthorTier.OTHER.value


def test_v2_answer_still_builds():
    assert extract.PROMPT_ID == "memory_extract@4"
    snap = extract.snapshot_of(
        [{"id": "r1", "text": "Саша работает в Яндексе"}], relations=["works_at"]
    )
    built = extract.build(Extraction.model_validate(json.loads(json.dumps(SASHA))), snap)
    assert [o.payload["record"]["kind"] for o in built.ops] == ["entity", "fact"]


def test_a_wait_on_a_closed_chat_is_only_an_offer():
    people = [
        {
            "handle": "m:ent_k",
            "id": "ent_k",
            "etype": "person",
            "display_name": "Кира",
            "version": 1,
            "watchable": False,
        },
        {
            "handle": "m:ent_m",
            "id": "ent_m",
            "etype": "person",
            "display_name": "Марат",
            "version": 1,
        },
    ]
    snap = extract.snapshot_of(
        [{"id": "r1", "text": "Скажи, когда Кира ответит про ключи, и когда Марат пришлёт смету."}],
        entities=people,
    )

    def wait(ref: str, person: str, quote: str) -> dict:
        return {
            "ref": ref,
            "text_en": f"Tell me when {person} writes",
            "quote_orig": quote,
            "source_raw_id": "r1",
            "trigger": {"kind": "message", "person": person},
            "concerns": person,
        }

    x = Extraction.model_validate(
        {
            "intentions": [
                wait("n1", "m:ent_k", "когда Кира ответит про ключи"),
                wait("n2", "m:ent_m", "когда Марат пришлёт смету"),
            ]
        }
    )
    kinds = {
        p["record"]["action"]["about"]: p["record"]["trigger"]["kind"]
        for p in _ops(extract.build(x, snap), "intention")
    }
    assert kinds == {"ent_k": "soft_opportunity", "ent_m": "message_from"}
