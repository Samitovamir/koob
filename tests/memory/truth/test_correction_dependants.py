"""What stops a correction from reaching its dependants. Several of the user's records holding
the retired value all retire (and their dependants go to review) instead of a «which one?»;
a possessive name form («Верин») names the stored person, so no duplicate is minted; a
decision and its reason in one message («X, так что Y», "X, so Y") are linked in build().
Fictional people, own wordings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from koob.memory import api, entities, ledger
from koob.memory.model import AuthorTier, Decision, Domain
from koob.memory.workers import extract

from ..inspect._world import TS, add, fact, person, raw, status, turn

VERA, ZOYA = "ent-vera", "ent-zoya"
LATER = "2026-10-14T09:00:00.000000Z"


@pytest.fixture
def db(mem):
    ledger.seed_relations(mem)
    ledger.seed_owner(mem)
    return mem


def _just(*members: str) -> list[dict]:
    return [{"group_no": 0, "member": m, "provenance": "stated"} for m in members]


def _decision(did: str, text: str, quote: str, at: str = TS) -> Decision:
    return Decision(
        id=did, author_tier=AuthorTier.USER, recorded_at=at, labels_final=True,
        domain=Domain.SOCIAL, chosen_text=text, decided_at=at, origin_quote=quote,
    )  # fmt: skip


def _asthma(conn) -> None:
    """Two messages each state Вера's asthma; a hotel choice rests on one copy, her «no pool»
    habit on the other."""
    add(conn, person(VERA, "Вера"))
    add(conn, fact("f-asthma-1", VERA, "has_condition", "астма", "Vera has asthma.",
                   "у Веры астма"))  # fmt: skip
    add(conn, fact("f-asthma-2", VERA, "has_condition", "астма", "Vera has asthma.",
                   "из-за астмы", recorded=LATER))  # fmt: skip
    add(conn, fact("f-pool", VERA, "other", "бассейн", "Vera avoids the pool.",
                   "Вера не ходит в бассейн"),
        justifications=_just("f-asthma-1"))  # fmt: skip
    add(conn, _decision("d-hotel", "They book the room without carpets at «Берёзка».",
                        "берём номер без ковров в «Берёзке» — из-за астмы", at=LATER),
        justifications=_just("f-asthma-2"))  # fmt: skip


def test_every_copy_of_the_retired_value_retires_and_both_dependants_go_to_review(db):
    _asthma(db)
    words = "Исправь: у Веры не астма, а аллергия на пыль."
    res = api.fix(words, words, turn=turn(words, n=2), because_quote=words, store=db)
    assert res.outcome == "committed", res
    assert status(db, "f-asthma-1") == "superseded"
    assert status(db, "f-asthma-2") == "superseded"
    assert status(db, "d-hotel") == "needs_review"
    assert status(db, "f-pool") == "needs_review"
    assert res.dependents_to_review == 2
    current = db.execute(
        "SELECT f.value_text FROM facts f JOIN records r ON r.id = f.id WHERE f.subject_id = ?"
        " AND f.relation = 'has_condition' AND r.status = 'active'",
        (VERA,),
    ).fetchall()
    assert [r[0] for r in current] and all("астм" not in r[0] for r in current)


def test_a_copy_already_merged_into_the_claim_carries_its_dependants_to_review(db):
    from koob.memory import store
    from koob.memory.model import OpKind, Proposal, ProposalOp

    _asthma(db)
    op = ProposalOp(op=OpKind.MERGE, target="f-asthma-1", group=VERA,
                    payload={"drop": "f-asthma-2"})  # fmt: skip
    assert store.commit(db, Proposal(ops=[op], author_tier=AuthorTier.SYSTEM), ledger.apply_op).ok
    assert status(db, "f-asthma-2") == "superseded" and status(db, "d-hotel") == "active"
    words = "Исправь: у Веры не астма, а аллергия на пыль."
    res = api.fix(words, words, turn=turn(words, n=2), because_quote=words, store=db)
    assert res.outcome == "committed", res
    assert status(db, "f-asthma-1") == "superseded"
    assert status(db, "d-hotel") == "needs_review"  # it rested on the merged-away copy
    assert status(db, "f-pool") == "needs_review"
    assert res.dependents_to_review == 2


def test_another_persons_copy_and_a_foreign_claim_are_never_retired(db):
    _asthma(db)
    add(db, person(ZOYA, "Зоя"))
    add(db, fact("f-zoya-asthma", ZOYA, "has_condition", "астма", "Zoya has asthma.",
                 "у Зои тоже астма"))  # fmt: skip
    raw(db, "r-zoya", "Зоя: у Веры астма, точно знаю", tier=AuthorTier.OTHER, author=ZOYA,
        chat_id="tg:77")  # fmt: skip
    add(db, fact("f-zoya-says", VERA, "has_condition", "астма", "Zoya says Vera has asthma.",
                 "у Веры астма, точно знаю", tier=AuthorTier.OTHER, speaker=ZOYA, raw="r-zoya"),
        tier=AuthorTier.OTHER)  # fmt: skip
    words = "Поправь: у Веры не астма, а аллергия на пыль."
    res = api.fix(words, words, turn=turn(words, n=2), because_quote=words, store=db)
    assert res.outcome == "committed", res
    assert status(db, "f-asthma-1") == "superseded"
    assert status(db, "f-zoya-asthma") == "active"  # another person's own fact
    assert status(db, "f-zoya-says") == "active"  # a foreign claim never retires by user's fix


def test_copies_on_two_people_his_words_do_not_name_ask_first(db):
    add(db, person(VERA, "Вера"))
    add(db, person(ZOYA, "Зоя"))
    add(db, fact("f-v", VERA, "has_condition", "мигрень", "Vera has migraines.", "у Веры мигрень"))
    add(db, fact("f-z", ZOYA, "has_condition", "мигрень", "Zoya has migraines.", "у Зои мигрень"))
    words = "Исправь: не мигрень, а просто усталость."
    res = api.fix(words, words, turn=turn(words, n=2), because_quote=words, store=db)
    assert res.outcome == "ask_first"
    assert status(db, "f-v") == "active" and status(db, "f-z") == "active"


def test_possessive_forms_name_their_nominative_and_stay_out_of_scope_resolution(db):
    for form in ("Верин", "Вериной", "Верину"):
        assert "вера" in entities.possessive_bases(form), form
    assert entities.possessive_bases("Петиного") >= {"петя"}
    assert entities.possessive_bases("Вера") == set()
    assert "вера" in entities.name_bases("Вере")  # a case form too
    # not a resolve tier: «забудь про Верину ссору» keeps its words as the forget's content
    add(db, person(VERA, "Вера"))
    add(db, person("ent-martin", "Мартин"))
    assert entities.resolve_rows(db, "Верину", "any") == []
    assert [r.entity_id for r in entities.resolve_rows(db, "Мартин", "any")] == ["ent-martin"]


def _ent(name: str, etype: str = "person") -> SimpleNamespace:
    return SimpleNamespace(display_name=name, etype=etype, aliases=[])


def test_snapshot_carries_form_people_outside_the_model_input_and_build_binds_them(db):
    add(db, person(VERA, "Вера"))
    raw(db, "r-coat", "Купили Верину куртку на распродаже", tier=AuthorTier.USER)
    snap, refs = extract.snapshot(db, ["r-coat"])
    assert VERA not in {e["id"] for e in snap["entities"]}
    assert [e["id"] for e in snap["form_entities"]] == [VERA]
    assert VERA not in extract.user_text(snap)  # the model's input is unchanged
    assert VERA in {r.id for r in refs}
    assert extract._form_entity(_ent("Вера"), snap) == VERA
    assert extract._form_entity(_ent("Вера Ким"), snap) is None  # a surname: the model's call
    assert extract._form_entity(_ent("Вера", "org"), snap) is None
    twin = {**snap, "form_entities": [*snap["form_entities"], {**snap["form_entities"][0],
            "id": "ent-vera-2"}]}  # fmt: skip
    assert extract._form_entity(_ent("Вера"), twin) is None  # namesakes stay apart
    seen = {**snap, "entities": [*snap["entities"], {"id": "ent-v3", "etype": "person",
            "display_name": "Вера", "handle": "m:ent-v3", "version": 1}]}  # fmt: skip
    assert extract._form_entity(_ent("Вера"), seen) is None  # the model saw a «Вера»: its call


@pytest.mark.parametrize(
    ("text", "reason", "choice", "linked"),
    [
        ("Там только веганское меню, так что «Ель» отпадает, берём «Клён».",
         "только веганское меню", "берём «Клён»", True),
        ("Аренда выросла до 40 тысяч, поэтому съезжаем в мае.", "аренда выросла до 40 тысяч",
         "съезжаем в мае", True),
        ("The ferry is cancelled on Sundays, so we drive on Saturday.",
         "the ferry is cancelled on sundays", "we drive on saturday", True),
        ("We drive on Saturday because the ferry is cancelled on Sundays.",
         "the ferry is cancelled on sundays", "we drive on saturday", True),
        ("So far the ferry runs daily and we drive on Saturday.", "the ferry runs daily",
         "we drive on saturday", False),
        ("Берём «Клён», так что меню там веганское.", "меню там веганское", "берём «Клён»",
         False),
    ],
)  # fmt: skip
def test_reason_of_reads_the_connective_and_its_sides(text, reason, choice, linked):
    assert extract.reason_of(text, reason, choice) is linked


GYM = "ent-gym"
LATE2 = "2026-11-20T09:00:00.000000Z"


def _org(eid: str, name: str):
    return person(eid, name).model_copy(update={"etype": "org"})


def _gym(conn) -> None:
    add(conn, _org(GYM, "«Атлант»"))
    add(conn, fact("f-fee", GYM, "other", "4 тысячи в месяц", "«Атлант» costs 4k a month.",
                   "дёшево, 4 тысячи в месяц"))  # fmt: skip
    add(conn, fact("f-walk", GYM, "other", "до дома 7 минут", "«Атлант» is 7 minutes from home.",
                   "до дома 7 минут"))  # fmt: skip
    add(conn, _decision("d-gym", "She keeps the «Атлант» membership.", "оставляю «Атлант»"),
        justifications=_just("f-fee", "f-walk"))  # fmt: skip


def test_a_later_amount_of_the_same_unit_replaces_the_earlier_other_fact(db):
    _gym(db)
    add(db, fact("f-fee2", GYM, "other", "9 тысяч с марта", "«Атлант» costs 9k from March.",
                 "«Атлант» с марта 9 тысяч", recorded=LATE2))  # fmt: skip
    assert status(db, "f-fee") == "superseded"
    assert status(db, "f-walk") == "active"  # another measure of the same place
    assert status(db, "f-fee2") == "active"


@pytest.mark.parametrize(
    ("value", "quote"),
    [
        ("40 тысяч в год", "за год 40 тысяч"),  # another period: two measures
        ("до метро 3 минуты", "до метро 3 минуты"),  # another distance
        ("9 тысяч с марта", "было 4 тысячи, с марта 9 тысяч для новых"),  # both said as current
    ],
)
def test_two_amounts_that_are_both_current_coexist(db, value, quote):
    _gym(db)
    add(db, fact("f-x", GYM, "other", value, "Another amount.", quote, recorded=LATE2))
    assert status(db, "f-fee") == "active" and status(db, "f-walk") == "active"


def test_a_number_his_word_order_reads_as_a_date_still_holds_the_value():
    raws = {"r1": {"text": "Да, «Атлант» с марта 9 тысяч, пришло письмо", "tier": "user"}}
    quote = "«Атлант» с марта 9 тысяч"
    assert extract._claim_tier(quote, "9 тысяч с марта", [], raws) is None  # keeps the user's tier
    assert extract._claim_tier(quote, "19 тысяч с марта", [], raws) is AuthorTier.SYSTEM


def test_his_own_other_facts_never_replace_each_other_by_unit(db):
    add(db, fact("f-sleep", "self", "other", "сплю 7 часов", "He sleeps 7 hours.", "сплю 7 часов"))
    add(db, fact("f-work", "self", "other", "работаю 9 часов", "He works 9 hours.",
                 "работаю 9 часов", recorded=LATE2))  # fmt: skip
    assert status(db, "f-sleep") == "active" and status(db, "f-work") == "active"


def test_a_decision_tail_names_its_own_place_premises(db):
    from koob.memory.model import Label

    add(db, _org(GYM, "«Атлант»"))
    add(db, fact("f-near", GYM, "other", "у метро Сокол", "«Атлант» is by Sokol metro.",
                 "у метро Сокол", labels=Label.LOCATION_PRECISE))  # fmt: skip
    dec = _decision("d-near", "She keeps «Атлант» by Sokol.", "оставляю «Атлант» у Сокола")
    add(db, dec.model_copy(update={"labels": Label.LOCATION_PRECISE}),
        justifications=_just("f-near"))  # fmt: skip
    basket = api.recall(turn("Оставлять «Атлант» у Сокола?", n=3), store=db)
    line = next((i.text for i in basket.items if i.id == "d-near"), "")
    assert "Sokol metro" in line, [i.text for i in basket.items]
