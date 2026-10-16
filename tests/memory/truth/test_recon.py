"""§8.4 outcomes of a change for the conclusions that rest on it.

Spec marker: `[T truth/test_recon::<case>]`: AND-reasons ("cheap and on time"), a
paraphrased "second source", a move that must not flip unrelated facts. Fictional people."""

from __future__ import annotations

import pytest

from koob.memory import api, justify, ledger
from koob.memory.model import OWNER_ENTITY_ID, AuthorTier, Decision, Domain

from ..inspect._world import TS, add, fact, person, raw, status, turn

DIMA, SEREGA, LENA = "ent-dima", "ent-serega", "ent-lena"
OTHER = AuthorTier.OTHER


@pytest.fixture
def db(mem):
    ledger.seed_relations(mem)
    ledger.seed_owner(mem)
    return mem


def _just(*groups: list[str], prov: str = "stated") -> list[dict]:
    return [
        {"group_no": g, "member": m, "provenance": prov} for g, ms in enumerate(groups) for m in ms
    ]


def _decision(did: str, text: str, quote: str) -> Decision:
    return Decision(
        id=did, author_tier=AuthorTier.USER, recorded_at=TS, labels_final=True,
        domain=Domain.TRAVEL, chosen_text=text, decided_at=TS, origin_quote=quote,
    )  # fmt: skip


def _fix(conn, target: str, words: str, *, n: int):
    res = api.fix(f"m:{target}", words, turn=turn(words, n=n), because_quote=words, store=conn)
    assert res.outcome == "committed", res
    return res


def _review(conn, rid: str) -> int:
    return conn.execute("SELECT review FROM records WHERE id = ?", (rid,)).fetchone()[0]


def test_and_reasons(db):
    """§8.4: AND-reasons ("cheap and on time"): one broken member breaks the group."""
    raw(db, "r-fl", "Рейс в 18:05 дешёвый, 4 тысячи, и я успеваю к началу")
    add(db, fact("f-cheap", OWNER_ENTITY_ID, "other", "4000 rub", "The 18:05 flight is cheap",
                 "Рейс в 18:05 дешёвый", raw="r-fl"))  # fmt: skip
    add(db, fact("f-ontime", OWNER_ENTITY_ID, "other", "on time", "The 18:05 flight arrives in time",
                 "я успеваю к началу", raw="r-fl"))  # fmt: skip
    add(db, _decision("d-flight", "Take the 18:05 flight", "беру рейс в 18:05: дёшево и успевает"),
        justifications=_just(["f-cheap", "f-ontime"]))  # fmt: skip
    assert justify.supported(db, "d-flight")

    res = _fix(db, "f-ontime", "нет, я не успеваю: рейс прилетает в 20:40", n=2)
    assert res.dependents_to_review == 1
    assert status(db, "d-flight") == "needs_review"  # one member broke the AND-group
    assert justify.outcome(db, "d-flight", []) == "unsupported"
    assert status(db, "f-cheap") == "active"  # the other reason is untouched
    jobs = db.execute("SELECT count(*) FROM jobs WHERE jkind = ?", (justify.REDERIVE_JOB,))
    assert jobs.fetchone()[0] == 1  # re-derivation queued


def _sources(conn, second: str) -> None:
    for eid, name in ((DIMA, "Дима"), (SEREGA, "Серёга"), (LENA, "Лена")):
        add(conn, person(eid, name))
    raw(conn, "r-dima", "Дима: проводы в пятницу в 19", tier=OTHER, author=DIMA, chat_id="tg:1")
    raw(conn, "r-ser", "Серёга: Дима сказал, проводы в пятницу вечером", tier=OTHER,
        author=SEREGA, chat_id="tg:2")  # fmt: skip
    raw(conn, "r-lena", "Лена: проводы Димы в пятницу, я уже купила торт", tier=OTHER,
        author=LENA, chat_id="tg:3")  # fmt: skip
    add(conn, fact("f-dima", DIMA, "other", "farewell on Friday", "Dima's farewell is on Friday",
                   "проводы в пятницу в 19", tier=OTHER, speaker=DIMA, raw="r-dima"))  # fmt: skip
    add(conn, fact("f-ser", DIMA, "other", "farewell on Friday", "Dima's farewell is on Friday",
                   "проводы в пятницу вечером", tier=OTHER, speaker=DIMA, raw="r-ser"))  # fmt: skip
    add(conn, fact("f-lena", DIMA, "other", "farewell on Friday", "Dima's farewell is on Friday",
                   "проводы Димы в пятницу", tier=OTHER, speaker=LENA, raw="r-lena"))  # fmt: skip
    add(conn, _decision("d-friday", "Keep Friday evening free for Dima's farewell",
                        "в пятницу вечером я на проводах Димы"),
        justifications=_just(["f-dima"], [second], prov="rule"))  # fmt: skip


def test_paraphrased_second_source(db):
    """§8.4: a paraphrased retelling is not a second independent source."""
    _sources(db, "f-ser")
    assert not justify.independent(db, "d-friday", (0, 1))
    _fix(db, "f-dima", "нет, Дима перенёс проводы на субботу", n=2)
    assert status(db, "f-ser") == "active"  # the retelling stays as the user's record of it
    assert status(db, "d-friday") == "needs_review"  # but it is no second source


def test_independent_second_source_keeps_the_conclusion(db):
    """§8.1 + §8.4: the same refutation with an independent second group changes nothing."""
    _sources(db, "f-lena")
    assert justify.independent(db, "d-friday", (0, 1))
    _fix(db, "f-dima", "нет, Дима перенёс проводы на субботу", n=2)
    assert status(db, "d-friday") == "active" and _review(db, "d-friday") == 0
    assert justify.outcome(db, "d-friday", []) == "unchanged"


def test_move_keeps_unrelated_facts(db):
    """§8.4: a move puts dependent facts in review and flips no unrelated fact."""
    raw(db, "r-home", "Я живу в Москве, хожу в зал на Тверской")
    add(db, fact("f-moscow", OWNER_ENTITY_ID, "lives_in", "Moscow", "The owner lives in Moscow",
                 "Я живу в Москве", raw="r-home"))  # fmt: skip
    add(db, fact("f-tz", OWNER_ENTITY_ID, "timezone", "MSK", "The owner's timezone is MSK",
                 "Я живу в Москве"),
        justifications=_just(["f-moscow"], prov="rule"))  # fmt: skip
    add(db, fact("f-gym", OWNER_ENTITY_ID, "other", "gym on Tverskaya",
                 "The owner trains at the gym on Tverskaya", "хожу в зал на Тверской"),
        justifications=_just(["f-moscow"]))  # fmt: skip
    add(db, fact("f-coffee", OWNER_ENTITY_ID, "other", "flat white", "The owner drinks flat white",
                 "пью флэт уайт"))  # fmt: skip
    add(db, fact("f-job", OWNER_ENTITY_ID, "works_at", "Acme", "The owner works at Acme",
                 "работаю в Acme"))  # fmt: skip
    before = {r: db.execute("SELECT version FROM records WHERE id = ?", (r,)).fetchone()[0]
              for r in ("f-coffee", "f-job")}  # fmt: skip

    add(db, fact("f-tver", OWNER_ENTITY_ID, "lives_in", "Tver", "The owner lives in Tver",
                 "я переехал в Тверь"))  # fmt: skip
    assert status(db, "f-moscow") == "superseded"  # §8.3 by rule, not a fix
    assert justify.outcome(db, "f-tz", ["f-moscow"]) == "review"
    done = justify.run_cascade(db, ["f-moscow"])  # the worker / night path
    assert done == {"f-tz": "review", "f-gym": "review"}
    for rid in ("f-tz", "f-gym"):  # usable, in doubt, never auto-retracted
        assert status(db, rid) == "active" and _review(db, rid) == 1
    for rid, ver in before.items():
        assert status(db, rid) == "active" and _review(db, rid) == 0
        assert db.execute("SELECT version FROM records WHERE id = ?", (rid,)).fetchone()[0] == ver
