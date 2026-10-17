"""The basket's recheck of the user's own raw messages, and facts learned from a revised message.

A raw message of the user's that the use-now verdict judged not applicable is re-read the way
the user's records are: through the records learned from it (their English lines and both
keyword lists) and the names of the question, so a question in one language finds their words
in the other. A strong label (health, a minor) still needs the exact ask. A fact learned from a
message the user's later message revised is marked past with the revised value cut, never served
as current.
"""

from __future__ import annotations

import pytest

from koob.memory import api, policy
from koob.memory.model import Label, TurnCtx, TurnOrigin
from koob.memory.recall import basket, lexical, vectors
from koob.memory.recall import cue as cue_mod

from .._ledger import add, entity, fact, open_seeded, raw

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    basket._NO_CONTRAST.clear()
    basket._NO_REVISION.clear()
    c = open_seeded(tmp_path)
    yield c
    c.close()
    basket._NO_CONTRAST.clear()
    basket._NO_REVISION.clear()


def t(words: str) -> TurnCtx:
    return TurnCtx(
        turn_id="turn-1",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=CLOCK,
    )


def _own_raw(conn, rid: str, text: str, labels: Label, *, day: str = "2026-10-01") -> None:
    raw(conn, rid, text, ts_=f"{day}T10:00:00Z", labels=int(labels))
    conn.execute("UPDATE raw_chunks SET labels_final = 1 WHERE id = ?", (rid,))
    lexical.index_raw(conn, rid)


def _recheck(conn, ids: list[str], words: str, cue: cue_mod.Cue | None = None) -> dict:
    """The basket's use-now step for these ids: the verdict, then the recheck."""
    turn = t(words)
    rows = basket._hydrate(conn, ids, CLOCK[:10])
    verdicts = policy.decide_many(conn, ids, turn)
    cue = cue or cue_mod.Cue(words=words)
    basket._recheck(
        rows,
        verdicts,
        {i: {"raw"} for i in ids},
        cue,
        lexical.query_stems(words),
        set(),
        conn=conn,
        name_stems=lexical.name_stems(words),
    )
    return verdicts


def _bakery(conn, labels: Label) -> None:
    assert add(conn, entity("e-denis", "Denis Smith")).ok
    _own_raw(conn, "r-bake", "Денис Смит дал мне в долг на пекарню, верну к лету.", labels)
    assert add(
        conn,
        fact(
            "f-bake",
            "self",
            "other",
            "a loan for the bakery",
            text="Denis Smith lent the owner money for the bakery, to be repaid by summer",
            keywords_ru="пекарня, долг",
            keywords_en="bakery, loan",
            quote_orig="Денис Смит дал мне в долг на пекарню",
            source_raw_id="r-bake",
            labels=labels,
            labels_final=True,
        ),
    ).ok


def test_own_russian_raw_found_by_an_english_question(conn):
    _bakery(conn, Label.MONEY)
    words = "what did I say about the bakery?"
    assert policy.decide(conn, "r-bake", t(words)).reason == "own_not_applicable"
    v = _recheck(conn, ["r-bake"], words)["r-bake"]
    assert v.allowed and v.reason == "own_asked"


def test_own_raw_found_by_a_name_in_the_other_script(conn):
    _own_raw(conn, "r-call", "Денис Смит просил перезвонить после обеда.", Label.LEGAL)
    words = "what did Denis want?"
    assert policy.decide(conn, "r-call", t(words)).reason == "own_not_applicable"
    assert _recheck(conn, ["r-call"], words)["r-call"].allowed


def test_own_raw_on_another_matter_stays_out(conn):
    _bakery(conn, Label.MONEY)
    v = _recheck(conn, ["r-bake"], "what was the name of that hiking trail?")["r-bake"]
    assert not v.allowed and v.reason == "own_not_applicable"


def _cast(conn) -> None:
    assert add(conn, entity("e-denis", "Denis Smith")).ok
    _own_raw(conn, "r-cast", "Денис Смит сломал руку, гипс снимут в мае.", Label.HEALTH)
    assert add(
        conn,
        fact(
            "f-cast",
            "e-denis",
            "has_condition",
            "a broken arm",
            text="Denis Smith has a broken arm, the cast comes off in May",
            keywords_ru="перелом, гипс",
            keywords_en="broken arm, cast",
            quote_orig="Денис Смит сломал руку",
            source_raw_id="r-cast",
            labels=Label.HEALTH,
            labels_final=True,
        ),
    ).ok


def test_own_health_raw_needs_more_than_the_name(conn):
    _cast(conn)
    words = "what is new with Denis?"
    v = _recheck(conn, ["r-cast"], words, cue_mod.Cue(words=words, entity_ids=["e-denis"]))
    assert not v["r-cast"].allowed


def test_own_health_raw_opens_on_the_exact_ask(conn):
    _cast(conn)
    words = "what condition does Denis have?"
    cue = cue_mod.Cue(words=words, entity_ids=["e-denis"], relations=["has_condition"])
    v = _recheck(conn, ["r-cast"], words, cue)["r-cast"]
    assert v.allowed and v.reason == "own_asked"


def _birthday(conn) -> None:
    assert add(conn, entity("e-denis", "Денис Смит")).ok
    _own_raw(
        conn, "r-old", "У Дениса день рождения 14 мая. Напомни купить подарок.", Label.NONE
    )
    assert add(
        conn,
        fact(
            "f-bday",
            "e-denis",
            "birthday",
            "14 мая",
            text="Denis Smith's birthday is on 14 May",
            quote_orig="У Дениса день рождения 14 мая",
            source_raw_id="r-old",
            labels_final=True,
        ),
    ).ok
    _own_raw(
        conn, "r-fix", "Нет, у Дениса день рождения 17 мая, я перепутал.", Label.NONE,
        day="2026-10-05",
    )  # fmt: skip


def test_fact_of_a_revised_raw_is_marked_past_with_the_value_cut(conn):
    _birthday(conn)
    rows = basket._hydrate(conn, ["f-bday"], CLOCK[:10])
    row = rows["f-bday"]
    assert row["past"] == "2026-10-05" and "r-fix" in row["corrected_by"]
    assert "14" not in row["text"] and "[сменилось]" in row["text"]
    assert "14" not in (row["value_text"] or "") and "14" not in row["quote_orig"]
    line = basket.line(basket._item(row, None, None))
    assert "сменилась" in line and "14" not in line


def test_revised_fact_is_never_shown_as_current_in_the_basket(conn):
    _birthday(conn)
    b = api.recall(t("Когда у Дениса день рождения?"), store=conn)
    lines = {i.id: basket.line(i) for i in b.items}
    assert not any("14 мая" in x or "14 May" in x for x in lines.values())
    assert "r-fix" in lines and "17 мая" in lines["r-fix"]


def test_fact_holding_another_value_of_the_revised_raw_stays_current(conn):
    _birthday(conn)
    assert add(
        conn,
        fact(
            "f-gift",
            "self",
            "other",
            "buy a present",
            text="The owner wants a reminder to buy a present",
            quote_orig="Напомни купить подарок",
            source_raw_id="r-old",
            labels_final=True,
        ),
    ).ok
    row = basket._hydrate(conn, ["f-gift"], CLOCK[:10])["f-gift"]
    assert row.get("past") is None and row["text"].endswith("present")
