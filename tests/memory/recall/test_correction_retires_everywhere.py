"""A correction retires the old value everywhere, before any worker extracts it and with no
retraction written: the user's own earlier message whose value a later message corrects is
shown with the value cut (basket and search), and everything learned or inferred from the
revised message - the user's facts, system-inferred facts, episodes - stops carrying the old
value in the basket, in search and in the core view's people section."""

from __future__ import annotations

import pytest

from koob.memory import api
from koob.memory.model import AuthorTier, Episode, TurnCtx, TurnOrigin
from koob.memory.recall import basket, core_view, vectors

from .._ledger import add, entity, fact, open_seeded, ts

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


def t(words: str, n: int = 9) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=CLOCK,
    )


def said(conn, n: int, text: str, day: str) -> None:
    api.ingest_turn(
        turn_id=f"turn-{n}",
        thread_id="thread-1",
        origin="owner",
        status="done",
        rows=[{"ts": f"{day}T10:00:00Z", "role": "owner", "text": text}],
        store=conn,
    )


CASES = [
    (
        "Денис переезжает в Тверь в ноябре, работать в новом офисе.",
        "Поправка: Денис переезжает не в Тверь, а в Самару.",
        "Куда переезжает Денис?",
        "Тверь",
    ),
    (
        "Denis is moving to Leeds in November for the new office.",
        "Correction: Denis is moving to York, not Leeds.",
        "Where is Denis moving?",
        "Leeds",
    ),
    (
        "Denis is moving to Leeds in November for the new office.",
        "Wait, I got it wrong - Denis is moving to York in November.",
        "Where is Denis moving?",
        "Leeds",
    ),
    (
        "Мой новый номер телефона +7 912 345-67-89, запиши.",
        "Ой, я ошибся: мой номер +7 912 345-67-98.",
        "Какой у меня номер телефона?",
        "345-67-89",
    ),
    (
        "Встреча с Денисом Смитом в четверг в 15:00 в кафе на углу.",
        "Перенёс: с Денисом Смитом встречаемся в 17:30.",
        "Во сколько встреча с Денисом Смитом?",
        "15:00",
    ),
    (
        "I'll be at the dentist on 14 May at 9am.",
        "Actually the dentist moved me to May 21.",
        "When is my dentist appointment?",
        "14 May",
    ),
]


@pytest.mark.parametrize("old,fix,ask,stale", CASES)
def test_own_raw_corrected_by_his_later_raw_is_cut_in_basket_and_search(conn, old, fix, ask, stale):
    said(conn, 1, old, "2026-10-01")
    said(conn, 2, fix, "2026-10-05")
    b = api.recall(t(ask), store=conn)
    lines = [basket.line(i) for i in b.items]
    found = api.search(ask, turn=t(ask, 10), store=conn)
    slines = [i.text for i in found]
    assert not any(stale in x for x in lines), lines
    assert not any(stale in x for x in slines), slines


MARK = "ent-mark"


def _learned(conn, *, inferred_text: str, episode_text: str, fact_text: str, value: str) -> None:
    """The user's message about Mark, a fact learned from it, a system-inferred fact and a system
    episode built on that fact (no source raw of their own)."""
    said(conn, 1, "Марк Иванов переезжает в Тверь 14 ноября, зовёт на новоселье.", "2026-10-01")
    rid = conn.execute("SELECT id FROM raw_chunks WHERE turn_id = 'turn-1'").fetchone()[0]
    add(conn, entity(MARK, "Марк Иванов"))
    add(conn, fact("f-move", MARK, "moves_on", value, text=fact_text, source_raw_id=rid,
                   quote_orig="Марк Иванов переезжает в Тверь 14 ноября"))  # fmt: skip
    add(conn, fact("f-lives", MARK, "lives_in", "Tver", text=inferred_text,
                   tier=AuthorTier.SYSTEM, origin_quote=None), derived_from=["f-move"])  # fmt: skip
    ep = Episode(id="ep-move", author_tier=AuthorTier.SYSTEM, recorded_at=ts(),
                 summary_en=episode_text)  # fmt: skip
    add(conn, ep, derived_from=["f-move"], mentions=[MARK])


def _everywhere(conn, ask: str) -> list[str]:
    b = api.recall(t(ask), store=conn)
    found = api.search(ask, turn=t(ask, 10), store=conn)
    people = [ln for sec in core_view.compute(conn) for ln in core_view.render([sec]).splitlines()]
    return [*(basket.line(i) for i in b.items), *(i.text for i in found), *people]


def test_a_revised_date_leaves_inferred_facts_and_episodes_of_that_message(conn):
    _learned(
        conn,
        fact_text="Mark Ivanov is moving to Tver on 14 November.",
        inferred_text="Mark Ivanov will live in Tver from 14 November.",
        episode_text="Mark Ivanov moves to Tver on 14 November; housewarming planned.",
        value="14 ноября",
    )
    said(conn, 2, "Ошибся: Марк Иванов переезжает 21 ноября, билеты сдвинули.", "2026-10-05")
    lines = _everywhere(conn, "Когда переезжает Марк Иванов?")
    assert any("21 ноября" in x for x in lines), lines
    assert not any("14 November" in x or "14 ноября" in x for x in lines), [
        x for x in lines if "14 November" in x or "14 ноября" in x
    ]


def test_a_contrast_about_another_person_retires_the_old_place_in_derived_items(conn):
    _learned(
        conn,
        fact_text="Mark Ivanov is moving to Tver in November.",
        inferred_text="Mark Ivanov will live in Tver.",
        episode_text="Mark Ivanov moves to Tver; housewarming planned.",
        value="Тверь",
    )
    said(conn, 2, "Марк Иванов переезжает не в Тверь, а в Самару.", "2026-10-05")
    lines = _everywhere(conn, "Куда переезжает Марк Иванов?")
    assert any("Самару" in x for x in lines), lines
    assert not any("Tver" in x or "Тверь" in x for x in lines), [
        x for x in lines if "Tver" in x or "Тверь" in x
    ]
