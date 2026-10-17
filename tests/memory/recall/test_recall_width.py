"""Basket width: a question that asks to recall gets a wider basket, the person it names is not
capped as crowding, and its lines keep the matching sentence and one reading of each date."""

from __future__ import annotations

import pytest

from koob.memory import api
from koob.memory.model import TurnCtx, TurnOrigin
from koob.memory.recall import basket, gate, lexical, vectors
from koob.memory.recall import cue as cue_mod

from .._ledger import add, entity, fact, open_seeded

CLOCK = "2026-10-12T09:00:00.000000Z"
SCORE = 1.0 / (gate.RRF_K + 1)


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str) -> TurnCtx:
    return TurnCtx(
        turn_id="turn-1",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=CLOCK,
    )


def cand(i: str, *, strong: bool = True, tokens: int = 40, **kw) -> gate.Cand:
    return gate.Cand(
        id=i, kind="fact", version=1, text=i, tokens=tokens, strong=strong, lanes={"lexical"}, **kw
    )


@pytest.mark.parametrize(
    "words",
    [
        "What did Denis Smith say about the new flat?",
        "When did Denis start the pottery course?",
        "Which cities has Denis visited?",
        "What kinds of music does Denis like?",
        "Do you remember when Denis and I went hiking?",
        "Что Денис говорил про переезд?",
        "Когда Денис ездил в Тверь?",
        "Какие книги Денис советовал?",
    ],
)
def test_question_that_asks_to_recall_is_marked(words):
    assert cue_mod.parse(t(words), now=CLOCK).recall


@pytest.mark.parametrize(
    "words",
    [
        "Где живёт Денис?",
        "What's Denis Smith's phone number?",
        "Привет, как дела?",
        "Когда у Дениса день рождения?",
        "when is the pottery class?",
        "Напиши Денису, что я опоздаю",
        "Denis called and said he is running late.",
    ],
)
def test_lookup_greeting_or_order_keeps_the_narrow_basket(words):
    assert not cue_mod.parse(t(words), now=CLOCK).recall


def test_recall_width_holds_more_grounded_lines_than_narrow(conn):
    cands = {f"s{k}": cand(f"s{k}") for k in range(20)}
    ranked = [(f"s{k}", SCORE - k * 1e-5) for k in range(20)]
    narrow = gate.gate(conn, ranked, thread_id="x", cands=cands)
    wide = gate.gate(conn, ranked, thread_id="x", cands=cands, width=gate.RECALL)
    assert len(narrow) == gate.MAX_ITEMS
    assert len(wide) == gate.RECALL.items
    assert wide[: len(narrow)] == narrow  # the same best lines first


def test_recall_width_fills_to_its_line_cap_inside_its_token_budget(conn):
    strong = {f"s{k}": cand(f"s{k}", tokens=50) for k in range(4)}
    weak = {f"w{k}": cand(f"w{k}", strong=False, tokens=50) for k in range(30)}
    ranked = [(i, SCORE) for i in [*strong, *weak]]
    out = gate.gate(conn, ranked, thread_id="x", cands={**strong, **weak}, width=gate.RECALL)
    assert len(out) == gate.RECALL.lines
    assert sum(50 for _ in out) <= gate.RECALL.fill_tokens
    heavy = {f"h{k}": cand(f"h{k}", tokens=200) for k in range(10)}
    out = gate.gate(
        conn, [(i, SCORE) for i in heavy], thread_id="x", cands=heavy, width=gate.RECALL
    )
    assert len(out) * 200 <= gate.RECALL.tokens


def test_named_subject_is_not_capped_per_entity_but_a_third_person_is(conn):
    about = {f"d{k}": cand(f"d{k}", entity="e-denis") for k in range(8)}
    other = {f"o{k}": cand(f"o{k}", entity="e-other") for k in range(8)}
    ranked = [(i, SCORE) for i in [*other, *about]]
    cands = {**about, **other}
    unnamed = gate.gate(conn, ranked, thread_id="x", cands=cands, width=gate.RECALL)
    assert sum(i.startswith("o") for i in unnamed) == gate.PER_ENTITY
    assert sum(i.startswith("d") for i in unnamed) == gate.PER_ENTITY
    named = gate.gate(
        conn, ranked, thread_id="x", cands=cands, width=gate.RECALL, subjects={"e-denis"}
    )
    assert sum(i.startswith("o") for i in named) == gate.PER_ENTITY
    assert sum(i.startswith("d") for i in named) == len(about)


def test_recall_question_about_a_person_gets_more_than_per_entity_of_his_lines(conn):
    assert add(conn, entity("e-denis", "Denis Smith")).ok
    things = ["oven", "flour supplier", "lease", "sign", "first hire", "opening day"]
    for k, thing in enumerate(things):
        text = f"Denis Smith said the bakery {thing} was sorted out"
        assert add(conn, fact(f"f-{k}", "e-denis", "other", thing, text=text)).ok
    b = api.recall(t("What did Denis Smith say about the bakery?"), store=conn)
    ids = [i.id for i in b.items]
    assert len(ids) > gate.PER_ENTITY
    assert set(ids) <= {f"f-{k}" for k in range(len(things))}


def test_long_message_is_cut_around_the_sentence_the_question_asks_about():
    text = "Morning chat about the weather and the traffic. " * 8 + (
        "Denis signed the lease for the bakery on Elm Street. That was the big news."
    )
    focus = lexical.query_stems("When did Denis sign the bakery lease?")
    clipped = basket._clip(text, 200, None, focus)
    assert "signed the lease for the bakery" in clipped
    assert clipped.startswith("…") and len(clipped) <= 200
    assert basket._clip(text, 200).startswith("Morning chat")  # no focus: the head


def _row(text: str, **dates) -> dict:
    return {"id": "f-1", "kind": "fact", "version": 1, "text": text, "author_tier": None, **dates}


def test_resolved_date_drops_the_relative_word_next_to_it():
    row = _row(
        "Denis lost his phone yesterday.",
        valid_from="2026-03-09",
        recorded_at="2026-03-10T10:00:00Z",
    )
    item = basket._item(row, None, None)
    assert item.date == "2026-03-09"
    assert "yesterday" not in item.text and item.text.startswith("Denis lost his phone")
    ru = basket._drop_relative("На прошлой неделе Денис купил велосипед, вчера забрал.")
    assert "прошлой неделе" not in ru and "вчера" not in ru and "купил велосипед" in ru


def test_date_of_the_day_it_was_said_keeps_the_relative_word():
    row = _row("Denis lost his phone yesterday.", recorded_at="2026-03-10T10:00:00Z")
    assert "yesterday" in basket._item(row, None, None).text
    same = _row(
        "Denis lost his phone yesterday.",
        valid_from="2026-03-10",
        recorded_at="2026-03-10T10:00:00Z",
    )
    assert "yesterday" in basket._item(same, None, None).text
