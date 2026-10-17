""""What do you remember about me" in any phrasing lists the user's own records with their dates,
the user's own sensitive ones included (§5.5: they apply to a question about the whole user);
other people's sensitive claims about the user stay behind the user's direct question; English
short names meet their full names; rent and housing payments carry MONEY; a broad question's
basket grows inside the token budget."""

from __future__ import annotations

import pytest

from koob.memory import commands, entities, policy
from koob.memory.model import AuthorTier, Label
from koob.memory.recall import gate

from ._world import DIMA, add, fact, first_record, person, turn

OWN_SHOE = "fact-own-shoe"
OWN_PILLS = "fact-own-pills"
OTHER_DEBT = "fact-other-debt"


def _me(conn) -> None:
    first_record(conn)
    add(conn, person(DIMA, "Дима"))
    add(
        conn,
        fact(OWN_SHOE, "user", "shoe_size", "43", "His shoe size is 43", "размер обуви у меня 43"),
    )
    add(
        conn,
        fact(
            OWN_PILLS,
            "user",
            "takes",
            "магний",
            "He takes magnesium tablets for cramps",
            "пью магний от судорог, назначил терапевт",
            labels=Label.HEALTH,
        ),
    )
    add(
        conn,
        fact(
            OTHER_DEBT,
            "user",
            "owes",
            "Диме",
            "He owes Dima money",
            "он мне должен денег",
            tier=AuthorTier.OTHER,
            speaker=DIMA,
            labels=Label.MONEY,
        ),
    )


@pytest.mark.parametrize(
    "ask",
    [
        "Что ты обо мне помнишь?",
        "Что ты про меня уже знаешь?",
        "What do you know about me so far?",
        "what do you remember about me already",
    ],
)
def test_self_inspect_lists_his_own_records_with_dates(mem, ask):
    _me(mem)
    text = "\n".join(commands.what(mem, ask, turn(ask)))
    assert "43" in text
    assert "магний" in text  # the user's own sensitive record applies to a question about the whole user
    # with the date it was recorded, in the language of the user's question
    assert "12 октября 2026" in text or "October 12, 2026" in text
    assert "ничего не помню" not in text and "remember nothing" not in text
    # someone else's sensitive claim about the user stays behind the user's
    # direct question
    assert "должен" not in text and "owes" not in text.casefold()
    assert "личн" in text or "private" in text


def test_self_filler_never_widens_a_topic_question(mem):
    _me(mem)
    text = "\n".join(commands.what(mem, "что ты уже знаешь про мой размер обуви", turn("x")))
    assert "43" in text and "магний" not in text


@pytest.mark.parametrize(
    ("short", "full"),
    [("Bobby", "Robert"), ("Rob", "Robert"), ("Lizzie", "Elizabeth"), ("Bill", "William")],
)
def test_english_short_names_meet_their_family(short, full):
    assert full in entities.family_of(short)
    assert full in entities.name_family(short)


def test_a_plain_word_has_no_family():
    assert entities.family_of("Okonkwo") == [] and entities.family_of("Amy") == []


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("Will T", "Bill Thornton", True),  # family + an initial
        ("Liz Moreno", "Elizabeth Moreno", True),
        ("Саша Белова", "Александра Белова", True),
        ("Will D", "Bill Thornton", False),  # the initial disagrees
        ("Will", "Bill Thornton", False),  # a first name alone never makes two people one
        ("Bill Ng", "Bob Ng", False),  # another family, one surname
    ],
)
def test_same_person_needs_more_than_a_first_name(a, b, same):
    assert entities.same_person_names(a, b) is same


@pytest.mark.parametrize(
    "text",
    [
        "аренда подорожала",
        "квартплата пришла",
        "коммуналку оплатил",
        "the rent goes up in March",
        "utilities are split three ways",
    ],
)
def test_housing_payments_carry_money(text):
    assert policy.lexicon_label(text) & Label.MONEY


def _cand(i: int, *, strong: bool = True) -> gate.Cand:
    return gate.Cand(
        id=f"c{i}", kind="fact", version=1, text="t", tokens=40, lanes={"a", "b"}, strong=strong
    )


def test_a_broad_question_gets_more_lines_inside_the_budget(mem):
    first_record(mem)
    ranked = [(f"c{i}", gate.HIGH_SCORE + 0.01) for i in range(12)]
    cands = {f"c{i}": _cand(i) for i in range(12)}
    got = gate.gate(mem, ranked, thread_id="th-x", cands=cands)
    assert gate.MAX_ITEMS < len(got) <= gate.MAX_ITEMS_BROAD
    tight = gate.gate(mem, ranked, thread_id="th-x", cands=cands, budget_tokens=200)
    assert len(tight) == 5  # the token budget still holds


def test_a_narrow_question_keeps_its_short_basket(mem):
    first_record(mem)
    ranked = [(f"c{i}", gate.HIGH_SCORE + 0.01) for i in range(3)]
    ranked += [(f"c{i}", gate.MIN_SCORE + 0.001) for i in range(3, 12)]
    cands = {f"c{i}": _cand(i) for i in range(12)}
    assert len(gate.gate(mem, ranked, thread_id="th-x", cands=cands)) == gate.MAX_ITEMS
