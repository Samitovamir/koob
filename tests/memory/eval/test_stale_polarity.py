"""A stale check reads polarity: a value its own clause negates or overrides
(«сегодня без кинзы», "not Boulder anymore", «уже не в Твери») is not shown as current; a
bare or affirmed mention (a sly «не забудь кинзу», "Boulder, not Denver") still is.
The same rule for every SUT: the graders' must_not_include / forbidden checks and the bench's
view judge read it alike."""

from __future__ import annotations

import pytest

from koob.memory.eval.bench import judge_view
from koob.memory.eval.graders import check_exclude, check_forbidden
from koob.memory.eval.matchers import shown_as_current
from koob.memory.eval.observe import Observation
from koob.memory.eval.schema import Expect, Probe

KINZA = ["кинз", "cilantro"]
TVER = ["твер", "tver"]
BOULDER = "boulder"

NEGATED = [
    ("Сегодня без кинзы.", KINZA),
    ("Today: no cilantro.", KINZA),
    ("Pasta without cilantro tonight.", KINZA),
    ("Ни кинзы, ни лука.", KINZA),
    ("Ты уже не в Твери, а в Москве.", TVER),
    ("Живёт не в Твери — в Москве.", TVER),
    ("Ты больше не живёшь в Твери.", TVER),
    ("Тверь — уже нет, теперь Москва.", TVER),
    ("Вместо кинзы петрушка.", KINZA),
    ("Not Boulder anymore: Denver now.", BOULDER),
    ("You're not in Boulder anymore.", BOULDER),
    ("You don't live in Boulder anymore.", BOULDER),
    ("You no longer live in Boulder.", BOULDER),
    ("Home isn't Boulder, it's Denver.", BOULDER),
    ("Denver instead of Boulder.", BOULDER),
    ("Встреча не 14 мая, а 10 мая.", {"date": "--05-14"}),
    ("Budget is not $1500 but $1800.", {"money": 1500, "cur": "USD"}),
]

AFFIRMED = [
    ("Не забудь кинзу.", KINZA),  # «не» negates the verb, not the herb
    ("Don't forget the cilantro.", KINZA),
    ("Don't forget the cilantro anymore.", KINZA),  # a verb that flips polarity back
    ("Больше не забывай кинзу.", KINZA),
    ("Не без кинзы, конечно.", KINZA),  # a double negation affirms
    ("Boulder, not Denver.", BOULDER),
    ("Not Denver but Boulder.", BOULDER),
    ("Not only Boulder: Denver too.", BOULDER),
    ("Не только Тверь, но и Москва.", TVER),
    ("Ты больше не живёшь в Москве, а в Твери.", TVER),  # the contrast affirms it
    ("No, Boulder.", BOULDER),
    ("Нет, в Твери.", TVER),
    ("Ты не в Твери?", TVER),  # a negative question suggests the value
    ("Isn't it Boulder?", BOULDER),
    ("Сегодня без кинзы, но вчера кинза была.", KINZA),  # one bare mention is enough
    ("Кинза. Без петрушки.", KINZA),
    ("Не больше $1500.", {"money": 1500, "cur": "USD"}),  # a ceiling, not a negation
    ("No later than 14 May.", {"date": "--05-14"}),
    ("Встреча 14 мая.", {"date": "--05-14"}),
]


@pytest.mark.parametrize("text,value", NEGATED)
def test_a_value_its_own_clause_negates_is_not_shown_as_current(text, value):
    assert not shown_as_current(text, value, history=False), text
    assert not shown_as_current(text, value), text  # `allow: history` reads it the same way


@pytest.mark.parametrize("text,value", AFFIRMED)
def test_a_bare_or_affirmed_mention_still_is(text, value):
    assert shown_as_current(text, value, history=False), text
    assert shown_as_current(text, value), text


def test_a_history_word_alone_is_still_stale_without_allow_history():
    """Never loosen anything else: «раньше …» / "earlier …, wrong" is not a negation."""
    for text in ("Раньше Тверь, теперь Москва.", "Earlier note said Tver - wrong."):
        assert shown_as_current(text, TVER, history=False), text
        assert not shown_as_current(text, TVER), text


def _obs(text: str) -> Observation:
    return Observation(level="e2e", primary=text, answer=text)


def test_graders_must_not_include_and_forbidden_read_polarity():
    exp = Expect.model_validate({
        "behaviour": "answer", "must_not_include": [KINZA],
        "forbidden": [{"values": [KINZA], "channels": ["answer"]}],
    })  # fmt: skip
    for text, ok in (("Сегодня без кинзы.", True), ("Не забудь кинзу.", False)):
        got = check_exclude(_obs(text), exp.must_not_include) + check_forbidden(_obs(text), exp)
        assert [c.status for c in got] == [("pass" if ok else "fail")] * 2, (text, got)


def test_bench_view_judge_reads_polarity_like_the_graders():
    p = Probe.model_validate({
        "id": "p1", "at": "2026-05-12T09:00", "level": "memory", "ask": ["Где я живу?"],
        "expect": {"must_include": ["denver"], "must_not_include": [BOULDER], "forbidden": [
            {"values": [BOULDER], "channels": ["core", "basket"]},
        ]},
    })  # fmt: skip
    assert judge_view(p, "Lives in Denver, not Boulder anymore.")[:2] == (True, False)
    assert judge_view(p, "Lives in Denver. Boulder, not Denver, was the office.")[1] is True
    assert judge_view(p, "Lives in Denver; earlier note said Boulder.")[1] is True
