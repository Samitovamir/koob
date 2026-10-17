"""The user's forget or cancel inside real speech: an order said in passing in a longer message,
a new value after it in the same message, a cancel said without a cancel verb, the slot named
by an English question word, and «забудь это» right after the items were shown. Synthetic
names and wordings, Russian and English."""

from __future__ import annotations

import pytest

from koob.memory import api, commands
from koob.memory.recall import safe_text
from koob.memory.writers import hot

from . import _seed as s
from .test_forget_orders import _status, _turn, _world

clock = s.clock  # fixture

_ALL = ("f-boat", "ep-boat", "f-dog", "ep-meet", "f-bakery")


def _forget(mem, words: str, n: int = 1):
    return api.forget(words, turn=_turn(words, n), because_quote=words, store=mem)


def _only(mem, *gone: str) -> None:
    for rid in _ALL:
        assert _status(mem, rid) == ("forgotten" if rid in gone else "active"), rid


@pytest.mark.parametrize(
    "words",
    [
        "Сегодня полдня просидела в библиотеке, голова гудит. И да, про лодку Арсения забудь, "
        "я ошиблась. Вечером созвонимся.",
        "Rough morning, the bus was late again. Btw forget what I told you about Arseny's boat, "
        "I mixed it up. See you later.",
    ],
)
def test_order_in_passing_takes_only_its_own_sentence(mem, clock, words):
    _world(mem)
    res = _forget(mem, words)
    assert res.outcome == "stopped", res.line
    _only(mem, "f-boat", "ep-boat")


@pytest.mark.parametrize(
    "words",
    [
        "Забудь про пекарню — Вероника теперь работает в цветочном.",
        "Forget the bakery thing, Veronika is at a flower shop now.",
        "Забудь прежнее место работы Вероники, теперь она в цветочном.",
    ],
)
def test_new_value_after_the_forget_is_not_its_target(mem, clock, words):
    _world(mem)
    res = _forget(mem, words)
    assert res.outcome == "stopped", res.line
    _only(mem, "f-bakery")


@pytest.mark.parametrize(
    "words",
    [
        "Отбой по встрече с Вероникой.",
        "Встреча с Вероникой в пятницу уже не нужна.",
        "Пятничная встреча с Вероникой неактуальна.",
        "Встречу с Вероникой можно вычеркнуть.",
        "never mind the Veronika meeting",
        "scrap Friday",
        "nix the meeting with Veronika",
    ],
)
def test_cancel_said_without_a_cancel_verb_retires_the_plan(mem, clock, words):
    _world(mem)
    res = _forget(mem, words)
    assert res.outcome == "stopped", res.line
    _only(mem, "ep-meet")


def test_said_cancel_stays_out_of_the_safety_net_and_fenced_reads():
    """Outside the forget call «не надо» and "never mind" are ordinary speech."""
    for words in ("never mind the playlist, just put on lofi", "Не надо, я сама донесу"):
        assert safe_text.order_kind(words) is None, words
        assert "forget" not in hot.detect(_turn(words), ""), words
    assert commands.said_cancel("never mind the dentist")
    assert not commands.said_cancel("Отмени встречу, не надо уже")  # the plain verb leads


@pytest.mark.parametrize(
    "words", ["forget where Veronika works", "forget which bakery Veronika works at"]
)
def test_english_question_word_names_the_slot(mem, clock, words):
    _world(mem)
    res = _forget(mem, words)
    assert res.outcome == "stopped", res.line
    _only(mem, "f-bakery")


def _shown(mem, clock, turn_id: str, *ids: str) -> None:
    for rid in ids:
        mem.execute(
            "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id,"
            " via, created_at) VALUES ('thread-forget', 'session-forget', ?, 'fact', 1, ?, 'tool', ?)",
            (rid, turn_id, s.now_iso(clock)),
        )
    mem.commit()


@pytest.mark.parametrize("words", ["Забудь это.", "drop that", "Не держи это в голове"])
def test_object_less_forget_takes_what_was_just_shown(mem, clock, words):
    _world(mem)
    _shown(mem, clock, "turn-0", "f-bakery")
    clock.advance(20)
    res = _forget(mem, words, 2)
    assert res.outcome == "stopped", res.line
    _only(mem, "f-bakery")


def test_a_long_list_shown_is_not_this(mem, clock):
    """After more than a few items, «забудь это» names nothing in particular."""
    _world(mem)
    _shown(mem, clock, "turn-0", *_ALL)
    clock.advance(20)
    res = _forget(mem, "Забудь это.", 2)
    assert res.outcome != "stopped"
    _only(mem)


def test_correction_call_with_said_cancel_cancels_the_plan(mem, clock):
    _world(mem)
    words = "Встреча с Вероникой в пятницу отменяется."
    w = api.fix(words, words, turn=_turn(words), because_quote=words, store=mem)
    assert w.outcome == "committed", w.line
    _only(mem, "ep-meet")


def test_said_cancel_with_a_new_day_is_a_move_not_a_cancel():
    assert commands.brings_new_value("Встречу в пятницу не надо, давай в понедельник")
    assert commands.brings_new_value("never mind Friday, make it Monday instead")
    assert not commands.brings_new_value("Встреча в пятницу отменяется")
