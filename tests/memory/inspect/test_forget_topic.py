"""«забудь всё про <тему>» ("forget everything about <topic>") resolves to every record and raw
message of that topic (the user's words are Russian, the records English), «всё» ("everything")
takes all of them, a question comes only for genuinely different things, a newer value of the
slot is never taken (A14); a committed fix retires the value it corrects.

Fictional people and codes only."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from koob.memory import api, forget
from koob.memory.clock import ManualClock
from koob.memory.model import OWNER_ENTITY_ID, AuthorTier

from ._world import (
    DIMA,
    LIVES_OWN,
    MOVE,
    SASHA_K,
    SASHA_K_WORK,
    add,
    fact,
    person,
    raw,
    second_sasha,
    status,
    turn,
    world,
)

ALYA, MARK, KOSTYA, LERA, VOVA = "ent-alya", "ent-mark", "ent-kostya", "ent-lera", "ent-vova"
SAVE, MONTHLY, DEPOSIT, CLASS, FLAT = "f-save", "f-monthly", "f-deposit", "f-class", "f-flat"
FIGHT, NICK, PROTO, SUSP, TALK, CHESS = (
    "f-fight",
    "f-nick",
    "f-proto",
    "f-susp",
    "f-talk",
    "f-chess",
)
VAC, MOCK, TRIP, VOVA_MOVE = "f-vac", "f-mock", "f-trip", "f-vova-move"
T = "2026-10-12T08:{:02d}:00.000000Z"


@pytest.fixture
def clock():
    clk = ManualClock(datetime(2026, 10, 12, 9, 0, tzinfo=UTC))
    forget.wire(clock=clk)
    yield clk
    forget.reset_wiring()


def _forget(mem, words, *, n=7, **kw):
    """The eval harness's shape: the user's whole message as target and quote."""
    return api.forget(words, turn=turn(words, n=n), because_quote=words, store=mem, **kw)


def _raw_status(mem, rid):
    return mem.execute("SELECT status FROM raw_chunks WHERE id = ?", (rid,)).fetchone()[0]


def _live_raw_with(mem, needle):
    return [
        r[0]
        for r in mem.execute("SELECT text FROM raw_chunks WHERE status = 'active'")
        if needle.casefold() in (r[0] or "").casefold()
    ]


def _with(rec, **kw):
    return rec.model_copy(update=kw)


def savings(mem):
    """Savings with Alya (three facts from one message, the user's follow-up question and the Cube's
    reply that repeats the deposit code), Mark's class right after, Dima's flat a week ago."""
    raw(mem, "r-s1", "Мы с Алей решили копить на квартиру: откладываем 300 тысяч в месяц"
        " на депозит «ЮУАЦФМШГ-7311» (в приложении QWZXRT-7311).",
        chat_id="cube", turn_id="t-s1", source_kind="owner", ts=T.format(0))  # fmt: skip
    add(mem, person(ALYA, "Аля"))
    for fid, rel, value, en, quote in (
        (SAVE, "plans_to", "save up for an apartment",
         "The owner and Alya decided to save up for an apartment", "решили копить на квартиру"),
        (MONTHLY, "other", "300 thousand per month",
         "They put aside 300 thousand a month into the deposit", "откладываем 300 тысяч в месяц"),
        (DEPOSIT, "other", "ЮУАЦФМШГ-7311", "The owner has the deposit ЮУАЦФМШГ-7311",
         "на депозит «ЮУАЦФМШГ-7311»"),
    ):  # fmt: skip
        add(mem, fact(fid, OWNER_ENTITY_ID, rel, value, en, quote, raw="r-s1"))
    raw(mem, "r-s2", "Сколько мы накопим к лету при таких темпах?",
        chat_id="cube", turn_id="t-s2", source_kind="owner", ts=T.format(1))  # fmt: skip
    raw(mem, "r-s3", "К июню выйдет около трёх миллионов на депозите «ЮУАЦФМШГ-7311».",
        chat_id="cube", source_kind="cube", tier=AuthorTier.SYSTEM, ts=T.format(2))  # fmt: skip
    raw(mem, "r-m1", "Марк зовёт в субботу на гончарный мастер-класс, к 11.",
        chat_id="cube", turn_id="t-m1", source_kind="owner", ts=T.format(3))  # fmt: skip
    add(mem, person(MARK, "Марк"))
    add(mem, fact(CLASS, MARK, "other", "Saturday at 11",
                  "Mark invites the owner to a pottery class on Saturday at 11",
                  "Марк зовёт в субботу на гончарный мастер-класс", raw="r-m1"))  # fmt: skip
    raw(mem, "r-d1", "Дима снял квартиру в Самаре.", chat_id="tg:9", source_kind="owner",
        ts="2026-10-05T10:00:00.000000Z")  # fmt: skip
    add(mem, person(DIMA, "Дима"))
    add(mem, fact(FLAT, DIMA, "other", "a flat in Samara", "Dima rents a flat in Samara",
                  "Дима снял квартиру в Самаре", raw="r-d1"))  # fmt: skip


def test_everything_about_a_topic_takes_its_records_and_messages(mem, clock):
    """inspect-d06: no entity is named; the topic is the savings conversation."""
    savings(mem)
    ask = turn("Сколько мы откладываем на депозит ЮУАЦФМШГ-7311?", n=9)
    assert "7311" in repr(api.recall(ask, store=mem))  # the basket served it before
    res = _forget(mem, "Забудь всё про наши накопления на квартиру.")
    assert res.outcome == "stopped", res
    assert "7311" not in repr(api.recall(ask, store=mem))  # G5: the basket is clean
    assert "7311" not in repr(api.core(thread_id="thread-owner", store=mem))
    assert {SAVE, MONTHLY, DEPOSIT} <= set(res.record_ids)
    assert [status(mem, i) for i in (SAVE, MONTHLY, DEPOSIT)] == ["forgotten"] * 3
    assert {_raw_status(mem, r) for r in ("r-s1", "r-s2", "r-s3")} == {"forgotten"}
    assert _live_raw_with(mem, "7311") == []  # the Cube's reply with the code too
    # another topic in the next message, and a flat from another conversation, stay
    assert status(mem, CLASS) == "active" and _raw_status(mem, "r-m1") == "active"
    assert status(mem, FLAT) == "active" and _raw_status(mem, "r-d1") == "active"
    assert status(mem, ALYA) == "active"  # «всё про накопления» is not «всё про Алю»
    assert "Перестал использовать" in res.line and "накопления" in res.line


def test_a_bare_topic_word_after_filler_forgets_the_topic_forever(mem, clock):
    """«Всё-таки забудь про накопления насовсем»: filler before the verb is not the target;
    the topic word reaches the money facts through the question, the reply and the code."""
    savings(mem)
    res = _forget(mem, "Ладно, всё-таки забудь про накопления насовсем.", forever=True)
    assert res.outcome == "stopped", res
    assert {SAVE, MONTHLY, DEPOSIT} <= set(res.record_ids)
    assert "Удаляю полностью" in res.line
    assert status(mem, CLASS) == "active" and status(mem, FLAT) == "active"


def kostya(mem):
    """The user's story about Kostya's fight (codes inside), the user's question about it, and an
    older, unrelated fact about Kostya."""
    add(mem, person(KOSTYA, "Костя"))
    raw(mem, "r-k0", "Костя ходит на шахматы по средам.", chat_id="cube", turn_id="t-k0",
        source_kind="owner", ts="2026-10-01T10:00:00.000000Z")  # fmt: skip
    add(mem, fact(CHESS, KOSTYA, "other", "chess on Wednesdays", "Kostya plays chess on Wednesdays",
                  "Костя ходит на шахматы по средам", raw="r-k0"))  # fmt: skip
    raw(mem, "r-k1", "Костя подрался в школе с мальчиком по кличке ЖУКОВ-5521 (в сети"
        " QPLMN-5521). Маму вызывали к директору, протокол № 5521-0907. Костю отстранили"
        " на неделю.", chat_id="cube", turn_id="t-k1", source_kind="owner", ts=T.format(10))  # fmt: skip
    add(mem, _with(fact(FIGHT, KOSTYA, "other", "fight at school",
                        "Kostya got into a fight at school with a boy nicknamed ЖУКОВ-5521",
                        "Костя подрался в школе с мальчиком по кличке ЖУКОВ-5521", raw="r-k1"),
                   keywords_ru="драка, школа"))  # fmt: skip
    for fid, value, en, quote in (
        (NICK, "QPLMN-5521", "The boy's online handle is QPLMN-5521", "в сети QPLMN-5521"),
        (PROTO, "протокол № 5521-0907", "The mother was summoned, protocol 5521-0907",
         "протокол № 5521-0907"),
        (SUSP, "a week", "Kostya was suspended for a week", "Костю отстранили на неделю"),
    ):  # fmt: skip
        add(mem, fact(fid, KOSTYA, "other", value, en, quote, raw="r-k1"))
    raw(mem, "r-k2", "Как мне поговорить с Костей про драку, чтобы он не закрылся?",
        chat_id="cube", turn_id="t-k2", source_kind="owner", ts=T.format(11))  # fmt: skip
    add(mem, _with(fact(TALK, OWNER_ENTITY_ID, "other", "talk with Kostya",
                        "The owner wants to talk with Kostya about the fight",
                        "поговорить с Костей про драку", raw="r-k2"),
                   value_entity_id=KOSTYA))  # fmt: skip


@pytest.mark.parametrize(
    "words",
    [
        "Забудь всё, что я тебе рассказывал про Костину драку.",
        "забудь всё про драку Кости",
    ],
)
def test_all_about_a_story_takes_every_item_without_a_question(mem, clock, words):
    """keeps_true-d03: three tied items are one story - «всё» takes them all, and the codes in
    the story's other sentences go with it; an older fact about Kostya stays."""
    kostya(mem)
    res = _forget(mem, words)
    assert res.outcome == "stopped", res
    assert [status(mem, i) for i in (FIGHT, NICK, PROTO, SUSP, TALK)] == ["forgotten"] * 5
    assert _live_raw_with(mem, "5521") == [] and _live_raw_with(mem, "драку") == []
    assert status(mem, CHESS) == "active" and status(mem, KOSTYA) == "active"


def test_unrelated_things_with_one_word_still_ask(mem, clock):
    """No «всё»: two moves of two people from two messages are different things → one
    question; nothing is forgotten."""
    world(mem)
    add(mem, person(VOVA, "Вова"))
    raw(mem, "r-v1", "Вова переезжает в Пермь", chat_id="cube", turn_id="t-v1")
    add(mem, fact(VOVA_MOVE, VOVA, "moves_to", "Perm", "Vova is moving to Perm",
                  "Вова переезжает в Пермь", raw="r-v1"))  # fmt: skip
    res = _forget(mem, "забудь про переезд")
    assert res.outcome == "ask_first", res
    assert f"m:{MOVE}" in res.line and f"m:{VOVA_MOVE}" in res.line
    assert status(mem, MOVE) == "active" and status(mem, VOVA_MOVE) == "active"


def test_a_slot_question_names_the_fact(mem, clock):
    """inspect-d05: «до какого числа Лера в отпуске» names her vacation's end, not the mockups
    from the same message."""
    add(mem, person(LERA, "Лера Пак"))
    raw(mem, "r-l1", "Лера Пак в отпуске до 20 июня, макеты пока делаю я.", chat_id="cube",
        turn_id="t-l1", source_kind="owner")  # fmt: skip
    add(mem, _with(fact(VAC, LERA, "other", "until 20 June", "Lera Pak is on vacation until 20 June",
                        "Лера Пак в отпуске до 20 июня", raw="r-l1"), valid_to="2026-06-20"))  # fmt: skip
    add(mem, fact(MOCK, OWNER_ENTITY_ID, "works_on", "mockups",
                  "The owner does the mockups while Lera Pak is on vacation",
                  "макеты пока делаю я", raw="r-l1"))  # fmt: skip
    res = _forget(mem, "Сотри, до какого числа Лера в отпуске.")
    assert res.outcome == "stopped", res
    assert status(mem, VAC) == "forgotten" and status(mem, MOCK) == "active"


def test_object_less_forget_after_another_sentence_is_not_a_guess(mem, clock):
    """«Всё, обсудили. Забудь насовсем.» names nothing: the earlier sentence is not the
    target, so no record that shares a word with it is forgotten."""
    add(mem, fact("f-mom", OWNER_ENTITY_ID, "other", "discuss with mom",
                  "The owner still has to discuss it with mom", "мне ещё с мамой это обсуждать"))  # fmt: skip
    res = _forget(mem, "Всё, обсудили. Забудь насовсем.", forever=True)
    assert res.outcome == "not_found", res
    assert status(mem, "f-mom") == "active"


def test_describing_word_keeps_everything_about_the_person(mem, clock):
    world(mem)
    res = _forget(mem, "забудь всё про моего коллегу Диму")
    assert res.outcome == "stopped", res
    assert status(mem, "fact-dima-work") == "forgotten" and status(mem, MOVE) == "active"


def _trip(mem):
    add(mem, person(DIMA, "Дима"))
    raw(mem, "r-t1", "Дима Орлов с 3 по 14 июня в командировке в Твери.", chat_id="cube",
        turn_id="t-t1", source_kind="owner", ts=T.format(20))  # fmt: skip
    add(mem, fact(TRIP, DIMA, "lives_in", "Tver", "Dima is on a business trip in Tver in June",
                  "с 3 по 14 июня в командировке в Твери", raw="r-t1"))  # fmt: skip
    words = "Нет, Дима не в Твери, а в Самаре, я спутала."
    raw(mem, "r-t2", words, chat_id="cube", turn_id="turn-5", source_kind="owner", ts=T.format(21))
    return words


def test_a_committed_fix_retires_the_value_it_corrects(mem, clock):
    """«не в Твери, а в Самаре» → the new value is Samara, its text does not carry Tver, the old
    fact is superseded and dated, the fix keeps the user's message as lineage; only Samara is
    current."""
    words = _trip(mem)
    res = api.fix(words, words, turn=turn(words, n=5), because_quote=words, store=mem)
    assert res.outcome == "committed", res
    new = res.record_ids[0]
    row = mem.execute(
        "SELECT value_text, f.text_en, source_raw_id FROM facts f WHERE f.id = ?", (new,)
    ).fetchone()
    assert "Самар" in row[0] and "Твер" not in row[0] and "Твер" not in row[1]
    assert row[2] == "r-t2"
    assert status(mem, TRIP) == "superseded"
    old_to = mem.execute("SELECT valid_to FROM facts WHERE id = ?", (TRIP,)).fetchone()[0]
    assert old_to == "2026-10-12"
    live = mem.execute(
        "SELECT f.value_text, f.text_en FROM facts f JOIN records r ON r.id = f.id"
        " WHERE f.subject_id = ? AND r.status IN ('active', 'archived')",
        (DIMA,),
    ).fetchall()
    assert live and not any(
        "Твер" in (v or "") + (t or "") or "Tver" in (t or "") for v, t in live
    )
    assert "Самар" in res.line and "Твер" not in res.line


def test_topic_forget_never_takes_the_newer_value(mem, clock):
    """A14: «забудь всё про Тверь» takes the Tver fact (history) and its message, never the
    Samara value that superseded it, though the user's correction says «не в Твери»."""
    words = _trip(mem)
    fixed = api.fix(words, words, turn=turn(words, n=5), because_quote=words, store=mem)
    new = fixed.record_ids[0]
    res = _forget(mem, "забудь всё про Тверь", n=8)
    assert res.outcome == "stopped", res
    assert status(mem, TRIP) == "forgotten" and status(mem, new) == "active"
    assert _raw_status(mem, "r-t1") == "forgotten"


def test_a_back_reference_keeps_the_sentence_before_the_verb():
    from koob.memory import commands

    phrase, whole = commands._forget_query("Дима переехал. Забудь его старый адрес.")
    assert "Дима" in phrase and not whole
    phrase, _ = commands._forget_query("Всё, обсудили. Забудь насовсем.")
    assert "обсудили" not in phrase
    _, whole = commands._forget_query("Забудь всё, что я тебе говорила о поездке.")
    assert whole


def test_topic_of_one_namesake_is_hers_alone(mem, clock):
    """«всё про переезд Саши» with two Sashas, only one of them moving: her move only."""
    world(mem)
    second_sasha(mem)
    res = _forget(mem, "забудь всё про переезд Саши")
    assert res.outcome == "stopped", res
    assert status(mem, MOVE) == "forgotten" and status(mem, SASHA_K_WORK) == "active"
    assert status(mem, LIVES_OWN) == "active"  # her other facts are another topic


def test_topic_of_two_namesakes_asks_which_person(mem, clock):
    """Both Sashas move: different people → one question, nothing forgotten."""
    world(mem)
    second_sasha(mem)
    raw(mem, "r-k-move", "Саша К. переезжает в Сочи", chat_id="cube", turn_id="t-km")
    add(mem, fact("f-k-move", SASHA_K, "moves_to", "Sochi", "Sasha K. is moving to Sochi",
                  "Саша К. переезжает в Сочи", raw="r-k-move"))  # fmt: skip
    res = _forget(mem, "забудь всё про переезд Саши")
    assert res.outcome == "ask_first", res
    assert f"m:{SASHA_K}" in res.line
    assert status(mem, MOVE) == "active" and status(mem, "f-k-move") == "active"
