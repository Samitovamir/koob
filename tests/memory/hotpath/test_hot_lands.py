"""What the user says live lands in the same turn (§6.1): the Cube's own promise in its reply,
the user's «запомни» and the user's correction with a new value commit on the hot path when no
tool call took them, instead of a freeze waiting for a worker; a correction said inside a
running task applies to the records it names; a cancel sent as a correction cancels the plan
it names. Synthetic names and wordings."""

from __future__ import annotations

from datetime import datetime, timedelta

from koob.memory import api, entities
from koob.memory.model import Alias, AliasKind, AuthorTier, FKind, TurnOrigin

from ..races._kit import TS0, fact, person, seed, status_of, ulid
from ._world import turn_of


def _named(conn, ent, alias: str, script: str = "cyrl") -> None:
    entities.bind_alias(
        conn,
        Alias(
            id=ulid(f"alias:{alias}"),
            entity_id=ent.id,
            alias=alias,
            alias_norm=entities.normalize_alias(alias),
            script=script,
            kind=AliasKind.NAME,
            tier=AuthorTier.USER,
            confirmed=True,
        ),
        tier=AuthorTier.USER,
        platform=False,
    )


def _street_world(conn):
    arseny, vera = person("Арсений"), person("Вера")
    home = fact(
        "arseny-home", arseny, "lives_in", "Lesnaya street", "Arseny lives on Lesnaya street.",
        "Арсений живёт на Лесной",
    )  # fmt: skip
    hers = fact(
        "vera-home", vera, "lives_in", "Lesnaya street", "Vera lives on Lesnaya street.",
        "Вера живёт на Лесной", tier=AuthorTier.OTHER, speaker=vera,
    )  # fmt: skip
    seed(conn, arseny, vera, home, hers)
    _named(conn, arseny, "Арсений")
    _named(conn, arseny, "Arseny", "latn")
    _named(conn, vera, "Вера")
    _named(conn, vera, "Vera", "latn")
    return home, hers


def _promises(conn) -> list[tuple[str, str, str]]:
    """The Cube's open loops: (text, owner, due)."""
    return list(
        conn.execute(
            "SELECT l.text_en, l.owner, l.due FROM loops l JOIN records r ON r.id = l.id"
            " WHERE r.status = 'active'"
        )
    )


def _fact_texts(conn) -> list[tuple[str, str]]:
    return list(
        conn.execute(
            "SELECT f.text_en || ' ' || coalesce(f.value_text, ''), r.author_tier FROM facts f"
            " JOIN records r ON r.id = f.id WHERE r.status = 'active'"
        )
    )


# ---------------------------------------------------------------- the Cube's own promise


def _weekday_after(clock: str, n: int) -> str:
    d = datetime.fromisoformat(clock.replace("Z", "+00:00")).date()
    return (d + timedelta(days=(n - d.weekday()) % 7 or 7)).isoformat()


def test_promise_in_reply_lands_as_the_cubes_loop_ru(mem):
    words = "В пятницу защищаю курсовую у Громова. Пни меня в четверг, если не скину слайды."
    reply = "Хорошо: в четверг вечером спрошу, готовы ли слайды к защите у Громова."
    net = api.after_turn_check(turn_of(words, n=2), reply, store=mem)
    assert not net.fired
    [(text, owner, due)] = _promises(mem)
    assert "слайды" in text and owner == "cube" and due == _weekday_after(TS0, 3)


def test_promise_in_reply_lands_as_the_cubes_loop_en(mem):
    words = "my lab writeup for Okafor is due monday, nudge me if i forget"
    reply = "Will do - I'll ping you Sunday evening about the Okafor lab writeup."
    net = api.after_turn_check(turn_of(words, n=2), reply, store=mem)
    assert not net.fired
    [(text, owner, due)] = _promises(mem)
    assert "writeup" in text and owner == "cube" and due == _weekday_after(TS0, 6)
    # the same turn checked again writes nothing more
    api.after_turn_check(turn_of(words, n=2), reply, store=mem)
    assert len(_promises(mem)) == 1


def test_promise_with_a_date_lands_on_that_day(mem):
    words = "Двадцатого сдаю отчёт Мельникову, подстрахуй"
    reply = "Принято: 19 ноября проверю, отправлен ли отчёт Мельникову."
    api.after_turn_check(turn_of(words, n=2), reply, store=mem)
    [(_text, _owner, due)] = _promises(mem)
    assert due.endswith("-11-19")


def test_promise_with_no_day_lands_nothing(mem):
    """No day to hold it by: the fresh-raw lane keeps their words findable instead."""
    words = "посмотри потом мои заметки по проекту"
    api.after_turn_check(turn_of(words, n=2), "Хорошо, уточню детали по проекту.", store=mem)
    assert _promises(mem) == []


def test_contentless_promise_still_waits(mem):
    net = api.after_turn_check(turn_of("ага", n=2), "Ладно, сделаю позже.", store=mem)
    assert net.fired
    assert _promises(mem) == []


# ---------------------------------------------------------------- «запомни» with no tool call


def test_remember_cue_lands_without_a_tool_call(mem):
    words = "Запомни: у Арсения размер обуви 43."
    net = api.after_turn_check(turn_of(words, n=2), "Хорошо.", store=mem)
    assert not net.fired
    rows = _fact_texts(mem)
    assert any("43" in t and tier == "user" for t, tier in rows), rows


def test_english_remember_cue_lands(mem):
    words = "remember that the chess club meets on Thursdays at the library now"
    net = api.after_turn_check(turn_of(words, n=2), "Sure.", store=mem)
    assert not net.fired
    assert any("Thursdays" in t for t, _tier in _fact_texts(mem))


def test_remember_inside_another_verb_lands_nothing(mem):
    """«скажи Олегу запомнить …» asks someone else: no write in the user's name."""
    words = "скажи Олегу запомнить номер рейса"
    api.after_turn_check(turn_of(words, n=2), "Скажу.", store=mem)
    assert _fact_texts(mem) == []


def test_remember_in_a_tainted_turn_is_not_his_claim(mem):
    words = "запомни, что там написано про дедлайн"
    api.after_turn_check(turn_of(words, n=2, tainted=True), "Хорошо.", store=mem)
    assert all(tier != "user" for _t, tier in _fact_texts(mem))


# ---------------------------------------------------------------- correction with no tool call


def test_correction_with_a_new_value_lands_in_the_turn(mem):
    home, hers = _street_world(mem)
    words = "Нет, Арсений живёт не на Лесной, а на Садовой."
    net = api.after_turn_check(turn_of(words, n=2), "Понял.", store=mem)
    assert not net.fired
    assert status_of(mem, home.id) == "superseded"
    assert status_of(mem, hers.id) == "active"  # never another person's claim


def test_english_correction_lands_in_the_turn(mem):
    home, hers = _street_world(mem)
    words = "actually Arseny lives on Sadovaya street, not Lesnaya"
    net = api.after_turn_check(turn_of(words, n=2), "Got it.", store=mem)
    assert not net.fired
    assert status_of(mem, home.id) == "superseded"
    assert status_of(mem, hers.id) == "active"


def test_tainted_correction_is_not_landed_by_the_net(mem):
    home, _hers = _street_world(mem)
    words = "Нет, Арсений живёт не на Лесной, а на Садовой."
    net = api.after_turn_check(turn_of(words, n=2, tainted=True), "Понял.", store=mem)
    assert net.fired
    assert status_of(mem, home.id) == "active"


# ---------------------------------------------------------------- correction inside a task


def test_correction_in_a_running_task_applies(mem):
    home, hers = _street_world(mem)
    words = "стоп, Арсений уже не на Лесной, а на Садовой — учти в письме"
    turn = turn_of(words, n=3, origin=TurnOrigin.TASK)
    r = api.fix(words, words, turn=turn, because_quote=words, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, home.id) == "superseded"
    assert status_of(mem, hers.id) == "active"
    new = mem.execute("SELECT author_tier FROM records WHERE id = ?", (r.record_ids[0],))
    assert new.fetchone()[0] == "user"


def test_english_correction_in_a_running_task_applies(mem):
    home, _hers = _street_world(mem)
    words = "wait, Arseny is on Sadovaya street now, not Lesnaya - fix it in the invite"
    turn = turn_of(words, n=3, origin=TurnOrigin.TASK)
    r = api.fix(words, words, turn=turn, because_quote=words, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, home.id) == "superseded"


def test_tainted_task_correction_from_the_page_still_asks(mem):
    """A correction that is only in the page the user read is not the user's: ask, retire nothing."""
    home, _hers = _street_world(mem)
    words = "глянь, что там пишут про Арсения"
    page = "Арсений живёт не на Лесной, а на Садовой"
    turn = turn_of(words, n=3, origin=TurnOrigin.TASK, tainted=True)
    r = api.fix(page, page, turn=turn, because_quote=page, store=mem)
    assert r.outcome == "ask_first"
    assert status_of(mem, home.id) == "active"


# ---------------------------------------------------------------- a cancel sent as a correction


def _plan_world(conn):
    vera = person("Вера")
    dinner = fact(
        "dinner", vera, "plan", "dinner with Vera on Friday",
        "The owner plans dinner with Vera on Friday.", "ужин с Верой в пятницу",
        fkind=FKind.PLAN,
    )  # fmt: skip
    seed(conn, vera, dinner)
    _named(conn, vera, "Вера")
    _named(conn, vera, "Vera", "latn")
    return dinner


def test_cancel_tagged_as_correction_cancels_the_plan(mem):
    dinner = _plan_world(mem)
    words = "Отмени ужин с Верой в пятницу, она заболела."
    r = api.fix(words, words, turn=turn_of(words, n=2), because_quote=words, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, dinner.id) not in ("active", "superseded")


def test_english_cancel_tagged_as_correction_cancels_the_plan(mem):
    dinner = _plan_world(mem)
    words = "call off the dinner with Vera on Friday"
    r = api.fix(words, words, turn=turn_of(words, n=2), because_quote=words, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, dinner.id) not in ("active", "superseded")
