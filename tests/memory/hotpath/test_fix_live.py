"""A correction the user makes live finds and retires the record it corrects (§6.1):
a tie breaks by the record that holds the value the user retires; times, weekdays and English
contrasts; a correction with no explicit target finds its record by subject and value kind.
Synthetic names and wordings."""

from __future__ import annotations

import pytest

from koob.memory import api, entities
from koob.memory.model import Alias, AliasKind, AuthorTier, FKind
from koob.memory.writers import hot

from ..races._kit import fact, person, seed, status_of, ulid
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


def _trip_world(conn):
    tim = person("Тимофей")
    stay = fact(
        "stay",
        tim,
        "other",
        "chess tournament in Tver",
        "Timofey is at a chess tournament in Tver this week.",
        "на турнире по шахматам в Твери",
        fkind=FKind.STATE,
    ).model_copy(update={"valid_to": "2026-06-12"})
    back = fact(
        "back",
        tim,
        "other",
        "Friday, 12 June",
        "Timofey will come back on Friday, 12 June.",
        "пятница, 12 июня",
        fkind=FKind.PLAN,
        value_norm="2026-06-12",
    ).model_copy(update={"valid_from": "2026-06-12"})
    seed(conn, tim, stay, back)
    _named(conn, tim, "Тимофей")
    return tim, stay, back


def test_tie_breaks_by_the_record_holding_the_retired_value(mem):
    _tim, stay, back = _trip_world(mem)
    msg = "Нет, Тимофей приезжает не в пятницу, а в субботу, 13-го: турнир продлили."
    r = api.fix(msg, msg, turn=turn_of(msg, n=2), because_quote=msg, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, back.id) == "superseded"
    assert status_of(mem, stay.id) == "active"


def test_english_tie_breaks_the_same_way(mem):
    _tim, stay, back = _trip_world(mem)
    msg = "Actually Timofey gets back Saturday the 13th, not Friday - the tournament ran long."
    r = api.fix(msg, msg, turn=turn_of(msg, n=2), because_quote=msg, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, back.id) == "superseded"
    assert status_of(mem, stay.id) == "active"


def test_tie_never_resolves_onto_someone_elses_claim(mem):
    """The record that holds the retired value is another person's claim, not the user's: no
    tie-break onto it."""
    tim = person("Тимофей")
    liza = person("Лиза")
    stay = fact(
        "stay2",
        tim,
        "other",
        "chess tournament in Tver",
        "Timofey is at a chess tournament in Tver this week.",
        "на турнире по шахматам в Твери",
        fkind=FKind.STATE,
    )
    back = fact(
        "back2",
        tim,
        "other",
        "Friday",
        "Liza says Timofey comes back on Friday.",
        "Тимофей вернётся в пятницу",
        tier=AuthorTier.OTHER,
        speaker=liza,
        fkind=FKind.PLAN,
    )
    seed(mem, tim, liza, stay, back)
    _named(mem, tim, "Тимофей")
    msg = "Нет, Тимофей приезжает не в пятницу, а в субботу: турнир продлили."
    api.fix(msg, msg, turn=turn_of(msg, n=2), because_quote=msg, store=mem)
    assert status_of(mem, back.id) == "active"


@pytest.mark.parametrize(
    ("words", "old", "new"),
    [
        ("Нет, приём не в 14:20, а в 15:00.", "в 14:20", "в 15:00"),
        ("Встреча в 9:30, а не в 10:15 — перенесли.", "в 10:15", "Встреча в 9:30"),
        ("Сдача не 12.06, а 13.06.", "12.06", "13.06"),
        ("The call is at 4:45, not 3:30 - they moved it.", "3:30", "4:45"),
        ("Pick it up by Thursday, not Friday.", "Friday", "Thursday"),
        ("It is not Monday but Tuesday the 9th.", "Monday", "Tuesday the 9th"),
        ("Не в пятницу, а в субботу, тринадцатого.", "в пятницу", "в субботу"),
    ],
)
def test_contrast_forms(words, old, new):
    c = hot.contrast(words)
    assert c is not None and (c.old, c.new) == (old, new)


@pytest.mark.parametrize(
    "words",
    ["It is not ready yet.", "Dad not coming tonight, sadly.", "I'm not sure, honestly."],
)
def test_negated_verb_is_no_contrast(words):
    assert hot.contrast(words) is None


@pytest.mark.parametrize(
    ("words", "new"),
    [
        ("hold on, the recital got pushed - it's thu 11/19 at 6, same hall", "thu 11/19 at 6"),
        ("the standup moved to 9:15", "9:15"),
        ("it's Friday now", "Friday"),
        ("Созвон перенесли на четверг, в 18:00", "четверг"),
        ("теперь в 16:30", "в 16:30"),
    ],
)
def test_shifted_names_the_new_value(words, new):
    assert hot.shifted(words) == new


def test_shift_without_target_finds_the_record_it_moves(mem):
    q = "remember that my violin recital is next thu the 12th at 6, hall B at the arts center"
    r = api.remember(q, turn=turn_of(q, n=1), because_quote=q, store=mem)
    assert r.outcome == "committed"
    msg = "hold on, the recital got pushed - it's fri 11/13 at 6, same hall"
    f = api.fix(msg, msg, turn=turn_of(msg, n=2), because_quote=msg, store=mem)
    assert f.outcome == "committed", f.line
    assert status_of(mem, r.record_ids[0]) == "superseded"


def _lesson_world(conn):
    vera = person("Вера")
    lesson = fact(
        "lesson",
        vera,
        "other",
        "Monday",
        "Vera's piano lesson is on Monday.",
        "у Веры урок фортепиано в понедельник",
        fkind=FKind.PLAN,
    )
    likes = fact(
        "likes",
        vera,
        "other",
        "Chopin",
        "Vera loves playing Chopin.",
        "Вера обожает Шопена",
    )
    seed(conn, vera, lesson, likes)
    _named(conn, vera, "Вера")
    return vera, lesson, likes


@pytest.mark.parametrize(
    "msg", ["Нет, у Веры это в среду.", "Actually for Вера it's Wednesday now."]
)
def test_no_target_finds_the_record_by_subject_and_value_kind(mem, msg):
    """The user's words name only the person and the new day: the one record of hers that holds
    a day is the one being corrected; the one without a day stays."""
    _vera, lesson, likes = _lesson_world(mem)
    r = api.fix(msg, msg, turn=turn_of(msg, n=2), because_quote=msg, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, lesson.id) == "superseded"
    assert status_of(mem, likes.id) == "active"


def test_no_target_and_no_value_kind_finds_nothing(mem):
    _lesson_world(mem)
    msg = "Нет, у Веры всё иначе."
    r = api.fix(msg, msg, turn=turn_of(msg, n=2), because_quote=msg, store=mem)
    assert r.outcome in ("rejected", "ask_first")


def _core(conn) -> str:
    from datetime import date

    from koob.memory.recall import core_view

    return "\n".join(s.text for s in core_view.compute(conn, today=date(2026, 6, 9)))


def test_core_never_shows_the_retired_day_as_a_bound(mem):
    """After «не в пятницу, а в субботу, 13-го» the stay's «until 2026-06-12» is the old plan's
    date: the core drops the bound, keeps the record and the new plan."""
    _tim, stay, back = _trip_world(mem)
    assert "2026-06-12" in _core(mem)
    msg = "Нет, Тимофей приезжает не в пятницу, а в субботу, 13-го: турнир продлили."
    r = api.fix(msg, msg, turn=turn_of(msg, n=2), because_quote=msg, store=mem)
    assert r.outcome == "committed", r.line
    core = _core(mem)
    assert "2026-06-12" not in core and "12 июня" not in core
    assert "chess tournament" in core and "13-го" in core


def test_time_only_fix_keeps_the_date_bound(mem):
    """A fix that moves only the time (17:05, not 16:30) retires no date for that person."""
    tim = person("Тимофей")
    stay = fact(
        "stay3",
        tim,
        "other",
        "visiting grandma in Tver",
        "Timofey is visiting his grandma in Tver.",
        "у бабушки в Твери",
        fkind=FKind.STATE,
    ).model_copy(update={"valid_to": "2026-06-12"})
    train = fact(
        "train",
        tim,
        "other",
        "12 June 16:30",
        "Timofey's train arrives on 12 June at 16:30.",
        "поезд 12 июня в 16:30",
        fkind=FKind.PLAN,
    ).model_copy(update={"valid_from": "2026-06-12"})
    seed(mem, tim, stay, train)
    _named(mem, tim, "Тимофей")
    msg = "Тимофей: поезд приходит в 17:05, not 16:30 - shifted the timetable."
    r = api.fix(msg, msg, turn=turn_of(msg, n=2), because_quote=msg, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, train.id) == "superseded"
    core = _core(mem)
    assert "until 2026-06-12" in core and "16:30" not in core
