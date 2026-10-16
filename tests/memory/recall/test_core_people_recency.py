"""The always-on core after every sender became an entity.

People are «recent» by the user's interactions (the user wrote to them, about them, or named
them), not by their messages arriving; a retracted record touches no one; an undated one-off is
dated by when the user said it. The recall conftest pins the day to Monday 2026-10-12;
synthetic data only."""

from __future__ import annotations

from datetime import date

import pytest

from koob.memory.model import AuthorTier, FKind, OpKind, Stability
from koob.memory.recall import core_view

from .._ledger import add, commit, entity, fact, op, open_seeded, status

OLD = "2026-05-03T10:00:00.000000Z"
RECENT = "2026-10-11T08:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    yield c
    c.close()


def _ok(res) -> None:
    assert res.ok, res


def _section(conn, name: str, today: date | None = None) -> str:
    return next(s.text for s in core_view.compute(conn, today=today) if s.name == name)


def _raw(conn, rid, text, *, tier="user", chat=None, author=None, at=RECENT):
    conn.execute(
        "INSERT INTO raw_chunks (id, source_kind, chat_id, author_tier, author_entity_id, ts,"
        " text, content_hmac, labels, policy_version, created_at, updated_at)"
        " VALUES (?, 'chat', ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)",
        (rid, chat, tier, author, at, text, f"h-{rid}", at, at),
    )


def _sender(conn, eid, name):
    rec = entity(eid, name, tier=AuthorTier.OTHER).model_copy(update={"recorded_at": RECENT})
    _ok(add(conn, rec))


def test_people_recent_by_his_interactions_not_their_messages(conn):
    # senders whose messages flood in this week; the user never wrote to or about them
    _sender(conn, "e-varvara", "Варвара")
    _sender(conn, "e-timofey", "Тимофей")
    _raw(
        conn,
        "r-v1",
        "Всем привет, кто идёт на стендап?",
        tier="other",
        chat="tg:-500",
        author="e-varvara",
    )
    _raw(
        conn, "r-t1", "Скинул макеты в общий чат", tier="other", chat="tg:-500", author="e-timofey"
    )
    _ok(
        add(
            conn,
            fact(
                "f-tim",
                "e-timofey",
                "works_on",
                "макеты",
                text="Timofey works on mockups",
                tier=AuthorTier.OTHER,
            ),
        )
    )
    # a sender the user answered in their one-to-one chat
    _sender(conn, "e-zlata", "Злата")
    _raw(conn, "r-z1", "Ты сможешь в пятницу?", tier="other", chat="tg:7001", author="e-zlata")
    _raw(conn, "r-z2", "Смогу после обеда", chat="tg:7001")
    # an old contact the user named this week in their own words (a declined form)
    _ok(add(conn, entity("e-gordey", "Гордей").model_copy(update={"recorded_at": OLD})))
    _raw(conn, "r-g1", "Надо вернуть Гордею дрель до выходных")
    # an old contact with a fresh fact in the user's words
    _ok(add(conn, entity("e-yarik", "Ярослав").model_copy(update={"recorded_at": OLD})))
    _ok(add(conn, fact("f-yar", "e-yarik", "moved_to", "Тверь", text="Yaroslav moved to Tver")))
    people = _section(conn, "people")
    assert "Злата" in people and "Гордей" in people and "Ярослав" in people
    assert "Варвара" not in people and "Тимофей" not in people


def test_a_retracted_record_keeps_no_one_in_the_core(conn):
    _ok(add(conn, entity("e-filipp", "Филипп").model_copy(update={"recorded_at": OLD})))
    _ok(add(conn, fact("f-fil", "e-filipp", "ran", "бенчмарк", text="Filipp ran the benchmark")))
    assert "Филипп" in _section(conn, "people")
    _ok(commit(conn, op(OpKind.RETRACT, target="f-fil")))
    assert status(conn, "f-fil") == "retracted"
    assert "Филипп" not in _section(conn, "people")


def test_undated_one_off_is_dated_by_when_he_said_it(conn):
    old = fact(
        "f-call",
        "self",
        "cancelled",
        "созвон",
        text="The sync with the vendor was cancelled",
        fkind=FKind.EVENT,
        stability=Stability.ONE_OFF,
    )
    _ok(add(conn, old.model_copy(update={"recorded_at": OLD})))
    _ok(
        add(
            conn,
            fact(
                "f-gift",
                "self",
                "buys",
                "цветы",
                text="Buys flowers on the way home",
                fkind=FKind.PLAN,
                stability=Stability.ONE_OFF,
            ),
        )
    )
    about = _section(conn, "about_you")
    assert "vendor" not in about
    assert "Buys flowers" in about  # said this week: it may still be about today


def test_habit_names_its_day_in_its_own_words_or_its_message(conn):
    """A routine with no period field: its day comes from its words or the message it rests
    on; on a Monday a Thursday routine is not about any turn, on Wednesday evening it is."""
    _raw(conn, "r-pool", "По четвергам вожу Мирона в бассейн к семи, сам жду в машине с книгой.")
    _ok(
        add(
            conn,
            fact(
                "f-pool",
                "self",
                "drives_to",
                "бассейн",
                text="Drives Miron to the pool on Thursdays",
                stability=Stability.HABIT,
                source_raw_id="r-pool",
            ),
        )
    )
    _ok(
        add(
            conn,
            fact(
                "f-book",
                "self",
                "waits",
                "в машине",
                text="Waits in the car with a book",
                stability=Stability.HABIT,
                source_raw_id="r-pool",
            ),
        )
    )
    _ok(
        add(
            conn,
            fact(
                "f-run",
                "self",
                "runs",
                "утром",
                text="Runs every morning",
                stability=Stability.HABIT,
            ),
        )
    )
    monday = _section(conn, "about_you")
    assert "pool" not in monday and "car with a book" not in monday
    assert "Runs every morning" in monday  # no day named: every day
    wednesday = _section(conn, "about_you", today=date(2026, 10, 14))
    assert "pool" in wednesday and "car with a book" in wednesday


def test_a_person_he_quotes_is_his_interaction(conn):
    _ok(add(conn, entity("e-milena", "Милена").model_copy(update={"recorded_at": OLD})))
    said = fact(
        "f-said",
        "self",
        "advised",
        "PostgreSQL",
        text="Milena advised PostgreSQL for the ledger",
        speaker_entity_id="e-milena",
    )
    _ok(add(conn, said))
    assert "Милена" in _section(conn, "people")


def test_a_shared_first_name_goes_to_the_one_his_message_names(conn):
    for eid, name in (
        ("e-t1", "Тоня Лисина"),
        ("e-t2", "Тоня Орлова"),
        ("e-l1", "Лёва"),
        ("e-l2", "Лёва Гусев"),
    ):
        _ok(add(conn, entity(eid, name).model_copy(update={"recorded_at": OLD})))
    _raw(conn, "r-ton", "Надо перезвонить Тоне Лисиной насчёт ключей")
    _raw(conn, "r-lev", "Лёва опять опоздал")
    people = _section(conn, "people")
    assert "Тоня Лисина" in people and "Тоня Орлова" not in people
    assert "Лёва Гусев" in people  # «Лёва» alone names no one in particular: both stay
