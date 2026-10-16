"""The always-on core shows a thing only when it applies to any turn today.

Regression tests for the stale wordings of the design notes
(c3, knows-d02/p4, understands-d01/d02/d03, policy_leaks-d02/p2-p3, knows-d01/p3). The test
clock is Monday 2026-10-12; synthetic data only."""

from __future__ import annotations

from datetime import date

import pytest

from koob.memory import api, policy
from koob.memory.model import FKind, Label, Loop, Polarity, Stability, TurnCtx, TurnOrigin
from koob.memory.recall import core_view

from .._ledger import add, entity, fact, open_seeded, pref, ts

CLOCK = "2026-10-12T09:00:00.000000Z"
OLD = "2026-02-12T10:00:00.000000Z"  # eight months before the test clock


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    yield c
    c.close()


def _ok(res) -> None:
    assert res.ok, res


def _old(rec):
    return rec.model_copy(update={"recorded_at": OLD})


def _text(conn, today: date | None = None) -> str:
    return core_view.render(core_view.compute(conn, today=today))


def _section(conn, name: str, today: date | None = None) -> str:
    return next(s.text for s in core_view.compute(conn, today=today) if s.name == name)


def test_people_only_when_recently_active(conn):
    """c3-d01/p1, c3-d02/p2, knows-d02/p4: a «who was that» question no longer gets every
    candidate from months ago; the people of this week stay."""
    _ok(add(conn, _old(entity("e-mark", "Марк"))))
    _ok(add(conn, _old(fact("f-mark", "e-mark", "has", "студия", text="Mark has a studio"))))
    _ok(add(conn, entity("e-ilya", "Илья")))
    _ok(add(conn, fact("f-ilya", "e-ilya", "has", "мастерская", text="Ilya has a workshop")))
    people = _section(conn, "people")
    assert "Илья" in people and "workshop" in people
    assert "Марк" not in people and "studio" not in people
    # a week later Ilya is no longer recent either; nothing refills the room
    assert _section(conn, "people", today=date(2026, 10, 25)) == ""


def test_preference_for_someone_else_stays_out(conn):
    """understands-d01/p2: the client style is not the user's style for everyone; a rule for
    Kostya is not about the user; the general dislike of long messages stays."""
    _ok(
        add(
            conn,
            pref(
                "p-formal",
                "self",
                "writes",
                "на «вы», без смайликов",
                text="Formal vy",
                applicability={"audience": "клиенты Polka"},
            ),
        )
    )
    _ok(
        add(
            conn,
            pref(
                "p-kostya",
                "self",
                "explains",
                "совсем просто",
                text="Simple for Kostya",
                applicability={"for_whom": "e-kostya"},
            ),
        )
    )
    _ok(
        add(
            conn,
            pref(
                "p-long",
                "self",
                "dislikes",
                "длинные сообщения",
                text="Does not like long messages",
            ),
        )
    )
    _ok(
        add(
            conn,
            pref(
                "p-mine",
                "self",
                "likes",
                "чай",
                text="Likes tea",
                applicability={"for_whom": "self", "audience": "все"},
            ),
        )
    )
    about = _section(conn, "about_you")
    assert "Does not like long messages" in about and "Likes tea" in about
    assert "Formal vy" not in about and "Polka" not in about and "клиенты" not in about
    assert "Kostya" not in about


def test_todays_override_beats_the_standing_rule(conn):
    """understands-d02/p1-p2: «сегодня не хочу кинзу» hides «обожаю кинзу» today only."""
    _ok(
        add(
            conn,
            pref(
                "p-love",
                "self",
                "likes",
                "кинза",
                text="Loves cilantro",
                stability=Stability.LASTING,
            ),
        )
    )
    _ok(
        add(
            conn,
            pref(
                "p-more",
                "self",
                "wants",
                "побольше кинзы",
                text="Plenty of cilantro",
                applicability={"task": "recipe suggestions"},
            ),
        )
    )
    _ok(
        add(
            conn,
            pref(
                "p-today",
                "self",
                "wants",
                "кинза",
                text="Not today: cilantro",
                polarity=Polarity.NEGATED,
                stability=Stability.ONE_OFF,
                valid_to="2026-10-12T23:59:00Z",
                applicability={"period": "2026-10-12"},
            ),
        )
    )
    _ok(add(conn, pref("p-tea", "self", "likes", "зелёный чай", text="Likes green tea")))
    today = _section(conn, "about_you")
    assert "Not today: cilantro" in today and "Likes green tea" in today
    assert "Loves cilantro" not in today and "Plenty of cilantro" not in today
    tomorrow = _section(conn, "about_you", today=date(2026, 10, 13))
    assert "Loves cilantro" in tomorrow and "Plenty of cilantro" in tomorrow
    assert "Not today" not in tomorrow  # an expired override is not shown


def test_day_bound_habit_only_near_its_day(conn):
    """understands-d03/p2: «каждую субботу вожу Армана на гольф» is not on a Monday."""
    _ok(
        add(
            conn,
            fact(
                "f-golf",
                "self",
                "drives_to",
                "гольф",
                text="Drives Arman to golf",
                stability=Stability.HABIT,
                applicability={"period": "каждую субботу"},
            ),
        )
    )
    _ok(
        add(
            conn,
            fact(
                "f-gym",
                "self",
                "goes_to",
                "зал",
                text="Goes to the gym",
                stability=Stability.HABIT,
                applicability={"period": "on weekdays"},
            ),
        )
    )
    monday = _section(conn, "about_you")
    assert "golf" not in monday and "Goes to the gym" in monday
    friday = _section(conn, "about_you", today=date(2026, 10, 16))  # the evening before
    assert "Drives Arman to golf" in friday
    assert "golf" in _section(conn, "about_you", today=date(2026, 10, 17))
    assert "golf" not in _section(conn, "about_you", today=date(2026, 10, 19))


def test_future_plan_and_expired_one_off_stay_out(conn):
    """knows-d01/p3: a move months ahead is not where the user lives now; a one-off long past is
    not about the user today, nor is an undated one-off said months ago; this week's plan and
    the user's current home stay."""
    _ok(add(conn, fact("f-home", "self", "lives_in", "Динамо", text="Lives at Dinamo")))
    _ok(
        add(
            conn,
            fact(
                "f-dorm",
                "self",
                "moves_to",
                "общежитие",
                text="Will live in the dorm",
                fkind=FKind.PLAN,
                valid_from="2026-12-01",
            ),
        )
    )
    _ok(
        add(
            conn,
            fact(
                "f-call",
                "self",
                "calls",
                "мама",
                text="Calls mum on Wednesday",
                fkind=FKind.PLAN,
                valid_from="2026-10-14",
                stability=Stability.ONE_OFF,
            ),
        )
    )
    _ok(
        add(
            conn,
            fact(
                "f-fair",
                "self",
                "visited",
                "ярмарка",
                text="Visited the fair",
                fkind=FKind.EVENT,
                valid_from="2026-09-01",
                stability=Stability.ONE_OFF,
            ),
        )
    )
    _ok(
        add(
            conn,
            _old(
                fact(
                    "f-meet",
                    "self",
                    "meets",
                    "Илья",
                    text="Meets Ilya on Saturday",
                    fkind=FKind.PLAN,
                    stability=Stability.ONE_OFF,
                )
            ),
        )
    )
    about = _section(conn, "about_you")
    assert "Lives at Dinamo" in about and "Calls mum on Wednesday" in about
    assert "dorm" not in about and "fair" not in about
    # An undated one-off is dated by when the user said it; eight months on it is over
    assert "Meets Ilya on Saturday" not in about
    assert "Will live in the dorm" in _section(conn, "about_you", today=date(2026, 11, 28))


def test_sensitive_never_always_on_but_answers_a_direct_question(conn):
    """policy_leaks-d02/p2-p3: the user's gastritis is not in the core of a weekend or a send
    turn; the question-driven path still may use it (the fence lets the user's own item through)."""
    _ok(
        add(
            conn,
            fact(
                "f-gast",
                "self",
                "has_condition",
                "гастрит",
                text="The owner's gastritis has flared up",
                labels=Label.HEALTH,
            ),
        )
    )
    _ok(
        add(
            conn,
            fact(
                "f-pay",
                "self",
                "has_hearing",
                "суд",
                text="Has a court hearing in November",
                labels=Label.LEGAL,
            ),
        )
    )
    _ok(add(conn, fact("f-work", "self", "works_at", "Kopilka", text="Works at Kopilka")))
    loop = Loop(
        id="l-iron",
        owner="user",
        text_en="Take the iron pills",
        due="2026-10-13",
        author_tier="user",
        recorded_at=ts(),
        origin_quote="пить железо",
        labels=Label.HEALTH,
    )
    _ok(add(conn, loop))
    text = api.core(thread_id="t1", session_id="s1", store=conn).text
    assert "Works at Kopilka" in text
    for word in ("gastritis", "гастрит", "court hearing", "iron"):
        assert word not in text
    turn = TurnCtx(
        turn_id="t2",
        thread_id="t1",
        session_id="s1",
        origin=TurnOrigin.OWNER,
        owner_words="Что у меня с гастритом?",
        clock=CLOCK,
    )
    assert policy.decide_many(conn, ["f-gast"], turn)["f-gast"].allowed


def test_budget_kept_and_bytes_stable(conn):
    """The section budgets hold, and the scoped core is still a pure read of the day."""
    for i in range(200):
        _ok(add(conn, pref(f"p-{i:03d}", "self", "prefers", f"вещь {i}", text=f"Thing {i} " * 12)))
        _ok(
            add(
                conn,
                pref(
                    f"q-{i:03d}",
                    "self",
                    "writes",
                    f"стиль {i}",
                    text=f"Style {i} " * 12,
                    applicability={"audience": f"group {i}"},
                ),
            )
        )
    first = _text(conn)
    assert core_view.tokens(first) <= core_view.BUDGET_TOKENS
    assert "Style" not in first  # freed room is not refilled with out-of-scope lines
    assert _text(conn) == first
