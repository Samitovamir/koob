"""A moved day reaches the plans that rest on it.

When the user's correction moves a dated record's day, every record that rests on it (it
justifies them, they are derived from it, or their own words give a reason naming its subject)
follows: a window taken from the moved day moves with it, a window their words state stays and
its line says the day moved instead of showing the stale day as current. A record that is only
dated near the old day, with no such link, is not touched. The recall conftest pins the day to
Monday 2026-10-12; synthetic people only.
"""

from __future__ import annotations

from datetime import date

import pytest

from koob.memory import ledger
from koob.memory.model import FKind, OpKind, Stability

from .._ledger import add, commit, entity, fact, op, open_seeded, served, status

SAT, SUN = "2026-10-17", "2026-10-18"


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    yield c
    c.close()


def _ok(res) -> None:
    assert res.ok, res


def _about(conn) -> str:
    from koob.memory.recall import core_view

    return next(s.text for s in core_view.compute(conn) if s.name == "about_you")


def _line(text: str, needle: str) -> str:
    return next(ln for ln in text.splitlines() if needle in ln)


def _bounds(conn, rid: str) -> tuple[str | None, str | None]:
    r = conn.execute("SELECT valid_from, valid_to FROM facts WHERE id = ?", (rid,)).fetchone()
    return r[0], r[1]


def _person(conn, eid: str, name: str) -> None:
    _ok(add(conn, entity(eid, name)))


def _premise(conn, fid: str, subject: str, value: str, text: str, quote: str, day: str) -> None:
    _ok(
        add(
            conn,
            fact(fid, subject, "other", value, text=text, fkind=FKind.EVENT,
                 stability=Stability.ONE_OFF, valid_from=day, quote_orig=quote),
        )
    )  # fmt: skip


def _plan(conn, fid: str, text: str, quote: str, start: str, end: str | None = None, **payload):
    rec = fact(fid, "self", "plans_to", text, text=text, fkind=FKind.PLAN,
               stability=Stability.ONE_OFF, valid_from=start, valid_to=end, quote_orig=quote)  # fmt: skip
    _ok(add(conn, rec, **payload))


def _correct(conn, old: str, new: str, subject: str, value: str, quote: str) -> None:
    rec = fact(new, subject, "other", value, text=quote, fkind=FKind.EVENT,
               stability=Stability.ONE_OFF, quote_orig=quote)  # fmt: skip
    _ok(commit(conn, op(OpKind.SUPERSEDE, target=old, group=subject,
                        record=rec.model_dump(mode="json"), fix=True)))  # fmt: skip
    assert status(conn, old) == "superseded"


def test_a_window_taken_from_the_moved_day_moves_with_it(conn):
    """A plan derived from the user's flight (its own words name no day) takes the new day."""
    _person(conn, "p-lena", "Lena")
    _premise(conn, "f-land", "p-lena", "Saturday, 17 October", "Lena lands on Saturday, 17 October",
             "Lena lands on Saturday, 17 October", SAT)  # fmt: skip
    _plan(conn, "f-pick", "Pick Lena up at the airport", "I'll pick her up at the airport",
          SAT, SAT, derived_from=["f-land"])  # fmt: skip
    served(conn, "t-trip", "f-pick", 1)
    _correct(conn, "f-land", "f-land2", "p-lena", "Sunday, 18 October",
             "Lena lands on Sunday, 18 October, not Saturday")  # fmt: skip
    assert _bounds(conn, "f-pick") == (SUN, SUN)
    line = _line(_about(conn), "Pick Lena up")
    assert f"from {SUN}" in line and SAT not in line
    pending = conn.execute(
        "SELECT 1 FROM pending_updates WHERE thread_id = 't-trip' AND item_id = 'f-pick'"
    ).fetchone()
    assert pending is not None  # the thread that was served the old day hears of the move


def test_a_link_across_messages_moves_a_window_he_did_not_date(conn):
    """«Встретить Тимура на вокзале» rests on his return (a justification from another
    message); the bare-day correction «не в субботу, а в воскресенье, 18-го» moves it."""
    _person(conn, "p-timur", "Тимур")
    _premise(conn, "f-back", "p-timur", "17 октября", "Timur returns from the trip on 17 October",
             "Тимур возвращается из поездки 17 октября", SAT)  # fmt: skip
    _plan(conn, "f-meet", "Meet Timur at the station", "Встречу Тимура на вокзале", SAT,
          justifications=[{"group_no": 0, "member": "f-back", "provenance": "stated"}])  # fmt: skip
    _correct(conn, "f-back", "f-back2", "p-timur", "в воскресенье, 18-го",
             "Нет, Тимур возвращается не в субботу, а в воскресенье, 18-го")  # fmt: skip
    assert _bounds(conn, "f-meet")[0] == SUN
    assert f"from {SUN}" in _line(_about(conn), "Meet Timur")


@pytest.mark.parametrize(
    ("name", "text", "quote"),
    [
        ("Вера", "Dacha trip on 17–18 October so Vera can come too",
         "Поедем на дачу 17–18 октября, чтобы Вера тоже успела"),
        ("Dana", "Board games night on October 17-18 because Dana is back",
         "Board games night on October 17-18 because Dana is back"),
        ("Тимур", "Picnic on 17 to 18 October so that Timur can join",
         "Picnic on 17 to 18 October so that Timur can join"),
    ],
)  # fmt: skip
def test_a_reason_in_the_plans_own_words_carries_the_moved_day(conn, name, text, quote):
    """The user's own window stays (both days were named), but the moved Saturday is not shown
    as current and the line says whose day moved."""
    _person(conn, "p-x", name)
    _premise(conn, "f-x", "p-x", "Saturday, 17 October", f"{name} is back on Saturday, 17 October",
             f"{name} is back on Saturday, 17 October", SAT)  # fmt: skip
    _plan(conn, "f-weekend", text, quote, SAT, SUN)
    _correct(conn, "f-x", "f-x2", "p-x", "Sunday", f"{name} is back on Sunday, not Saturday")
    assert _bounds(conn, "f-weekend") == (SAT, SUN)  # the user's stated window is kept
    line = _line(_about(conn), text[:12])
    assert SAT not in line
    assert f"until {SUN}" in line and f"{name}'s day moved to {SUN}" in line


def test_a_premise_whose_end_the_fix_closed_still_moves_its_plans(conn):
    """The hot fix ends the old record on the day of the fix; the day their words named is
    what moved («до субботы, 17 октября» -> «в воскресенье»)."""
    _person(conn, "p-g", "Гоша")
    rec = fact("f-camp", "p-g", "other", "training camp", fkind=FKind.STATE,
               text="Gosha is at a training camp until Saturday, 17 October",
               quote_orig="Гоша на сборах до субботы, 17 октября",
               valid_from="2026-10-12", valid_to=SAT)  # fmt: skip
    _ok(add(conn, rec))
    _plan(conn, "f-pic", "Picnic with the kids", "Пикник с детьми 17–18 октября, чтобы Гоша успел",
          SAT, SUN)  # fmt: skip
    _correct(conn, "f-camp", "f-camp2", "p-g", "воскресенье",
             "Гоша вернётся не в субботу, а в воскресенье")  # fmt: skip
    conn.execute("UPDATE facts SET valid_to = '2026-10-12' WHERE id = 'f-camp'")
    line = _line(_about(conn), "Picnic with the kids")
    assert SAT not in line and f"Гоша's day moved to {SUN}" in line


def test_a_plan_only_dated_near_the_old_day_stays(conn):
    """No link, no reason naming her: a haircut that Saturday and a dinner that only
    mentions her keep their day."""
    _person(conn, "p-lena", "Lena")
    _premise(conn, "f-land", "p-lena", "17 October", "Lena lands on 17 October",
             "Лена прилетает 17 октября", SAT)  # fmt: skip
    _plan(conn, "f-hair", "Haircut on Saturday 17 October", "Стрижка в субботу 17-го", SAT, SAT)
    _plan(conn, "f-dinner", "Dinner with Lena at Mama's", "Ужин с Леной у мамы", SAT)
    _correct(
        conn, "f-land", "f-land2", "p-lena", "18 October", "Лена прилетает не 17, а 18 октября"
    )
    assert _bounds(conn, "f-hair") == (SAT, SAT)
    assert _bounds(conn, "f-dinner") == (SAT, None)
    about = _about(conn)
    assert f"from {SAT}" in _line(about, "Haircut") and "moved" not in _line(about, "Haircut")
    assert f"from {SAT}" in _line(about, "Dinner with Lena")
    assert "moved" not in _line(about, "Dinner with Lena")


def test_a_time_only_fix_moves_no_day(conn):
    _person(conn, "p-lena", "Lena")
    _premise(conn, "f-land", "p-lena", "17 October at 14:20", "Lena lands on 17 October at 14:20",
             "Lena lands on 17 October at 14:20", SAT)  # fmt: skip
    _plan(conn, "f-pick", "Pick Lena up at the airport", "so I can pick Lena up", SAT,
          derived_from=["f-land"])  # fmt: skip
    _correct(conn, "f-land", "f-land2", "p-lena", "17 October at 15:40",
             "Lena lands on 17 October at 15:40, not 14:20")  # fmt: skip
    assert _bounds(conn, "f-pick") == (SAT, None)
    assert "moved" not in _line(_about(conn), "Pick Lena up")


def test_said_days_reads_ranges_bare_days_and_a_lone_weekday():
    assert ledger.said_days(["на выходные 9–10 мая"], 2026) == {date(2026, 5, 9), date(2026, 5, 10)}
    assert ledger.said_days(["May 9-10"], 2026) == {date(2026, 5, 9), date(2026, 5, 10)}
    assert ledger.said_days(["в воскресенье, 10-го"], 2026, month=5) == {date(2026, 5, 10)}
    assert ledger.said_days(["on Sunday"], 2026, near=date(2026, 5, 9)) == {date(2026, 5, 10)}
    assert ledger.said_days(["at 14:20"], 2026, month=5) == set()
