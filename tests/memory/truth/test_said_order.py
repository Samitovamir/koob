"""RAW-TRUTH: a correction and the value it corrects meet in the order the user SAID them,
whatever order extraction commits them in. A late extraction of an early message never
resurrects a retired value; a correction committed before its target still wins when the
target lands."""

from __future__ import annotations

from koob.memory import ledger
from koob.memory.model import AuthorTier, FKind, Stability

from .._ledger import add, edges, entity, fact, open_seeded, raw, status


def _people(conn):
    add(conn, entity("e-lina", "Lina"))
    add(conn, entity("e-vik", "Viktor"))


def _plan(fid, subject, value, quote, src, **kw):
    return fact(
        fid, subject, "other", value, fkind=FKind.PLAN, stability=Stability.ONE_OFF,
        quote_orig=quote, source_raw_id=src, **kw,
    )  # fmt: skip


def test_correction_committed_first_wins_when_its_target_lands_late(tmp_path):
    conn = open_seeded(tmp_path)
    _people(conn)
    raw(conn, "r-early", "Lina's ferry gets in at 11:15 on Friday", ts_="2026-04-02T09:00:00Z")
    raw(conn, "r-fix", "Lina's ferry gets in at 12:45, not 11:15", ts_="2026-04-02T18:00:00Z")
    add(conn, _plan("f-new", "e-lina", "12:45", "ferry gets in at 12:45, not 11:15", "r-fix"))
    # the earlier message is extracted only now (a queue, a retry, a night batch)
    add(conn, _plan("f-old", "e-lina", "11:15", "ferry gets in at 11:15", "r-early"))
    assert status(conn, "f-old") == "superseded"
    assert status(conn, "f-new") == "active"
    assert ("f-new", "f-old") in edges(conn, "supersedes")


def test_his_contrast_retires_an_earlier_one_off_value_of_the_same_person(tmp_path):
    conn = open_seeded(tmp_path)
    _people(conn)
    raw(conn, "r1", "Виктор приедет в четверг вечером", ts_="2026-04-03T08:00:00Z")
    raw(conn, "r2", "Виктор приедет не в четверг, а в пятницу", ts_="2026-04-03T12:00:00Z")
    raw(conn, "r3", "Лина уезжает в четверг", ts_="2026-04-03T09:00:00Z")
    add(conn, _plan("f-thu", "e-vik", "в четверг вечером", "приедет в четверг вечером", "r1"))
    add(conn, _plan("f-lina", "e-lina", "в четверг", "уезжает в четверг", "r3"))
    add(conn, _plan("f-fri", "e-vik", "в пятницу", "приедет не в четверг, а в пятницу", "r2"))
    assert status(conn, "f-thu") == "superseded"
    assert status(conn, "f-fri") == "active"
    assert status(conn, "f-lina") == "active"  # another person's Thursday is not the user's correction


def test_late_copy_of_a_replaced_value_lands_as_history(tmp_path):
    conn = open_seeded(tmp_path)
    _people(conn)
    raw(conn, "r1", "Lina lives in Tromso now", ts_="2026-02-01T10:00:00Z")
    raw(conn, "r2", "Lina still loves Tromso, her flat there is tiny", ts_="2026-02-05T10:00:00Z")
    raw(conn, "r3", "Lina moved to Bergen last month", ts_="2026-03-10T10:00:00Z")
    add(conn, fact("f1", "e-lina", "lives_in", "Tromso", source_raw_id="r1"))
    add(conn, fact("f3", "e-lina", "lives_in", "Bergen", source_raw_id="r3"))
    assert status(conn, "f1") == "superseded"
    # a message said BEFORE the move, extracted after it: never the current city again
    add(conn, fact("f2", "e-lina", "lives_in", "Tromso", source_raw_id="r2"))
    assert status(conn, "f2") == "superseded"
    assert status(conn, "f3") == "active"
    assert ("f3", "f2") in edges(conn, "supersedes")


def test_a_later_said_value_still_replaces_by_rule(tmp_path):
    conn = open_seeded(tmp_path)
    _people(conn)
    raw(conn, "r1", "Lina lives in Tromso", ts_="2026-02-01T10:00:00Z")
    raw(conn, "r2", "Lina lives in Bergen", ts_="2026-03-01T10:00:00Z")
    add(conn, fact("f2", "e-lina", "lives_in", "Bergen", source_raw_id="r2"))
    add(conn, fact("f1", "e-lina", "lives_in", "Tromso", source_raw_id="r1"))
    assert status(conn, "f1") == "superseded"  # said first, committed last: history
    assert status(conn, "f2") == "active"


def test_foreign_contrast_never_retires_his_value(tmp_path):
    conn = open_seeded(tmp_path)
    _people(conn)
    raw(conn, "r1", "Viktor's talk is at 16:30", ts_="2026-04-03T08:00:00Z")
    raw(conn, "r2", "the talk is at 17:00, not 16:30", tier="other", ts_="2026-04-03T12:00:00Z")
    add(conn, _plan("f-his", "e-vik", "16:30", "talk is at 16:30", "r1"))
    add(
        conn,
        _plan("f-them", "e-vik", "17:00", "the talk is at 17:00, not 16:30", "r2",
              tier=AuthorTier.OTHER),
        tier=AuthorTier.OTHER,
    )  # fmt: skip
    assert status(conn, "f-his") == "active"


def test_retired_sides_reads_contrasts_in_both_languages():
    sides = ledger.retired_sides
    assert sides("Встреча не в среду, а в четверг") == ["в среду"]
    assert sides("в четверг, а не в среду") == ["в среду"]
    assert sides("the dentist moved to 9:40, not 8:50 - they called") == ["8:50"]
    assert sides("It's Oslo, not Bergen and the date is 4/9 not 4/2") == ["Bergen", "4/2"]
    assert sides("not on Monday but on Tuesday") == ["on Monday"]
    assert sides("it is not ready yet") == []
    assert sides("I do not like it") == []
    assert ledger.holds_value("8:50", "at 8:50 sharp")
    assert ledger.holds_value("не 7-го", "7 апреля")
    assert ledger.holds_value("the 21st", "Mon, May 21")
    assert not ledger.holds_value("7-го", "17 апреля")


def test_a_late_extraction_never_overrides_his_hot_word(tmp_path):
    conn = open_seeded(tmp_path)
    _people(conn)
    raw(conn, "r-early", "Lina lives in Tromso", ts_="2026-01-05T10:00:00+05:00")
    # the user's hot word today (no raw yet: the turn is ingested after the write)
    add(conn, fact("f-hot", "e-lina", "lives_in", "Bergen", origin_quote="Лина теперь в Бергене"))
    add(conn, fact("f-late", "e-lina", "lives_in", "Tromso", source_raw_id="r-early"))
    assert status(conn, "f-hot") == "active"
    assert status(conn, "f-late") == "superseded"
