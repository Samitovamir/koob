"""The truth overlay over raw messages: a later message from the user that revises the same slot
(the same topic, a different value of the same kind) makes an earlier raw past, even when no
record holds either value. A past raw of the user's is shown with the revised value cut
(«[сменилось]») and the revising message beside it; a Cube output built on an overtaken value
is left out. An older message on another topic is not touched."""

from __future__ import annotations

import pytest

from koob.memory import api, policy
from koob.memory.model import Label, TurnCtx, TurnOrigin
from koob.memory.recall import basket, lexical, vectors

from .._ledger import add, fact, open_seeded, raw

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    _forget_parse_caches()
    c = open_seeded(tmp_path)
    yield c
    c.close()
    _forget_parse_caches()


def _forget_parse_caches() -> None:
    """The per-message parse caches are keyed by (raw id, rev); test stores reuse short ids."""
    basket._NO_CONTRAST.clear()
    basket._NO_REVISION.clear()


def t(words: str) -> TurnCtx:
    return TurnCtx(
        turn_id="turn-1",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=CLOCK,
    )


def _raw(conn, rid: str, text: str, day: str, tier: str = "user") -> None:
    raw(conn, rid, text, tier=tier, ts_=f"{day}T10:00:00Z")
    lexical.index_raw(conn, rid)


def _lines(conn, words: str) -> dict[str, str]:
    b = api.recall(t(words), store=conn)
    return {i.id: basket.line(i) for i in b.items}


def test_old_dated_raw_is_not_served_as_current_after_his_correction(conn):
    _raw(
        conn,
        "r-old",
        "У Марка день рождения 14 мая. Напомни купить подарок за неделю.",
        "2026-10-01",
    )
    _raw(conn, "r-fix", "Нет, у Марка день рождения 17 мая, я перепутал.", "2026-10-05")
    lines = _lines(conn, "Когда у Марка день рождения?")
    assert "r-fix" in lines and "17 мая" in lines["r-fix"]
    assert not any("14 мая" in line for line in lines.values())
    if "r-old" in lines:  # the rest of the old message stays; the revised date is cut
        assert "[сменилось]" in lines["r-old"] and "подарок" in lines["r-old"]


def test_use_note_of_a_revised_raw_never_quotes_the_cut_value(conn):
    """The «учёл» note quotes the item's words: for a revised raw, the masked words."""
    _raw(conn, "r-old", "У Марка день рождения 14 мая, купить торт.", "2026-10-01")
    _raw(conn, "r-fix", "Нет, у Марка день рождения 17 мая, я перепутал.", "2026-10-05")
    # a third-party label makes the user's own raw ranked in with a use note
    conn.execute(
        "UPDATE raw_chunks SET labels = ?, labels_final = 1", (int(Label.THIRD_PARTY_PRIVATE),)
    )
    b = api.recall(t("Что купить Марку на день рождения?"), store=conn)
    old = [basket.line(i) for i in b.items if i.id == "r-old"]
    assert old and "учёл" in old[0] and "[сменилось]" in old[0]
    assert not any("14 мая" in basket.line(i) for i in b.items)


def test_restated_value_without_correction_words_still_revises(conn):
    _raw(conn, "r-old", "Созвон с Ниной по проекту в 15:00.", "2026-10-01")
    _raw(conn, "r-new", "Созвон с Ниной по проекту в 16:30.", "2026-10-03")
    lines = _lines(conn, "Во сколько созвон с Ниной по проекту?")
    assert "r-new" in lines
    assert not any("15:00" in line for line in lines.values())


def test_cube_plan_overtaken_by_his_correction_is_left_out(conn):
    _raw(conn, "r-ask", "Составь план на выходные, чтобы Олег успел с нами.", "2026-10-02")
    _raw(
        conn,
        "r-plan",
        "Суббота, 9 мая: встречаем Олега с вокзала, вечером ужин. Воскресенье: парк.",
        "2026-10-02",
        tier="system",
    )
    _raw(
        conn,
        "r-fix",
        "Нет, Олег приезжает не в субботу, а в воскресенье, 10-го: поезд перенесли.",
        "2026-10-03",
    )
    lines = _lines(conn, "Когда встречаем Олега?")
    assert "r-fix" in lines
    assert "r-plan" not in lines


def test_unrelated_older_raw_on_another_topic_is_not_marked_past(conn):
    _raw(conn, "r-exam", "У Марка экзамен по физике 20 мая.", "2026-10-01")
    _raw(conn, "r-fix", "Нет, у Марка день рождения 17 мая, я перепутал.", "2026-10-05")
    lines = _lines(conn, "Когда у Марка экзамен по физике?")
    assert "r-exam" in lines
    assert "20 мая" in lines["r-exam"] and "[сменилось]" not in lines["r-exam"]


def test_a_later_message_that_keeps_the_old_value_revises_nothing(conn):
    _raw(conn, "r-old", "Встреча с Ниной 14 мая в кафе.", "2026-10-01")
    _raw(conn, "r-add", "Нет, встреча с Ниной 14 мая и ещё 21 мая.", "2026-10-03")
    lines = _lines(conn, "Когда встреча с Ниной?")
    assert "r-old" in lines and "14 мая" in lines["r-old"]


def test_scope_that_the_question_names_is_not_tagged_as_someone_elses(conn):
    """A preference scoped to «клиенты Northwind» applies to a letter to a Northwind client:
    an inflected group name still names it, and the line carries no «для:» tag."""
    assert add(
        conn,
        fact(
            "f-tone",
            "self",
            "prefers",
            "formal tone",
            text="Writes to Northwind clients formally, without emoji",
            origin_quote="С клиентами Northwind пишу официально, без смайликов",
            applicability={"audience": "клиенты Northwind"},
        ),
    ).ok
    lines = _lines(conn, "Напиши письмо клиенту Northwind про сроки")
    assert "f-tone" in lines and "для:" not in lines["f-tone"]


def test_correction_of_a_shown_raw_comes_with_it_within_open_labels(conn):
    """A correction denied only for relevance comes beside the raw it corrects; one that
    carries a sensitive label the corrected raw lacks (beyond the open ones) stays out."""
    _raw(conn, "r-old", "Ужин с Марком 14 мая.", "2026-10-01")
    _raw(conn, "r-fix", "Нет, ужин с Марком 17 мая, его брат занят.", "2026-10-05")
    _raw(conn, "r-hl", "Нет, ужин с Марком 18 мая, у меня приём у врача.", "2026-10-06")
    conn.execute(
        "UPDATE raw_chunks SET labels = ? WHERE id = 'r-fix'", (int(Label.THIRD_PARTY_PRIVATE),)
    )
    conn.execute("UPDATE raw_chunks SET labels = ? WHERE id = 'r-hl'", (int(Label.HEALTH),))
    rows = {
        "r-old": {"text": "Ужин с Марком [сменилось].", "corrected_by": ["r-fix", "r-hl"]},
        "r-fix": {"text": "Нет, ужин с Марком 17 мая, его брат занят."},
        "r-hl": {"text": "Нет, ужин с Марком 18 мая, у меня приём у врача."},
    }
    verdicts = {
        "r-old": policy.Use(True, "not_sensitive"),
        "r-fix": policy.Use(False, "own_not_applicable"),
        "r-hl": policy.Use(False, "own_not_applicable"),
    }
    basket._recheck_fixers(conn, rows, verdicts)
    assert verdicts["r-fix"].allowed and verdicts["r-fix"].reason == "own_corrects"
    assert not verdicts["r-hl"].allowed
