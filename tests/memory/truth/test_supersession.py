"""Supersession by mechanism, both languages, synthetic names.

- «больше не» / "no longer" retires the habit it names, across a scope the general word covers;
- a partial change supersedes only the part that changed (an explicit «вместо X» side, an
  added «тоже» value);
- a lapse of a part never retires the whole;
- an unlabelled topic-sensitive item stays out of the always-on core."""

from __future__ import annotations

import pytest

from koob.memory.model import OWNER_ENTITY_ID, Polarity
from koob.memory.recall import core_view

from .._ledger import add, fact, open_seeded, pref, status

ME = OWNER_ENTITY_ID


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    yield c
    c.close()


def _ok(res) -> None:
    assert res.ok, res


def test_no_longer_retires_a_scoped_taste_in_other_words(conn):
    """«Обожаю тархун в лимонадах» (scoped) ends with a general «больше не люблю тархуна»: the
    general word covers its scoped instance, the case ending differs, the relation is a peer."""
    _ok(
        add(
            conn,
            pref(
                "p-tar",
                ME,
                "likes",
                "тархун",
                applicability={"situation": "лимонады"},
                quote_orig="Обожаю тархун в лимонадах",
            ),
        )
    )
    _ok(add(conn, pref("p-mint", ME, "likes", "мята", quote_orig="мяту люблю")))
    _ok(
        add(
            conn,
            pref(
                "p-no",
                ME,
                "prefers",
                "тархуна",
                polarity=Polarity.NEGATED,
                quote_orig="Знаешь, я больше не люблю тархуна, надоел",
            ),
        )
    )
    assert status(conn, "p-tar") == "superseded"
    assert status(conn, "p-mint") == "active"  # a rival value never lapses


def test_no_longer_retires_an_english_habit(conn):
    _ok(
        add(
            conn,
            fact("f-swim", ME, "hobby", "evening swimming", quote_orig="I go evening swimming"),
        )
    )
    _ok(add(conn, fact("f-chess", ME, "hobby", "chess", quote_orig="I play chess")))
    _ok(
        add(
            conn,
            fact(
                "f-stop",
                ME,
                "hobby",
                "evening swims",
                polarity=Polarity.NEGATED,
                quote_orig="I don't do evening swims anymore",
            ),
        )
    )
    assert status(conn, "f-swim") == "superseded"
    assert status(conn, "f-chess") == "active"


def test_lapse_of_a_part_keeps_the_whole(conn):
    """ "No longer smoked fish" says nothing about fish in general."""
    _ok(add(conn, fact("f-fish", ME, "eats", "fish", quote_orig="I eat fish")))
    _ok(
        add(
            conn,
            fact(
                "f-smoked",
                ME,
                "eats",
                "smoked fish",
                polarity=Polarity.NEGATED,
                quote_orig="I no longer eat smoked fish",
            ),
        )
    )
    assert status(conn, "f-fish") == "active"


def test_a_scoped_lapse_does_not_cross_to_the_general(conn):
    _ok(add(conn, pref("p-jazz", ME, "likes", "джаз", quote_orig="люблю джаз")))
    _ok(
        add(
            conn,
            pref(
                "p-jazz-car",
                ME,
                "likes",
                "джаз",
                polarity=Polarity.NEGATED,
                applicability={"situation": "в машине"},
                quote_orig="в машине джаз больше не слушаю",
            ),
        )
    )
    assert status(conn, "p-jazz") == "active"


def test_partial_change_supersedes_only_the_named_side(conn):
    _ok(add(conn, fact("f-box", ME, "hobby", "бокс", quote_orig="хожу на бокс")))
    _ok(add(conn, fact("f-draw", ME, "hobby", "рисование", quote_orig="рисую по вечерам")))
    _ok(
        add(
            conn,
            fact(
                "f-climb",
                ME,
                "hobby",
                "скалолазание",
                quote_orig="Теперь вместо бокса скалолазание",
            ),
        )
    )
    assert status(conn, "f-box") == "superseded"
    assert status(conn, "f-draw") == "active"


def test_an_added_value_supersedes_nothing(conn):
    _ok(add(conn, fact("f-piano", ME, "hobby", "piano", quote_orig="I play piano")))
    _ok(
        add(
            conn,
            fact(
                "f-pottery", ME, "hobby", "pottery", quote_orig="Now I also do pottery on Sundays"
            ),
        )
    )
    assert status(conn, "f-piano") == "active"


def test_unlabelled_diagnosis_stays_out_of_the_core(conn):
    _ok(
        add(
            conn,
            fact(
                "f-dx",
                ME,
                "has",
                "диагноз",
                text="The owner got a diagnosis of migraine",
                quote_orig="мне поставили диагноз мигрень",
            ),
        )
    )
    _ok(
        add(
            conn,
            fact(
                "f-city",
                ME,
                "lives_in",
                "Tula",
                text="The owner lives in Tula",
                quote_orig="живу в Туле",
            ),
        )
    )
    text = core_view.render(core_view.compute(conn))
    assert "Tula" in text
    assert "migraine" not in text and "диагноз" not in text


def test_his_lapse_leaves_what_he_buys_for_someone_else(conn):
    kid = pref(
        "p-kid",
        ME,
        "likes",
        "marshmallows",
        applicability={"for_whom": "nephew"},
        quote_orig="I buy marshmallows for my nephew",
    )
    _ok(add(conn, kid))
    _ok(add(conn, pref("p-me", ME, "likes", "marshmallows", quote_orig="I love marshmallows")))
    off = pref(
        "p-off",
        ME,
        "likes",
        "marshmallows",
        polarity=Polarity.NEGATED,
        quote_orig="I no longer like marshmallows",
    )
    _ok(add(conn, off))
    assert status(conn, "p-me") == "superseded"
    assert status(conn, "p-kid") == "active"
