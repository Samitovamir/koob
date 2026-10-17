"""Relative day words in basket lines name the day they mean, counted from the line's date."""

from __future__ import annotations

import pytest

from koob.memory.recall import basket
from koob.memory.recall.reldate import resolve_relative

SAID = "2024-03-15T18:20:00+00:00"  # a Friday


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Denis Smith sold his bike yesterday.", "yesterday (2024-03-14)"),
        ("the day before yesterday we met", "the day before yesterday (2024-03-13)"),
        ("I fly out tomorrow", "tomorrow (2024-03-16)"),
        ("we talked three days ago", "three days ago (2024-03-12)"),
        ("it started 10 days ago", "10 days ago (2024-03-05)"),
        ("I moved two weeks ago", "two weeks ago (week of 2024-02-26)"),
        ("we hiked last weekend", "last weekend (weekend of 2024-03-09)"),
        ("the course ended last week", "last week (week of 2024-03-04)"),
        ("I saw it last Friday", "last Friday (2024-03-08)"),
        ("see you next Monday", "next Monday (2024-03-18)"),
        ("she joined last month", "last month (2024-02)"),
        ("the shop opened last year", "last year (2023)"),
        ("we went there last summer", "last summer (summer 2023)"),
        ("Денис вчера продал велосипед", "вчера (2024-03-14)"),
        ("это было 3 дня назад", "3 дня назад (2024-03-12)"),
        ("на прошлой неделе был дождь", "на прошлой неделе (week of 2024-03-04)"),
        ("в прошлом году он переехал", "в прошлом году (2023)"),
    ],
)
def test_relative_word_is_followed_by_the_day_it_means(text, want):
    assert want in resolve_relative(text, SAID)


@pytest.mark.parametrize(
    "text",
    ["I did it today", "a few days ago it rained", "the other day I saw him", "сегодня тепло"],
)
def test_vague_words_and_the_day_itself_stay_as_they_are(text):
    assert resolve_relative(text, SAID) == text


def test_already_resolved_word_is_not_resolved_twice():
    text = "yesterday (2024-03-14) was long"
    assert resolve_relative(text, SAID) == text


def test_text_without_a_date_stays_as_it_is():
    assert resolve_relative("yesterday", None) == "yesterday"
    assert resolve_relative("yesterday", "not a date") == "yesterday"


def _row(**kw):
    return {"id": "r1", "kind": "raw", "version": 1, "author_tier": None, **kw}


def test_raw_line_shows_the_resolved_day_next_to_its_message_date():
    item = basket._item(
        _row(text="Lost my job yesterday, so I'm free.", recorded_at=SAID), None, None
    )
    assert item.date == "2024-03-15"
    assert "yesterday (2024-03-14)" in item.text
    assert "yesterday (2024-03-14)" in basket.line(item)


def test_record_with_a_resolved_date_keeps_dropping_the_relative_word():
    row = _row(
        kind="fact", text="Denis Smith sold his bike yesterday.", valid_from="2024-03-14",
        observed_at=SAID,
    )  # fmt: skip
    item = basket._item(row, None, None)
    assert item.date == "2024-03-14"
    assert "yesterday" not in item.text
