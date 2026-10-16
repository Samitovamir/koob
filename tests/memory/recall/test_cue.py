"""Cue parse and RRF by code: the memory spec §10.2 steps 1 and 4, §15.3 (basket)."""

from __future__ import annotations

from koob.memory.model import TurnCtx, TurnOrigin
from koob.memory.recall import cue, fusion

NOW = "2026-10-14T15:00:00.000000Z"  # a Wednesday


def _t(words: str, **kw) -> TurnCtx:
    return TurnCtx(turn_id="t", thread_id="th", owner_words=words, clock=NOW, **kw)


def test_relative_dates_with_the_turn_clock():
    day = cue.turn_day(NOW)
    assert cue.parse_dates("что было вчера?", now=NOW) == [
        (day.replace(day=day.day - 1)).isoformat()
    ]
    assert cue.parse_dates("what about tomorrow", now=NOW) == [
        day.replace(day=day.day + 1).isoformat()
    ]
    last_week = cue.parse_dates("на прошлой неделе", now=NOW)[0]
    a, b = last_week.split("..")
    assert a < b < day.isoformat()


def test_named_dates_and_months():
    assert "2026-05-12" in cue.parse_dates("у Саши день рождения 12 мая", now=NOW)
    assert "2026-05-01..2026-05-31" in cue.parse_dates("что мы делали в мае", now=NOW)
    assert "2025-12-01..2025-12-31" in cue.parse_dates("в декабре", now=NOW)  # last December
    assert "2026-03-05" in cue.parse_dates("встреча 2026-03-05", now=NOW)
    assert "2026-09-01" in cue.parse_dates("с 01.09", now=NOW)
    assert cue.parse_dates("привет", now=NOW) == []


def test_episode_hint_and_constraints():
    c = cue.parse(_t("помнишь, мы хотели встретиться, но сорвалось из-за поездки?"), now=NOW)
    assert c.episode_hint
    assert c.constraints["event_type"] == "meeting"
    assert c.constraints["outcome"] == "cancelled"
    assert c.constraints["cause"].startswith("поездки")
    plain = cue.parse(_t("запомни, что я люблю кофе"), now=NOW)
    assert not plain.episode_hint and plain.constraints == {}


def test_lexicons_and_scope():
    c = cue.parse(_t("отправь Диме мой диагноз"), now=NOW)
    assert c.send_turn
    assert "health" in c.sensitive_topics
    assert c.scope["thread"] == "th"
    task = cue.parse(
        TurnCtx(turn_id="t", thread_id="th", origin=TurnOrigin.TASK, task_words="напиши отчёт"),
        now=NOW,
    )
    assert task.scope["task"] == "1" and task.words == "напиши отчёт"
    assert cue.parse(_t("что угодно", origin=TurnOrigin.DUTY), now=NOW).words == ""


def test_rrf_ranks_agreement_first_and_bonus_only_reorders():
    lanes = {"lexical": [("a", 9.0), ("b", 5.0), ("a", 1.0)], "names": [("b", 1.0)], "vector": []}
    ranked = fusion.rrf(lanes)
    assert [i for i, _ in ranked] == ["b", "a"]
    assert abs(dict(ranked)["a"] - 1 / 61) < 1e-12  # a duplicate in a lane counts once
    boosted = fusion.rrf(lanes, bonuses={"a": 1.0, "zzz": 5.0})
    assert [i for i, _ in boosted] == ["a", "b"]  # a bonus never creates an item
