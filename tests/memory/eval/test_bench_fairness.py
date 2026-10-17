"""Benchmark exam fairness: probe timing after a racing worker, numeric dates by the scenario's
locale, bracket emoticons and negated wants, the same for every SUT."""

from __future__ import annotations

from koob.memory.eval.matchers import (
    DATE_ORDER,
    date_order,
    find,
    locale_date_order,
    shown_as_current,
)
from koob.memory.eval.runner import Runner
from koob.memory.eval.schema import Scenario

from .helpers import PERSONAS, FakeApi, FakeSUT, scenario_dict


def _race(**over) -> dict:
    d = scenario_dict(setup={"workers": "manual"}, **over)
    d["timeline"] = [
        {"id": "e1", "ev": "user_msg", "text": "Я живу в Твери."},
        {"id": "w1", "ev": "worker", "job": "summary", "snapshot_at": "e1", "commit_after": "e3"},
        {"id": "e2", "ev": "user_msg", "at": "+1d", "text": "Как дела?"},
        {"id": "e3", "ev": "correction", "at": "+2d", "text": "Нет, я живу в Твери."},
        {"id": "e4", "ev": "user_msg", "at": "+3d", "text": "Спасибо."},
    ]
    d["probes"][0].pop("at")
    d["probes"][0]["after"] = "w1"
    return d


def _order(runner: Runner) -> list[str]:
    return [s.ev.id if s.kind == "event" else f"probe:{s.probe.id}" for s in runner._plan()]


def test_a_probe_after_a_racing_worker_waits_for_its_commit():
    scn = Scenario.model_validate(_race())
    order = _order(Runner(scn, PERSONAS, sut=FakeSUT(FakeApi())))
    assert order.index("probe:p1") > order.index("e3")  # the commit runs right after e3
    assert order.index("probe:p1") < order.index("e4")  # and not later than needed
    api = FakeApi()
    res = Runner(scn, PERSONAS, sut=FakeSUT(api)).run()
    assert api.workers == ["summary:snapshot", "summary:commit"]
    assert res.status == "pass"


def test_a_probe_after_a_plain_event_keeps_its_place():
    d = _race()
    d["probes"][0]["after"] = "e2"
    order = _order(Runner(Scenario.model_validate(d), PERSONAS, sut=FakeSUT(FakeApi())))
    assert order.index("e2") < order.index("probe:p1") < order.index("e3")


def test_numeric_dates_read_by_the_locale():
    nov3, mar11 = {"date": "--11-03"}, {"date": "--03-11"}
    with date_order("mdy"):
        assert find("starts 11/3", nov3) and not find("starts 11/3", mar11)
        assert find("on 11/3/2026", {"date": "2026-11-03"})
        assert find("November 3", nov3) and find("2026-11-03", nov3)
    with date_order("dmy"):
        assert find("с 11/3", mar11) and not find("с 11/3", nov3)
        assert find("с 11.03", mar11) and find("11 марта", mar11)
    assert find("11/3", nov3) and find("11/3", mar11)  # no locale: both readings, as before
    assert DATE_ORDER.get() is None


def test_the_locale_comes_from_the_persona_or_the_scenario():
    assert locale_date_order("America/Chicago", ["en"]) == "mdy"
    assert locale_date_order("America/Chicago", ["ru", "en"]) == "dmy"
    assert locale_date_order("Europe/Moscow", ["ru", "en"]) == "dmy"
    assert locale_date_order("Europe/London", ["en"]) == "dmy"
    assert locale_date_order("America/Sao_Paulo", None) is None
    us = {**PERSONAS, "riley": {**PERSONAS["mira"], "tz": "America/Chicago", "langs": ["en"]}}
    scn = Scenario.model_validate(scenario_dict(persona="riley"))
    r = Runner(scn, us, sut=FakeSUT(FakeApi()))
    assert r.date_order() == "mdy"
    seen = []
    r._run = lambda: seen.append(DATE_ORDER.get())  # type: ignore[method-assign]
    r.run()
    assert seen == ["mdy"] and DATE_ORDER.get() is None
    scn = Scenario.model_validate(scenario_dict(persona="riley", date_order="dmy"))
    assert Runner(scn, us, sut=FakeSUT(FakeApi())).date_order() == "dmy"


def test_a_bracket_emoticon_is_not_a_closing_bracket():
    assert not find("Хорошо, буду обращаться к вам (на «вы»).", ")")
    assert not find("Okay, I'll keep it formal (using «вы»).", ")")
    assert not find("1) купить; 2) позвонить", ")")
    assert not find("Смотри (вот)", "(")
    for text, tok in (("Привет)", ")"), ("Привет :)", ")"), ("Привет :)", ":)"), ("(ок))", ")"),
                      ("ok (see above). Thanks)", ")"), ("Жаль ((", "(("), ("Ну вот(", "(")):  # fmt: skip
        assert find(text, tok), (text, tok)
        assert shown_as_current(text, tok, history=False)


def test_negated_wants_are_not_shown_as_current():
    for text, v in (
        ("She does not want cilantro.", "cilantro"),
        ("He doesn't want cilantro in the salad.", "cilantro"),
        ("She does not want to eat cilantro.", "cilantro"),
        ("No cilantro, please.", "cilantro"),
        ("Мира не хочет кинзу.", "кинз"),
        ("Я не хочу сегодня кинзу.", "кинз"),
        ("Без кинзы сегодня.", "кинз"),
    ):
        assert not shown_as_current(text, v), text
        assert not shown_as_current(text, v, history=False), text
    for text, v in (
        ("She wants cilantro.", "cilantro"),
        ("Мира хочет кинзу.", "кинз"),
        ("She doesn't want to forget cilantro.", "cilantro"),
        ("Мира не хочет без кинзы.", "кинз"),
        ("Ты не хочешь кинзу?", "кинз"),
        ("I don't want Boulder, I want Denver", "Denver"),
    ):
        assert shown_as_current(text, v), text
