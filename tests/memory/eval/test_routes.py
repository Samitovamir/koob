"""Tool-routed probes.

An inspect or scout probe (or one with `route:`) is answered by the memory tools; the
view-level check also counts what the tool returns for the question, for every system that
has the tool. A system with no such tool is reported n/a and judged on its view alone."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from koob.memory.eval import bench
from koob.memory.eval.routes import GROUP_ROUTE, probe_route, route_tools
from koob.memory.eval.schema import Probe, Scenario

from .helpers import PERSONAS, scenario_dict


def _scn(group: str, route: str | None = None, sid: str | None = None) -> Scenario:
    probe = {
        "id": "p1",
        "at": "2026-02-01T10:00",
        "level": "memory",
        "ask": ["Откуда это?"],
        "expect": {"behaviour": "answer", "must_include": [["Oksana", "Оксан"]]},
    }
    if route:
        probe["route"] = route
    d = scenario_dict(
        id=sid or f"{group}-d90",
        group=group,
        subgroup={"inspect": "where_from", "scout": "practice"}.get(group, "as_of"),
        timeline=[{"id": "e1", "ev": "user_msg", "text": "Переезд офиса с 1 августа."}],
        probes=[probe],
    )
    return Scenario.model_validate(d)


class _ToolSUT(bench.PlainFileSUT):
    """A plain file whose memory tool knows who said it (the view does not)."""

    def tool_view(self, route, wording, turn=None):
        return "Oksana wrote it in the Kopilka chat" if route_tools(route) else None


def _record(scn: Scenario, sut) -> dict:
    r = bench.BenchRunner(scn, PERSONAS, sut=sut, seed=1)
    res = r.run()
    assert res.error is None, res.error
    return r.records[-1]


def test_route_defaults_by_group_and_explicit_route_wins():
    assert GROUP_ROUTE == {"inspect": "tool", "scout": "tool"}
    p = SimpleNamespace(route=None)
    assert probe_route(p, "inspect") == "tool" and probe_route(p, "scout") == "tool"
    assert probe_route(p, "knows") == "view"
    assert probe_route(SimpleNamespace(route="view"), "inspect") == "view"
    assert route_tools("tool") == ("memory_what", "memory_where_from", "memory_resolve")
    assert route_tools("memory_where_from") == ("memory_where_from",) and route_tools("view") == ()
    with pytest.raises(ValidationError):
        Probe.model_validate(
            {"id": "p", "at": "2026-02-01T10:00", "route": "memory_guess", "expect": {}}
        )


def test_tool_result_counts_only_where_the_route_is_a_tool():
    rec = _record(_scn("inspect"), _ToolSUT())
    assert rec["route"] == "tool" and rec["hit"] is True and rec["tool"]
    rec = _record(_scn("knows"), _ToolSUT())  # not routed: the view alone
    assert rec["route"] == "view" and rec["hit"] is False and rec["tool"] is None
    rec = _record(_scn("inspect", route="view"), _ToolSUT())  # the probe says view
    assert rec["hit"] is False


def test_a_system_without_the_tool_is_na_not_a_free_pass():
    rec = _record(_scn("inspect"), bench.PlainFileSUT())
    assert rec["tool"] == "n/a" and rec["hit"] is False


def test_tool_text_is_judged_for_stale_values_too():
    class Stale(_ToolSUT):
        def tool_view(self, route, wording, turn=None):
            return "Oksana said so; office moves 14 May" if route_tools(route) else None

    scn = _scn("inspect")
    scn.probes[0].expect.must_not_include = [{"date": "--05-14"}]
    rec = _record(scn, Stale())
    assert rec["hit"] is True and rec["stale"] is True


def test_legacy_equivalent_reads_the_notes_with_source_and_scenario_date(tmp_path):
    """The old Cube's memory_search + memory_read: the note opened whole, `source` and
    `created` in scenario time; memory_resolve has no legacy equivalent (n/a)."""
    sut = bench.LegacySUT()
    clock = SimpleNamespace(now=lambda: __import__("datetime").datetime(2026, 7, 6, 9, 0))
    sut.start(None, clock, None, tmp_path)  # type: ignore[arg-type]
    try:
        sut.api.sweep(
            rows=[
                {
                    "role": "other",
                    "sender": "oksana",
                    "chat_id": "kopilka-backend",
                    "ts": "2026-07-01T10:00",
                    "text": "С 1 августа переезжаем в Сапфир-Плаза",
                }
            ]
        )
        sut.api.ingest_turn(
            turn_id="t1",
            rows=[
                {"role": "owner", "text": "Когда переезд офиса Сапфир?", "ts": "2026-07-06T09:00"}
            ],
        )
        tool = sut.tool_view("memory_where_from", "Откуда это?")
        assert "kopilka-backend" in tool and "created: 2026-07-01" in tool
        assert sut.tool_view("memory_resolve", "Откуда это?") is None
        assert sut.tool_view("view", "Откуда это?") is None
    finally:
        sut.stop()


def test_api_where_from_reads_what_the_thread_was_shown_before_this_turn():
    """«Откуда это?» ("where is this from?"): the item the last turn served, its lineage as
    memory_where_from gives it; the probe's own turn (the bench view) is not «это»."""
    from koob.memory import store
    from koob.memory.eval.routes import TOOLS, api_tool_text
    from koob.memory.model import TurnCtx

    t = "2026-07-01T10:00:00"
    conn = store.open(":memory:")
    conn.execute(
        "INSERT INTO records (id, kind, version, status, author_tier, recorded_at, created_at,"
        " updated_at, origin_quote) VALUES ('f1', 'fact', 1, 'active', 'other', ?, ?, ?, ?)",
        (t, t, t, "с 1 августа переезжаем в Сапфир-Плаза"),
    )
    conn.execute(
        "INSERT INTO facts (id, subject_id, relation, value_text, text_en, fkind) SELECT 'f1',"
        " 'self', relation, 'Сапфир-Плаза', 'The office moves to Sapphire Plaza', 'plan'"
        " FROM relations LIMIT 1"
    )
    served = (
        "INSERT INTO served (thread_id, item_id, item_kind, version, turn_id, via, created_at)"
        " VALUES (?, 'f1', 'fact', 1, ?, 'basket', ?)"
    )
    turn = TurnCtx(turn_id="bench", thread_id="th-main", session_id="s", owner_words="Откуда это?")
    assert "Сапфир" not in api_tool_text(conn, "Откуда это?", turn, TOOLS)
    conn.execute(served, ("th-main", "bench", t))  # served to this very turn: not «это» ("this")
    assert "Сапфир" not in api_tool_text(conn, "Откуда это?", turn, ("memory_where_from",))
    conn.execute(served, ("th-main", "t4", t))
    text = api_tool_text(conn, "Откуда это?", turn, ("memory_where_from",))
    assert "Сапфир" in text and "1 июля" in text
    other = TurnCtx(turn_id="bench", thread_id="th-new", session_id="s2", owner_words="Откуда это?")
    assert "Сапфир" not in api_tool_text(conn, "Откуда это?", other, TOOLS)  # another thread


def test_followup_where_from_reaches_record_the_previous_bench_wording_was_shown(monkeypatch):
    """Each bench wording is its own turn: the view opens it, the tools of that wording read in
    it, so a chained «Откуда это?» reaches the record the previous wording's answer showed."""
    from koob.memory import store
    from koob.memory.eval import sut as sut_mod
    from koob.memory.eval.routes import api_tool_text

    conn = store.open(":memory:")
    t = "2026-03-02T09:00:00"
    conn.execute(
        "INSERT INTO records (id, kind, version, status, author_tier, recorded_at, created_at,"
        " updated_at, origin_quote) VALUES ('f1', 'fact', 1, 'active', 'user', ?, ?, ?, ?)",
        (t, t, t, "Запомни: мой размер перчаток 8."),
    )
    conn.execute(
        "INSERT INTO facts (id, subject_id, relation, value_text, text_en, fkind) SELECT 'f1',"
        " 'self', relation, '8', 'Denis Smith wears size 8 gloves', 'state'"
        " FROM relations LIMIT 1"
    )
    seen: list = []

    def tool_view(self, route, wording, turn=None):  # noqa: ANN001, ANN202
        seen.append(turn)
        return api_tool_text(conn, wording, turn, ("memory_where_from",))

    monkeypatch.setattr(sut_mod.ApiSUT, "tool_view", tool_view)
    sut = bench.ApiBenchSUT.__new__(bench.ApiBenchSUT)
    sut._turn_n, sut._turn = 0, None
    first = sut._new_turn("Что ты обо мне помнишь?")
    sut.tool_view("tool", "Что ты обо мне помнишь?")
    conn.execute(
        "INSERT INTO served (thread_id, item_id, item_kind, version, turn_id, via, created_at)"
        " VALUES (?, 'f1', 'fact', 1, ?, 'tool', ?)",
        (first.thread_id, first.turn_id, t),
    )
    second = sut._new_turn("Откуда это?")
    text = sut.tool_view("tool", "Откуда это?")
    assert seen == [first, second] and first.turn_id != second.turn_id
    assert "перчаток" in text and "2 марта" in text
