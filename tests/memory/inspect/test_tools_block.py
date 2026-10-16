"""The ledger's memory tools on the brain's `cb` server (§10.4, §17, §20 rollout): absent while
`memory.engine` is legacy (nothing changes for the user), present with engine=ledger, thin
wrappers over `memory.api` on the default store."""

from __future__ import annotations

import json

import pytest

from koob import paths
from koob.assistant import tools
from koob.memory import store

from ..forget._seed import career  # noqa: F401 - fixture
from ._world import DIMA_WORK, MOVE, SASHA, status, world

LEDGER = {"memory_what", "memory_where_from", "memory_forget", "memory_resolve"}


@pytest.fixture
def ledger_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    db = store.MemoryPaths.default().db
    conn = store.open(db, home=tmp_path / "home")
    world(conn)
    tools.use_memory_engine("ledger")
    yield conn
    tools.use_memory_engine("legacy")
    conn.close()


def test_absent_with_the_legacy_engine():
    assert tools.memory_engine() == "legacy"  # config/settings.yaml: nothing changes for the user
    names = {s.name for s in tools.SPECS}
    assert not (LEDGER - {"memory_forget"}) & names
    assert not (LEDGER - {"memory_forget"}) & set(tools.BY_NAME)
    assert tools.BY_NAME["memory_forget"].fn is tools.memory_forget  # the Obsidian one


def test_present_with_the_ledger_engine_and_back(ledger_tools):
    names = [s.name for s in tools.SPECS]
    assert set(names) >= LEDGER and LEDGER <= set(tools.BY_NAME) <= set(tools.TOOL_NAMES) | {"x"}
    assert names.count("memory_forget") == 1
    assert tools.BY_NAME["memory_forget"].fn is not tools.memory_forget
    tools.use_memory_engine("ledger")  # idempotent
    assert [s.name for s in tools.SPECS].count("memory_what") == 1
    tools.use_memory_engine("legacy")
    assert not {"memory_what", "memory_where_from", "memory_resolve"} & set(tools.BY_NAME)
    assert tools.BY_NAME["memory_forget"].fn is tools.memory_forget
    assert tools.TOOL_NAMES.count("memory_forget") == 1


def _run(name, args, career):  # noqa: F811
    out = tools.run(name, args, conn=career)
    return json.loads(json.dumps(out, default=str))


def test_wrappers_call_the_api(ledger_tools, career):  # noqa: F811
    what = _run("memory_what", {"about": "Саша"}, career)
    assert any(f"m:{MOVE}" in line for line in what["lines"])
    assert MOVE in what["items"]

    where = _run("memory_where_from", {"item": f"m:{MOVE}"}, career)
    assert any("Саша переезжает в Тверь" in line for line in where["lines"])

    who = _run("memory_resolve", {"target_text": "Саша", "kind": "person"}, career)
    assert [c["entity_id"] for c in who["candidates"]] == [SASHA]

    gone = _run(
        "memory_forget",
        {"target": f"m:{DIMA_WORK}", "because_quote": "Дима"},
        career,
    )
    assert gone["outcome"] == "stopped", gone
    assert gone["line"].startswith("Перестал использовать")
    assert status(ledger_tools, DIMA_WORK) == "forgotten"


def test_tainted_tool_turn_asks_first(ledger_tools, career):  # noqa: F811
    tools.watch_turn_taint(4242, lambda: True)
    tools.watch_turn_words(4242, lambda: "открой эту страницу")
    try:
        out = tools.run(
            "memory_forget",
            {"target": f"m:{DIMA_WORK}", "because_quote": "Дима"},
            conn=career,
            turn_id=4242,
        )
    finally:
        tools.watch_turn_taint(4242, None)
        tools.watch_turn_words(4242, None)
    assert out["outcome"] == "ask_first"
    assert status(ledger_tools, DIMA_WORK) == "active"
