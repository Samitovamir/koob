"""Memory comparison harness: batching SUTs, cached role calls, answer grading. No model, no Ollama."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from koob.memory.eval import compare as c

ROOT = Path(__file__).resolve().parents[3] / "evals" / "memory"


@pytest.fixture(scope="module")
def loaded():
    return c.load_compare_set(ROOT)


def test_compare_set_adds_codes_scenario(loaded):
    ids = {s.id for s in loaded.scenarios}
    assert "codes-c01" in ids and "keeps_true-d01" in ids
    probes = c.probe_index(loaded)
    missing = [k for k in c.ANSWER_SET if k not in probes]
    assert not missing, missing  # every panel / accuracy probe exists in the set


def test_grep_lines_matches_stems():
    notes = "# Дима\n- день рождения 17 мая\n- любит настолки\n# Нина\n- работает в Polka"
    # 5-letter stems: «Димы» does not reach «Дима»; «рождения» and «день» do
    assert c.grep_lines(notes, "Когда у Димы день рождения?") == ["- день рождения 17 мая"]
    assert c.grep_lines(notes, "что?") == []


def test_chunk_lines_caps_lines_and_chars():
    batch = [f"line {i}" for i in range(70)]
    chunks = c.chunk_lines(batch, max_lines=30)
    assert [len(x) for x in chunks] == [30, 30, 10]
    big = ["x" * 2000, "y" * 2000, "z" * 9000]
    chunks = c.chunk_lines(big, max_chars=2500)
    assert len(chunks) == 3 and len(chunks[2][0]) == 2500


class _FakeUpdater:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def __call__(self, text: str) -> dict[str, Any]:
        self.inputs.append(text)
        return {"notes": f"notes v{len(self.inputs)}: Дима день рождения 17 мая"}


def test_filemem_folds_at_command_and_schedule_points(loaded):
    up = _FakeUpdater()
    res = c.run_compare(
        loaded, lambda: c.FileMemoryAgentSUT(updater=up), scenario_ids=["keeps_true-d01"]
    )
    assert res["stats"]["flushes"] == len(up.inputs) >= 1
    first = up.inputs[0]
    assert "<new_messages>" in first and "17 мая" in first  # the correction reached the model
    assert "assistant (its reply)" in first or "owner:" in first
    # p1 comes before any fold (as for the api SUT: extraction has not run): empty notes;
    # the memory-level probe at the end sees the folded notes
    assert [r["tokens"] for r in res["records"] if r["probe"] == "p1"][0] == 0
    assert [r["tokens"] for r in res["records"] if r["probe"] == "p3"][-1] > 0


def test_banned_chat_rows_never_reach_the_batch(loaded):
    up = _FakeUpdater()
    res = c.run_compare(
        loaded, lambda: c.FileMemoryAgentSUT(updater=up), scenario_ids=["policy_leaks-d01"]
    )
    assert res["stats"]["dropped"] >= 1
    joined = "\n".join(up.inputs)
    assert "VORTELL" not in joined and "Ригу" not in joined
    assert "mode 'banned'" in joined  # the setting itself is told to the notes keeper


class _Out(BaseModel):
    notes: str


def test_role_call_caches_on_disk(tmp_path):
    calls: list[str] = []

    def runner(role: str, out_model: Any, text: str, **kw: Any):
        calls.append(text)
        return out_model(notes=text.upper()), {"cost_usd": 0.5}

    rc = c.RoleCall("memory_filemem", _Out, "m", tmp_path, runner=runner)
    assert rc("abc") == {"notes": "ABC"}
    assert rc("abc") == {"notes": "ABC"}
    rc2 = c.RoleCall("memory_filemem", _Out, "m", tmp_path, runner=runner)
    assert rc2("abc") == {"notes": "ABC"}
    assert calls == ["abc"]
    assert rc.stats()["calls"] == 1 and rc.stats()["hits"] == 1 and rc2.hits == 1
    assert rc.stats()["cost_usd"] == 0.5


def test_grade_answers_code_first_then_judge(loaded):
    probes = c.probe_index(loaded)
    base = {"now": "2026-04-13T10:00:00+00:00", "view": "", "guarantee": "G5"}
    captured = [
        {**base, "scenario": "keeps_true-d01", "probe": "p1", "wording": "Когда?"},
        {**base, "scenario": "keeps_true-d01", "probe": "p1", "wording": "Когда?!"},
    ]
    answers = iter(["17 мая.", "14 мая."])
    judged: list[str] = []

    def brain(text: str) -> dict[str, Any]:
        return {"answer": next(answers)}

    def judge(text: str) -> dict[str, Any]:
        judged.append(text)
        return {"verdict": "fail", "reason": "old date as current"}

    rows = c.grade_answers(captured, probes, brain, judge)
    assert [r["passed"] for r in rows] == [True, False]
    assert [r["by"] for r in rows] == ["code", "judge"]
    assert len(judged) == 1 and "forbidden" in judged[0]


def test_panel_view_rules(loaded):
    probes = c.probe_index(loaded)
    captured = [
        {"scenario": "keeps_true-d01", "probe": "p1", "guarantee": "G5", "view": "17 мая и 14 мая"},
        {"scenario": "keeps_true-d01", "probe": "p1", "guarantee": "G5", "view": "17 мая"},
        {"scenario": "knows-d01", "probe": "p1", "guarantee": None, "view": ""},
    ]
    out = c.panel_view(captured, probes)
    assert [r["ok"] for r in out] == [False, True]


def test_merge_runs_sums_and_keeps_max_wall():
    a = {
        "sut": "x",
        "records": [],
        "captured": [1],
        "errors": [],
        "stats": {"calls": 2, "wall_s": 5},
    }
    b = {
        "sut": "x",
        "records": [],
        "captured": [2],
        "errors": ["e"],
        "stats": {"calls": 3, "wall_s": 9},
    }
    m = c.merge_runs([a, b])
    assert m["stats"] == {"calls": 5, "wall_s": 9} and m["captured"] == [1, 2]


def test_ollama_guard(monkeypatch):
    from koob.memory.eval import judge as j

    monkeypatch.setattr(j, "swap_used_gb", lambda: 0.0)
    monkeypatch.setattr(j, "free_memory_gb", lambda: 10.0)
    tags = {
        "models": [
            {"name": "llm", "size": 7e9},
            {"name": "emb", "size": 1e9},
            {"name": "big", "size": 2e10},
        ]
    }

    def http(loaded: list[str]):
        return lambda p: tags if p == "/api/tags" else {"models": [{"name": n} for n in loaded]}

    c.ollama_guard("llm", http(["llm", "emb"]))  # a small embedder next to the LLM is fine
    with pytest.raises(c.GuardAbort, match="another big model"):
        c.ollama_guard("llm", http(["llm", "big"]))
    with pytest.raises(c.GuardAbort, match="GB free"):
        c.ollama_guard("llm", http([]))
    monkeypatch.setattr(j, "swap_used_gb", lambda: 13.0)
    with pytest.raises(c.GuardAbort, match="swap"):
        c.ollama_guard("llm", http(["llm"]))
