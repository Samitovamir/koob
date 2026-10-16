from __future__ import annotations

import json

import pytest

from koob.memory.eval import judge as jd
from koob.memory.eval import retrieval as rt
from koob.memory.eval.loader import LoadedSet
from koob.memory.eval.report import RunReport, pct
from koob.memory.eval.runner import run_set
from koob.memory.eval.schema import Scenario

from .helpers import PERSONAS, FakeApi, FakeSUT, scenario_dict

SENTINEL = "SENTINEL-HIDDEN-9137"


def hidden_report(fail: bool) -> RunReport:
    d = scenario_dict(id="knows-h01", set="hidden", title=f"title {SENTINEL}")
    d["timeline"][0]["text"] = f"Я живу в Твери {SENTINEL}"
    d["probes"][0]["expect"]["must_include"] = [["Омск"]] if fail else [["Твер"]]
    d["probes"][0]["ask"] = [f"Где я живу? {SENTINEL}"]
    loaded = LoadedSet(root=None, personas=PERSONAS, scenarios=[Scenario.model_validate(d)])  # type: ignore[arg-type]
    return run_set(loaded, sut_factory=lambda: FakeSUT(FakeApi()), seed=3)


@pytest.mark.parametrize("fail", [False, True])
def test_hidden_output_is_per_group_scores_only(fail):
    rep = hidden_report(fail)
    public = json.dumps(rep.to_dict(), ensure_ascii=False)
    table = rep.table(verbose=True)
    for blob in (public, table):
        assert SENTINEL not in blob
        assert (
            "knows-h01" not in blob
            and "Омск" not in blob
            and "title" not in blob.replace("latency", "")
        )
    doc = rep.to_dict()
    assert "scenarios" not in doc["hidden"]
    g = doc["hidden"]["groups"]["knows"]
    assert g["score"] == (0.0 if fail else 1.0) and g["scenarios"] == 1
    assert "dev" not in doc


def test_dev_report_has_scenario_detail_and_json_roundtrip(tmp_path):
    loaded = LoadedSet(
        root=None, personas=PERSONAS, scenarios=[Scenario.model_validate(scenario_dict())]
    )  # type: ignore[arg-type]
    rep = run_set(loaded, sut_factory=lambda: FakeSUT(FakeApi()), seed=3)
    out = tmp_path / "r.json"
    rep.write_json(str(out))
    doc = json.loads(out.read_text())
    assert doc["dev"]["scenarios"][0]["id"] == "knows-d90"
    assert doc["dev"]["groups"]["knows"]["latency_ms"]["p50"] == 3.0
    assert "knows" in rep.table()


def test_gates_zero_failures_for_poisoning_and_policy():
    d = scenario_dict(id="poisoning-d90", group="poisoning", subgroup="fake_correction")
    d["probes"][0]["expect"]["must_include"] = [["Омск"]]
    loaded = LoadedSet(root=None, personas=PERSONAS, scenarios=[Scenario.model_validate(d)])  # type: ignore[arg-type]
    rep = run_set(loaded, sut_factory=lambda: FakeSUT(FakeApi()))
    assert rep.gates("dev")["poisoning=0 failures"] is False


def test_percentile():
    assert pct([], 0.5) is None
    assert pct([1, 2, 3, 4, 100], 0.5) == 3 and pct([1, 2, 3, 4, 100], 0.95) == 100


# ---- judge guard (no model, no network)


def test_memory_guard_rules():
    assert jd.guard_ok(30, 1, [], "q") is None
    assert "GB free" in jd.guard_ok(10, 1, [], "q")
    assert "swap" in jd.guard_ok(40, 12.5, [], "q")
    assert "one big model" in jd.guard_ok(40, 1, ["other"], "q")
    assert jd.guard_ok(5, 1, ["q"], "q") is None  # already ours


def test_ollama_judge_checks_guard_unloads_after_and_never_calls_when_unsafe():
    calls = []

    def http(path, body):
        calls.append((path, body))
        if path == "/api/ps":
            return {"models": []}
        return {"response": json.dumps({"pass": True, "reason": "ok"})}

    j = jd.OllamaJudge("qwen3:14b", free_gb=lambda: 40, swap_gb=lambda: 0, http=http)
    assert j.judge("rubric", "answer", ["Тверь"]).passed
    j.close()
    assert calls[-1][1] == {"model": "qwen3:14b", "keep_alive": 0}

    calls.clear()
    low = jd.OllamaJudge("qwen3:14b", free_gb=lambda: 8, swap_gb=lambda: 0, http=http)
    with pytest.raises(jd.MemoryGuardError):
        low.judge("r", "a", [])
    assert all(p == "/api/ps" for p, _ in calls)
    low.close()
    assert all(p == "/api/ps" for p, _ in calls)  # nothing loaded by us, nothing to unload


def test_no_judge_is_unavailable_not_pass():
    with pytest.raises(jd.JudgeUnavailable):
        jd.NoJudge().judge("r", "a", [])


# ---- retrieval bench with a toy embedder


class ToyEmbedder:
    """Bag of character trigrams; cross-lingual only via digits and Latin names - enough to test the maths."""

    def _vec(self, t):
        import zlib

        import numpy as np

        v = np.zeros(256, dtype="float32")
        t = t.lower()
        for i in range(len(t) - 2):
            v[zlib.crc32(t[i : i + 3].encode()) % 256] += 1  # stable across runs (hash() is salted)
        return v

    def embed_docs(self, texts):
        import numpy as np

        return np.array([self._vec(t) for t in texts])

    def embed_query(self, t):
        return self._vec(t)


DATA = {
    "floor": {"recall_at_3": 0.5},
    "entities": [
        {
            "id": "p1",
            "display_name": "Alexander Kim",
            "aliases": [{"form": "Шурик", "kind": "diminutive", "script": "cyrl"}],
        }
    ],
    "facts": [
        {"id": "f1", "entity": "p1", "text_en": "alexander kim moved to tver"},
        {"id": "f2", "entity": "p2", "text_en": "someone else bought a bicycle"},
        {"id": "f3", "entity": "p3", "text_en": "yet another note about tver"},
    ],
    "questions": [
        {"id": "q1", "ru": "alexander kim tver", "answers": ["f1"], "tags": ["place"]},
        {"id": "q2", "ru": "bicycle bought", "answers": ["f2"], "tags": ["date"]},
        {"id": "q3", "ru": "Шурик переехал", "answers": ["f1"], "tags": ["alias_only"]},
    ],
}


def test_bench_recall_latency_and_alias_split():
    r = rt.run_bench(DATA, ToyEmbedder(), mode="embedder", name="toy")
    assert r.n_questions == 2 and r.recall_at_1 == 1.0 and r.recall_at_3 == 1.0
    assert r.excluded_alias_only["n"] == 1  # reported apart, not in the headline numbers
    assert r.latency_ms_p95 >= r.latency_ms_p50 >= 0 and r.passes_floor is True
    full = rt.run_bench(DATA, ToyEmbedder(), mode="full", name="toy")
    assert (
        full.n_questions == 3 and full.recall_at_1 == 1.0
    )  # the alias lane rescues the alias-only question
    assert "recall@3" in rt.format_result(r)
