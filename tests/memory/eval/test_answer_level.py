"""Answer level of the comparison: probe sample, outcome classes, score table, cache identity."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from koob.memory.eval import answer_level as al
from koob.memory.eval import compare as c

ROOT = Path(__file__).resolve().parents[3] / "evals" / "memory"


@pytest.fixture(scope="module")
def loaded():
    return c.load_compare_set(ROOT)


def _askable(loaded) -> Counter[str]:
    n: Counter[str] = Counter()
    for s in loaded.scenarios:
        n[s.group] += sum(1 for p in s.probes if p.ask)
    return n


def test_sample_is_deterministic_and_keeps_old_probes(loaded):
    a = al.stratified_sample(loaded.scenarios, 80, 7, keep=set(c.ANSWER_SET))
    b = al.stratified_sample(loaded.scenarios, 80, 7, keep=set(c.ANSWER_SET))
    other = al.stratified_sample(loaded.scenarios, 80, 8, keep=set(c.ANSWER_SET))
    assert a == b
    assert a != other
    assert set(c.ANSWER_SET) <= a
    assert len(a) == 80


def test_sample_has_a_floor_per_group_and_silent_groups(loaded):
    pick = al.stratified_sample(loaded.scenarios, 80, 1, keep=set(c.ANSWER_SET))
    group_of = {s.id: s.group for s in loaded.scenarios}
    got = Counter(group_of[sid] for sid, _ in pick)
    have = _askable(loaded)
    for g, size in have.items():
        if size:
            assert got[g] >= min(3, size), (g, got[g], size)
    for g in ("policy_leaks", "poisoning", "safety_net", "keeps_true"):
        assert got[g] >= 3


def test_sample_is_proportional_beyond_the_floor(loaded):
    pick = al.stratified_sample(loaded.scenarios, 80, 3, keep=set())
    group_of = {s.id: s.group for s in loaded.scenarios}
    got = Counter(group_of[sid] for sid, _ in pick)
    assert got["safety_net"] > got["c1"]  # 16 askable probes against 4


def test_sample_only_takes_probes_with_a_wording(loaded):
    asked = {(s.id, p.id) for s in loaded.scenarios for p in s.probes if p.ask}
    pick = al.stratified_sample(loaded.scenarios, 80, 5, keep=set(c.ANSWER_SET))
    assert pick <= asked


def test_outcome_classes():
    assert al.outcome({"passed": True}, "knows") == "correct"
    assert al.outcome({"passed": False, "code_stale": True}, "keeps_true") == "stale"
    assert al.outcome({"passed": False, "code_stale": True}, "policy_leaks") == "leak"
    assert al.outcome({"passed": False, "code_stale": True, "guarantee": "G3"}, "x") == "leak"
    assert al.outcome({"passed": False, "code_stale": False}, "knows") == "miss"
    assert al.outcome({"passed": False, "code_stale": None}, None) == "miss"


def test_score_table_counts_per_group_and_total():
    rows = [
        {"group": "knows", "class": "correct", "noisy": False},
        {"group": "knows", "class": "miss", "noisy": True},
        {"group": "poisoning", "class": "leak", "noisy": False},
    ]
    table = al.score_table({"api@abc": {"answers": rows}, "plain@abc": {"answers": rows[:1]}})
    assert table["knows"]["api@abc"] == {
        "correct": 1,
        "stale": 0,
        "leak": 0,
        "miss": 1,
        "noisy": 1,
        "n": 2,
    }
    assert table["ALL"]["api@abc"]["n"] == 3 and table["ALL"]["api@abc"]["leak"] == 1
    assert "poisoning" not in table or "plain@abc" not in table["poisoning"]
    text = al.format_table(table, ["api@abc", "plain@abc"])
    assert "knows" in text and "ALL" in text


class _Out(BaseModel):
    answer: str


def test_cache_key_includes_the_commit(tmp_path):
    def runner(role: str, out: type, text: str, **kw: Any):
        return _Out(answer=text), {}

    a = c.RoleCall("r", _Out, "m", tmp_path, runner=runner, commit="aaa")
    b = c.RoleCall("r", _Out, "m", tmp_path, runner=runner, commit="bbb")
    assert a.path("same input") != b.path("same input")
    a("same input")
    b("same input")
    assert (a.calls, b.calls) == (1, 1)  # the second commit is not served from the first
    a("same input")
    assert a.hits == 1


def test_checkout_commit_names_the_checkout():
    sha = al.checkout_commit()
    assert sha == "unknown" or len(sha.split("-")[0]) == 10


def test_grade_answers_marks_noise_and_class(loaded):
    probes = c.probe_index(loaded)
    item = {
        "now": "2026-04-13T10:00:00+00:00",
        "view": "",
        "guarantee": "G5",
        "group": "keeps_true",
        "scenario": "keeps_true-d01",
        "probe": "p1",
        "wording": "Когда?",
    }
    rows = c.grade_answers(
        [item],
        probes,
        lambda t: {"answer": "14 мая."},
        lambda t: {"verdict": "fail", "reason": "old", "noisy": True},
    )
    assert rows[0]["class"] == "stale"  # a retired value, not a policy leak
    assert rows[0]["noisy"] is True
