"""night: the nightly exam from the raw archive, cross-family judge, same routing table
(MEMORY.md §5.4, §18). No live model: fakes only."""

from __future__ import annotations

import pytest

from koob.memory.model import AuthorTier, Label, Route
from koob.memory.workers import exam

from .. import _ledger as L


class FakeJudge:
    def __init__(self, family: str = "qwen", route: Route = Route.LOCAL, score: float = 1.0):
        self.family, self.route, self.score, self.calls = family, route, score, []

    def grade(self, *, question: str, expected: str, answer: str) -> float:
        self.calls.append((question, expected))
        return self.score


def _seed(conn, value: str = "Berlin", text: str = "Sasha lives in Berlin") -> None:
    L.add(conn, L.entity("sasha", "Sasha"))
    L.raw(conn, "r1", f"я знаю, что {text}")
    L.add(
        conn,
        L.fact("f1", "sasha", "lives_in", value, text=text, source_raw_id="r1"),
        derived_from=["r1"],
    )


def test_same_family_judge_refused(mem):
    with pytest.raises(ValueError):
        exam.use_judge(FakeJudge(family="claude"))
    with pytest.raises(ValueError):
        exam.run_exam(mem, judge=FakeJudge(family="Anthropic"))


def test_code_check_on_exact_values(mem):
    _seed(mem)
    got = exam.run_exam(mem, recall=lambda c, q: ["Sasha lives in Berlin"], seed=1)
    assert got["sampled"] == 1 and got["code_checked"] == 1 and got["kept"] == 1.0
    lost = exam.run_exam(mem, recall=lambda c, q: ["Sasha lives in Paris"], seed=1)
    assert lost["kept"] == 0.0


def test_free_text_judge_obeys_routing(mem):
    value = "a small flat near the old harbour with a view"
    L.add(mem, L.entity("sasha", "Sasha"))
    L.raw(mem, "r1", f"Sasha: I live in {value}", tier="other", labels=int(Label.HEALTH))
    fact = L.fact(
        "f1",
        "sasha",
        "lives_in",
        value,
        text=f"Sasha lives in {value}",
        source_raw_id="r1",
        tier=AuthorTier.OTHER,
        speaker_entity_id="sasha",
    )
    L.add(mem, fact, derived_from=["r1"])
    mem.execute("UPDATE raw_chunks SET labels_final = 1")
    mem.execute("UPDATE records SET labels = labels | 1, labels_final = 1 WHERE id = 'f1'")
    cloud = FakeJudge(route=Route.CLOUD)  # a health label routes local only (§5.4)
    got = exam.run_exam(mem, judge=cloud, recall=lambda c, q: [value], seed=1)
    assert cloud.calls == [] and got["skipped_route"] == 1 and got["probes"] == 0
    local = FakeJudge(route=Route.LOCAL)
    got = exam.run_exam(mem, judge=local, recall=lambda c, q: [value], seed=1)
    assert got["judged"] == 1 and got["kept"] == 1.0
    assert local.calls == [("Sasha lives in", value)]  # the exact expected value
