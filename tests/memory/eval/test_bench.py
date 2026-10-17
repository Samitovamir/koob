"""The two baseline SUTs and the memory-level metrics (offline, no model)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from koob.memory.eval import bench
from koob.memory.eval.loader import load_set
from koob.memory.eval.schema import Expect, Scenario

DEV = Path(__file__).resolve().parents[3] / "evals" / "memory" / "dev"


def probe(**expect):
    return SimpleNamespace(expect=Expect.model_validate(expect), expect_any=[])


def owner(text, **kw):
    return {"role": "owner", "text": text, "ts": "t", **kw}


def test_plain_file_takes_everything_and_rolls_back():
    sut = bench.PlainFileSUT()
    sut.api.sweep(rows=[{"role": "other", "sender": "dima", "text": "I live in Tver"}])
    sut.api.ingest_turn(turn_id="t1", rows=[owner("hi"), {"role": "cube", "text": "scripted"}])
    mark = sut.snapshot()
    sut.api.sweep(rows=[owner("later")])
    assert "later" in sut.view("q")
    sut.rollback(mark)
    text = sut.view("q")
    assert "Tver" in text and "hi" in text and "later" not in text and "scripted" not in text
    sut.api.forget("x", turn=SimpleNamespace(turn_id="t2"), because_quote="x")
    assert sut.misses == {"forget": 1}  # a file has no forget: the value stays and is counted


def test_judge_view_hit_stale_and_na():
    p = probe(must_include=["Tver", ["Omsk", "Tomsk"]], must_not_include=["Perm"])
    assert bench.judge_view(p, "Tver and Omsk") == (True, False, [])
    hit, stale, why = bench.judge_view(p, "Tver, Perm")
    assert (hit, stale) == (False, True) and any("Perm" in w for w in why)
    # a forbidden value only counts on channels the brain sees; refs and other channels do not
    f = probe(forbidden=[{"values": ["Perm", "contact:dima"], "channels": ["answer"]}])
    assert bench.judge_view(f, "Perm") == (None, None, [])
    f = probe(forbidden=[{"values": ["Perm", "contact:dima"], "channels": ["basket"]}])
    assert bench.judge_view(f, "Perm")[:2] == (None, True)
    assert bench.judge_view(probe(behaviour="silence"), "Perm")[:2] == (None, None)


class FakeMem:
    """The legacy adapter's surface, in a list."""

    def __init__(self, vault: Path) -> None:
        self.notes: dict[str, str] = {}
        self.closed = False
        self.deleted: list[tuple[str, str]] = []

    def add(self, text, source="owner"):
        slug = f"n{len(self.notes)}"
        self.notes[slug] = text
        return slug

    def top_note(self, query):
        words = set(query.lower().split())
        best = [(len(words & set(t.lower().split())), s) for s, t in self.notes.items()]
        best = [b for b in best if b[0]]
        return max(best)[1] if best else None

    def overwrite(self, slug, text, source="owner"):
        self.notes[slug] = text

    def delete(self, slug):
        self.deleted.append((slug, self.notes.pop(slug)))
        return True

    def restore_last_delete(self):
        if not self.deleted:
            return False
        slug, text = self.deleted.pop()
        self.notes[slug] = text
        return True

    def view(self, query):
        return "\n".join(self.notes.values())

    def mark(self):
        return 0

    def rollback(self, mark):
        pass

    def close(self):
        self.closed = True


def test_legacy_sut_semantics(tmp_path):
    sut = bench.LegacySUT(memory_factory=FakeMem)
    sut.start(None, None, None, tmp_path)  # type: ignore[arg-type]
    turn = SimpleNamespace(turn_id="t1")
    sut.api.sweep(rows=[{"role": "other", "chat_id": "tg:a", "text": "Dima lives in Perm"}])
    sut.api.fix("x", "Dima lives in Tver", turn=turn, because_quote="")
    sut.api.ingest_turn(turn_id="t1", rows=[owner("Dima lives in Tver")])  # handled: not a note
    assert list(sut.mem.notes.values()) == ["Dima lives in Tver"]
    sut.api.forget("Dima Tver", turn=SimpleNamespace(turn_id="t2"), because_quote="")
    assert sut.mem.notes == {} and sut.misses == {}
    sut.api.forget("nobody", turn=SimpleNamespace(turn_id="t3"), because_quote="")
    sut.api.set_chat_mode("tg:a", "banned", turn=turn)
    assert sut.misses == {"forget": 1, "chat_mode": 1}  # no hit / no legacy equivalent: a miss
    sut.stop()
    assert sut.mem is None


def test_legacy_adapter_real_vault(tmp_path):
    """The real legacy recall over a tmp vault: a saved note comes back before the turn."""
    sut = bench.LegacySUT()
    sut.start(None, None, None, tmp_path)  # type: ignore[arg-type]
    try:
        sut.api.ingest_turn(
            turn_id="a", rows=[owner("My dentist is Doctor Orlova on Birch Street")]
        )
        sut.api.ingest_turn(turn_id="b", rows=[owner("Pizza on Friday with the team")])
        mark = sut.snapshot()
        seen = sut.view("who is my dentist doctor Orlova")
        assert "Orlova" in seen and "Главное" in seen  # recall block + always-on notes
        sut.api.forget("dentist Orlova", turn=SimpleNamespace(turn_id="c"), because_quote="")
        assert "Orlova" not in sut.view("who is my dentist doctor Orlova")
        sut.rollback(mark)
        assert "Orlova" in sut.view("who is my dentist doctor Orlova")
    finally:
        sut.stop()


def test_run_bench_replays_a_group_and_summarises():
    loaded = load_set(DEV, expect_set="dev")
    res = bench.run_bench(loaded, bench.PlainFileSUT, groups=["knows"], per_group=1, seed=1)
    assert res["scenarios"] == 1 and not res["errors"] and res["records"]
    again = bench.run_bench(loaded, bench.PlainFileSUT, groups=["knows"], per_group=1, seed=1)
    assert [r["probe"] for r in again["records"]] == [r["probe"] for r in res["records"]]
    s = bench.summarize(res["records"])
    assert set(s) == {"knows", "ALL"} and s["ALL"]["hit"] is not None
    assert s["ALL"]["tokens_mean"] > 0


def test_summarize_and_markdown():
    recs = [
        {"group": "g", "hit": True, "stale": False, "tokens": 10, "ms": 1.0},
        {"group": "g", "hit": False, "stale": None, "tokens": 30, "ms": 3.0},
        {"group": "g", "hit": None, "stale": None, "tokens": 5, "ms": 0.5},
    ]
    s = bench.summarize(recs)["g"]
    assert (s["hit"], s["stale"], s["na"], s["tokens_mean"]) == (0.5, 0.0, 1, 20.0)
    data = {
        "meta": {"date": "d", "method": "m", "commands": ["c"]},
        "runs": {"plain": {"1x": {"summary": bench.summarize(recs), "misses": {"forget": 2}}}},
    }
    md = bench.render_markdown(data)
    assert "| g | 50% |" in md and "plain 1x" in md and "forget x2" in md


def test_an_e2e_only_probe_is_not_judged_at_memory_level(tmp_path):
    from .helpers import PERSONAS, scenario_dict

    d = scenario_dict()
    d["probes"][0]["level"] = "e2e"
    d["probes"].append({**d["probes"][0], "id": "p2", "level": "both"})
    r = bench.BenchRunner(Scenario.model_validate(d), PERSONAS, sut=bench.PlainFileSUT())
    res = r.run()
    assert res.error is None, res.error
    by = {p.id: p for p in res.probes}
    assert by["p1"].status == "skip" and all(w.skipped for w in by["p1"].wordings)
    assert by["p2"].status in ("pass", "fail")  # `both` still has its memory part judged
    assert {rec["probe"] for rec in r.records} == {"p2"}
