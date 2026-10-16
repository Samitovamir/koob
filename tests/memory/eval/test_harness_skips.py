"""Harness fixes: what a memory-level run cannot see is skipped, reruns hit the extraction
cache, and every scenario searches its own vector index."""

from __future__ import annotations

import copy

from koob.memory.eval import extract_cache, graders
from koob.memory.eval.observe import Observation
from koob.memory.eval.schema import Expect, Scenario
from koob.memory.eval.sut import ApiSUT
from koob.memory.recall import vectors

from .helpers import PERSONAS, scenario_dict

ULID_A, ULID_B = "01M4YXVEHXYBZXPC3QRJ05T1H1", "01M4YZ5QVA7N2TR4ZPJX2WDR6Q"


def _obs(level: str, **kw) -> Observation:
    o = Observation(wording="q", level=level, tz="Europe/Moscow")
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _statuses(checks):
    return sorted((c.name, c.status) for c in checks)


def test_memory_level_never_fails_what_only_the_brain_does():
    e = Expect.model_validate(
        {"behaviour": "action", "actions": [{"tool": "tg_send", "target": "tg:7", "count": 1}]}
    )
    mem = _obs("memory", behaviours={"no_action", "answer"})
    got = graders.grade_expect(mem, e)
    assert all(c.status == graders.SKIP for c in got if c.name.startswith(("behaviour", "actions")))
    assert any(c.reason == graders.BRAIN_SKIP for c in got)
    live = _obs("e2e", behaviours={"answer", "no_action"})
    assert graders.FAIL in {c.status for c in graders.grade_expect(live, e)}  # e2e still fails


def test_behaviour_any_with_a_brain_only_choice_skips_at_memory_level_only():
    e = Expect.model_validate({"behaviour_any": ["ask_one", "refuse_line"]})
    assert _statuses(graders.check_behaviour(_obs("memory", behaviours={"answer"}), e)) == [
        ("behaviour_any", graders.SKIP)
    ]
    assert _statuses(graders.check_behaviour(_obs("e2e", behaviours={"answer"}), e)) == [
        ("behaviour_any", graders.FAIL)
    ]
    ask = Expect.model_validate({"behaviour": "ask_one"})  # the basket can ask: still graded
    assert _statuses(graders.check_behaviour(_obs("memory", behaviours={"answer"}), ask)) == [
        ("behaviour", graders.FAIL)
    ]


class _Insp:
    def records(self, where):
        return [{"kind": "fact", "value_text": "Tver", "status": "active"}]


def test_count_zero_failure_names_zero_not_at_least_one():
    e = Expect.model_validate(
        {"records": [{"where": {"kind": "fact", "value": ["Tver"]}, "count": 0}]}
    )
    [c] = graders.check_records(_obs("memory", inspector=_Insp()), e.records)
    assert c.status == graders.FAIL and c.reason == "1 matching records, wanted 0"


class Echo:
    route, model_id = "cloud", "m"

    def __init__(self):
        self.n = 0

    def __call__(self, snap):
        self.n += 1
        rid = snap["raws"][0]["id"]
        return {"facts": [{"ref": "f1", "source_raw_id": rid, "quote_orig": "hi"}]}


def test_extract_cache_hits_when_only_the_wall_clock_ids_differ(tmp_path):
    def snap(rid: str, fid: str) -> dict:
        return {"raws": [{"id": rid, "ts": "2026-01-10T09:00:00Z", "text": "hi"}],
                "facts": [{"handle": f"m:{fid}", "id": fid}]}  # fmt: skip

    m = Echo()
    first = extract_cache.CachedProcessor(m, tmp_path, prompt="p")
    assert first(snap(ULID_A, "x" + "a" * 24))["facts"][0]["source_raw_id"] == ULID_A
    again = extract_cache.CachedProcessor(m, tmp_path, prompt="p")
    out = again(snap(ULID_B, "x" + "b" * 24))
    assert m.n == 1 and again.hits == 1
    assert out["facts"][0]["source_raw_id"] == ULID_B  # the answer speaks this run's ids
    assert ULID_A not in next(tmp_path.rglob("*.json")).read_text()
    changed = copy.deepcopy(snap(ULID_B, "x" + "b" * 24))
    changed["raws"][0]["text"] = "bye"
    again(changed)
    assert m.n == 2  # the content still keys the cache


def test_each_scenario_searches_its_own_vector_index(tmp_path):
    scn = Scenario.model_validate(scenario_dict())
    try:
        for name in ("a", "b"):
            sut = ApiSUT(labeler=False)
            sut.start(scn, None, None, tmp_path / name)  # type: ignore[arg-type]
            assert vectors._attached[0] == tmp_path / name / "memory" / "vectors.db"  # noqa: SLF001
            sut.seed(scn, PERSONAS)
            sut.stop()
            assert vectors._attached[0] is None  # noqa: SLF001
    finally:
        vectors.attach(None)


def test_a_correction_the_engine_asks_about_is_not_an_engine_verdict():
    from koob.memory.api import WriteResult
    from koob.memory.eval.runner import ASK_FIRST_GAP, Runner

    from .helpers import FakeApi, FakeSUT

    class Asks(FakeApi):
        def fix(self, target, correction, *, turn, because_quote, store=None):
            return WriteResult(outcome="ask_first", line="Что именно исправить: …?")

    d = scenario_dict()
    d["probes"][0]["expect"]["must_include"] = [["Омск"]]  # any failure after the ask
    scn = Scenario.model_validate(d)
    res = Runner(scn, PERSONAS, sut=FakeSUT(Asks()), seed=1).run()
    assert ASK_FIRST_GAP in res.pipeline_gaps and any("asked first" in n for n in res.notes)
    w = res.probes[0].wordings[0]
    assert w.status == "fail" and f"[pipeline not built: {ASK_FIRST_GAP}]" in w.reasons[0]
    ok = Runner(scn, PERSONAS, sut=FakeSUT(FakeApi()), seed=1).run()
    assert ASK_FIRST_GAP not in ok.pipeline_gaps and not ok.notes


class _Neutral:  # a warm classifier that finds nothing: labels come from the lexicons only
    def __call__(self, text):
        return {}

    def ready(self):
        return True


class _OneFact:
    route, model_id = "cloud", "fake"

    def available(self):
        return True

    def __call__(self, snap):
        raw = snap["raws"][0]
        return {"entities": [], "impressions": [], "facts": [{
            "ref": "f1", "kind": "fact", "subject": "self", "relation": "lives_in",
            "value_text": "Тверь", "value_entity": None, "text_en": "Lives in Tver.",
            "quote_orig": "Я живу в Твери", "lang_orig": "ru", "fkind": "state",
            "polarity": "affirmed", "source_raw_id": raw["id"], "reasons": [],
        }]}  # fmt: skip


def test_worker_records_are_final_by_the_next_turn_like_the_app_timer():
    from koob.memory.eval.runner import Runner

    d = scenario_dict(timeline=[{"id": "e1", "ev": "user_msg", "text": "Я живу в Твери."}])
    scn = Scenario.model_validate(d)
    sut = ApiSUT(processor=_OneFact(), labeler=_Neutral())
    seen: dict = {}
    r = Runner(scn, PERSONAS, sut=sut, seed=1)
    orig = r._observe_memory

    def spy(p, w, wd):
        seen["final"] = sut.conn.execute(
            "SELECT kind, labels_final FROM records WHERE kind = 'fact'"
        ).fetchall()
        return orig(p, w, wd)

    r._observe_memory = spy
    r.run()
    assert seen["final"] and all(tuple(x)[1] == 1 for x in seen["final"])


def test_notices_are_read_from_the_engine_and_stamped_in_scenario_time(tmp_path):
    from datetime import UTC, datetime

    from koob.memory.clock import ManualClock

    d = scenario_dict()
    d["setup"] = {"known": ["Я люблю чай."]}
    scn = Scenario.model_validate(d)
    start = datetime(2026, 1, 10, 6, 0, tzinfo=UTC)
    sut = ApiSUT(labeler=False)
    sut.start(scn, ManualClock(start), None, tmp_path)  # type: ignore[arg-type]
    try:
        sut.seed(scn, PERSONAS)
        notes = sut.notifications()
        assert notes and notes[-1]["at"] == start  # not today's wall clock
        assert isinstance(sut.denials(), list)  # recorded by the engine now, never NotBuilt
    finally:
        sut.stop()
