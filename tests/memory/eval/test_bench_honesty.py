"""The yardstick is honest: the labeler is on in every new-memory bench run, each scenario starts
from a clean, settled store whatever ran before it, the extraction cache keys on what the model
reads, a value shown as history is not stale for any system, and a probe after a `during` event
has a time."""

from __future__ import annotations

import json
import threading
import time

from koob.memory import labeler
from koob.memory.eval import bench, extract_cache
from koob.memory.eval.bench import ApiBenchSUT, judge_view
from koob.memory.eval.matchers import shown_as_current
from koob.memory.eval.runner import Runner
from koob.memory.eval.schema import Probe, Scenario
from koob.memory.eval.sut import ApiSUT

from .helpers import PERSONAS, scenario_dict

# ------------------------------------------------------------------ the labeler is on


def test_bench_sut_runs_with_the_labeler_outside_pytest(monkeypatch):
    assert ApiBenchSUT().labeler_on() is False  # under pytest: no real model load
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert ApiBenchSUT().labeler_on() is True  # every bench run outside the tests
    assert ApiBenchSUT(labeler=False).labeler_on() is False  # only when asked off


# ------------------------------------------------------------------ hermetic per scenario


class _SlowNeutral:
    """A warm classifier that finds nothing and takes its time, so an unwaited background
    label pass is still running when the next step reads the store."""

    def __call__(self, text):
        time.sleep(0.03)
        return {}

    def ready(self):
        return True


class _Recorder:
    """Fake cloud processor: one fact per batch; records what it was handed and whether a
    label pass was still pending at that moment."""

    route, model_id = "cloud", "fake"

    def __init__(self):
        self.snaps: list[str] = []
        self.pending: list[bool] = []

    def available(self):
        return True

    def __call__(self, snap):
        with labeler._jobs_lock:
            self.pending.append(any(not f.done() for f in labeler._jobs))
        self.snaps.append(extract_cache.canonical(snap)[0])  # versions and finality included
        raw = snap["raws"][-1]
        return {"entities": [], "impressions": [], "facts": [{
            "ref": "f1", "kind": "fact", "subject": "self", "relation": "lives_in",
            "value_text": raw["text"][:20], "value_entity": None, "text_en": "A fact.",
            "quote_orig": raw["text"][:20], "lang_orig": "ru", "fkind": "state",
            "polarity": "affirmed", "source_raw_id": raw["id"], "reasons": [],
        }]}  # fmt: skip


def _scn(sid: str, texts: list[str]) -> Scenario:
    d = scenario_dict(
        id=sid,
        timeline=[
            {"id": f"e{i}", "ev": "user_msg", "at": f"+{i}h", "text": t}
            for i, t in enumerate(texts, 1)
        ],
    )
    return Scenario.model_validate(d)


A = ["Я живу в Твери.", "Мой кот Бублик.", "Работаю в «Полке».", "Люблю чай."]
B = ["Сестру зовут Аля.", "Мы переехали на Динамо.", "Нет, я живу в Твери."]


def _run(scenarios: list[Scenario], seed: int) -> dict[str, tuple]:
    out = {}
    for scn in scenarios:
        rec = _Recorder()
        sut = ApiSUT(processor=rec, labeler=_SlowNeutral())
        r = Runner(scn, PERSONAS, sut=sut, seed=seed)
        seen: dict = {}
        orig = r._observe_memory

        def spy(p, w, wd, sut=sut, seen=seen, orig=orig):
            seen["rows"] = sut.conn.execute(
                "SELECT kind, status, version, labels, labels_final FROM records"
                " ORDER BY kind, version, labels"
            ).fetchall()
            return orig(p, w, wd)

        r._observe_memory = spy
        res = r.run()
        assert res.error is None, res.error
        with labeler._jobs_lock:  # nothing of this scenario may land in the next one
            assert all(f.done() for f in labeler._jobs)
        out[scn.id] = (rec.snaps, [tuple(x) for x in seen["rows"]], rec.pending)
    return out


def test_scenario_is_hermetic_whatever_ran_before_and_with_which_seed():
    a, b = _scn("knows-d91", A), _scn("knows-d92", B)
    alone = _run([b], seed=7)["knows-d92"]
    after = _run([a, b], seed=3)["knows-d92"]  # another scenario and another seed first
    again = _run([b], seed=7)["knows-d92"]
    assert alone[0], "the processor was never called: the test proves nothing"
    assert not any(alone[2]) and not any(after[2])  # no label pass in flight at a snapshot
    assert alone[0] == after[0] == again[0]  # the model saw the same input, bookkeeping too
    assert alone[1] == after[1] == again[1]  # and the store held the same records


def test_rollback_waits_for_the_label_passes_of_the_wording(tmp_path):
    from koob.memory.clock import ManualClock
    from koob.memory.model import TurnCtx

    scn = _scn("knows-d93", ["Я живу в Твери."])
    sut = ApiSUT(labeler=_SlowNeutral())
    from datetime import UTC, datetime

    sut.start(scn, ManualClock(datetime(2026, 1, 10, 6, 0, tzinfo=UTC)), None, tmp_path)  # type: ignore[arg-type]
    try:
        snap = sut.snapshot()
        before = sut.conn.execute("SELECT COUNT(*) FROM raw_chunks").fetchone()[0]
        recs = sut.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        turn = TurnCtx(turn_id="t1", thread_id="th", owner_words="Я люблю кофе.")
        sut.api.remember(
            "Я люблю кофе.", turn=turn, because_quote="Я люблю кофе.", store=sut.store()
        )
        sut.rollback(snap)
        with labeler._jobs_lock:
            assert all(f.done() for f in labeler._jobs)
        assert sut.conn.execute("SELECT COUNT(*) FROM raw_chunks").fetchone()[0] == before
        assert sut.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == recs
    finally:
        sut.stop()


def test_facade_calls_settle_label_passes_before_returning():
    gate = threading.Event()
    calls = []

    class _Api:
        def poke(self):
            calls.append("poke")

    from koob.memory.eval.sut import Guarded

    g = Guarded(_Api(), after=lambda: (calls.append("settle"), gate.set()))
    g.poke()
    assert calls == ["poke", "settle"] and gate.is_set()


# ------------------------------------------------------------------ the cache key


def _snap(order: tuple[int, int], versions: tuple[int, int], final: bool = True) -> dict:
    ents = [
        {"handle": "m:01K00000000000000000000AAA", "id": "01K00000000000000000000AAA",
         "etype": "product", "display_name": "Maze", "version": versions[0]},
        {"handle": "m:01K00000000000000000000BBB", "id": "01K00000000000000000000BBB",
         "etype": "product", "display_name": "Lookback", "version": versions[1]},
    ]  # fmt: skip
    return {
        "raws": [{"id": "01K00000000000000000000CCC", "rev": 1, "tier": "user",
                  "speaker": "self", "ts": "2026-09-23T09:00:00Z", "text": "Беру Maze.",
                  "chat": None, "source": "owner"}],
        "entities": [ents[i] for i in order],
        "facts": [{"handle": "m:01K00000000000000000000DDD", "id": "01K00000000000000000000DDD",
                   "subject": "m:01K00000000000000000000BBB", "relation": "other",
                   "value_text": "платный", "text_en": "Paid.", "version": 3, "labels": 0,
                   "labels_final": final, "tier": "user"}],
        "relations": [{"relation": "other", "family": "x", "cardinality": "many"}],
    }  # fmt: skip


def test_cache_key_is_what_the_model_reads(tmp_path):
    base = _snap((0, 1), (5, 8))
    same_input = [_snap((1, 0), (5, 8)), _snap((0, 1), (6, 11)), _snap((0, 1), (5, 8), False)]
    key = extract_cache.view_hash(base)
    assert all(extract_cache.view_hash(s) == key for s in same_input)
    other = _snap((0, 1), (5, 8))
    other["raws"][0]["text"] = "Беру Lookback."
    assert extract_cache.view_hash(other) != key
    dec = _snap((0, 1), (5, 8))
    dec["decisions"] = [{"handle": "m:01K00000000000000000000EEE", "id": "01K00000000000000000000EEE",
                         "chosen_text": "Maze", "domain": "tools"}]  # fmt: skip
    assert extract_cache.view_hash(dec) != key  # whatever the model reads moves the key

    class _Model:
        route, model_id = "cloud", "m"
        calls = 0

        def __call__(self, snap):
            _Model.calls += 1
            return {"facts": [{"subject": snap["entities"][0]["handle"]}]}

    cp = extract_cache.CachedProcessor(_Model(), tmp_path, prompt="p")
    first = cp(base)
    for s in same_input:
        out = extract_cache.CachedProcessor(_Model(), tmp_path, prompt="p")(s)
        assert out == first  # ids map back to the same entity, whatever the id order
    assert _Model.calls == 1


def test_old_cache_still_answers_through_its_bookkeeping_variants(tmp_path):
    gate = _snap((1, 0), (7, 9))  # what an earlier run keyed the whole-snapshot cache on
    cp = extract_cache.CachedProcessor(object(), tmp_path, prompt="p")
    cp.model_id = "m"
    path = cp._path(gate)
    path.parent.mkdir(parents=True)
    ids = extract_cache.canonical(gate)[1]
    path.write_text(
        json.dumps({"facts": [{"subject": f"m:{ids['01K00000000000000000000AAA']}"}]}),
        encoding="utf-8",
    )
    now = _snap((0, 1), (5, 8))  # same model input, other id order and versions
    assert cp.lookup(now) is None  # the exact keys miss
    out = cp.lookup(now, fuzzy=True)
    assert out == {"facts": [{"subject": "m:01K00000000000000000000AAA"}]}


# ------------------------------------------------------------------ history is not stale


def test_a_value_shown_as_history_is_not_shown_as_current():
    old = {"date": "--05-14"}
    for text in (
        "Встреча 10 мая. Раньше было 14 мая.",
        "Meeting on 10 May; earlier note said 14 May, wrong.",
        "14 мая — неверно, теперь 10 мая.",
        "Date: 14 May → 10 May",
        "- previously «14.05»\n- now 10 May",
    ):
        assert not shown_as_current(text, old), text
    for text in (
        "Встреча 14 мая.",
        "14 мая. Раньше было 10 мая.",  # the mark belongs to another clause
        "Meeting 14 May, previously 10 May",  # the past word comes after the old value
    ):
        assert shown_as_current(text, old), text


def test_bench_and_graders_read_allow_history_alike():
    from koob.memory.eval.graders import _occurrence_allowed

    p = Probe.model_validate({
        "id": "p1", "at": "2026-05-12T09:00", "level": "memory", "ask": ["Когда встреча?"],
        "expect": {"must_include": [{"date": "--05-10"}], "forbidden": [
            {"values": [{"date": "--05-14"}], "channels": ["core", "basket"], "allow": "history"},
        ]},
    })  # fmt: skip
    hist = "Встреча 10 мая (earlier note said 14 May - wrong)."
    cur = "Встреча 14 мая."
    assert judge_view(p, hist)[:2] == (True, False)
    assert judge_view(p, cur)[1] is True
    assert _occurrence_allowed("core", hist, {"date": "--05-14"})
    assert not _occurrence_allowed("core", cur, {"date": "--05-14"})
    q = p.model_copy(deep=True)
    q.expect.forbidden[0].allow = None  # «never», not «not as current»: history still counts
    assert judge_view(q, hist)[1] is True


# ------------------------------------------------------------------ the KeyError 'e3' crash


def test_probe_after_a_during_event_runs_at_its_host_time():
    d = scenario_dict(
        timeline=[
            {"id": "e1", "ev": "user_msg", "text": "Я живу в Твери.", "reply": "Понял."},
            {"id": "e2", "ev": "user_msg", "at": "+1h", "text": "Глянь чат с Глебом.",
             "reply": "Смотрю."},
            {"id": "e3", "ev": "other_msg", "during": "e2", "from": "gleb", "chat": "gleb",
             "msg_id": 1, "text": "Я в Твери."},
        ],
        probes=[{"id": "p1", "after": "e3", "level": "memory", "ask": ["Где я живу?"],
                 "expect": {"must_include": [["Твер"]]}}],
    )  # fmt: skip
    scn = Scenario.model_validate(d)
    r = Runner(scn, PERSONAS, sut=bench.PlainFileSUT(), seed=1)
    plan = r._plan()
    assert r.event_time["e3"] == r.event_time["e2"]
    kinds = [(s.kind, s.ev.id if s.ev else s.probe.id) for s in plan]  # type: ignore[union-attr]
    assert kinds.index(("probe", "p1")) > kinds.index(("event", "e2"))
    res = bench.BenchRunner(scn, PERSONAS, sut=bench.PlainFileSUT(), seed=1).run()
    assert res.error is None, res.error
