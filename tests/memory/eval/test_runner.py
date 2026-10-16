from __future__ import annotations

import copy

from koob.memory.eval.loader import LoadedSet
from koob.memory.eval.runner import Runner, run_set
from koob.memory.eval.schema import Scenario
from koob.memory.eval.sut import ApiSUT

from .helpers import PERSONAS, FakeApi, FakeSUT, scenario_dict, stores_in


def run(d: dict, api: FakeApi | None = None, stores=None, seed: int = 1):
    api = api or FakeApi()
    scn = Scenario.model_validate(d)
    return Runner(scn, PERSONAS, sut=FakeSUT(api, stores), seed=seed).run(), api


def test_correction_replay_and_paraphrases_pass():
    res, api = run(scenario_dict())
    assert res.status == "pass", res
    p = res.probes[0]
    assert p.status == "pass" and p.score == 1.0 and len(p.wordings) == 2
    assert api.city == "Твери"  # state restored to the post-timeline state after both wordings
    assert res.basket_ms == [3.0, 3.0]


def test_a_wrong_expectation_fails_with_the_reason():
    d = scenario_dict()
    d["probes"][0]["expect"]["must_include"] = [["Омск"]]
    res, _ = run(d)
    assert res.status == "fail"
    assert "missing: Омск" in res.probes[0].wordings[0].reasons[0]


def test_during_event_lands_inside_the_turn_and_order_is_kept():
    d = scenario_dict()
    d["timeline"] = [
        {"id": "e0", "ev": "user_msg", "text": "Я живу в Твери."},
        {"id": "e1", "ev": "user_msg", "text": "Составь мне план на неделю.", "reply": "Готово."},
        {"id": "e2", "ev": "correction", "text": "Нет, я живу в Твери.", "during": "e1"},
    ]
    res, api = run(d)
    assert res.status == "pass", res.probes[0].wordings[0].reasons
    assert api.city == "Твери" and api.history[0] == "Твери"
    assert api.calls.count("startup") == 0


def test_clock_is_injected_and_scheduled_workers_run_on_the_way():
    d = scenario_dict()
    d["timeline"].append({"id": "e3", "ev": "advance", "by": "+4d"})
    api = FakeApi()
    res, _ = run(d, api)
    assert res.status == "pass"
    assert "night:run" in api.workers  # 03:00 crossed with new data


def test_manual_workers_run_only_on_worker_events():
    d = scenario_dict(setup={"workers": "manual"})
    d["timeline"].append({"id": "w1", "ev": "worker", "job": "night", "at": "+1d"})
    api = FakeApi()
    run(d, api)
    assert api.workers == ["night:run"]


def test_worker_interleave_points_use_snapshot_and_commit_phases():
    d = scenario_dict(setup={"workers": "manual"})
    d["timeline"] = [
        {"id": "e1", "ev": "user_msg", "text": "Я живу в Твери."},
        {"id": "w1", "ev": "worker", "job": "extract", "snapshot_at": "e1", "commit_after": "e3"},
        {"id": "e2", "ev": "user_msg", "text": "Нет, я живу в Твери."},
        {"id": "e3", "ev": "correction", "text": "Я живу в Твери."},
    ]
    api = FakeApi()
    res, _ = run(d, api)
    assert api.workers == ["extract:snapshot", "extract:commit"]
    assert res.status == "pass"


def test_backup_restore_and_restart():
    d = scenario_dict()
    d["timeline"] = [
        {"id": "e1", "ev": "user_msg", "text": "Я живу в Твери."},
        {"id": "b1", "ev": "backup", "name": "before"},
        {"id": "e2", "ev": "correction", "text": "Нет, я живу в Твери."},
        {"id": "r1", "ev": "restore", "name": "before"},
        {"id": "x1", "ev": "restart", "kind": "app"},
    ]
    d["probes"][0]["expect"] = {
        "behaviour": "answer",
        "must_include": ["Твер"],
        "must_not_include": ["Твер"],
    }
    api = FakeApi()
    res, _ = run(d, api)
    assert res.status == "pass", res.probes[0].wordings[0].reasons
    assert "startup" in api.calls


def test_chained_probe_runs_in_the_same_trial_per_wording():
    d = scenario_dict()
    d["probes"].append({
        "id": "p2", "after": "p1", "level": "memory", "ev_after": [{"ev": "restart", "kind": "session"}],
        "ask": ["А ещё раз?"], "expect": {"must_include": ["Твер"]},
    })  # fmt: skip
    res, _ = run(d)
    assert [p.id for p in res.probes] == ["p1", "p2"]
    assert all(p.status == "pass" for p in res.probes)
    assert len(res.probes[1].wordings) == 2  # cycles over the parent's wordings


def test_noise_is_replayed_in_bulk_and_scaled():
    d = scenario_dict()
    d["timeline"].insert(1, {
        "ev": "noise", "seed": 7, "count": 40, "span": {"from": "2026-01-11", "to": "2026-01-20"},
        "chats": {"tg:family": 1.0}, "kinds": {"smalltalk": 1.0}, "cast": ["gleb"],
    })  # fmt: skip
    api = FakeApi()
    log = []
    orig = api.sweep
    api.sweep = lambda **kw: log.append(len(kw["rows"])) or orig(**kw)
    scn = Scenario.model_validate(d)
    Runner(scn, PERSONAS, sut=FakeSUT(api), scale="10x", seed=1).run()
    assert sum(log) == 400


def test_probe_level_e2e_is_skipped_without_live_and_both_runs_memory_part():
    d = scenario_dict()
    d["probes"][0]["level"] = "e2e"
    res, _ = run(d)
    assert res.probes[0].status == "skip" and res.status == "skip" and res.score is None
    d["probes"][0]["level"] = "both"
    res, _ = run(d)
    assert res.probes[0].status == "pass" and res.probes[0].e2e_pending


def test_canary_banned_chat_passes_and_a_leaky_system_fails(tmp_path):
    d = scenario_dict(
        group="policy_leaks", subgroup="banned", id="policy_leaks-d90",
        setup={"chat_policy": {"tg:gleb": "banned"}},
        canaries=[{"id": "c1", "lat": "VORTELL-5521", "cyr": "ВОРТЕЛЛ-5521", "dig": "5521-3307", "allowed": ["none"]}],
    )  # fmt: skip
    d["timeline"] = [
        {
            "id": "e1",
            "ev": "other_msg",
            "from": "gleb",
            "chat": "tg:gleb",
            "msg_id": 1,
            "text": "код {{c1.lat}} / {{c1.cyr}} / {{c1.dig}}",
        }
    ]
    d["probes"] = [
        {
            "id": "p1",
            "at": "2026-02-01T10:00",
            "level": "memory",
            "gate": True,
            "expect": {"scan": {"canaries": ["c1"], "stores": "all"}},
        }
    ]

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    res, _ = run(copy.deepcopy(d), FakeApi(clean_dir / "raw.log"), stores_in(clean_dir))
    assert res.status == "pass", res.probes[0].wordings[0].reasons

    leaky_dir = tmp_path / "leaky"
    leaky_dir.mkdir()
    res, _ = run(copy.deepcopy(d), FakeApi(leaky_dir / "raw.log", leak=True), stores_in(leaky_dir))
    assert res.status == "fail" and res.gate_failures == 1
    assert "unlisted hit" in res.probes[0].wordings[0].reasons[0]


def test_tokens_differ_between_runs(tmp_path):
    d = scenario_dict(
        canaries=[{"id": "c1", "lat": "AAAA-1", "cyr": "ААА-1", "dig": "1-2", "allowed": ["none"]}]
    )
    d["timeline"][0]["text"] = "Я живу в Твери {{c1.lat}}"
    a = Runner(Scenario.model_validate(d), PERSONAS, sut=FakeSUT(FakeApi()), seed=1)
    b = Runner(Scenario.model_validate(d), PERSONAS, sut=FakeSUT(FakeApi()), seed=2)
    assert a.canaries[0].lat != b.canaries[0].lat
    assert a.scn.timeline[0].text.endswith(a.canaries[0].lat)


def test_real_unbuilt_api_names_the_lanes_and_grades_the_rest():
    scn = Scenario.model_validate(scenario_dict())
    res = Runner(scn, PERSONAS, sut=ApiSUT(), seed=1).run()
    assert res.not_built and all(isinstance(x, str) for x in res.not_built)
    w = res.probes[0].wordings[0]
    assert w.status == "fail"  # no model: nothing extracts, so the wording fails on a real check
    assert not w.reasons[0].startswith("not built:")  # graded, not blanket-failed
    if res.pipeline_gaps:  # a failure next to a stage that is still unbuilt names it
        assert "pipeline not built" in w.reasons[0]


def test_run_set_filters_groups():
    loaded = LoadedSet(
        root=None, personas=PERSONAS, scenarios=[Scenario.model_validate(scenario_dict())]
    )  # type: ignore[arg-type]
    rep = run_set(loaded, groups=["c1"], sut_factory=lambda: FakeSUT(FakeApi()))
    assert rep.results == []
    rep = run_set(loaded, groups=["knows"], sut_factory=lambda: FakeSUT(FakeApi()))
    assert len(rep.results) == 1
