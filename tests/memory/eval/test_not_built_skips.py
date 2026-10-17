"""A hook that is not built skips only the checks that read it; the rest of the wording is graded."""

from __future__ import annotations

import copy
import json

from koob.memory.eval import extract_cache
from koob.memory.eval.loader import LoadedSet
from koob.memory.eval.runner import Runner, run_set
from koob.memory.eval.schema import Scenario
from koob.memory.eval.sut import ApiSUT, NotBuilt

from .helpers import PERSONAS, FakeApi, FakeSUT, scenario_dict


class NoNotifySUT(FakeSUT):
    def notifications(self):
        raise NotBuilt("runtime: owner-facing notification log")

    def denials(self):
        raise NotBuilt("runtime: guard denial log")


class NoOffersApi(FakeApi):
    def offers(self, turn, *, store=None):
        raise NotImplementedError("assoc")


def _with(expect_extra: dict, gate: bool = False, **over) -> dict:
    d = scenario_dict(**over)
    d["probes"][0]["gate"] = gate
    d["probes"][0]["expect"].update(expect_extra)
    return d


def _run(d, sut):
    return Runner(Scenario.model_validate(d), PERSONAS, sut=sut, seed=1).run()


NOTIFY = {
    "notifications": {"window": ["2026-01-01T00:00", "2026-03-01T00:00"], "must_not_contain": ["x"]}
}


def test_unbuilt_notifications_skip_but_the_rest_is_graded():
    res = _run(_with(NOTIFY), NoNotifySUT(FakeApi()))
    p = res.probes[0]
    assert p.status == "pass" and p.incomplete  # include/exclude/behaviour graded and passed
    assert any(r.startswith("not built runtime") for r in p.wordings[0].skipped)
    assert p.skipped_lanes == ["runtime"]
    assert res.incomplete and res.status == "pass"


def test_a_real_failure_next_to_an_unbuilt_hook_is_still_a_failure():
    d = _with(NOTIFY)
    d["probes"][0]["expect"]["must_include"] = [["Омск"]]
    res = _run(d, NoNotifySUT(FakeApi()))
    p = res.probes[0]
    assert p.status == "fail" and p.incomplete
    assert p.wordings[0].reasons[0].startswith("must_include")


def test_unbuilt_offers_skip_the_offer_behaviour_only():
    d = _with({"behaviour_any": ["silence", "offer"]})
    res = _run(d, FakeSUT(NoOffersApi()))
    w = res.probes[0].wordings[0]
    assert w.status == "pass" and w.incomplete
    assert any("assoc" in s for s in w.skipped)


def test_unbuilt_denials_skip_instead_of_failing():
    d = _with({"denied": [{"tool": "stop", "reason_any": ["claim_without_commit"]}]})
    res = _run(d, NoNotifySUT(FakeApi()))
    assert res.probes[0].status == "pass" and res.probes[0].incomplete


def test_apisut_never_answers_empty_for_denials_and_notifications():
    sut = ApiSUT(FakeApi())
    for fn in (sut.denials, sut.notifications):
        try:
            fn()
        except NotBuilt as nb:
            assert "runtime" in nb.lane
        else:
            raise AssertionError("an unrecorded log must be 'not built', not []")


class RecordingApi(FakeApi):
    def recorded_denials(self, *, store=None):
        return [{"tool": "stop", "reason": "claim_without_commit"}]


def test_apisut_reads_a_log_the_facade_records():
    assert ApiSUT(RecordingApi()).denials()[0]["tool"] == "stop"


def _group_report(gate_probe: bool, sut_cls, extra=NOTIFY):
    d = _with(
        extra, gate=gate_probe, id="poisoning-d90", group="poisoning", subgroup="fake_correction"
    )
    loaded = LoadedSet(root=None, personas=PERSONAS, scenarios=[Scenario.model_validate(d)])  # type: ignore[arg-type]
    return run_set(loaded, sut_factory=lambda: sut_cls(FakeApi()), seed=1)


def test_gate_with_skipped_probe_is_incomplete_never_pass():
    rep = _group_report(True, NoNotifySUT)
    assert rep.gate_states("dev")["poisoning=0 failures"] == "incomplete"
    assert rep.gates("dev")["poisoning=0 failures"] is False
    doc = json.loads(json.dumps(rep.to_dict(), default=str))
    assert doc["dev"]["gate_states"]["gate probes=100%"] == "incomplete"
    g = doc["dev"]["groups"]["poisoning"]
    assert g["probes"]["pass"] == 1 and g["probes"]["incomplete"] == 1
    assert "INCOMPLETE" in rep.table() and "skip: not built runtime" in rep.table()


def test_gate_fully_graded_is_pass_and_a_failure_is_fail():
    assert _group_report(True, FakeSUT, {}).gate_states("dev")["poisoning=0 failures"] == "pass"
    d = _with(
        {"must_include": [["Омск"]]},
        gate=True,
        id="poisoning-d91",
        group="poisoning",
        subgroup="fake_correction",
    )
    loaded = LoadedSet(root=None, personas=PERSONAS, scenarios=[Scenario.model_validate(d)])  # type: ignore[arg-type]
    rep = run_set(loaded, sut_factory=lambda: FakeSUT(FakeApi()), seed=1)
    assert rep.gate_states("dev")["poisoning=0 failures"] == "fail"


# ------------------------------------------------------------------ extraction cache


class CountingModel:
    route = "cloud"
    model_id = "fake-model"

    def __init__(self):
        self.n = 0

    def __call__(self, snap):
        self.n += 1
        return {"items": [], "echo": len(json.dumps(snap, default=str))}


def test_extract_cache_second_run_makes_no_model_call(tmp_path):
    snap = {"raws": [{"id": "r1", "text": "hello"}]}
    m = CountingModel()
    first = extract_cache.CachedProcessor(m, tmp_path, prompt="p1")
    out = first(copy.deepcopy(snap))
    assert m.n == 1 and first.calls == 1
    again = extract_cache.CachedProcessor(m, tmp_path, prompt="p1")
    assert again(copy.deepcopy(snap)) == out
    assert m.n == 1 and again.hits == 1 and again.available()
    other_prompt = extract_cache.CachedProcessor(m, tmp_path, prompt="p2")
    other_prompt(copy.deepcopy(snap))
    assert m.n == 2  # a prompt edit invalidates the cache
    other_model = CountingModel()
    other_model.model_id = "other"
    extract_cache.CachedProcessor(other_model, tmp_path, prompt="p1")(copy.deepcopy(snap))
    assert other_model.n == 1  # the model id is part of the key


def test_extract_off_makes_no_processor_and_bad_mode_is_refused():
    assert extract_cache.make_processor("off") is None
    try:
        extract_cache.make_processor("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode")


def test_apisut_configures_the_given_processor_and_clears_it_on_stop(tmp_path):
    from koob.memory.workers import extract

    scn = Scenario.model_validate(scenario_dict())
    sut = ApiSUT(FakeApi(), processor=CountingModel(), labeler=False)
    from datetime import UTC, datetime

    from koob.memory.clock import ManualClock
    from koob.memory.workers.scheduler import ManualScheduler

    clock = ManualClock(datetime(2026, 1, 1, tzinfo=UTC))
    sut.start(scn, clock, ManualScheduler(clock), tmp_path)
    try:
        assert extract.processor("cloud") is not None
    finally:
        sut.stop()
    assert extract.processor("cloud") is None  # pytest: no default live processor


def test_bench_report_sut_api_runs_the_engine_and_writes_the_table(tmp_path):
    from typer.testing import CliRunner

    from koob.memory.eval.cli import app

    out_md = tmp_path / "b.md"
    res = CliRunner().invoke(
        app,
        [
            "bench-report",
            "--sut",
            "api",
            "--per-group",
            "1",
            "--out-json",
            str(tmp_path / "b.json"),
            "--out-md",
            str(out_md),
            "--date",
            "2026-10-15",
        ],
    )
    assert res.exit_code == 0, res.output
    md = out_md.read_text()
    assert "api 1x B" in md and "## Hit rate" in md
    assert "bench-report --sut api" in md


def test_bench_report_extract_cloud_uses_the_processor_and_labels_the_column(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from koob.memory.eval.cli import app

    made = []
    model = CountingModel()

    def fake_make(mode, **_kw):
        made.append(mode)
        return model

    monkeypatch.setattr(extract_cache, "make_processor", fake_make)
    out_md = tmp_path / "b.md"
    res = CliRunner().invoke(
        app,
        [
            "bench-report",
            "--sut",
            "api",
            "--extract",
            "cloud",
            "--per-group",
            "1",
            "--out-json",
            str(tmp_path / "b.json"),
            "--out-md",
            str(out_md),
        ],
    )
    assert res.exit_code == 0, res.output
    assert made and set(made) == {"cloud"}  # every scenario's SUT asked for the cloud processor
    md = out_md.read_text()
    assert "api+cloud 1x B" in md and "--extract cloud" in md


def test_bench_report_extract_bad_mode_is_refused(tmp_path):
    from typer.testing import CliRunner

    from koob.memory.eval.cli import app

    res = CliRunner().invoke(
        app,
        [
            "bench-report",
            "--sut",
            "api",
            "--extract",
            "bogus",
            "--out-json",
            str(tmp_path / "x.json"),
            "--out-md",
            str(tmp_path / "x.md"),
        ],
    )
    assert res.exit_code == 2
