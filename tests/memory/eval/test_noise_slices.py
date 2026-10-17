"""A noise block without `at` is spread over its span, interleaved with the timeline by time."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from koob.memory.eval.runner import Runner
from koob.memory.eval.schema import Scenario
from koob.memory.eval.timeutil import to_utc

from .helpers import PERSONAS, FakeApi, FakeSUT, scenario_dict

MSK = ZoneInfo("Europe/Moscow")
NOISE = {
    "id": "n1", "ev": "noise", "seed": 11, "count": 60,
    "span": {"from": "2026-01-11", "to": "2026-01-20"},
    "chats": {"tg:family": 1.0}, "kinds": {"smalltalk": 1.0}, "cast": ["gleb"],
}  # fmt: skip


def _local(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=MSK).astimezone(UTC)


class _Logged(Runner):
    """Logs every noise delivery and every probe with the clock at that moment."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.log: list[tuple[str, datetime, list[datetime]]] = []

    def _deliver_noise(self, msgs) -> None:
        self.log.append(("noise", self.clock.now(), [to_utc(m.at, self.tz) for m in msgs]))
        super()._deliver_noise(msgs)

    def _run_probe_tree(self, probe, workdir):
        self.log.append(("probe:" + probe.id, self.clock.now(), []))
        return super()._run_probe_tree(probe, workdir)


def _sent_before(r: _Logged, probe: str) -> list[datetime]:
    upto = [k for k, *_ in r.log].index("probe:" + probe)
    return [t for _, _, ts in r.log[:upto] for t in ts]


def _run(d: dict, **kw) -> tuple[_Logged, object, FakeApi]:
    api = FakeApi()
    r = _Logged(Scenario.model_validate(d), PERSONAS, sut=FakeSUT(api), seed=1, **kw)
    return r, r.run(), api


def _scn(**over) -> dict:
    d = scenario_dict(**over)
    d["timeline"] = [
        {"id": "e1", "ev": "user_msg", "text": "Я живу в Твери."},
        dict(NOISE),
        {"id": "e2", "ev": "correction", "at": "2026-01-15T12:00", "text": "Нет, я живу в Твери."},
    ]
    d["probes"][0]["at"] = "2026-01-16T10:00"
    return d


def test_an_event_dated_inside_a_noise_span_keeps_its_own_time():
    r, res, _ = _run(_scn())
    assert r.event_time["e2"] == _local("2026-01-15T12:00")
    assert res.status == "pass", res  # the probe on the 16th sees the move of the 15th


def test_a_probe_inside_the_span_sees_the_noise_sent_so_far_and_nothing_later():
    r, res, _ = _run(_scn())
    assert res.error is None, res.error
    probe_at = _local("2026-01-16T10:00")
    before = _sent_before(r, "p1")
    after = [t for kind, _, ts in r.log if kind == "noise" for t in ts][len(before) :]
    assert before and after  # the noise arrives on both sides of the probe
    assert max(before) < probe_at <= min(after)
    assert len(before) + len(after) == NOISE["count"]


def test_no_message_arrives_after_a_later_step_or_worker_run():
    """Each slice is delivered at its last message's time, and no event, probe or scheduled
    worker run (evening 21:00, night 03:00) falls between a message and its delivery."""
    r, _, _ = _run(_scn())
    cuts = sorted(s.t for s in r._plan() if s.kind != "noise")
    lo, hi = _local("2026-01-11T00:00"), _local("2026-01-21T00:00")
    cuts += [t for t, _ in r._schedule_points(lo, hi)]
    slices = [(now, ts) for kind, now, ts in r.log if kind == "noise"]
    assert len(slices) > 10  # spread over the days, not one block at the end
    for now, ts in slices:
        assert max(ts) == now
        assert not [c for c in cuts if min(ts) < c <= now]


def test_the_night_pass_runs_between_noise_days():
    r, _, api = _run(_scn())
    assert api.workers.count("night:run") >= 5  # each night of the span has new noise to see


def test_a_relative_time_right_after_noise_counts_from_the_end_of_the_span():
    d = _scn()
    d["timeline"][2] = {"id": "e2", "ev": "correction", "at": "+1h", "text": "Нет, я живу в Твери."}
    d["timeline"].append({"id": "e3", "ev": "user_msg", "at": "+2h", "text": "Привет."})
    r, _, _ = _run(d)
    end = _local("2026-01-21T00:00")
    assert r.event_time["n1"] == end
    assert r.event_time["e2"] == end + timedelta(hours=1)
    assert r.event_time["e3"] == end + timedelta(hours=3)  # counts from e2, not from the noise


def test_a_probe_after_the_noise_block_sees_all_of_it_and_its_hooks_run_once():
    d = _scn()
    d["probes"][0].pop("at")
    d["probes"][0]["after"] = "n1"
    d["timeline"].append(
        {"id": "w1", "ev": "worker", "job": "night", "at": "2026-01-12T10:00", "snapshot_at": "n1"}
    )
    r, _, api = _run(d)
    assert len(_sent_before(r, "p1")) == NOISE["count"]
    assert api.workers.count("night:snapshot") == 1
