"""A replay of the same scenario gives the same store: ids in creation order, finder deadlines
out of reach of CPU load, label passes fully drained, the vector index loaded before probes."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from koob.memory import labeler, ledger
from koob.memory.clock import ManualClock
from koob.memory.eval import sut as sut_mod
from koob.memory.eval.runner import Runner
from koob.memory.eval.schema import Scenario
from koob.memory.eval.sut import EVAL_DEADLINE_MS, ApiSUT, OrderedIds
from koob.memory.recall import basket, vectors

from .._store_seed import add_fact
from .helpers import PERSONAS, scenario_dict


class _Neutral:  # a warm classifier that finds nothing: labels come from the lexicons only
    def __call__(self, text):
        return {}

    def ready(self):
        return True


class _Recorder:
    """A processor that answers nothing and keeps the raw texts of each batch, in call order."""

    route, model_id = "cloud", "fake"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def available(self):
        return True

    def __call__(self, snap):
        self.batches.append([r["text"] for r in snap["raws"]])
        return {"entities": [], "facts": [], "impressions": []}


def test_ordered_ids_sort_in_creation_order_and_repeat_per_run():
    clock = ManualClock(datetime(2026, 10, 1, 9, tzinfo=UTC))
    first = OrderedIds(clock)
    ids = [first() for _ in range(50)]  # one instant: a ULID's random part would shuffle them
    assert ids == sorted(ids) and len(set(ids)) == 50 and all(len(i) == 26 for i in ids)
    clock.advance(1)
    later = first()
    assert later > ids[-1]
    again = OrderedIds(ManualClock(datetime(2026, 10, 1, 9, tzinfo=UTC)))
    assert [again() for _ in range(50)] == ids


def test_settle_waits_for_passes_queued_by_a_running_pass(monkeypatch):
    pool = ThreadPoolExecutor(max_workers=1)
    jobs: list = []
    done: list[str] = []
    monkeypatch.setattr(labeler, "_jobs", jobs)

    def second():
        time.sleep(0.05)
        done.append("second")

    def first():
        time.sleep(0.05)
        with labeler._jobs_lock:  # noqa: SLF001
            jobs.append(pool.submit(second))
        done.append("first")

    jobs.append(pool.submit(first))
    sut_mod.settle(timeout=5)
    assert done == ["first", "second"]
    pool.shutdown()


def test_a_run_pins_ids_and_deadlines_and_stop_restores_them(tmp_path):
    real_new_id, deadlines = ledger.new_id, dict(basket.FINDER_DEADLINES_MS)
    scn = Scenario.model_validate(scenario_dict())
    sut = ApiSUT(labeler=False)
    sut.start(scn, ManualClock(), None, tmp_path)  # type: ignore[arg-type]
    try:
        assert isinstance(ledger.new_id, OrderedIds)
        assert min(basket.FINDER_DEADLINES_MS.values()) >= EVAL_DEADLINE_MS
    finally:
        sut.stop()
    assert ledger.new_id is real_new_id
    assert deadlines == basket.FINDER_DEADLINES_MS


@pytest.fixture
def hash_embedder():
    vectors.reset()
    vectors.set_embedder(vectors.HashEmbedder())
    yield
    vectors.reset()


def test_records_are_in_the_vector_index_before_a_probe(tmp_path, hash_embedder):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    scn = Scenario.model_validate(scenario_dict())
    sut = ApiSUT(labeler=False)
    sut.start(scn, ManualClock(), None, tmp_path / "link")  # type: ignore[arg-type]
    try:
        add_fact(sut.conn, "f-gym", "The gym closes at ten in the evening")
        sut.snapshot()  # the runner takes it before a probe's wordings
        assert "f-gym" in vectors._index[0].ids  # noqa: SLF001
    finally:
        sut.stop()


def _replay() -> tuple[list[list[str]], list[str]]:
    msgs = [f"Denis Smith note number {i}: the meeting moved to room {i}." for i in range(12)]
    d = scenario_dict(timeline=[{"id": f"e{i}", "ev": "user_msg", "text": t}
                                for i, t in enumerate(msgs)])  # fmt: skip
    proc = _Recorder()
    sut = ApiSUT(processor=proc, labeler=_Neutral())
    ids: list[str] = []
    r = Runner(Scenario.model_validate(d), PERSONAS, sut=sut, seed=1)
    orig = r._observe_memory  # noqa: SLF001

    def spy(p, w, wd):
        ids.extend(x[0] for x in sut.conn.execute("SELECT id FROM raw_chunks ORDER BY id"))
        return orig(p, w, wd)

    r._observe_memory = spy  # noqa: SLF001
    r.run()
    return proc.batches, ids


def test_batch_composition_and_ids_repeat_from_run_to_run():
    batches, ids = _replay()
    assert batches and ids
    assert _replay() == (batches, ids)
