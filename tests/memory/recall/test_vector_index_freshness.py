"""The in-RAM vector index follows the file the embed job writes: a store opened through a
symlinked folder, a background load overtaken by a job, a load before the file exists, and
the scheduler embedding on every tick."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from koob.memory import store
from koob.memory.clock import ManualClock
from koob.memory.policy import ALL_LABELS
from koob.memory.recall import vectors
from koob.memory.workers import scheduler

from .._store_seed import add_fact

ALL = int(ALL_LABELS)


@pytest.fixture(autouse=True)
def fake():
    vectors.reset()
    vectors.set_embedder(vectors.HashEmbedder())
    yield
    vectors.reset()


def _embed(mem) -> None:
    vectors.enqueue(mem)
    store.run_queued_jobs(mem, vectors.JOB_KIND)


def _found(query: str) -> list[str]:
    return [rid for rid, _ in vectors.search(vectors.embed_query(query), label_mask=ALL)]


def test_job_refreshes_ram_for_a_store_opened_through_a_symlinked_folder(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    mem = store.open(link / "memory" / "memory.db", home=link / "home")
    try:
        vectors.attach(link / "memory" / "vectors.db")  # the name the caller knows
        add_fact(mem, "f-gym", "The gym closes at ten in the evening")
        _embed(mem)
        assert _found("gym")[0] == "f-gym"
        add_fact(mem, "f-bday", "Denis Smith has his birthday on May 3")
        _embed(mem)  # the next turn sees what this job wrote
        assert _found("birthday")[0] == "f-bday"
        assert set(vectors._index[0].ids) == {"f-gym", "f-bday"}  # noqa: SLF001
    finally:
        mem.close()


def test_a_background_load_never_replaces_a_newer_index(mem, monkeypatch):
    add_fact(mem, "f-gym", "The gym closes at ten in the evening")
    _embed(mem)
    vpath = store.MemoryPaths.of(store.db_file(mem)).vectors
    vectors.attach(vpath)
    real_read = vectors._read_index  # noqa: SLF001

    def read_then_job_lands(vdb):
        old = real_read(vdb)
        monkeypatch.setattr(vectors, "_read_index", real_read)
        add_fact(mem, "f-bday", "Denis Smith has his birthday on May 3")
        _embed(mem)  # the job refreshes RAM while the load still holds the older rows
        return old

    monkeypatch.setattr(vectors, "_read_index", read_then_job_lands)
    vectors.load_index()
    assert set(vectors._index[0].ids) == {"f-gym", "f-bday"}  # noqa: SLF001


def test_a_load_before_the_file_exists_lets_the_next_search_retry(tmp_path):
    vectors.attach(tmp_path / "memory" / "vectors.db")
    vectors._load_started[0] = True  # noqa: SLF001 - as a search's background load sets it
    assert vectors.load_index() is False
    assert vectors._load_started[0] is False  # noqa: SLF001


def test_every_tick_embeds_after_its_jobs(mem, monkeypatch):
    calls: list[str] = []

    def run(conn, kind):
        calls.append(kind)
        return 1 if kind == scheduler.EXTRACT else 0

    monkeypatch.setattr(store, "run_queued_jobs", run)
    scheduler.tick(mem, ManualClock(datetime(2026, 10, 1, 9, tzinfo=UTC)))
    assert calls == [scheduler.SAFETY, scheduler.EXTRACT, scheduler.EMBED]


def test_records_written_by_a_tick_are_searchable_after_it(mem):
    add_fact(mem, "f-gym", "The gym closes at ten in the evening")
    scheduler.tick(mem, ManualClock(datetime(2026, 10, 1, 9, tzinfo=UTC)))
    assert _found("gym")[0] == "f-gym"
