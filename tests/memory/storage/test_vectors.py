"""Acceptance for the memory spec §9.1a (vectors.db orphan sweep).

Spec marker: `[T storage/test_vectors::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import api, store
from koob.memory.recall import vectors

from .._store_seed import add_fact


@pytest.fixture(autouse=True)
def _fake_embedder():
    vectors.reset()
    vectors.set_embedder(vectors.HashEmbedder())
    yield
    vectors.reset()


def test_orphans_swept_after_crash(mem):
    """§9.1a: a crash between the ledger commit and the vector delete is healed at the next open."""
    add_fact(mem, "f-keep", "The gym closes at ten in the evening")
    add_fact(mem, "f-gone", "Sasha birthday is on May 3")
    vpath = store.MemoryPaths.of(store.db_file(mem)).vectors
    vdb = vectors.open_db(vpath)
    assert vectors.embed_pending(mem, vdb) == 2
    vdb.close()
    # the ledger commit of a forget lands, then the process dies before the vector delete
    mem.execute("UPDATE records SET status = 'forgotten' WHERE id = 'f-gone'")
    vectors.reset()  # a new process: nothing in RAM
    vectors.set_embedder(vectors.HashEmbedder())

    report = api.startup(store=mem)

    assert report.vectors_swept == 1
    vdb = vectors.open_db(vpath)
    try:
        assert [r[0] for r in vdb.execute("SELECT record_id FROM vectors")] == ["f-keep"]
    finally:
        vdb.close()
    assert vectors.load_index()  # startup attached the hot path to this file
    hits = vectors.search(vectors.embed_query("Sasha birthday"), label_mask=-1)
    assert [h[0] for h in hits] == ["f-keep"]
