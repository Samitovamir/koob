"""vec: meaning search in a separate, rebuildable vectors.db (MEMORY.md §3.1, §3.2
invariants, §9.1a, §10.2 vector finder, §10.3, §18 G12). Offline: a deterministic hash
embedder is injected for every unit test; one real-model test skips without a cached model."""

from __future__ import annotations

import logging

import pytest

from koob.memory import forget, store
from koob.memory.model import Label
from koob.memory.policy import ALL_LABELS
from koob.memory.recall import vectors

from .._store_seed import add_fact

ALL = int(ALL_LABELS)


@pytest.fixture(autouse=True)
def fake():
    vectors.reset()
    emb = vectors.HashEmbedder()
    vectors.set_embedder(emb)
    yield emb
    vectors.reset()


@pytest.fixture
def vpath(mem):
    return store.MemoryPaths.of(store.db_file(mem)).vectors


@pytest.fixture
def vdb(vpath):
    conn = vectors.open_db(vpath)
    yield conn
    conn.close()


def _seed(mem):
    add_fact(mem, "f-bday", "Sasha birthday is on May 3")
    add_fact(mem, "f-gym", "The gym closes at ten in the evening")
    add_fact(mem, "f-old", "Sasha lived in Tver", status="superseded")


def _rows(vdb, embedder_id="hash@1"):
    return {
        r[0]: r[1]
        for r in vdb.execute(
            "SELECT record_id, version FROM vectors WHERE embedder_id = ?", (embedder_id,)
        )
    }


def _ids(hits):
    return [h[0] for h in hits]


def test_embed_pending_pulls_usable_records_once(mem, vdb):
    _seed(mem)
    assert vectors.embed_pending(mem, vdb) == 2
    assert vectors.embed_pending(mem, vdb) == 0
    assert set(_rows(vdb)) == {"f-bday", "f-gym"}  # superseded: not usable, never embedded
    assert vectors.serving_id(vdb) == "hash@1"
    assert mem.execute("SELECT embedder_id FROM meta").fetchone()[0] == "hash@1"


def test_search_ranks_by_meaning(mem, vdb):
    _seed(mem)
    vectors.embed_pending(mem, vdb)
    hits = vectors.search(vectors.embed_query("when is Sasha birthday"), label_mask=ALL)
    assert _ids(hits)[0] == "f-bday"
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert len(vectors.search(vectors.embed_query("gym"), label_mask=ALL, limit=1)) == 1


def test_label_mask_fences_others_items_not_his_own(mem, vdb):
    _seed(mem)
    add_fact(mem, "f-med", "Sasha birthday clinic visit", labels=int(Label.HEALTH))
    mem.execute("UPDATE records SET author_tier = 'other' WHERE id = 'f-med'")
    add_fact(mem, "f-mine", "my birthday clinic checkup", labels=int(Label.HEALTH))
    vectors.embed_pending(mem, vdb)
    q = vectors.embed_query("birthday clinic")
    got = _ids(vectors.search(q, label_mask=0))
    assert "f-med" not in got and {"f-mine", "f-bday"} <= set(got)
    assert "f-med" in _ids(vectors.search(q, label_mask=int(Label.HEALTH)))
    # labels refreshed by the next pull, no re-embedding needed
    mem.execute(
        "UPDATE records SET labels = ?, author_tier = 'other' WHERE id = 'f-bday'",
        (int(Label.HEALTH),),
    )
    assert vectors.embed_pending(mem, vdb) == 0
    assert "f-bday" not in _ids(vectors.search(q, label_mask=0))


def test_new_record_version_is_reembedded(mem, vdb):
    _seed(mem)
    vectors.embed_pending(mem, vdb)
    mem.execute("UPDATE records SET version = 2 WHERE id = 'f-bday'")
    mem.execute("UPDATE facts SET text_en = 'Sasha birthday is on June 9' WHERE id = 'f-bday'")
    assert vectors.embed_pending(mem, vdb) == 1
    assert _rows(vdb)["f-bday"] == 2


def test_deadline_gives_partial_or_empty(mem, vdb, monkeypatch):
    _seed(mem)
    add_fact(mem, "f-zoo", "Sasha birthday party at the zoo")
    vectors.embed_pending(mem, vdb)
    q = vectors.embed_query("Sasha birthday")
    assert vectors.search(q, label_mask=ALL, deadline_ms=0) == []
    monkeypatch.setattr(vectors, "CHUNK_ROWS", 1)
    tick = iter(i * 0.010 for i in range(1000))
    monkeypatch.setattr(vectors, "_now", lambda: next(tick))
    got = _ids(vectors.search(q, label_mask=ALL, deadline_ms=25))
    assert sorted(got) == ["f-bday", "f-gym"]  # rows ordered by id: two chunks fit in 25 ms


def test_no_fastembed_gives_empty_and_logs_once(mem, vdb, monkeypatch, caplog):
    _seed(mem)
    vectors.reset()  # no injected embedder: the configured fastembed model
    monkeypatch.setattr(vectors, "_fastembed_present", lambda: False)
    with caplog.at_level(logging.WARNING, logger=vectors.__name__):
        assert vectors.embed_query("Sasha") == []
        assert vectors.embed_query("Sasha") == []
        assert vectors.search([0.1] * 64, label_mask=ALL) == []
        assert vectors.embed_pending(mem, vdb) == 0
    assert sum("fastembed" in r.getMessage() for r in caplog.records) == 1


def test_embedder_change_reembeds_while_old_vectors_serve(mem, vdb, fake):
    """G12: a new embedder_id re-embeds in the background; the old vectors stay in use until
    the new ones cover every record, then serving switches and the old rows go."""
    _seed(mem)
    vectors.embed_pending(mem, vdb)
    new = vectors.HashEmbedder("hash@2", dim=32)
    vectors.set_embedder(new)

    def stop(_n):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        vectors.embed_pending(mem, vdb, batch_size=1, on_batch=stop)
    assert vectors.serving_id(vdb) == "hash@1"
    assert vectors.load_index()
    assert _ids(vectors.search(fake.embed_query("Sasha birthday").tolist(), label_mask=ALL))[0] == (
        "f-bday"
    )
    assert vectors.embed_query("Sasha birthday") == []  # another space: no mixed scores
    vectors.embed_pending(mem, vdb)
    assert vectors.serving_id(vdb) == "hash@2"
    assert _rows(vdb, "hash@1") == {} and set(_rows(vdb, "hash@2")) == {"f-bday", "f-gym"}
    assert mem.execute("SELECT embedder_id FROM meta").fetchone()[0] == "hash@2"
    q = vectors.embed_query("Sasha birthday")
    assert len(q) == 32 and _ids(vectors.search(q, label_mask=ALL))[0] == "f-bday"


def test_vectors_db_deletable_and_rebuilt_by_the_job(mem, vpath):
    _seed(mem)
    assert vectors.enqueue(mem) and not vectors.enqueue(mem)  # one waiting pull at a time
    assert store.run_queued_jobs(mem, vectors.JOB_KIND) == 1
    assert vpath.exists()
    for p in vpath.parent.glob("vectors.db*"):
        p.unlink()
    vectors.attach(None)
    assert vectors.enqueue(mem)
    assert store.run_queued_jobs(mem, vectors.JOB_KIND) == 1
    vdb = vectors.open_db(vpath)
    try:
        assert set(_rows(vdb)) == {"f-bday", "f-gym"}
    finally:
        vdb.close()
    assert _ids(vectors.search(vectors.embed_query("gym"), label_mask=ALL))[0] == "f-gym"


def test_embed_is_a_background_job_kind_not_startup():
    assert store.job_runner(vectors.JOB_KIND) is not None
    assert vectors.JOB_KIND not in store.startup_job_kinds()


def test_stop_hook_drops_forgotten_ids_from_ram(mem, vdb):
    _seed(mem)
    vectors.embed_pending(mem, vdb)
    q = vectors.embed_query("Sasha birthday")
    assert "f-bday" in _ids(vectors.search(q, label_mask=ALL))
    vectors.stop_hook(
        forget.StopEvent(
            forget_id="fg-1", record_ids=["f-bday"], raw_ids=[], turn_ids=[], thread_ids=[]
        )
    )
    assert "f-bday" not in _ids(vectors.search(q, label_mask=ALL))


def test_forgotten_records_vectors_gone_after_sweep(mem, vdb):
    _seed(mem)
    vectors.embed_pending(mem, vdb)
    mem.execute("UPDATE records SET status = 'forgotten' WHERE id = 'f-bday'")
    assert forget._sweep_vectors(mem)
    assert set(_rows(vdb)) == {"f-gym"}
    q = vectors.embed_query("Sasha birthday")
    assert "f-bday" not in _ids(vectors.search(q, label_mask=ALL))  # RAM swept too


def test_forget_delete_covers_the_vectors_table(mem, vdb):
    _seed(mem)
    vectors.embed_pending(mem, vdb)
    forget._delete_vectors(mem, ["f-gym"])
    assert set(_rows(vdb)) == {"f-bday"}


def test_record_forgotten_while_embedding_leaves_no_vector(mem, vdb):
    _seed(mem)
    base = vectors.HashEmbedder()

    class Racing(vectors.HashEmbedder):
        def embed_docs(self, texts):
            mem.execute("UPDATE records SET status = 'forgotten' WHERE id = 'f-gym'")
            return base.embed_docs(texts)

    vectors.set_embedder(Racing())
    assert vectors.embed_pending(mem, vdb) == 1
    assert set(_rows(vdb)) == {"f-bday"}


def test_search_never_raises(mem, vdb):
    assert vectors.search([0.1] * 64, label_mask=ALL) == []  # no index yet
    _seed(mem)
    vectors.embed_pending(mem, vdb)
    assert vectors.search([], label_mask=ALL) == []
    assert vectors.search([1.0, 2.0], label_mask=ALL) == []  # wrong dimension
    assert vectors.search([0.1] * 64, label_mask=ALL, limit=0) == []


def test_real_embedder_cross_lingual(mem, vdb):
    """The chosen model, when it is cached locally: a Russian question finds an English fact."""
    pytest.importorskip("fastembed")
    vectors.reset()
    if vectors._load(vectors.PRIMARY, allow_download=False) is None:
        pytest.skip("embeddinggemma-300m is not in the local model cache")
    _seed(mem)
    assert vectors.embed_pending(mem, vdb) == 2
    q = vectors.embed_query("когда у Саши день рождения")
    assert len(q) == vectors.PRIMARY.dim
    assert _ids(vectors.search(q, label_mask=ALL))[0] == "f-bday"
