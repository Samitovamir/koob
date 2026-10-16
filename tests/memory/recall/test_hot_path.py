"""The basket's hot path: the memory spec §10.2, §10.3.

Spec marker: `[T recall/test_hot_path::<case>]`: no model, no Ollama, no socket while the
basket is built; works with vectors absent (fastembed missing or cold)."""

from __future__ import annotations

import http.client
import socket
import urllib.request

import pytest

from koob.memory import api
from koob.memory.model import TurnCtx
from koob.memory.recall import basket, vectors

from .._ledger import add, entity, fact, open_seeded


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    c = open_seeded(tmp_path)
    assert add(c, entity("e-sasha", "Саша")).ok
    assert add(
        c, fact("f-bday", "e-sasha", "birthday", "May 12", text="Sasha's birthday is May 12")
    ).ok
    yield c
    c.close()
    vectors.set_embedder(None)
    vectors.reset()


def _tripwires(monkeypatch) -> list[str]:
    touched: list[str] = []

    def trip(name):
        def f(*a, **k):
            touched.append(name)
            raise AssertionError(f"hot path touched {name}")

        return f

    monkeypatch.setattr(socket.socket, "connect", trip("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", trip("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", trip("socket.create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", trip("socket.getaddrinfo"))
    monkeypatch.setattr(http.client.HTTPConnection, "connect", trip("http.connect"))
    monkeypatch.setattr(urllib.request, "urlopen", trip("urlopen"))  # Ollama's HTTP API
    from koob.llm import runner

    monkeypatch.setattr(runner, "run_role", trip("llm.runner.run_role"))
    return touched


def test_no_model_no_socket(turn, conn, monkeypatch):
    """§10.2: no llm/runner, Ollama or socket while the basket is built."""
    touched = _tripwires(monkeypatch)
    vectors.set_embedder(vectors.HashEmbedder())  # an in-process embedder may run
    b = api.recall(turn, store=conn)
    assert touched == []
    assert b.block and "[m:f-bday]" in b.block


def test_vectors_absent_still_recalls(conn, monkeypatch):
    """fastembed missing or cold: embed_query returns [] and the basket never waits."""
    touched = _tripwires(monkeypatch)
    vectors.set_embedder(None)
    monkeypatch.setattr(vectors, "_fastembed_present", lambda: False)
    turn = TurnCtx(turn_id="t2", thread_id="th-2", owner_words="когда день рождения у Саши?")
    b = api.recall(turn, store=conn)
    assert touched == [] and [i.id for i in b.items] == ["f-bday"]
    assert b.timings_ms["vector_wait"] < basket.FINDER_DEADLINES_MS["vector"] + 10


def test_slow_embedder_is_skipped_past_its_deadline(conn, monkeypatch):
    """§10.2: vector 40 ms, skipped past it; the other finders still answer."""
    import time

    class Slow(vectors.HashEmbedder):
        def embed_query(self, text):
            time.sleep(0.3)
            return super().embed_query(text)

    vectors.set_embedder(Slow())
    monkeypatch.setattr(vectors, "_embedder_for", lambda _id: Slow())
    turn = TurnCtx(turn_id="t3", thread_id="th-3", owner_words="что про Сашу")
    t0 = time.perf_counter()
    b = api.recall(turn, store=conn)
    assert (time.perf_counter() - t0) * 1000 < 150
    assert [i.id for i in b.items] == ["f-bday"]
