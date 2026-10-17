"""Gate: a line whose source messages the block already holds gives its place to new ones."""

from __future__ import annotations

import pytest

from koob.memory.recall import gate, vectors

from .._ledger import open_seeded

SCORE = 1.0 / (gate.RRF_K + 1)


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def cand(i: str, sources=(), kind: str = "fact", **kw) -> gate.Cand:
    return gate.Cand(
        id=i, kind=kind, version=1, text=i, tokens=20, strong=True,
        lanes={"lexical", "vector"}, sources=frozenset(sources), **kw,
    )  # fmt: skip


def test_fact_next_to_its_own_message_yields_the_place_to_another_message(conn):
    cands = {
        "raw1": cand("raw1", ["raw1"], kind="raw"),
        "fact1": cand("fact1", ["raw1"]),
        "raw2": cand("raw2", ["raw2"], kind="raw"),
    }
    ranked = [("raw1", 3 * SCORE), ("fact1", 2 * SCORE), ("raw2", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands, max_items=2) == ["raw1", "raw2"]


def test_repeat_fills_the_room_left_after_every_other_line(conn):
    cands = {
        "raw1": cand("raw1", ["raw1"], kind="raw"),
        "fact1": cand("fact1", ["raw1"]),
        "raw2": cand("raw2", ["raw2"], kind="raw"),
    }
    ranked = [("raw1", 3 * SCORE), ("fact1", 2 * SCORE), ("raw2", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands) == ["raw1", "raw2", "fact1"]


def test_fact_drawn_from_two_messages_is_new_while_one_is_missing(conn):
    cands = {
        "raw1": cand("raw1", ["raw1"], kind="raw"),
        "fact1": cand("fact1", ["raw1", "raw9"]),
        "raw2": cand("raw2", ["raw2"], kind="raw"),
    }
    ranked = [("raw1", 3 * SCORE), ("fact1", 2 * SCORE), ("raw2", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands, max_items=2) == ["raw1", "fact1"]


def test_single_valued_slot_value_keeps_its_place_beside_its_message(conn):
    cands = {
        "raw1": cand("raw1", ["raw1"], kind="raw"),
        "city_new": cand("city_new", ["raw1"], slot="denis|lives_in"),
        "city_old": cand("city_old", ["raw0"], slot="denis|lives_in"),
    }
    ranked = [("raw1", 3 * SCORE), ("city_new", 2 * SCORE), ("city_old", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands) == ["raw1", "city_new"]


def test_candidates_without_sources_are_never_repeats(conn):
    cands = {"a": cand("a"), "b": cand("b")}
    ranked = [("a", 2 * SCORE), ("b", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands, max_items=2) == ["a", "b"]
