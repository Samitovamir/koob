"""Another person's item allowed only under the uncertain band ranks by its real score.

Such an item is not known to be sensitive, so the basket ranks it like an unlabeled line:
among the clean lines by its own score, and the score floor, the high gate and the breadth
cap read that same score. Only an exact tie goes to the clean line.
"""

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


def cand(i: str, *, strong: bool = True, lanes=("lexical", "names"), **kw) -> gate.Cand:
    return gate.Cand(
        id=i, kind="fact", version=1, text=i, tokens=20, strong=strong, lanes=set(lanes), **kw
    )


def test_unsure_line_a_little_stronger_than_the_clean_ones_comes_first(conn):
    cands = {"u": cand("u", unsure=True), "a": cand("a"), "b": cand("b")}
    ranked = [("u", 1.2 * SCORE), ("a", SCORE), ("b", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands) == ["u", "a", "b"]


def test_unsure_line_weaker_than_a_clean_one_comes_after_it_and_before_weaker_ones(conn):
    cands = {"u": cand("u", unsure=True), "a": cand("a"), "b": cand("b")}
    ranked = [("a", 2 * SCORE), ("u", 1.5 * SCORE), ("b", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands) == ["a", "u", "b"]


def test_unsure_line_stronger_than_a_clean_one_takes_its_place_at_the_item_cap(conn):
    cands = {"u": cand("u", unsure=True), "a": cand("a"), "b": cand("b")}
    ranked = [("u", 1.2 * SCORE), ("a", SCORE), ("b", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands, max_items=2) == ["u", "a"]


def test_unsure_and_clean_lines_share_the_room_by_score(conn):
    cands = {"u": cand("u", unsure=True), "v": cand("v", unsure=True), "a": cand("a")}
    ranked = [("u", 1.5 * SCORE), ("v", 1.2 * SCORE), ("a", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands, max_items=2) == ["u", "v"]


def test_exact_tie_between_an_unsure_and_a_clean_line_goes_to_the_clean_one(conn):
    cands = {"u": cand("u", unsure=True), "a": cand("a")}
    ranked = [("u", SCORE), ("a", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands, max_items=1) == ["a"]


def test_unsure_line_at_the_score_floor_is_kept_and_one_below_it_dropped(conn):
    cands = {"u": cand("u", unsure=True), "v": cand("v", unsure=True), "a": cand("a")}
    ranked = [("a", SCORE), ("u", gate.MIN_SCORE), ("v", 0.99 * gate.MIN_SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands) == ["a", "u"]


def test_unsure_line_passes_the_high_gate_past_the_session_budget_on_its_own_score(conn):
    cands = {"u": cand("u", unsure=True), "a": cand("a")}
    ranked = [("u", gate.HIGH_SCORE), ("a", SCORE)]
    out = gate.gate(conn, ranked, thread_id="x", cands=cands, session_used=gate.SESSION_BUDGET)
    assert out == ["u"]


def test_unsure_lines_at_the_high_gate_widen_a_broad_question_like_clean_ones(conn):
    n = gate.MAX_ITEMS + 2
    cands = {f"u{k}": cand(f"u{k}", unsure=True) for k in range(n)}
    ranked = [(f"u{k}", gate.HIGH_SCORE) for k in range(n)]
    assert gate.breadth_cap(ranked, cands) == n
    assert len(gate.gate(conn, ranked, thread_id="x", cands=cands)) == n
