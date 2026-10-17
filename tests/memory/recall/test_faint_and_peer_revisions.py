"""Faint candidates and revisions of other people's messages.

A finder's hit under the evidence floors is ranked after every evidenced line and fills only
the room they leave; alone it never makes a basket. Another person's message whose value
their own later message (or the user's) in the same chat replaced is shown with that value cut.
"""

from __future__ import annotations

import pytest

from koob.memory.recall import basket, gate, vectors

from .._ledger import open_seeded

SCORE = 1.0 / (gate.RRF_K + 1)


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def cand(i: str, *, lanes=("lexical",), tokens: int = 20, **kw) -> gate.Cand:
    return gate.Cand(id=i, kind="fact", version=1, text=i, tokens=tokens, lanes=set(lanes), **kw)


def faint(i: str, **kw) -> gate.Cand:
    return cand(i, lanes=("vector",), evidence=False, faint=True, **kw)


def test_faint_hit_fills_room_after_the_evidenced_line(conn):
    cands = {"f": faint("f"), "e": cand("e")}
    ranked = [("f", 2 * SCORE), ("e", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands) == ["e", "f"]


def test_faint_hits_alone_leave_the_basket_empty(conn):
    cands = {"f1": faint("f1"), "f2": faint("f2")}
    assert gate.gate(conn, [("f1", SCORE), ("f2", SCORE)], thread_id="x", cands=cands) == []


def test_faint_hits_never_displace_evidenced_lines(conn):
    weak = {f"w{k}": cand(f"w{k}", tokens=40) for k in range(gate.MAX_LINES)}
    dim = {f"f{k}": faint(f"f{k}", tokens=40) for k in range(4)}
    ranked = [(i, 2 * SCORE) for i in dim] + [(i, SCORE) for i in weak]
    out = gate.gate(conn, ranked, thread_id="x", cands={**weak, **dim})
    assert out == list(weak)


def test_faint_fill_keeps_the_token_budget(conn):
    cands = {"e": cand("e", tokens=300), **{f"f{k}": faint(f"f{k}", tokens=200) for k in range(3)}}
    ranked = [("e", SCORE)] + [(f"f{k}", SCORE) for k in range(3)]
    out = gate.gate(conn, ranked, thread_id="x", cands=cands)
    assert out[0] == "e" and sum(cands[i].tokens for i in out) <= gate.WEAK_BUDGET


def test_raw_meaning_hits_under_the_bar_are_faint_not_lost():
    hits = [("a", 0.70), ("b", 0.60), ("c", 0.45), ("d", 0.36)]
    lane, dim = basket._split_raw_hits(hits)
    assert [i for i, _ in lane] == ["a", "b"]
    assert [i for i, _ in dim] == ["c", "d"]


def test_raw_lane_bar_never_drops_below_the_noise_floor():
    lane, dim = basket._split_raw_hits([("a", 0.38), ("b", 0.36)])
    assert [i for i, _ in lane] == ["a", "b"] and dim == []


def _msg(i: str, text: str, ts: str, *, tier: str = "other", who: str | None = "e-denis"):
    return {"id": i, "rev": 1, "ts": ts, "chat_id": "chat-team", "text": text,
            "author_tier": tier, "author_entity_id": who}  # fmt: skip


def test_a_persons_later_message_revises_their_earlier_value():
    old = _msg("r-old", "The team dinner is on May 14.", "2026-05-01T10:00:00Z")
    new = _msg("r-new", "The team dinner is on May 16 now.", "2026-05-03T10:00:00Z")
    out = basket._later_revisions(None, [old], [], [new])
    assert out["r-old"]["by"] == ["r-new"]
    assert out["r-old"]["sides"] == ["May 14"]
    assert basket._cut_sides(old["text"], out["r-old"]["sides"]) == (
        "The team dinner is on [сменилось]."
    )


def test_his_later_message_revises_another_persons_value():
    old = _msg("r-old", "The team dinner is on May 14.", "2026-05-01T10:00:00Z")
    his = _msg("r-his", "The team dinner is on May 16 now.", "2026-05-03T10:00:00Z",
               tier="user", who=None)  # fmt: skip
    assert basket._later_revisions(None, [old], [his], [])["r-old"]["by"] == ["r-his"]


def test_someone_elses_message_does_not_revise_a_persons_value():
    old = _msg("r-old", "The team dinner is on May 14.", "2026-05-01T10:00:00Z")
    other = _msg("r-x", "The team dinner is on May 16 now.", "2026-05-03T10:00:00Z", who="e-marta")
    assert basket._later_revisions(None, [old], [], [other]) == {}


def test_a_persons_message_does_not_revise_his_own_value():
    mine = _msg("r-mine", "The team dinner is on May 14.", "2026-05-01T10:00:00Z",
                tier="user", who=None)  # fmt: skip
    peer = _msg("r-peer", "The team dinner is on May 16 now.", "2026-05-03T10:00:00Z")
    assert basket._later_revisions(None, [mine], [], [peer]) == {}
