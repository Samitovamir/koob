"""Recall-first selection: what the finders found is ranked, not thrown away.

Only policy and truth deny an item; relevance orders it. The gate keeps the best allowed
candidates inside a line cap and a token budget, says the strong ones first, and returns
nothing only when no candidate carries real lane evidence.
"""

from __future__ import annotations

import pytest

from koob.memory import api, policy
from koob.memory.model import Label, TurnCtx, TurnOrigin
from koob.memory.recall import basket, gate, vectors

from .._ledger import add, entity, fact, open_seeded, raw

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str, **kw) -> TurnCtx:
    return TurnCtx(
        turn_id=kw.pop("turn_id", "turn-1"),
        thread_id=kw.pop("thread", "thread-1"),
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=CLOCK,
        **kw,
    )


def cand(i: str, *, strong: bool = False, lanes=("lexical",), tokens: int = 20, **kw):
    return gate.Cand(
        id=i, kind="fact", version=1, text=i, tokens=tokens, strong=strong, lanes=set(lanes), **kw
    )


SCORE = 1.0 / (gate.RRF_K + 1)


def test_weak_single_lane_candidate_is_kept_when_nothing_stronger(conn):
    """One lane found it and nothing better is there: the basket is not emptied."""
    cands = {"a": cand("a")}
    assert gate.gate(conn, [("a", SCORE)], thread_id="x", cands=cands) == ["a"]


def test_candidate_without_lane_evidence_stays_out(conn):
    """A meaning-only hit below the floor is no evidence: the cheap empty basket stays."""
    cands = {"a": cand("a", lanes=("vector",), evidence=False)}
    assert gate.gate(conn, [("a", SCORE)], thread_id="x", cands=cands) == []


def test_strong_lines_are_said_first(conn):
    """A grounded line leads even when a weak one fused higher."""
    cands = {"weak": cand("weak"), "strong": cand("strong", strong=True)}
    ranked = [("weak", 2 * SCORE), ("strong", SCORE)]
    assert gate.gate(conn, ranked, thread_id="x", cands=cands) == ["strong", "weak"]


def test_weak_fill_holds_the_line_cap_and_the_token_budget(conn):
    cands = {f"w{k}": cand(f"w{k}", tokens=90) for k in range(20)}
    ranked = [(f"w{k}", SCORE - k * 1e-5) for k in range(20)]
    out = gate.gate(conn, ranked, thread_id="x", cands=cands)
    assert 0 < len(out) <= gate.MAX_LINES
    assert sum(cands[i].tokens for i in out) <= gate.WEAK_BUDGET
    assert out == [f"w{k}" for k in range(len(out))]  # best first


def test_strong_lines_keep_their_room_before_weak_fill(conn):
    strong = {f"s{k}": cand(f"s{k}", strong=True, tokens=40) for k in range(4)}
    weak = {f"w{k}": cand(f"w{k}", tokens=40) for k in range(10)}
    ranked = [(i, SCORE) for i in [*weak, *strong]]
    out = gate.gate(conn, ranked, thread_id="x", cands={**strong, **weak})
    assert out[:4] == list(strong)
    assert len(out) <= gate.MAX_LINES


def test_basket_keeps_an_owner_fact_found_by_one_text_lane(conn):
    """The user's fact shares one word with a long question: one weak lane, still in the basket."""
    assert add(conn, entity("e-denis", "Denis Smith")).ok
    assert add(
        conn,
        fact(
            "f-keys",
            "self",
            "other",
            "spare keys at the neighbour",
            text="The spare keys are kept with the neighbour on the second floor",
            origin_quote="The spare keys are kept with the neighbour on the second floor",
        ),
    ).ok
    b = api.recall(
        t("before the long trip remind me what we settled about keys and plants"), store=conn
    )
    assert [i.id for i in b.items] == ["f-keys"]


def _ben_world(conn, labels: Label, *, final: bool) -> None:
    assert add(conn, entity("e-ben", "Ben")).ok
    assert add(
        conn,
        fact(
            "f-ben",
            "e-ben",
            "other",
            "a hard month at home",
            text="Ben is going through a hard month at home",
            origin_quote="Ben is going through a hard month at home",
            labels=labels,
            labels_final=final,
        ),
    ).ok
    assert add(
        conn, fact("f-club", "e-ben", "attends", "chess club", text="Ben attends a chess club")
    ).ok


def test_owner_health_fact_stays_out_of_a_question_about_something_else(conn):
    """A named topic keeps the topic test: the name lane finds the user's note on the person the
    user asks about, the weekend plan still does not get it."""
    _ben_world(conn, Label.HEALTH, final=True)
    ids = [i.id for i in api.recall(t("plan a Saturday with Ben"), store=conn).items]
    assert "f-club" in ids and "f-ben" not in ids


def test_owner_fact_the_classifier_left_unsure_is_ranked_in(conn):
    """A finished pass that named no topic (unsure) is no proof the item does not apply: the
    finder's hit is ranked, not denied."""
    _ben_world(conn, Label.UNKNOWN_SENSITIVE, final=True)
    ids = [i.id for i in api.recall(t("plan a Saturday with Ben"), store=conn).items]
    assert "f-ben" in ids


def test_owner_fact_with_pending_labels_keeps_the_topic_test(conn):
    """A label still pending is not yet known: the arrival guess (unsure) must not read as a
    finished unsure label, so the item stays out until the classifier has settled it."""
    _ben_world(conn, Label.UNKNOWN_SENSITIVE, final=False)
    ids = [i.id for i in api.recall(t("plan a Saturday with Ben"), store=conn).items]
    assert "f-club" in ids and "f-ben" not in ids


_VERDICT_CASES = [
    (Label.UNKNOWN_SENSITIVE, False, False),  # pending: the arrival guess decides nothing
    (Label.UNKNOWN_SENSITIVE, True, True),  # finished and unsure: ranked
    (Label.THIRD_PARTY_PRIVATE, True, True),
    (Label.HEALTH | Label.UNKNOWN_SENSITIVE, False, False),
    (Label.HEALTH, True, False),
    (Label.MONEY, True, False),
    (Label.LEGAL, True, False),
]


@pytest.mark.parametrize(("labels", "final", "allowed"), _VERDICT_CASES)
def test_owner_raw_verdict_ranks_only_finished_open_labels(conn, labels, final, allowed):
    """The user's own raw words under the use-now verdict: only a finished open label ranks
    without the topic test; a pending one and every named topic (money and legal too) keep it."""
    raw(conn, "r-ben", "Ben said the last weeks were rough", labels=int(labels))
    conn.execute("UPDATE raw_chunks SET labels_final = ? WHERE id = 'r-ben'", (int(final),))
    use = policy.decide_many(conn, ["r-ben"], t("plan a Saturday with Ben"))["r-ben"]
    assert use.allowed is allowed
    assert use.reason == ("own_ranked" if allowed else "own_not_applicable")


@pytest.mark.parametrize(("labels", "final", "allowed"), _VERDICT_CASES)
def test_owner_fact_verdict_ranks_only_finished_open_labels(conn, labels, final, allowed):
    """The same verdict for the user's own facts."""
    _ben_world(conn, labels, final=final)
    use = policy.decide_many(conn, ["f-ben"], t("plan a Saturday with Ben"))["f-ben"]
    assert use.allowed is allowed
    assert use.reason == ("own_ranked" if allowed else "own_not_applicable")


def test_fact_scoped_to_another_person_shows_its_scope(conn):
    """A choice made for someone else is shown with whom it is for, never as the user's own."""
    assert add(conn, entity("e-anna", "Anna")).ok
    assert add(
        conn,
        fact(
            "f-gift",
            "self",
            "prefers",
            "dark chocolate",
            text="Prefers dark chocolate as a gift",
            origin_quote="Prefers dark chocolate as a gift",
            applicability={"for_whom": "e-anna"},
        ),
    ).ok
    b = api.recall(t("what chocolate do I like as a gift"), store=conn)
    lines = [basket.line(i) for i in b.items if i.id == "f-gift"]
    assert lines and "Anna" in lines[0]
