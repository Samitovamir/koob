"""§8.3 assertion matching by rule: code sees a new assertion restate an old slot.

Spec marker: `[T truth/test_match::<case>]`, plus the rule's edges (cardinality, cues,
one_off, scope, plan vs state, the `other` relation)."""

from __future__ import annotations

import pytest

from koob.memory import ledger
from koob.memory.model import FKind, Polarity, Stability

from .._ledger import add, edges, fact, open_seeded, pref, row, status, version


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    yield c
    c.close()


def test_no_meat_supersedes_likes_meat(conn):
    """§8.3: «больше не ем мясо» supersedes «люблю мясо» by rule."""
    likes = pref("p-meat", "u", "prefers_food", "meat", quote_orig="люблю мясо")
    assert add(conn, likes).ok
    no_meat = fact(
        "f-nomeat",
        "u",
        "eats",
        "meat",
        fkind=FKind.OPINION,
        polarity=Polarity.NEGATED,
        stability=Stability.LASTING,
        quote_orig="больше не ем мясо",
    )
    assert add(conn, no_meat, rule_match=False).ok  # stored, not matched yet
    assert ledger.rule_match(conn, "f-nomeat") == ["p-meat"]

    again = fact(
        "f-nomeat2",
        "u",
        "eats",
        "meat",
        fkind=FKind.OPINION,
        polarity=Polarity.NEGATED,
        stability=Stability.LASTING,
        quote_orig="больше не ем мясо",
    )
    assert add(conn, again).ok  # the default path matches and supersedes at commit
    assert status(conn, "p-meat") == "superseded"
    assert ("f-nomeat2", "p-meat") in edges(conn, "supersedes")


def test_not_today_coexists(conn):
    """§8.3: a one_off coexists with the lasting preference, bounded by valid_to."""
    likes = pref("p-cil", "u", "prefers", "cilantro", quote_orig="люблю кинзу")
    assert add(conn, likes).ok
    today = pref(
        "p-today",
        "u",
        "prefers",
        "cilantro",
        polarity=Polarity.NEGATED,
        stability=Stability.ONE_OFF,
        valid_to="2026-10-12T23:59:59Z",
        quote_orig="сегодня не хочу кинзу",
    )
    assert add(conn, today).ok
    assert ledger.rule_match(conn, "p-today") == []
    assert status(conn, "p-cil") == "active" and status(conn, "p-today") == "active"

    lasting = pref(
        "p-nomore",
        "u",
        "prefers",
        "cilantro",
        polarity=Polarity.NEGATED,
        stability=Stability.LASTING,
        quote_orig="я больше не люблю кинзу",
    )
    assert add(conn, lasting).ok
    assert status(conn, "p-cil") == "superseded"
    assert status(conn, "p-today") == "active"  # a one_off is never a slot to supersede


def test_near_keys_give_review(conn):
    """§8.3: a model match on non-identical keys gives review, not supersession."""
    old = pref("p-a", "u", "prefers_food", "cilantro")
    assert add(conn, old).ok
    v = version(conn, "p-a")
    near = pref("p-b", "u", "prefers_food", "cilantro-free salads")
    assert add(conn, near, model_id="worker-x", model_matches=["p-a"]).ok
    r = row(conn, "p-a")
    assert r["status"] == "active" and r["review"] == 1
    assert r["version"] == v  # review is not version-bearing (§7.1)

    # The same claim through an explicit SUPERSEDE op from a model: still only review.
    other = pref("p-c", "u", "prefers_food", "coriander seeds")
    assert add(conn, other).ok
    from koob.memory.model import OpKind

    from .._ledger import commit, op

    res = commit(conn, op(OpKind.SUPERSEDE, target="p-c", by="p-b"), model_id="worker-x")
    assert res.ok
    assert status(conn, "p-c") == "active" and row(conn, "p-c")["review"] == 1


def test_single_cardinality_new_value_supersedes(conn):
    assert add(conn, fact("f-msk", "u", "lives_in", "Moscow")).ok
    assert add(conn, fact("f-kzn", "u", "lives_in", "Tver", fkind=FKind.EVENT)).ok
    assert status(conn, "f-msk") == "superseded"


def test_multi_needs_a_replacement_cue(conn):
    assert add(conn, fact("e-a", "u", "has_email", "a@x.org")).ok
    assert add(conn, fact("e-b", "u", "has_email", "b@x.org", quote_orig="вот ещё почта")).ok
    assert status(conn, "e-a") == "active"  # multi, no cue: both stay
    assert add(
        conn, fact("e-c", "u", "has_email", "c@x.org", quote_orig="Теперь моя почта c@x.org")
    ).ok
    assert status(conn, "e-a") == "superseded" and status(conn, "e-b") == "superseded"


def test_negation_with_cue_leaves_other_values(conn):
    assert add(conn, fact("f-fish", "u", "eats", "fish")).ok
    nomeat = fact(
        "f-nm", "u", "eats", "meat", polarity=Polarity.NEGATED, quote_orig="больше не ем мясо"
    )
    assert add(conn, nomeat).ok
    assert status(conn, "f-fish") == "active"


def test_plan_does_not_supersede_state(conn):
    """C0: «собираюсь переехать» is a plan; where the user lives is untouched."""
    assert add(conn, fact("f-home", "sasha", "lives_in", "Moscow")).ok
    plan = fact("f-plan", "sasha", "lives_in", "Berlin", fkind=FKind.PLAN)
    assert add(conn, plan).ok
    assert status(conn, "f-home") == "active"
    assert add(conn, fact("f-move", "sasha", "moves_to", "Berlin", fkind=FKind.PLAN)).ok
    assert status(conn, "f-home") == "active"


def test_choice_for_someone_else_is_not_his_preference(conn):
    """§4.1: «взял билеты на гольф для сына» is not the user's taste for golf."""
    assert add(conn, pref("p-golf", "u", "likes", "golf", polarity=Polarity.NEGATED)).ok
    for_son = pref("p-son", "u", "likes", "golf", applicability={"for_whom": "son"})
    assert add(conn, for_son).ok
    assert status(conn, "p-golf") == "active"


def test_other_relation_is_never_matched(conn):
    assert add(conn, fact("o-1", "u", "other", "x", text="some note")).ok
    assert add(
        conn, fact("o-2", "u", "other", "x", polarity=Polarity.NEGATED, quote_orig="теперь нет")
    ).ok
    assert ledger.rule_match(conn, "o-2") == []
    assert status(conn, "o-1") == "active"


def test_unknown_relation_stored_as_other(conn):
    assert add(conn, fact("f-odd", "u", "collects_stamps_from", "Peru")).ok
    rel = conn.execute("SELECT relation FROM facts WHERE id = 'f-odd'").fetchone()[0]
    assert rel == "other"


def test_older_assertion_does_not_supersede_newer(conn):
    newer = fact("f-new", "u", "lives_in", "Tver", valid_from="2026-09-01")
    assert add(conn, newer).ok
    older = fact("f-old", "u", "lives_in", "Moscow", valid_from="2025-01-01")
    assert add(conn, older).ok
    assert status(conn, "f-new") == "active"
