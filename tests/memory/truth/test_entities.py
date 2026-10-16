"""Entities (§10.3, §13.3, §16.1): aliases in every script, resolution order, merge and split."""

from __future__ import annotations

import json

import pytest

from koob.memory import entities, forget
from koob.memory.model import AuthorTier, ForgetScope, OpKind

from .._ledger import add, commit, edges, entity, fact, op, open_seeded, served, status, version


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    yield c
    c.close()


def test_normalize_alias():
    assert entities.normalize_alias("  Ｓａｓｈａ   K. ") == "sasha k."
    assert entities.normalize_alias("СЕРЁЖА") == "серёжа"


@pytest.mark.parametrize(
    ("name", "expect"),
    [
        ("Саша", {"sasha"}),
        ("Sasha", {"саша"}),
        ("Александр", {"aleksandr", "alexander"}),
        ("Alexander", {"александр"}),
        ("Дмитрий", {"dmitry", "dmitriy"}),
        ("Илья", {"ilya"}),
        ("Ilya", {"илья"}),
        ("Женя", {"zhenya"}),
        ("Zhenya", {"женя"}),
        ("Щукин", {"shchukin"}),
        ("Юлия", {"yulia", "yuliya"}),
    ],
)
def test_transliterations_both_ways(name, expect):
    got = {entities.normalize_alias(t) for t in entities.transliterations(name)}
    assert expect <= got
    assert entities.normalize_alias(name) not in got


def test_name_family_links_diminutives():
    fam = {entities.normalize_alias(n) for n in entities.name_family("Саша")}
    assert {"александр", "александра", "alexander"} <= fam
    fam = {entities.normalize_alias(n) for n in entities.name_family("Alexander")}
    assert "саша" in fam


def test_entity_creation_writes_aliases_in_both_scripts(conn):
    assert add(conn, entity("e-1", "Саша")).ok
    rows = conn.execute(
        "SELECT alias_norm, kind, confirmed FROM aliases WHERE entity_id='e-1'"
    ).fetchall()
    got = {(r[0], r[1], r[2]) for r in rows}
    assert ("саша", "name", 1) in got and ("sasha", "translit", 1) in got
    n = conn.execute(
        "SELECT count(*) FROM fts_names WHERE rowid IN"
        " (SELECT fts_rowid FROM aliases WHERE entity_id = 'e-1')"
    ).fetchone()[0]
    assert n == len(rows)


def test_resolve_order_and_forms(conn):
    assert add(conn, entity("e-sasha", "Саша")).ok
    assert add(conn, entity("e-ilya", "Илья")).ok
    assert add(conn, entity("e-cafe", "Саша", etype="place")).ok
    # Exact, translit and inflected forms land on the same person.
    for text in ("Саша", "sasha", "Саше", "Сашей", "SASHA"):
        rows = entities.resolve_rows(conn, text, "person")
        assert [r.entity_id for r in rows] == ["e-sasha"], text
    assert [r.entity_id for r in entities.resolve_rows(conn, "Ilya", "person")] == ["e-ilya"]
    assert [r.entity_id for r in entities.resolve_rows(conn, "Илье", "person")] == ["e-ilya"]
    # The full name finds the diminutive, but weaker than an exact name.
    fam = entities.resolve_rows(conn, "Александр", "person")
    assert [r.entity_id for r in fam] == ["e-sasha"] and fam[0].confidence < 0.7
    # kind filters; "" means any.
    assert {r.entity_id for r in entities.resolve_rows(conn, "Саша", "")} == {"e-sasha", "e-cafe"}
    assert entities.candidates(conn, "Саша", "place")[0][0] == "e-cafe"


def test_two_sashas_both_returned(conn):
    assert add(conn, entity("e-a", "Саша")).ok
    assert add(conn, entity("e-b", "Саша")).ok
    rows = entities.resolve_rows(conn, "Саша", "person")
    assert {r.entity_id for r in rows} == {"e-a", "e-b"}
    assert all(r.confidence >= 0.9 for r in rows)


def test_retired_entity_not_resolved(conn):
    assert add(conn, entity("e-a", "Саша")).ok
    forget.stop_use(conn, ["e-a"], scope=ForgetScope.ITEM, forever=False)
    assert entities.resolve_rows(conn, "Саша", "person") == []


def test_merge(conn):
    assert add(conn, entity("e-keep", "Ilya")).ok
    assert add(conn, entity("e-drop", "Илюша")).ok
    assert add(conn, fact("f-1", "e-drop", "lives_in", "Tver")).ok
    served(conn, "t-1", "f-1", version(conn, "f-1"))
    v1 = version(conn, "f-1")
    entities.merge(conn, "e-keep", "e-drop")
    assert status(conn, "e-drop") == "superseded"
    assert ("e-keep", "e-drop") in edges(conn, "supersedes")
    assert conn.execute("SELECT subject_id FROM facts WHERE id='f-1'").fetchone()[0] == "e-keep"
    assert version(conn, "f-1") > v1  # subject_id is content
    assert ("f-1", "e-keep") in edges(conn, "about")
    assert (
        conn.execute("SELECT count(*) FROM pending_updates WHERE item_id='f-1'").fetchone()[0] == 1
    )
    names = {r[0] for r in conn.execute("SELECT alias_norm FROM aliases WHERE entity_id='e-keep'")}
    assert {"ilya", "илья", "илюша"} <= names
    assert [r.entity_id for r in entities.resolve_rows(conn, "Илюша", "person")] == ["e-keep"]


def test_merge_by_a_worker_needs_a_shared_stable_id(conn):
    from koob.memory.ledger import CommitRejected
    from koob.memory.model import StableId

    sid = StableId(kind="email", value="ilya@x.org", tier=AuthorTier.USER)
    assert add(conn, entity("e-1", "Ilya", stable_ids=[sid])).ok
    assert add(conn, entity("e-2", "Илья K.")).ok
    # a worker's rejected group is dropped alone and reported (§6.3)
    res = commit(
        conn, op(OpKind.MERGE, target="e-1", drop="e-2"), tier=AuthorTier.SYSTEM, model_id="w"
    )
    assert not res.ok and res.rejected["g"].startswith("tier")
    with pytest.raises(CommitRejected):  # the user's own write hears the refusal
        commit(conn, op(OpKind.MERGE, target="e-1", drop="e-2"), tier=AuthorTier.TOOL)
    assert add(conn, entity("e-3", "I. K.", stable_ids=[sid])).ok
    assert commit(
        conn, op(OpKind.MERGE, target="e-1", drop="e-3"), tier=AuthorTier.SYSTEM, model_id="w"
    ).ok
    assert status(conn, "e-3") == "superseded"


def test_split(conn):
    """«это не тот Саша» ("not that Sasha"): the user's facts about the other Sasha move to a new
    entity."""
    assert add(conn, entity("e-s", "Саша")).ok
    assert add(conn, fact("f-work", "e-s", "works_at", "Acme")).ok
    assert add(conn, fact("f-home", "e-s", "lives_in", "Tver")).ok
    v = version(conn, "f-home")
    new_id = entities.split(conn, "e-s", ["f-home"])
    assert new_id != "e-s" and status(conn, new_id) == "active"
    subj = dict(conn.execute("SELECT id, subject_id FROM facts").fetchall())
    assert subj == {"f-work": "e-s", "f-home": new_id}
    assert version(conn, "f-home") > v
    assert (new_id, "e-s") in edges(conn, "derived_from")
    disp = conn.execute(
        "SELECT display_name, stable_ids FROM entities WHERE id=?", (new_id,)
    ).fetchone()
    assert disp[0] == "Саша" and json.loads(disp[1]) == []
    # Both Sashas now resolve: hands ask which one (§16.1).
    assert {r.entity_id for r in entities.resolve_rows(conn, "Саша", "person")} == {"e-s", new_id}
