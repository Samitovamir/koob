"""§7.1 versions: only version-bearing columns move a record's version.

Spec marker: `[T consistency/test_versions::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import ledger, store
from koob.memory.ledger import CommitRejected
from koob.memory.model import AuthorTier, Label, OpKind, Status

from .._ledger import add, commit, entity, fact, op, open_seeded, row, version


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    assert add(c, entity("e-1", "Sasha")).ok
    assert add(c, fact("f-1", "e-1", "birthday", "05-12", importance=0.4)).ok
    yield c
    c.close()


def _update(conn, target, tier=AuthorTier.USER, **fields):
    return commit(conn, op(OpKind.UPDATE, target=target, fields=fields), tier=tier)


def test_heat_does_not_bump(conn):
    """§7.1: heat, importance, confidence and review never bump a record's version."""
    v = version(conn, "f-1")
    assert _update(conn, "f-1", importance=0.9, confidence=0.7, review=True).ok
    conn.execute(
        "INSERT INTO record_stats (record_id, uses, last_used_at, heat) VALUES ('f-1', 5, 't', 3.5)"
    )
    commit(conn, op(OpKind.FLAG, target="f-1", review=False))
    assert version(conn, "f-1") == v
    r = conn.execute("SELECT importance, confidence, review FROM records WHERE id='f-1'").fetchone()
    assert tuple(r) == (0.9, 0.7, 0)
    assert store.memory_version(conn) > v  # the commits happened; the record did not move


def test_content_columns_bump(conn):
    v0 = version(conn, "f-1")
    assert _update(conn, "f-1", text_en="Sasha's birthday is 12 May").ok
    v1 = version(conn, "f-1")
    assert v1 > v0
    assert _update(conn, "f-1", valid_from="2001-05-12").ok
    v2 = version(conn, "f-1")
    assert v2 > v1
    assert _update(conn, "f-1", labels=int(Label.HEALTH)).ok  # labels are bearing
    v3 = version(conn, "f-1")
    assert v3 > v2
    commit(conn, op(OpKind.FLAG, target="f-1", status="archived"))
    v4 = version(conn, "f-1")
    assert v4 > v3 and row(conn, "f-1")["status"] == "archived"
    # A justification member is a content change of its target (§1, §7.1).
    assert add(conn, fact("f-2", "e-1", "lives_in", "Tver")).ok
    j = {"target": "f-1", "group_no": 1, "member": "f-2", "provenance": "inferred"}
    assert commit(conn, op(OpKind.LINK, justification=j)).ok
    assert version(conn, "f-1") > v4


def test_same_value_does_not_bump(conn):
    v = version(conn, "f-1")
    assert _update(conn, "f-1", value_text="05-12").ok  # unchanged
    assert version(conn, "f-1") == v


def test_entity_display_name_and_aliases_bump(conn):
    v = version(conn, "e-1")
    assert _update(conn, "e-1", display_name="Alexandra").ok
    v1 = version(conn, "e-1")
    assert v1 > v
    alias = {"alias": "Шура", "kind": "diminutive"}
    assert commit(conn, op(OpKind.UPDATE, target="e-1", add_aliases=[alias])).ok
    assert version(conn, "e-1") > v1


def test_labels_only_grow_unless_he_says(conn):
    assert _update(conn, "f-1", labels=int(Label.HEALTH | Label.MONEY)).ok
    with pytest.raises(CommitRejected) as e:
        _update(conn, "f-1", tier=AuthorTier.SYSTEM, labels=int(Label.HEALTH))
    assert e.value.reason == "labels"
    assert _update(
        conn, "f-1", tier=AuthorTier.SYSTEM, labels=int(Label.HEALTH | Label.MONEY | Label.LEGAL)
    ).ok
    assert _update(conn, "f-1", tier=AuthorTier.USER, labels=0).ok  # «исправь» (fix) may drop


def test_versions_never_go_backwards(conn):
    seen = []
    for i in range(4):
        _update(conn, "f-1", text_en=f"t{i}")
        seen.append(version(conn, "f-1"))
    assert seen == sorted(seen) and len(set(seen)) == 4
    ledger_set = conn.execute("SELECT max(version) FROM records").fetchone()[0]
    assert ledger_set <= store.memory_version(conn)


def test_set_status_is_bearing(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        v = conn.execute(
            "UPDATE meta SET memory_version = memory_version + 1 WHERE id = 1 RETURNING memory_version"
        ).fetchone()[0]
        from koob.memory.model import Proposal

        ctx = store.CommitCtx(
            conn=conn, version=v, ts="2026-10-12T12:00:00Z", proposal=Proposal(ops=[])
        )
        ledger.set_status(ctx, "f-1", Status.NEEDS_REVIEW)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    assert row(conn, "f-1")["status"] == "needs_review" and version(conn, "f-1") == v
