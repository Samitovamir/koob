"""§7.3 invalidation and the §7.6 mark: derived holders go invalid and queued; threads that
were served a changed item get a pending update.

Spec marker: `[T consistency/test_invalidation::<case>]`."""

from __future__ import annotations

import json

import pytest

from koob.memory import forget
from koob.memory.model import AuthorTier, ForgetScope, Label, OpKind, ReadRef, ReadSrc

from .._ledger import add, commit, entity, fact, op, open_seeded, raw, served, version


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    assert add(c, entity("e-s", "Sasha")).ok
    assert add(c, fact("f-bd", "e-s", "birthday", "05-12", text="Sasha's birthday is 12 May")).ok
    assert add(c, entity("e-gym", "Gym", etype="place")).ok
    yield c
    c.close()


def _derived(conn, did, content, refs, dkind="summary"):
    body = {"id": did, "dkind": dkind, "dkey": did, "content": content}
    read_set = [ReadRef(**r) for r in refs]  # the op's read set is the derived item's (§7.2)
    return commit(
        conn, op(OpKind.ADD, group=did, read_set=read_set, derived=body), tier=AuthorTier.SYSTEM
    )


def _refs(conn, *ids):
    return [
        ReadRef(
            id=i, version=conn.execute("SELECT version FROM records WHERE id=?", (i,)).fetchone()[0]
        ).model_dump(mode="json")
        for i in ids
    ]


def _valid(conn, did):
    return conn.execute("SELECT valid FROM derived WHERE id = ?", (did,)).fetchone()[0]


def test_alias_no_storm(conn):
    """§7.3: an alias change invalidates only derived text holding a changed token."""
    assert _derived(conn, "d-bd", "Sasha's birthday is 12 May.", _refs(conn, "e-s", "f-bd")).ok
    assert _derived(conn, "d-people", "Close people: 3. Next birthday soon.", _refs(conn, "e-s")).ok
    assert _derived(conn, "d-gym", "Gym on Tuesdays.", _refs(conn, "e-gym")).ok

    # A display-name change is non-semantic: only text holding the changed token goes.
    upd = op(OpKind.UPDATE, target="e-s", group="e-s", fields={"display_name": "Alexandra"})
    assert commit(conn, upd).ok
    assert (_valid(conn, "d-bd"), _valid(conn, "d-people"), _valid(conn, "d-gym")) == (0, 1, 1)
    jobs = [
        json.loads(r[0])
        for r in conn.execute("SELECT payload FROM jobs WHERE jkind = 'rebuild_derived'")
    ]
    assert [j["derived_id"] for j in jobs] == ["d-bd"]

    # A new alias: the same rule; no derived text holds «Шура».
    alias = {"alias": "Шура", "kind": "diminutive"}
    assert commit(conn, op(OpKind.UPDATE, target="e-s", add_aliases=[alias])).ok
    assert _valid(conn, "d-people") == 1

    # A content change of a read fact is semantic: every holder of it goes.
    assert _derived(conn, "d-bd2", "Birthday: 12 May.", _refs(conn, "f-bd")).ok
    assert commit(conn, op(OpKind.UPDATE, target="f-bd", fields={"value_norm": "05-13"})).ok
    assert _valid(conn, "d-bd2") == 0


def test_derived_reading_derived_goes_too(conn):
    assert _derived(conn, "d-sum", "Sasha: birthday 12 May.", _refs(conn, "f-bd")).ok
    ref = {"id": "d-sum", "version": 0, "src": "derived"}
    assert _derived(conn, "d-core", "People section", [ref], dkind="core_section").ok
    assert commit(conn, op(OpKind.RETRACT, target="f-bd")).ok
    assert (_valid(conn, "d-sum"), _valid(conn, "d-core")) == (0, 0)


def test_derived_inherits_labels_and_needs_usable_sources(conn):
    raw(conn, "r-h", "у Саши астма", labels=int(Label.HEALTH))
    refs = _refs(conn, "f-bd") + [{"id": "r-h", "version": 1, "src": ReadSrc.RAW.value}]
    assert _derived(conn, "d-h", "Sasha has asthma.", refs).ok
    labels = conn.execute("SELECT labels FROM derived WHERE id = 'd-h'").fetchone()[0]
    assert labels & int(Label.HEALTH)
    deps = conn.execute(
        "SELECT item_id, version FROM read_deps WHERE holder_id = 'd-h' ORDER BY item_id"
    ).fetchall()
    assert [tuple(d) for d in deps] == [("f-bd", version(conn, "f-bd")), ("r-h", 1)]

    from koob.memory.ledger import CommitRejected

    assert commit(conn, op(OpKind.RETRACT, target="f-bd")).ok
    with pytest.raises(CommitRejected):
        _derived(conn, "d-x", "from a retracted fact", _refs(conn, "f-bd"))


def test_pending_updates_for_served_threads(conn):
    v = version(conn, "f-bd")
    served(conn, "t-owner", "f-bd", v)
    served(conn, "t-task", "f-bd", v, via="core")
    served(conn, "t-other", "e-gym", version(conn, "e-gym"))
    assert add(conn, fact("f-bd2", "e-s", "birthday", "05-13")).ok  # supersedes f-bd by rule
    rows = conn.execute(
        "SELECT thread_id, item_id, old_version, new_version FROM pending_updates ORDER BY thread_id"
    ).fetchall()
    nv = version(conn, "f-bd")
    assert nv > v
    assert [tuple(r) for r in rows] == [("t-owner", "f-bd", v, nv), ("t-task", "f-bd", v, nv)]


def test_forgotten_ids_get_no_update_block(conn):
    served(conn, "t-owner", "f-bd", version(conn, "f-bd"))
    # every forget goes through stop_use (the ledger's FORGET op is refused)
    forget.stop_use(conn, ["f-bd"], scope=ForgetScope.ITEM, forever=False)
    assert conn.execute("SELECT count(*) FROM pending_updates").fetchone()[0] == 0
    fts = conn.execute(
        "SELECT count(*) FROM fts_records WHERE rowid = (SELECT fts_rowid FROM records WHERE id='f-bd')"
    ).fetchone()[0]
    assert fts == 0  # no FTS row for a forgotten item (§3.2 invariants)
