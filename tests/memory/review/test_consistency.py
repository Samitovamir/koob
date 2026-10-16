"""Consistency tests: versions, read sets and the pre-send check (§7.1-§7.4, G2, G5).

Covered fixes: derived `rev` (migration 0002) and payload read sets validated in
`store.commit`. Each test's docstring names the finding it covers in the design notes."""

from __future__ import annotations

from koob.memory import forget, store
from koob.memory.model import AuthorTier, ForgetScope, OpKind, ReadRef, ReadSrc

from .._ledger import add, commit, entity, fact, op, open_seeded
from ..forget import _seed as s

clock = s.clock  # fixture: a manual clock wired into forget


def _ref(conn, rid: str) -> ReadRef:
    (v,) = conn.execute("SELECT version FROM records WHERE id = ?", (rid,)).fetchone()
    return ReadRef(id=rid, version=v)


def _core(conn, content: str, refs: list[ReadRef]):
    body = {"id": "d-core", "dkind": "core_section", "dkey": "people", "content": content}
    return commit(
        conn, op(OpKind.ADD, group="d-core", read_set=refs, derived=body), tier=AuthorTier.SYSTEM
    )


def test_presend_refuses_a_core_section_rebuilt_after_a_forget(tmp_path, clock):
    """G2/G5: derived items used to carry no version; a rebuild upserts the same id with
    `valid = 1`, and `stale_refs` checked only `valid` for derived refs (ignoring the ref's
    version). A prompt prepared before a "forget" holds the old core section text; the core is
    rebuilt before the next turn (§10.1, G9: after a correction it rebuilds at once); the
    pre-send check must then refuse, or the forgotten fact reaches the model."""
    conn = open_seeded(tmp_path)
    assert add(conn, entity("e-s", "Sasha")).ok
    assert add(conn, fact("f-move", "e-s", "moving_to", "Berlin", text="Sasha moves to Berlin")).ok
    assert add(conn, fact("f-job", "e-s", "works_at", "Yandex", text="Sasha works at Yandex")).ok
    assert _core(conn, "Sasha moves to Berlin; works at Yandex", [_ref(conn, "f-move")]).ok
    prepared = [ReadRef(id="d-core", version=0, src=ReadSrc.DERIVED)]  # the prompt's read set
    assert store.pre_send_check(conn, thread_id="t-1", read_set=prepared).ok

    forget.stop_use(conn, ["f-move"], scope=ForgetScope.ITEM, forever=False)
    assert _core(conn, "Sasha works at Yandex", [_ref(conn, "f-job")]).ok  # rebuilt at once

    res = store.pre_send_check(conn, thread_id="t-1", read_set=prepared)
    assert not res.ok, "a prompt holding the pre-forget core text passed the pre-send check"


def test_derived_commit_validates_the_versions_in_its_payload_read_set(tmp_path, clock):
    """G2: `_add_derived` used to take `derived.read_set` from the payload when present and
    check only that each member is usable, never its version; `store.commit` validated only
    `op.read_set`. A summary built from "birthday 12 May" committed as valid after the fact
    was corrected to 14 May, when the worker put its read set in the payload."""
    conn = open_seeded(tmp_path)
    assert add(conn, entity("e-s", "Sasha")).ok
    assert add(conn, fact("f-bd", "e-s", "birthday", "05-12", text="Sasha's birthday is 12 May")).ok
    snapshot = _ref(conn, "f-bd").model_dump(mode="json")  # the worker reads version N
    upd = op(OpKind.UPDATE, target="f-bd", group="e-s", fields={"value_norm": "05-14"})
    assert commit(conn, upd).ok  # the user's correction lands while the worker runs
    body = {
        "id": "d-sum",
        "dkind": "summary",
        "dkey": "2026-10-12",
        "content": "Sasha's birthday is 12 May",
        "read_set": [snapshot],
    }
    res = commit(conn, op(OpKind.ADD, group="d-sum", derived=body), tier=AuthorTier.SYSTEM)
    valid = conn.execute("SELECT valid FROM derived WHERE id = 'd-sum'").fetchone()
    assert not res.ok or valid is None or valid[0] == 0
