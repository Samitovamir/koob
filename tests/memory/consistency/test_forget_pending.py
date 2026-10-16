"""the memory spec §7.6 (v0.4): a forget never leaves a thread's gate shut.
Spec marker: `[T consistency/test_forget_pending::<case>]`."""

from __future__ import annotations

from koob.memory import api, forget, store
from koob.memory.model import ForgetScope

from ..forget import _seed as s
from ..forget.test_about_entity import _world

clock = s.clock  # fixture


def _pending(conn, thread: str, item: str) -> None:
    conn.execute(
        "INSERT INTO pending_updates (thread_id, item_id, old_version, new_version, created_at)"
        " VALUES (?, ?, 1, 2, ?)",
        (thread, item, s.OLD),
    )


def test_forget_clears_pending_rows(mem, clock):
    """Stop-use deletes `pending_updates` of forgotten and masked ids in its own transaction;
    the thread's next pre-send check is not refused for them; the session restarts instead."""
    _world(mem)
    s.served(mem, "th-1", "f-subject", "t-1")
    s.served(mem, "th-1", "f-masha", "t-1")
    _pending(mem, "th-1", "f-subject")  # forgotten whole
    _pending(mem, "th-1", "f-masha")  # masked: still usable, X's name gone
    _pending(mem, "th-1", "raw-m")  # its raw, span-masked
    _pending(mem, "th-2", "f-value")  # forgotten too (X is its value)
    s.entity(mem, "e-ann", "Аня", [("Аня", "name")])
    s.fact(mem, "f-ann", "e-ann", "lives_in", value="Omsk", text_en="Anya lives in Omsk")
    _pending(mem, "th-3", "f-ann")  # unrelated: its gate stays shut
    mem.commit()
    assert not api.pre_send_check(thread_id="th-1", read_set=[], store=mem).ok

    events = []
    forget.wire(stop_hooks=[events.append])
    version = store.memory_version(mem)
    forget.stop_use(mem, ["ent-x"], scope=ForgetScope.ABOUT_ENTITY, forever=False)

    left = mem.execute("SELECT thread_id, item_id FROM pending_updates ORDER BY 1").fetchall()
    assert [tuple(r) for r in left] == [("th-3", "f-ann")]
    # one commit: the pending rows went with the stop-use's own version bump
    assert store.memory_version(mem) == version + 1
    for thread in ("th-1", "th-2"):
        assert api.pre_send_check(thread_id=thread, read_set=[], store=mem).ok, thread
    assert not api.pre_send_check(thread_id="th-3", read_set=[], store=mem).ok
    (event,) = events
    assert "th-1" in event.thread_ids  # the session restarts from a seed instead
