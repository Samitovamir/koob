"""Acceptance for the memory spec §7.6 (runtime): the thread's pending updates
reach a task session before its next model call, and a running session at its next
PostToolUse. Spec marker: `[T consistency/test_update_block::<case>]`."""

from __future__ import annotations

from koob.memory import api
from koob.memory.adapters import turn as turnmod

from ..adapters._turnkit import FIX_WORDS, fix_world, host, memory_db, task_turn
from ..races._kit import handle, owner_turn, ref, serve


def _fix(conn, bday):
    out = api.fix(
        handle(bday.id),
        "у Саши день рождения 14 мая",
        turn=owner_turn(FIX_WORDS, n=2),
        because_quote="у Саши день рождения 14 мая",
        store=conn,
    )
    assert out.outcome == "committed"


def test_reaches_task_session(tmp_path):
    """§7.6: a pending update reaches a task session before its next model call."""
    db_path = memory_db(tmp_path)
    from koob.memory import store

    conn = store.open(db_path)
    try:
        _sasha, _dima, bday, _city = fix_world(conn)
        serve(conn, thread="thread-task", session="s-task", item=ref(conn, bday.id), via="core")
        _fix(conn, bday)
        turn = task_turn("thread-task", "s-task")
        inj = turnmod.compute(conn, turn, first_question=False)
        assert inj.update is not None and f"[m:{bday.id}]" in inj.update and "14" in inj.update
        assert inj.basket is None  # recall rides on user turns only
        assert api.pre_send_check(thread_id="thread-task", read_set=[], store=conn).ok
        again = turnmod.compute(conn, turn, first_question=False)
        assert again.update is None  # delivered once, never twice
    finally:
        conn.close()


def test_mid_turn_via_post_tool(tmp_path):
    """§7.6: a correction landing mid-turn reaches the running session at its next PostToolUse."""
    db_path = memory_db(tmp_path)
    with host(db_path) as h:
        from koob.memory import store

        conn = store.open(db_path)
        try:
            _sasha, _dima, bday, _city = fix_world(conn)
            serve(conn, thread="thread-run", session="s-run", item=ref(conn, bday.id), via="core")
            assert h.mid_turn("thread-run", "s-run") is None  # nothing pending: one count
            _fix(conn, bday)  # lands while the session runs
            block = h.mid_turn("thread-run", "s-run")
            assert block is not None and f"[m:{bday.id}]" in block and "outdated" in block
            assert h.mid_turn("thread-run", "s-run") is None
        finally:
            conn.close()
