"""Acceptance for the memory spec §7.7 (runtime): a forget made by another
process reaches a live session of this host within one turn.
Spec marker: `[T consistency/test_cross_process::<case>]`."""

from __future__ import annotations

from koob.memory import store

from ..adapters._turnkit import (
    FORGET_WORDS,
    Restarts,
    career,
    forget_world,
    host,
    memory_db,
    task_turn,
)
from ..races._kit import handle, owner_turn, ref, run_child, serve


def test_cb_forget_reaches_live_session(tmp_path):
    """§7.7: a `cb memory` forget in another process reaches a live session within one turn."""
    db_path = memory_db(tmp_path)
    restarts = Restarts()
    with host(db_path, conversation=career(tmp_path, restarts), restarts=restarts) as h:
        conn = store.open(db_path)
        try:
            move, _city = forget_world(conn)
            serve(conn, thread="thread-live", session="s-live", item=ref(conn, move.id), via="core")
            turn = task_turn("thread-live", "s-live")
            assert h.before(turn, first_question=False).restart_session is False
            code, out = run_child(
                {
                    "db": str(db_path),
                    "home": str(tmp_path / "data" / "home"),
                    "kill_at": None,
                    "target": "api",
                    "call": "forget",
                    "args": [handle(move.id)],
                    "kwargs": {
                        "turn": owner_turn(FORGET_WORDS, n=2).model_dump(mode="json"),
                        "because_quote": "переезд Саши",
                        "scope": "item",
                    },
                }
            )
            assert code == 0 and out is not None and out["result"]["outcome"] == "stopped"
            assert restarts.calls == []  # the other process has no live session to restart
            inj = h.before(turn, first_question=False)  # this host's next turn
            assert inj.restart_session is True
            assert "thread-live" in restarts.threads or restarts.all == 1
            assert inj.update is None or move.id not in inj.update
        finally:
            conn.close()
