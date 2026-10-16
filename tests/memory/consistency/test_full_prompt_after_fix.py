"""Acceptance for the memory spec §7.5 (runtime): after a fix, every old value in
the full prompt the CLI reads (resumed transcript plus the new prompt) is followed by an
update block. Spec marker: `[T consistency/test_full_prompt_after_fix::<case>]`."""

from __future__ import annotations

from koob.memory import api, store
from koob.memory.adapters import turn as turnmod

from ..adapters._turnkit import FIX_WORDS, fix_world, memory_db
from ..races._kit import handle, owner_turn, ref, serve


def test_full_prompt_after_fix(tmp_path):
    """§7.5: after a fix the old value is absent or marked outdated in the full prompt the CLI reads."""
    conn = store.open(memory_db(tmp_path))
    try:
        _sasha, _dima, bday, _city = fix_world(conn)
        serve(conn, thread="thread-1", session="session-1", item=ref(conn, bday.id), via="core")
        # the fake CLI's resumed session: an old exchange that carried the old value
        transcript = "[user] когда у Саши день рождения?\n[assistant] Sasha's birthday is 12 May"
        api.fix(
            handle(bday.id),
            "у Саши день рождения 14 мая",
            turn=owner_turn(FIX_WORDS, n=2),
            because_quote="у Саши день рождения 14 мая",
            store=conn,
        )
        turn = owner_turn("а что подарить?", n=3, thread="thread-1", session="session-1")
        # a new CLI process resuming session-1: its first question
        inj = turnmod.compute(conn, turn, first_question=True, resumed=True)
        full = transcript + "\n[user] " + inj.prepend("а что подарить?")
        tail = full[full.index("12 May") :]  # the old value in the transcript ...
        assert "<memory-update>" in tail and "14" in tail
        assert any(w in tail for w in ("outdated", "retracted", "previously"))
        assert api.pre_send_check(thread_id="thread-1", read_set=[], store=conn).ok
    finally:
        conn.close()
