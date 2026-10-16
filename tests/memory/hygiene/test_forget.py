"""Acceptance for the memory spec §11, §9.1c (runtime): a forget of served or
transcribed content restarts every session that held it, the answer that streamed during
it is blanked, and the next call holds no trace - without touching SDK files.
Spec marker: `[T hygiene/test_forget::<case>]`."""

from __future__ import annotations

from koob import db as career_db
from koob.memory import api, store
from koob.memory.adapters import turn as turnmod
from koob.memory.adapters.conversation import FORGOTTEN

from ..adapters._turnkit import Restarts, _forget, career, forget_world, host, memory_db
from ..races._kit import owner_turn, ref, serve
from ..races.test_forget_during_answer import TOKEN_EN

TS = "2026-10-14T12:00:00+00:00"


def _world(tmp_path):
    db_path = memory_db(tmp_path)
    conn = store.open(db_path)
    move, _city = forget_world(conn)
    return db_path, conn, move


def test_all_sessions_restarted(tmp_path):
    """§9.1c: every live session that served a forgotten id or transcribed an affected turn restarts."""
    restarts = Restarts()
    db_path, conn, move = _world(tmp_path)
    try:
        with host(db_path, conversation=career(tmp_path, restarts)):
            for t in ("11", "12"):
                serve(conn, thread=t, session=f"s-{t}", item=ref(conn, move.id), via="basket")
            result = _forget(conn, move)
            assert result.outcome == "stopped"
            assert {"11", "12"} <= restarts.threads
    finally:
        conn.close()


def test_streaming_answer_blanked(tmp_path):
    """§9.1c: the answer that streamed during the forget is re-matched and blanked."""
    restarts = Restarts()
    db_path, conn, move = _world(tmp_path)
    conv = career(tmp_path, restarts)
    try:
        with host(db_path, conversation=conv) as h:
            _forget(conn, move)
            # the answer streamed during the forget lands its rows after it
            c = conv.conn
            tid = c.execute(
                "INSERT INTO koob_threads(title, created_at, updated_at) VALUES ('t', ?, ?)",
                (TS, TS),
            ).lastrowid
            c.execute(
                "INSERT INTO koob_turns(id, thread_id, status, started_at) VALUES (77, ?,"
                " 'done', ?)",
                (tid, TS),
            )
            c.execute(
                "INSERT INTO koob_messages(thread_id, turn_id, role, text, status, created_at)"
                " VALUES (?, 77, 'assistant', ?, 'done', ?)",
                (tid, f"Sasha is {TOKEN_EN}", TS),
            )
            c.commit()
            turnmod.HOST = h  # the module-level entry points use the started host
            turnmod.after(turn_id="77", thread_id=str(tid), origin="owner", status="done")
            assert h.flush(10)
            text = c.execute("SELECT text FROM koob_messages WHERE turn_id = 77").fetchone()[0]
            assert TOKEN_EN not in (text or "") and FORGOTTEN in (text or "")
    finally:
        turnmod.HOST = turnmod.Host()
        conn.close()


def test_correct_without_touching_sdk_files(tmp_path):
    """§11: correctness never relies on editing SDK transcript files."""
    restarts = Restarts()
    db_path, conn, move = _world(tmp_path)
    sdk = tmp_path / "claude-projects" / "session-1.jsonl"
    sdk.parent.mkdir()
    sdk.write_text(f'{{"text": "Sasha is {TOKEN_EN}"}}\n', encoding="utf-8")
    before = sdk.read_bytes()
    try:
        with host(db_path, conversation=career(tmp_path, restarts)):
            serve(conn, thread="21", session="session-1", item=ref(conn, move.id), via="core")
            _forget(conn, move)
            assert "21" in restarts.threads  # a restart from a seed, not a file edit
            assert sdk.read_bytes() == before
    finally:
        conn.close()


def test_next_call_clean(tmp_path):
    """§11: the next call of every affected session holds no trace, match keys included."""
    db_path, conn, move = _world(tmp_path)
    try:
        serve(conn, thread="31", session="s-31", item=ref(conn, move.id), via="core")
        _forget(conn, move)
        turn = owner_turn("куда переезжает Саша?", n=5, thread="31", session=None)
        inj = turnmod.compute(conn, turn, first_question=True, resumed=False)
        text = inj.prepend("куда переезжает Саша?") + (inj.core or "")
        for key in (TOKEN_EN, TOKEN_EN.lower(), move.id):
            assert key.casefold() not in text.casefold()
        assert api.pre_send_check(thread_id="31", read_set=inj.read_set, store=conn).ok
    finally:
        conn.close()


del career_db
