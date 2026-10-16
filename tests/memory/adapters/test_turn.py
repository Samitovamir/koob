"""The turn adapter's host (runtime): legacy starts nothing; shadow computes off the
first word's path, logs content-free counts and injects nothing; ledger timings."""

from __future__ import annotations

import time

from koob.memory import store
from koob.memory.adapters import turn as turnmod

from ..races._kit import handle, owner_turn, ref, serve
from ._turnkit import FIX_WORDS, fix_world, host, memory_db


def test_legacy_off_starts_nothing(tmp_path):
    h = turnmod.Host()
    path = tmp_path / "memory" / "memory.db"
    assert h.start(engine="legacy", shadow="off", db_path=path) is False
    assert h.before(owner_turn("привет"), first_question=True) == turnmod.Injection()
    assert h.mid_turn("thread-owner", None) is None
    h.after(turn_id="1", thread_id="1", origin="owner", status="done", pass_id=None)
    assert h._thread is None and not path.exists()  # noqa: SLF001


def test_shadow_injects_nothing_and_logs_counts(tmp_path):
    db_path = memory_db(tmp_path)
    conn = store.open(db_path)
    try:
        _sasha, _dima, bday, _city = fix_world(conn)
        serve(conn, thread="thread-owner", session=None, item=ref(conn, bday.id), via="core")
        from koob.memory import api

        api.fix(
            handle(bday.id),
            "у Саши день рождения 14 мая",
            turn=owner_turn(FIX_WORDS, n=2),
            because_quote="у Саши день рождения 14 мая",
            store=conn,
        )
        with host(db_path, shadow="on") as h:
            t0 = time.perf_counter()
            inj = h.before(owner_turn("когда у Саши день рождения?", n=3), first_question=True)
            enqueue_ms = (time.perf_counter() - t0) * 1000
            assert inj == turnmod.Injection()  # nothing reaches the model in shadow
            assert enqueue_ms < 5, enqueue_ms  # never on the first word's path
            assert h.flush(10)
            row = h.shadow_log[-1]
            assert {"core_tokens", "basket_items", "total_ms", "queued_ms"} <= set(row)
            assert all(isinstance(v, (int, float, bool, str)) for v in row.values())
            assert "Саш" not in str(row) and "14" not in str(row.get("turn_id"))
            shadow_rows = conn.execute(
                "SELECT count(*) FROM served WHERE thread_id LIKE 'shadow:%'"
            ).fetchone()[0]
            assert shadow_rows == 0
            # the real thread's update is still owed: shadow delivered nothing
            assert conn.execute(
                "SELECT count(*) FROM pending_updates WHERE thread_id = 'thread-owner'"
            ).fetchone()[0]
            print(f"\nshadow enqueue {enqueue_ms:.3f} ms; worker {row}")
    finally:
        conn.close()


def test_ledger_turn_path_timings(tmp_path):
    db_path = memory_db(tmp_path)
    conn = store.open(db_path)
    try:
        fix_world(conn)
        with host(db_path) as h:
            first = owner_turn("когда у Саши день рождения?", n=3, session=None)
            t0 = time.perf_counter()
            inj = h.before(first, first_question=True)
            first_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            h.before(owner_turn("а Дима?", n=4, session=None), first_question=False)
            next_ms = (time.perf_counter() - t0) * 1000
            print(f"\nledger first {first_ms:.1f} ms {inj.timings_ms}; next {next_ms:.1f} ms")
            assert first_ms < 1000 and next_ms < 1000
    finally:
        conn.close()


def test_task_turn_gets_a_basket(tmp_path):
    """RC-7 (MEMORY.md §10.2): a task turn recalls too, not only the user's own messages."""
    from koob.memory.model import TurnOrigin

    db_path = memory_db(tmp_path)
    conn = store.open(db_path)
    try:
        fix_world(conn)
        with host(db_path) as h:
            words = "напомни Саше про день рождения"
            task = owner_turn(words, n=5, session=None).model_copy(
                update={"origin": TurnOrigin.TASK, "task_words": words}
            )
            inj = h.before(task, first_question=False)
            assert "basket" in inj.timings_ms
    finally:
        conn.close()
