"""«Забудь» while an answer is being prepared or streams (spec §7.4, §7.6, §7.7, §9.1, §9.5):
the pre-send check blocks the forgotten item from going out, every session that held it is
named for a restart, a purge never claims «удалил полностью» while a reader still holds the
old pages, and a forget from another process reaches the host."""

from __future__ import annotations

import json
import threading

from koob.memory import api
from koob.memory.model import ForgetScope

from ._kit import (
    WAIT,
    add_raw,
    fact,
    handle,
    journal_path,
    lineage,
    open_db,
    owner_turn,
    person,
    ref,
    run_child,
    run_threads,
    scan,
    seed,
    serve,
    status_of,
)

TOKEN_EN = "Zebrafinch4471"
TOKEN_RU = "Зебрафинч4471"
FORGET_WORDS = "забудь про переезд Саши"


def _world(conn):
    sasha, dima = person("Саша"), person("Дима")
    raw = add_raw(
        conn,
        "move",
        f"Саша переезжает в Тверь, улица {TOKEN_RU}",
        chat_id=None,
        turn_id="turn-0",
    )
    move = fact(
        "sasha-move",
        sasha,
        "moves_to",
        "Tver",
        f"Sasha is moving to Tver, {TOKEN_EN} street",
        f"Саша переезжает в Тверь, улица {TOKEN_RU}",
        source_raw_id=raw,
    )
    city = fact("dima-city", dima, "lives_in", "Perm", "Dima lives in Perm", "Дима в Перми")
    seed(conn, sasha, dima, move, city, extra=lambda sql: lineage(sql, move.id, raw))
    return move, city


def _forget(conn, move, *, n: int = 2, forever: bool = False):
    return api.forget(
        handle(move.id),
        turn=owner_turn(FORGET_WORDS, n=n),
        because_quote="переезд Саши",
        scope=ForgetScope.ITEM,
        forever=forever,
        store=conn,
    )


def test_presend_blocks_item_forgotten_while_answer_prepares(db_path, conn):
    """§7.4, §9.1 1a: the basket was built with the item; the forget commits before the send;
    the pre-send check refuses the item and keeps the rest; the journal has the stop line."""
    move, city = _world(conn)
    basket = [ref(conn, move.id), ref(conn, city.id)]
    serve(conn, thread="thread-owner", session="session-owner", item=basket[0], turn_id="turn-1")
    built, forgotten = threading.Event(), threading.Event()

    def answer():
        a = open_db(db_path)
        try:
            built.set()
            assert forgotten.wait(WAIT)
            return api.pre_send_check(thread_id="thread-owner", read_set=basket, store=a)
        finally:
            a.close()

    def owner():
        assert built.wait(WAIT)
        f = open_db(db_path)
        try:
            return _forget(f, move)
        finally:
            f.close()
            forgotten.set()

    pre, result = run_threads(answer, owner)
    assert result.outcome == "stopped" and move.id in result.record_ids
    assert not pre.ok
    assert basket[0] in pre.stale and basket[1] not in pre.stale
    assert status_of(conn, move.id) == "forgotten"
    lines = [json.loads(x) for x in journal_path(db_path).read_text().splitlines() if x.strip()]
    stopped = [x for x in lines if x["stage"] == "stopped" and move.id in x["record_ids"]]
    assert stopped and stopped[-1]["forget_id"] == result.forget_id
    assert not scan(journal_path(db_path).parent, TOKEN_EN) + scan(
        journal_path(db_path).parent, "Тверь"
    )


def test_forget_racing_presend_checks_never_lets_it_through(db_path, conn):
    """Pre-send checks run back to back on another connection while the forget commits;
    every check that starts after `forget` returned refuses the item (no stale snapshot)."""
    move, city = _world(conn)
    basket = [ref(conn, move.id), ref(conn, city.id)]
    returned = threading.Event()
    started = threading.Event()
    seen: list[tuple[bool, bool]] = []

    def checker():
        a = open_db(db_path)
        try:
            after = 0
            while after < 5:
                was_after = returned.is_set()
                pre = api.pre_send_check(thread_id="thread-owner", read_set=basket, store=a)
                seen.append((was_after, pre.ok))
                started.set()
                after += was_after
        finally:
            a.close()

    def owner():
        assert started.wait(WAIT)
        f = open_db(db_path)
        try:
            _forget(f, move)
        finally:
            f.close()
            returned.set()

    run_threads(checker, owner)
    assert seen and seen[0] == (False, True)
    assert all(not ok for was_after, ok in seen if was_after)


def test_streaming_turn_and_every_serving_session_are_named(db_path, conn):
    """§9.1 1c, §9.5: the turn whose raw is in the lineage, the turn streaming an answer that
    was served the item, and every session (user and task) that holds it are returned for
    blanking and restart."""
    move, _city = _world(conn)
    m0 = ref(conn, move.id)
    serve(conn, thread="thread-owner", session="session-owner", item=m0, turn_id="turn-3")
    serve(conn, thread="thread-task", session="session-task", item=m0, via="core")
    result = _forget(conn, move, n=4)
    assert result.outcome == "stopped"
    assert {"turn-0", "turn-3"} <= set(result.turn_ids)
    assert {"session-owner", "session-task"} <= set(result.sessions_to_restart)


def test_purge_with_a_reader_open_is_not_claimed_purged(db_path, conn):
    """§9.1 stage 3: «удалил полностью» needs a verified `wal_checkpoint(TRUNCATE)`; while
    a streaming answer holds a read snapshot, the purge stays «удаляю» and retries; once the
    reader is gone it completes and no byte of the content is left in the live files."""
    move, _city = _world(conn)
    result = _forget(conn, move, forever=True)
    assert result.outcome == "stopped"

    reader = open_db(db_path)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT count(*) FROM records").fetchone()[0] > 0
        api.run_purges(store=conn)
        assert api.purge_status(result.forget_id, store=conn).state != "purged"
        assert status_of(reader, move.id) != "active"
    finally:
        reader.execute("ROLLBACK")
        reader.close()

    api.run_purges(store=conn)
    assert api.purge_status(result.forget_id, store=conn).state == "purged"
    root = db_path.parent
    for token in (TOKEN_EN, TOKEN_RU, TOKEN_EN.lower()):
        assert scan(root, token, skip=("backups",)) == [], token


def test_forgotten_item_is_never_restated_in_an_update_block(db_path, conn):
    """§7.6: forgotten ids get no update block (it would restate them); the session that held
    it restarts instead, and its gate does not stay shut on an undeliverable update."""
    move, _city = _world(conn)
    serve(conn, thread="thread-task", session="session-task", item=ref(conn, move.id), via="core")
    result = _forget(conn, move)
    upd = api.update_block(thread_id="thread-task", session_id="session-task", store=conn)
    assert upd.block is None or (move.id not in upd.block and "Tver" not in upd.block)
    assert upd.restart or "session-task" in result.sessions_to_restart
    assert (
        api.pre_send_check(thread_id="thread-task", read_set=[], store=conn).undelivered_updates
        == 0
    )


def test_forget_from_another_process_reaches_the_live_host(db_path, conn):
    """§7.7, §19.1 «a write from another process»: `cb memory forget` runs in its own process
    while the host holds a prepared read set; the host's watermark read shows the forget and
    its pre-send check refuses the item."""
    move, _city = _world(conn)
    prepared = [ref(conn, move.id)]
    v0 = api.changes_since(memory_version=0, last_event_id=0, store=conn)
    turn = owner_turn(FORGET_WORDS, n=2)
    code, out = run_child(
        {
            "db": str(db_path),
            "home": str(db_path.parent.parent / "home"),
            "kill_at": None,
            "target": "api",
            "call": "forget",
            "args": [handle(move.id)],
            "kwargs": {
                "turn": turn.model_dump(mode="json"),
                "because_quote": "переезд Саши",
                "scope": "item",
            },
        }
    )
    assert code == 0 and out is not None and out["result"]["outcome"] == "stopped"
    seen = api.changes_since(
        memory_version=v0.memory_version, last_event_id=v0.last_event_id, store=conn
    )
    assert seen.memory_version > v0.memory_version
    assert seen.last_event_id > v0.last_event_id and seen.forget_events
    pre = api.pre_send_check(thread_id="thread-owner", read_set=prepared, store=conn)
    assert not pre.ok and prepared[0] in pre.stale
