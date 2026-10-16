"""Event redelivery and a restart mid-fire (spec §12, §16.2): the same event arriving twice,
at the same instant on two connections, or again after the process died mid-fire, fires an
intention exactly once; the fire's action is reserved once and an unknown outcome is never
retried blind."""

from __future__ import annotations

import threading

from koob.memory import api, intentions
from koob.memory.model import AuthorTier, Intention, Trigger, TriggerKind

from .._contract import not_built
from ._kit import (
    TS0,
    WAIT,
    crash_runs,
    open_db,
    person,
    run_threads,
    seed,
    ulid,
)

DUE = "2026-10-13T09:00:00.000000Z"
DATE_KEY = "date:2026-10-13"


def _reminder(trigger: Trigger, key: str = "tickets") -> Intention:
    return Intention(
        id=ulid(f"intention:{key}"),
        author_tier=AuthorTier.USER,
        recorded_at=TS0,
        labels_final=True,
        origin_quote="напомни написать Диме про билеты завтра в девять",
        trigger=trigger,
        action={"tool": "tg_send", "stable_target": "tg:777", "text_en": "Ask Dima about tickets"},
        completion={"on": "fired"},
    )


def _arm_date(conn) -> str:
    return intentions.arm(conn, _reminder(Trigger(kind=TriggerKind.DATE_REACHED, ts=DUE)))


def _fire_date(conn):
    return api.on_event("clock", DATE_KEY, {"ts": DUE}, store=conn)


def _processed(conn, iid: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM intention_events WHERE intention_id = ?", (iid,)
    ).fetchone()[0]


def _state(conn, iid: str) -> str:
    return conn.execute("SELECT state FROM intentions WHERE id = ?", (iid,)).fetchone()[0]


def test_same_event_twice_fires_once(db_path, conn):
    """§12: `intention_events(intention_id, event_key)` is UNIQUE; a redelivered event collides."""
    iid = _arm_date(conn)
    first = _fire_date(conn)
    second = _fire_date(conn)
    assert [f.intention_id for f in first] == [iid]
    assert first[0].event_key == DATE_KEY and first[0].origin_key
    assert second == []
    assert _processed(conn, iid) == 1
    assert _state(conn, iid) in ("fired", "done")


def test_concurrent_redelivery_fires_once(db_path, conn):
    """The same event delivered at the same instant on two connections fires once."""
    iid = _arm_date(conn)
    gate = threading.Barrier(2, timeout=WAIT)

    def deliver():
        c = open_db(db_path)
        try:
            gate.wait()
            return _fire_date(c)
        finally:
            c.close()

    a, b = run_threads(deliver, deliver)
    assert len(a) + len(b) == 1
    assert _processed(conn, iid) == 1


def test_message_redelivered_after_fire_fires_once(db_path, conn):
    """§12 `message_from(entity, after_ts)`: the same Telegram message twice fires once; a later
    message does not fire the intention again (it is no longer armed)."""
    dima = person("Дима", tg_user_id="777")
    seed(conn, dima)
    iid = intentions.arm(
        conn,
        _reminder(Trigger(kind=TriggerKind.MESSAGE_FROM, entity_id=dima.id, after_ts=TS0), "reply"),
    )

    def message(mid: str):
        payload = {
            "chat_id": "tg:777",
            "sender_id": "tg:777",
            "sender_entity_id": dima.id,
            "message_id": mid,
            "ts": "2026-10-12T10:00:00.000000Z",
        }
        return api.on_event("telegram", f"telegram:tg:777:{mid}", payload, store=conn)

    assert [f.intention_id for f in message("m-42")] == [iid]
    assert message("m-42") == []
    assert message("m-43") == []
    assert _processed(conn, iid) == 1


def test_redelivery_after_restart_mid_fire_fires_once(db_path, conn, tmp_path):
    """§19.1 restart mid-fire: the process dies at points spread over `on_event`; on reopen the
    processed set and the intention state agree, and the redelivered event fires exactly the
    fires still owed: one in total."""
    iid = _arm_date(conn)
    conn.close()
    call = {"target": "api", "call": "on_event", "args": ["clock", DATE_KEY, {"ts": DUE}]}
    for db, code in crash_runs(db_path, tmp_path, call):
        c = open_db(db)
        try:
            done = _processed(c, iid)
            assert done in (0, 1)
            assert (_state(c, iid) == "armed") == (done == 0), f"state torn (exit {code})"
            again = _fire_date(c)
            assert len(again) == 1 - done
            assert _processed(c, iid) == 1
        finally:
            c.close()


@not_built("hands")  # intent part built; needs record_action / actions_at_startup
def test_fire_action_reserved_once_and_unknown_never_retried_blind(db_path, conn):
    """§12 + §16.2: the fire's send is reserved with `origin_key`; the process dies after
    Telegram delivered and before PostToolUse; at startup the action is `unknown`; the
    redelivered event does not fire again and the re-run step is denied, not retried."""
    iid = _arm_date(conn)
    (fire,) = _fire_date(conn)
    ticket = api.record_action(
        tool="tg_send",
        stable_target="tg:777",
        origin_key=fire.origin_key,
        read_set=[],
        intention_id=iid,
        store=conn,
    )
    assert ticket.allow and ticket.action_key
    conn.close()  # the process dies here: delivered, never closed

    c = open_db(db_path)
    try:
        assert api.actions_at_startup(store=c) == 1
        state = c.execute(
            "SELECT state FROM actions WHERE action_key = ?", (ticket.action_key,)
        ).fetchone()[0]
        assert state == "unknown"
        assert _fire_date(c) == []
        rerun = api.record_action(
            tool="tg_send",
            stable_target="tg:777",
            origin_key=fire.origin_key,
            read_set=[],
            intention_id=iid,
            store=c,
        )
        assert not rerun.allow and rerun.reason
        rows = c.execute(
            "SELECT count(*) FROM actions WHERE origin_key = ?", (fire.origin_key,)
        ).fetchone()[0]
        assert rows == 1
    finally:
        c.close()
