"""the memory spec §12: exactly once - the processed set, a restart after delivery, and
slot-bound triggers that follow supersession. Spec marker: `[T intentions/test_once::<case>]`."""

from __future__ import annotations

from koob.memory import api, intentions, store
from koob.memory.model import IntentionState, Trigger, TriggerKind, record_event_key

from ..races._kit import fact, person, seed, version_of
from ._kit import (
    DAY,
    DIMA,
    DUE,
    fix_value,
    from_dima,
    on_date,
    processed,
    reminder,
    state,
    tg_message,
)


def test_redelivery(mem):
    """§12: a redelivered event collides in intention_events."""
    seed(mem, DIMA)
    iid = intentions.arm(mem, reminder("reply", from_dima(), quote="напомни, когда Дима ответит"))
    key, payload = tg_message("m-42", resolved=False)  # the sender resolved by stable id
    first = api.on_event("telegram", key, payload, store=mem)
    assert [(f.intention_id, f.event_key) for f in first] == [(iid, key)]
    assert first[0].origin_key == intentions.origin_key(iid, key)
    assert api.on_event("telegram", key, payload, store=mem) == []
    assert intentions.fire(mem, iid, key) is False  # the processed set itself collides
    assert processed(mem, iid) == 1
    notes = [n["text"] for n in api.recorded_notifications(store=mem)]
    assert notes == ["Напоминание: когда Дима ответит"]
    # A message before the intention's T never fires it.
    early = reminder("later", from_dima(after="2026-10-20T00:00:00.000000Z"))
    later_id = intentions.arm(mem, early)
    key2, payload2 = tg_message("m-43")
    assert api.on_event("telegram", key2, payload2, store=mem) == []
    assert state(mem, later_id) == IntentionState.ARMED


def test_crash_after_telegram_delivered_before_post_tool(tmp_path):
    """§12: a crash after delivery and before PostToolUse never sends twice."""
    db = tmp_path / "memory" / "memory.db"
    home = tmp_path / "home"
    conn = store.open(db, home=home)
    action = {"tool": "tg_send", "stable_target": "tg:777", "text_en": "Ask Dima about tickets"}
    iid = intentions.arm(conn, reminder("tickets", on_date(), action=action))
    (fire,) = api.on_event("clock", DAY, {"ts": DUE}, store=conn)
    origin = fire.origin_key
    assert state(conn, iid) == IntentionState.FIRED
    conn.close()  # Telegram delivered; the process died before PostToolUse closed it

    c = store.open(db, home=home)
    try:
        assert state(c, iid) == IntentionState.FIRED  # never re-armed by the restart
        assert api.on_event("clock", DAY, {"ts": DUE}, store=c) == []
        assert intentions.origin_key(iid, DAY) == origin  # a re-run reserves the same key
        assert intentions.settle(c, iid, "unknown")  # §16.2 step 4: checked, never blind
        assert state(c, iid) == IntentionState.NEEDS_ATTENTION
        assert api.on_event("clock", DAY, {"ts": DUE}, store=c) == []
        assert processed(c, iid) == 1
    finally:
        c.close()


def test_trigger_follows_supersession(mem):
    """§12: a status_change trigger follows supersedes edges."""
    acme = person("Acme")
    stage = fact("stage", acme, "status", "interview", "Acme: interview", "в Acme интервью")
    seed(mem, acme, stage)
    trig = Trigger(
        kind=TriggerKind.STATUS_CHANGE, subject_id=acme.id, relation="status", to="offer"
    )
    iid = intentions.arm(mem, reminder("offer", trig, quote="напомни, когда будет оффер от Acme"))

    # The commit event of the old record arrives late, after a fix superseded it: the slot's
    # current head decides, not the payload's value.
    old_key = record_event_key(stage.id, version_of(mem, stage.id))
    offer = fix_value(mem, stage.id, "stage2", "offer", subject=acme, rel="status")
    stale = {"record_id": stage.id, "subject_id": acme.id, "relation": "status"}
    stale |= {"value_norm": "interview", "status": "active"}
    assert [f.intention_id for f in api.on_event("record", old_key, stale, store=mem)] == [iid]
    assert state(mem, iid) == IntentionState.DONE
    new_key = record_event_key(offer.id, version_of(mem, offer.id))
    payload = {"record_id": offer.id, "subject_id": acme.id, "relation": "status"}
    assert api.on_event("record", new_key, payload, store=mem) == []

    # A payload claiming the target value while the slot moved on does not fire.
    trig2 = trig.model_copy(update={"to": "interview"})
    other = intentions.arm(mem, reminder("again", trig2, quote="напомни про интервью"))
    lie = stale | {"value_norm": "interview"}
    assert api.on_event("record", old_key + "x", lie, store=mem) == []
    assert state(mem, other) == IntentionState.ARMED
