"""The intention events adapter (§12, G13): duties-shaped Telegram events, the clock and
ledger commits reach `api.on_event` with the §12 keys; a redelivery fires nothing."""

from __future__ import annotations

from datetime import UTC, datetime

from koob.memory import intentions
from koob.memory.adapters import intents
from koob.memory.model import IntentionState, Trigger, TriggerKind

from ..intentions._kit import DIMA, fix_value, from_dima, on_date, reminder, state
from ..races._kit import fact, person, seed


def test_duty_event_tick_and_record_fire_once(mem):
    seed(mem, DIMA)
    reply = intentions.arm(mem, reminder("reply", from_dima()))
    at = datetime(2026, 10, 12, 10, 0, tzinfo=UTC).timestamp()
    ev = {"chat_id": 777, "message_id": 42, "at": at}
    assert [f.event_key for f in intents.duty_event(ev, store=mem)] == ["telegram:tg:777:42"]
    assert intents.duty_event(ev, store=mem) == []
    assert intents.duty_event({"lost": "аккаунт вышел"}, store=mem) == []
    assert state(mem, reply) == IntentionState.DONE

    day = intentions.arm(mem, reminder("day", on_date()))
    noon = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    assert [f.event_key for f in intents.tick(noon, store=mem)] == ["date:2026-10-13"]
    assert intents.tick(noon, store=mem) == []
    assert state(mem, day) == IntentionState.DONE

    acme = person("Acme")
    stage = fact("stage", acme, "status", "interview", "Acme: interview", "интервью")
    seed(mem, acme, stage)
    trig = Trigger(
        kind=TriggerKind.STATUS_CHANGE, subject_id=acme.id, relation="status", to="offer"
    )
    offer_wait = intentions.arm(mem, reminder("offer", trig))
    offer = fix_value(mem, stage.id, "stage2", "Offer", subject=acme, rel="status")
    fired = intents.record_changed(mem, [offer.id, stage.id])
    assert [f.intention_id for f in fired] == [offer_wait]
