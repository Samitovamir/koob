"""Intention events from the runtime's side (§12, G13): one engine, not two.

The duties machinery (`assistant/duties.py`) already listens to Telegram and keeps a clock; its
sources hand over ids and times only ({"chat_id", "message_id", "at"}). This adapter turns
those, the clock and ledger commits into `api.on_event` calls with the §12 keys and payloads,
so a memory intention fires through the same listeners a duty uses. No message text crosses
here. Wiring it into `duties.register_source` / the bot's clock is a koob-side edit; nothing
in koob/ imports this yet.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from .. import api
from .. import clock as _clock
from ..clock import iso
from ..model import date_event_key, message_event_key, record_event_key


def _chat(value: Any) -> str:
    text = str(value)
    return text if ":" in text else f"tg:{text}"


def message(
    chat_id: Any,
    message_id: Any,
    *,
    sender_id: Any = None,
    ts: str | None = None,
    sender_entity_id: str | None = None,
    reply_to: Any = None,
    source: str = "telegram",
    store: sqlite3.Connection | None = None,
) -> list[api.Fire]:
    """A new message in a chat (any messenger by its name): §12 key and payload."""
    chat = _chat(chat_id)
    payload: dict[str, Any] = {
        "chat_id": chat,
        "sender_id": _chat(sender_id) if sender_id is not None else chat,
        "message_id": str(message_id),
        "ts": ts or iso(_clock.now()),
    }
    if sender_entity_id:
        payload["sender_entity_id"] = sender_entity_id
    if reply_to is not None:
        payload["reply_to"] = str(reply_to)
    key = message_event_key(source, chat, str(message_id))
    return api.on_event(source, key, payload, store=store)


def duty_event(
    ev: dict[str, Any], *, source: str = "telegram", store: sqlite3.Connection | None = None
) -> list[api.Fire]:
    """What a duties `Source` emits (`{"chat_id", "message_id", "at"}`, epoch seconds) as a
    memory event. A `lost` notice or an event without ids is not a message: nothing fires."""
    if "lost" in ev or ev.get("chat_id") is None or ev.get("message_id") is None:
        return []
    at = ev.get("at")
    ts = iso(datetime.fromtimestamp(float(at), tz=UTC)) if at is not None else None
    return message(
        ev["chat_id"],
        ev["message_id"],
        sender_id=ev.get("sender_id"),
        ts=ts,
        reply_to=ev.get("reply_to"),
        source=source,
        store=store,
    )


def tick(now: datetime | None = None, *, store: sqlite3.Connection | None = None) -> list[api.Fire]:
    """The clock: `date:<UTC date>`; later ticks on the same date never refire (§12)."""
    at = (now or _clock.now()).astimezone(UTC)
    return api.on_event(
        "clock", date_event_key(at.strftime("%Y-%m-%d")), {"ts": iso(at)}, store=store
    )


def record_changed(
    conn: sqlite3.Connection, record_ids: list[str], *, store: sqlite3.Connection | None = None
) -> list[api.Fire]:
    """After a commit touched slots: one `record` event per fact, payload from its row."""
    fires: list[api.Fire] = []
    for rid in dict.fromkeys(record_ids):
        row = conn.execute(
            "SELECT r.version, r.status, f.subject_id, f.relation, f.value_norm FROM records r"
            " JOIN facts f ON f.id = r.id WHERE r.id = ?",
            (rid,),
        ).fetchone()
        if row is None:
            continue
        payload = {
            "record_id": rid,
            "version": int(row[0]),
            "status": row[1],
            "subject_id": row[2],
            "relation": row[3],
            "value_norm": row[4],
        }
        key = record_event_key(rid, int(row[0]))
        fires += api.on_event("record", key, payload, store=store if store is not None else conn)
    return fires
