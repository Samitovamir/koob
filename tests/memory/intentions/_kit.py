"""Shared seeds for the §12 intention tests: a contact with a Telegram id, the contact's address,
a fix that supersedes a value, and reminders by trigger."""

from __future__ import annotations

from typing import Any

from koob.memory import ledger, store
from koob.memory.model import (
    AuthorTier,
    Fact,
    Intention,
    OpKind,
    Proposal,
    ProposalOp,
    Trigger,
    TriggerKind,
)

from ..races._kit import TS0, fact, person, seed, ulid, version_of

DUE = "2026-10-13T09:00:00.000000Z"
DAY = "date:2026-10-13"
DIMA = person("Дима", tg_user_id="777")


def seed_dima(conn: Any, address: str = "Ленина 5") -> Fact:
    addr = fact("addr", DIMA, "address", address, f"Dima lives at {address}", "Дима живёт ...")
    seed(conn, DIMA, addr)
    return addr


def fix_value(conn: Any, old_id: str, key: str, value: str, *, subject=DIMA, rel="address") -> Fact:
    """A correction: a new record in the slot supersedes the old one (fix path)."""
    new = fact(key, subject, rel, value, f"{rel}: {value}", f"на самом деле {value}")

    def apply(ctx: store.CommitCtx, op: ProposalOp) -> list[str]:
        ledger.insert_record(ctx, new)
        ledger.supersede(ctx, old_id, new.id)
        return [new.id, old_id]

    res = store.commit(conn, Proposal(ops=[ProposalOp(op=OpKind.ADD, group="fix")]), apply)
    assert res.ok
    return new


def reminder(
    key: str,
    trigger: Trigger,
    *,
    quote: str = "напомни написать Диме про билеты завтра в девять",
    action: dict[str, Any] | None = None,
    completion: dict[str, Any] | None = None,
    **header: Any,
) -> Intention:
    return Intention(
        id=ulid(f"intention:{key}"),
        author_tier=AuthorTier.USER,
        recorded_at=TS0,
        labels_final=True,
        origin_quote=quote,
        trigger=trigger,
        action=action,
        completion=completion,
        **header,
    )


def on_date(ts: str = DUE) -> Trigger:
    return Trigger(kind=TriggerKind.DATE_REACHED, ts=ts)


def from_dima(after: str = TS0) -> Trigger:
    return Trigger(kind=TriggerKind.MESSAGE_FROM, entity_id=DIMA.id, after_ts=after)


def address_action(conn: Any, addr: Fact) -> dict[str, Any]:
    return {
        "tool": "tg_send",
        "stable_target": "tg:777",
        "template": "Адрес: {0}",
        "text_en": f"Адрес: {addr.value_text}",
        "read_set": [{"id": addr.id, "version": version_of(conn, addr.id), "src": "record"}],
    }


def state(conn: Any, iid: str) -> str:
    return conn.execute("SELECT state FROM intentions WHERE id = ?", (iid,)).fetchone()[0]


def processed(conn: Any, iid: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM intention_events WHERE intention_id = ?", (iid,)
    ).fetchone()[0]


def deps(conn: Any, holder_kind: str, holder_id: str) -> list[tuple[str, int]]:
    return [
        (r[0], r[1])
        for r in conn.execute(
            "SELECT item_id, version FROM read_deps WHERE holder_kind = ? AND holder_id = ?",
            (holder_kind, holder_id),
        )
    ]


def tg_message(mid: str, *, ts: str = "2026-10-12T10:00:00.000000Z", resolved: bool = True):
    payload = {
        "chat_id": "tg:777",
        "sender_id": "tg:777",
        "message_id": mid,
        "ts": ts,
    }
    if resolved:
        payload["sender_entity_id"] = DIMA.id
    return f"telegram:tg:777:{mid}", payload
