"""Helpers for the arrival tests (§6.2): rows in the eval harness's shape."""

from __future__ import annotations

TS = "2026-10-12T09:00:00.000000Z"


def owner(text: str, **kw) -> dict:
    return {"role": "owner", "ts": TS, "text": text, **kw}


def cube(text: str, **kw) -> dict:
    return {"role": "cube", "ts": TS, "text": text, **kw}


def other(text: str, *, chat: str = "tg:777", msg_id: int = 42, **kw) -> dict:
    return {
        "role": "other",
        "ts": TS,
        "source": "telegram",
        "chat_id": chat,
        "msg_id": msg_id,
        "sender": "Дима",
        "sender_tg_id": 777,
        "text": text,
        **kw,
    }


def raws(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM raw_chunks ORDER BY fts_rowid")]


def jobs(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM jobs WHERE jkind = 'extract'")]


def queued_ids(conn) -> set[str]:
    import json

    return {i for j in jobs(conn) for i in json.loads(j["payload"])["raw_ids"]}


def run_turn(conn, rows, *, turn_id: str = "turn-1", thread_id: str = "thread-1") -> int:
    from koob.memory import api

    return api.ingest_turn(
        turn_id=turn_id, thread_id=thread_id, origin="owner", status="done", rows=rows, store=conn
    )
