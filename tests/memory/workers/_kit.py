"""Helpers for the extract tests: a seeded store with raw, gold extractors standing in for
the model (no network), and the processors / routing / classifier switched per test.

Not a test module."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from koob.memory import clock as clockmod
from koob.memory import policy
from koob.memory.clock import ManualClock
from koob.memory.workers import extract, scheduler

from .. import _ledger as L


class Gold:
    """A fake processor: returns a gold extraction (a dict or a function of the snapshot)."""

    def __init__(
        self, gold: dict | Callable[[dict], dict], *, route: str = "cloud", up: bool = True
    ) -> None:
        self.gold, self.route, self.up = gold, route, up
        self.model_id = f"gold-{route}"
        self.snaps: list[dict[str, Any]] = []

    def available(self) -> bool:
        return self.up

    def __call__(self, snap: dict[str, Any]) -> dict[str, Any]:
        self.snaps.append(snap)
        return self.gold(snap) if callable(self.gold) else self.gold


def fact(
    ref: str, subject: str, relation: str, value: str | None, quote: str, raw: str, **kw: Any
) -> dict[str, Any]:
    return {
        "ref": ref,
        "subject": subject,
        "relation": relation,
        "value_text": value,
        "text_en": f"{subject} {relation} {value}",
        "quote_orig": quote,
        "fkind": kw.pop("fkind", "state"),
        "source_raw_id": raw,
        **kw,
    }


SASHA = {
    "entities": [{"ref": "e1", "etype": "person", "display_name": "Sasha", "aliases": ["Саша"]}],
    "facts": [fact("f1", "e1", "works_at", "Yandex", "Саша работает в Яндексе", "r1")],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A seeded store, a manual clock wired as the process clock, labels finalizable."""
    conn = L.open_seeded(tmp_path)
    clk = ManualClock()
    prev = clockmod._current
    clockmod.use(clk)
    policy.set_classifier(lambda text: {})
    yield conn, clk
    extract.configure()
    policy.set_classifier(None)
    policy.configure()
    clockmod.use(prev)
    conn.close()


def raw(
    conn, rid: str, text: str, *, tier: str = "user", labels: int = 0, turn: str | None = None
) -> None:
    L.raw(conn, rid, text, tier=tier, labels=labels)
    if turn:
        conn.execute("UPDATE raw_chunks SET turn_id = ? WHERE id = ?", (turn, rid))
    conn.commit()


def batch(conn, raw_ids: list[str], turn: str | None = "t1") -> str:
    return scheduler.enqueue(
        conn,
        scheduler.EXTRACT,
        {"raw_ids": raw_ids, "turn_id": turn},
        idempotency_key=f"extract:{raw_ids[0]}",
    )


def jobs(conn) -> list[dict[str, Any]]:
    return [
        {"id": r[0], "jkind": r[1], "payload": json.loads(r[2]), "state": r[3]}
        for r in conn.execute(
            "SELECT id, jkind, payload, state FROM jobs WHERE jkind != 'embed'"
            " ORDER BY created_at, id"
        )
    ]


def facts(conn) -> dict[str, str]:
    return {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT f.relation, f.value_text FROM facts f JOIN records r ON r.id = f.id"
            " WHERE r.status = 'active'"
        )
    }


def audit(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT event FROM audit WHERE actor = 'worker'")]
