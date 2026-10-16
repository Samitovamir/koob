"""A toy memory behind the facade, for exercising the harness offline (the real api is unbuilt)."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from koob.memory.api import Basket, Item, SafetyNet, WriteResult
from koob.memory.eval.canary import StoreMap, StoreSpec
from koob.memory.eval.sut import ApiSUT


class FakeApi:
    """Remembers where the user lives (latest wins), drops banned chats, optionally leaks everything."""

    def __init__(self, raw_log: Path | None = None, leak: bool = False) -> None:
        self.city: str | None = None
        self.history: list[str] = []
        self.banned: set[str] = set()
        self.raw_log = raw_log
        self.leak = leak
        self.calls: list[str] = []
        self.backups: dict[str, dict] = {}
        self.workers: list[str] = []

    # state for paraphrase isolation
    def state(self) -> dict:
        return copy.deepcopy({"city": self.city, "history": self.history, "banned": self.banned})

    def load(self, st: dict) -> None:
        self.city, self.history, self.banned = st["city"], st["history"], st["banned"]

    def _learn(self, text: str) -> None:
        m = re.search(r"живу в (\w+)", text)
        if m:
            if self.city:
                self.history.append(self.city)
            self.city = m.group(1)

    def _log(self, text: str) -> None:
        if self.raw_log:
            with self.raw_log.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")

    def set_chat_mode(self, chat_id, mode, *, turn, store=None):
        self.calls.append(f"mode:{chat_id}:{mode.value}")
        (self.banned.add if mode.value == "banned" else self.banned.discard)(chat_id)
        return SimpleNamespace(chat_id=chat_id, mode=mode, policy_version=1, forget_id=None)

    def sweep(self, *, rows=None, store=None):
        for r in rows or []:
            if r.get("chat_id") in self.banned and not self.leak:
                continue
            self._log(r.get("text", ""))
            if r.get("role") == "owner":
                self._learn(r["text"])
        return len(rows or [])

    def record_turn_source(self, *a, **k):
        return None

    def ingest_turn(self, *, rows=None, **k):
        for r in rows or []:
            if r.get("role") == "tool" and not self.leak:
                self._log(
                    "[tool read stored in turn conversation]"
                )  # allowed place in real life; here: no token
                continue
            self._log(r.get("text", ""))
            if r.get("role") == "owner":
                self._learn(r["text"])
        return len(rows or [])

    def fix(self, target, correction, *, turn, because_quote, store=None):
        self._learn(correction)
        return WriteResult(outcome="committed")

    def after_turn_check(self, turn, reply_text, store=None):
        return SafetyNet(fired=False)

    def actions_at_startup(self, *, store=None):
        self.calls.append("startup")
        return 0

    def recall(self, turn, *, store=None):
        items = (
            [Item(id="m1", kind="fact", version=1, text=f"Живёт в {self.city}")]
            if self.city
            else []
        )
        return Basket(
            block="\n".join(i.text for i in items) or None,
            items=items,
            tokens=10,
            timings_ms={"total": 3.0},
        )

    def core(self, *, thread_id, session_id=None, store=None):
        from koob.memory.api import CoreView

        return CoreView(text="Профиль: Мира", tokens=5)

    def update_block(self, *, thread_id, session_id=None, store=None):
        from koob.memory.api import UpdateBlock

        return UpdateBlock(block=None)

    def search(self, query, *, turn, **k):
        return (
            [Item(id="m1", kind="fact", version=1, text=f"Живёт в {self.city}")]
            if self.city
            else []
        )

    def offers(self, turn, *, store=None):
        return []

    def run_nightly(self, *, store=None):
        self.workers.append("night")
        return {}

    def run_purges(self, *, store=None):
        return []

    def backup(self, *, store=None):
        key = f"set{len(self.backups)}"
        self.backups[key] = self.state()
        return key

    def restore(self, backup_set, *, store=None):
        self.load(self.backups[backup_set])
        return 99


class FakeSUT(ApiSUT):
    def __init__(self, api: FakeApi, stores: StoreMap | None = None) -> None:
        super().__init__(api, stores=stores)
        self.fake = api

    def snapshot(self):
        return self.fake.state()

    def rollback(self, token) -> None:
        self.fake.load(token)

    def run_worker(self, job, phase="run", token=None):
        self.fake.workers.append(f"{job}:{phase}")


def stores_in(tmp: Path) -> StoreMap:
    return StoreMap([StoreSpec("memory_db", [str(tmp / "*")])])


def scenario_dict(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "knows-d90",
        "group": "knows",
        "subgroup": "as_of",
        "set": "dev",
        "title": "Synthetic",
        "persona": "mira",
        "clock": {"start": "2026-01-10T09:00", "tz": "Europe/Moscow"},
        "timeline": [
            {"id": "e1", "ev": "user_msg", "text": "Я живу в Твери."},
            {"id": "e2", "ev": "correction", "at": "+2d", "text": "Нет, я живу в Твери."},
        ],
        "probes": [
            {
                "id": "p1",
                "at": "2026-02-01T10:00",
                "level": "memory",
                "ask": ["Где я живу?", "В каком я городе?"],
                "expect": {
                    "behaviour": "answer",
                    "must_include": [["Твер"]],
                    "must_not_include": ["Твер"],
                },
            }
        ],
    }
    base.update(over)
    return base


PERSONAS = {
    "mira": {
        "contacts": {"gleb": {"en": "Gleb", "ru": "Глеб", "ids": {"tg_user_id": 7100040}}},
        "chats": {},
    }
}
