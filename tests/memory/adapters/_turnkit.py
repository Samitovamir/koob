"""Shared pieces for the turn-adapter tests (runtime): a memory.db, a koob.db
conversation store with a recording restarter, a started host, and the two race worlds."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from koob import db as career_db
from koob.memory import forget, store
from koob.memory.adapters import turn as turnmod
from koob.memory.adapters.conversation import PerThreadConversation
from koob.memory.model import TurnCtx, TurnOrigin
from koob.memory.recall import vectors

from ..races.test_fix_during_worker import FIX_WORDS
from ..races.test_fix_during_worker import _world as fix_world  # noqa: F401
from ..races.test_forget_during_answer import FORGET_WORDS, _forget  # noqa: F401
from ..races.test_forget_during_answer import _world as forget_world  # noqa: F401


def memory_db(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "memory" / "memory.db"
    path.parent.mkdir(parents=True)
    (tmp_path / "data" / "home").mkdir()
    store.open(path, home=tmp_path / "data" / "home").close()
    return path


class Restarts:
    """The runtime's restart path, recorded: thread ids per call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.all = 0

    def __call__(self, thread_ids: list[str], exclude_turns: list[str]) -> list[str]:
        self.calls.append(list(thread_ids))
        return [f"sdk-{t}" for t in thread_ids]

    def restart_all(self) -> list[str]:
        self.all += 1
        return []

    @property
    def threads(self) -> set[str]:
        return {t for c in self.calls for t in c}


def career(tmp_path: Path, restarts: Restarts) -> PerThreadConversation:
    """koob.db as the runtime wires it: one connection per calling thread."""
    path = tmp_path / "koob.db"
    career_db.init(path).close()
    return PerThreadConversation(lambda: career_db.connect(path), restarter=restarts)


def task_turn(thread: str, session: str | None, *, n: int = 9, origin=TurnOrigin.TASK) -> TurnCtx:
    return TurnCtx(turn_id=f"turn-{n}", thread_id=thread, session_id=session, origin=origin)


@contextmanager
def host(
    db_path: Path, *, conversation=None, restarts: Restarts | None = None, shadow: str = "off"
) -> Iterator[turnmod.Host]:
    h = turnmod.Host()
    h.start(
        engine="ledger",
        shadow=shadow,
        conversation=conversation,
        db_path=db_path,
        every_s=3600,
        restart_all=restarts.restart_all if restarts else None,
    )
    try:
        assert h.wait_ready(10)
        yield h
    finally:
        h.stop()
        forget.reset_wiring()
        vectors.reset()


__all__ = ["FIX_WORDS", "FORGET_WORDS"]
