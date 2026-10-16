"""Turn adapter. **before** wraps `_with_context(_with_recall(...))` (`runtime.py:1959`):
core, update block, basket, pre-send. **after** runs at the end of `_finish` and in
`_finish_bare`: ingestion, safety net, re-match of pending forgets.

Spec: §16.4, §7.6, §10.1-§10.2, §6.1, §9.1c. Seams: the memory seams note A.4.

The host (`HOST`) is the process's memory side, started once from `Runtime.ensure_started`
(`start`): `api.wire` then `api.startup` on its own worker thread, which then runs
`api.run_jobs` on a timer and the after-turn work from a queue. The turn path reads through
a second connection: the basket installs a progress-handler deadline on its connection, so
the turn path and the workers never share one. `memory.engine: legacy` with `memory.shadow:
off` starts nothing and the runtime's path stays byte for byte unchanged.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..model import ReadRef, TurnCtx, TurnOrigin

log = logging.getLogger(__name__)

JOBS_EVERY_S = 30.0  # the scheduler's tick (§6.3); the queue wakes it at once
READY_WAIT_S = 0.15  # the turn path waits this long for startup, then goes without memory
SHADOW_PREFIX = "shadow:"  # shadow computes under this thread id and removes its rows after


class Injection(BaseModel):
    """What the runtime prepends to the user turn (never into the cached prefix)."""

    core: str | None = None  # only on a process's first question
    update: str | None = None  # <memory-update>
    basket: str | None = None  # <memory-recall id=...>
    # a forget or restore this session held (another process's, §7.7): restart before the call
    restart_session: bool = False
    # §7.6 overflow: the full update rides on this question; the next process starts from a seed
    restart_next: bool = False
    read_set: list[ReadRef] = Field(default_factory=list)
    basket_items: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)

    def prepend(self, prompt: str) -> str:
        """Update block, then the basket, over the question (§7.6, §11 findable markers)."""
        parts = [p for p in (self.update, self.basket) if p]
        return "\n\n".join([*parts, prompt]) if parts else prompt


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def compute(
    conn: sqlite3.Connection, turn: TurnCtx, *, first_question: bool, resumed: bool = True
) -> Injection:
    """The injection for one model call (pure over `conn`; served rows are written).

    First question of a process: the full core when this session holds none of it, else
    `core_diff` (changed lines, or a restart). Every call: the thread's pending updates.
    Owner turns: the basket, revalidated right before the model call (§7.4)."""
    from .. import api
    from ..recall import basket as _basket
    from ..recall import core_view

    t_all = time.perf_counter()
    inj = Injection()
    updates: list[str] = []
    if first_question:
        t = time.perf_counter()
        held = (
            core_view.served_core(conn, thread_id=turn.thread_id, session_id=turn.session_id)
            if resumed and turn.session_id
            else {}
        )
        if held:
            diff = api.core_diff(thread_id=turn.thread_id, session_id=turn.session_id, store=conn)
            if diff.restart:
                inj.restart_next = True
            if diff.block:
                updates.append(diff.block)
        else:
            cv = api.core(thread_id=turn.thread_id, session_id=turn.session_id, store=conn)
            refs = [r for rs in cv.sections.values() for r in rs]
            pre = api.pre_send_check(thread_id=turn.thread_id, read_set=refs, store=conn)
            if pre.stale:  # §7.4: a stale core section is rebuilt once, else omitted
                cv = api.core(thread_id=turn.thread_id, session_id=turn.session_id, store=conn)
                refs = [r for rs in cv.sections.values() for r in rs]
                pre = api.pre_send_check(thread_id=turn.thread_id, read_set=refs, store=conn)
            if not pre.stale and cv.text:
                inj.core = cv.text
                inj.read_set += refs
        inj.timings_ms["core"] = _ms(t)
    t = time.perf_counter()
    upd = api.update_block(thread_id=turn.thread_id, session_id=turn.session_id, store=conn)
    if upd.block:
        updates.append(upd.block)
    inj.restart_next = inj.restart_next or upd.restart
    inj.update = "\n".join(updates) or None
    inj.timings_ms["update"] = _ms(t)
    if turn.origin in (TurnOrigin.OWNER, TurnOrigin.TASK):  # §10.2: task turns recall too
        t = time.perf_counter()
        b = api.recall(turn, store=conn)
        b = _basket.revalidate(conn, b, turn_id=turn.turn_id)  # right before the model call
        inj.basket = b.block
        inj.basket_items = len(b.items) if b.block else 0
        inj.read_set += b.read_set
        inj.timings_ms["basket"] = _ms(t)
    inj.timings_ms["total"] = _ms(t_all)
    return inj


class Host:
    """The process's memory side (one per server process, `HOST`)."""

    def __init__(self) -> None:
        self.engine = "legacy"
        self.shadow = "off"
        self.db_path: Path | None = None
        self.every_s = JOBS_EVERY_S
        self.restart: Callable[[list[str], list[str]], list[str]] | None = None
        self.restart_all: Callable[[], list[str]] | None = None
        self.shadow_log: deque[dict[str, Any]] = deque(maxlen=200)
        self.mark = (0, 0)  # (memory_version, last event id) this host has seen (§7.7)
        self._tconn: sqlite3.Connection | None = None
        self._tlock = threading.Lock()
        self._ready = threading.Event()
        self._failed = False
        self._q: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # thread id -> when its fresh session (no SDK id yet) took its first core (store.now)
        self._fresh_at: dict[str, str] = {}

    @property
    def active(self) -> bool:
        return self.engine == "ledger" or self.shadow == "on"

    @property
    def injects(self) -> bool:
        """Ledger on and shadow off: the core, updates and basket reach the model."""
        return self.engine == "ledger" and self.shadow != "on"

    # ------------------------------------------------------------ lifecycle

    def start(
        self,
        *,
        engine: str,
        shadow: str,
        conversation: Any = None,
        clock: Any = None,
        stop_hooks: Iterable[Any] = (),
        purge_steps: dict[str, Any] | None = None,
        db_path: Path | None = None,
        every_s: float = JOBS_EVERY_S,
        restart_all: Callable[[], list[str]] | None = None,
    ) -> bool:
        """`api.wire` now, `api.startup` and the job timer on the worker thread. Returns
        False (and does nothing) for legacy + shadow off. Idempotent per process."""
        self.engine, self.shadow = engine, "on" if shadow in ("on", True) else "off"
        if not self.active or self._thread is not None:
            return self.active
        from .. import api
        from .. import store as _store
        from ..recall import (
            core_view,  # noqa: F401 - its rebuild hooks register on import
            vectors,
        )

        self.db_path = Path(db_path) if db_path else _store.MemoryPaths.default().db
        self.every_s = every_s
        self.restart_all = restart_all
        api.wire(
            conversation=conversation,
            clock=clock,
            stop_hooks=[vectors.stop_hook, *stop_hooks],
            purge_steps=purge_steps,
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cb-memory-jobs", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._q.put(("stop", {}))
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None
        if self._tconn is not None:
            self._tconn.close()
            self._tconn = None
        self._ready.clear()
        self._failed = False

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout) and not self._failed

    def _run(self) -> None:
        from .. import api
        from .. import store as _store

        assert self.db_path is not None
        try:
            conn = _store.open(self.db_path)
            report = api.startup(store=conn)
            self._tconn = _store.open(self.db_path)
            ch = api.changes_since(memory_version=0, last_event_id=0, store=self._tconn)
            self.mark = (ch.memory_version, ch.last_event_id)
            log.info("memory: started (%s)", report.model_dump_json())
        except Exception:  # noqa: BLE001 - memory never stops the Cube; the turn goes without it
            log.exception("memory: startup failed; turns go without memory")
            self._failed = True
            self._ready.set()
            return
        self._ready.set()
        due = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    kind, body = self._q.get(timeout=max(0.0, due - time.monotonic()))
                except queue.Empty:
                    kind, body = "", {}
                if kind == "stop":
                    break
                if kind:
                    self._work(conn, kind, body)
                if time.monotonic() >= due:
                    self._tick(conn)
                    due = time.monotonic() + self.every_s
        finally:
            conn.close()

    def _tick(self, conn: sqlite3.Connection) -> None:
        from .. import api

        try:
            api.run_jobs(store=conn)
        except Exception:  # noqa: BLE001 - one bad tick never stops the timer
            log.exception("memory: job tick failed")
        try:
            api.sweep(store=conn)  # §6.2 rows with no turn (catch-up archiving)
        except NotImplementedError:
            pass
        except Exception:  # noqa: BLE001
            log.debug("memory: sweep failed", exc_info=True)

    def _work(self, conn: sqlite3.Connection, kind: str, body: dict[str, Any]) -> None:
        from .. import api, forget

        try:
            if kind == "after":
                try:
                    forget.rematch_turn(conn, body["turn_id"])  # §9.1c the answer that streamed
                finally:
                    api.ingest_turn(store=conn, **body)  # §6.2 arrival, after the re-match
            elif kind == "sync":
                body["ev"].set()
            elif kind == "session":
                self._backfill(conn, **body)
            elif kind == "shadow":
                self._shadow(conn, body["turn"], body["first"], body["t0"])
        except Exception:  # noqa: BLE001 - after-turn work never reaches the turn
            log.exception("memory: %s work failed", kind)

    # ------------------------------------------------------------ turn path

    def before(self, turn: TurnCtx, *, first_question: bool, resumed: bool = True) -> Injection:
        """The injection for this call; empty when off, in shadow (computed later, off the
        critical path) or while startup has not finished."""
        if not self.active:
            return Injection()
        if not self.injects:
            self._q.put(
                ("shadow", {"turn": turn, "first": first_question, "t0": time.perf_counter()})
            )
            return Injection()
        if not self.wait_ready(READY_WAIT_S) or self._tconn is None:
            return Injection()
        if turn.session_id is None and turn.thread_id not in self._fresh_at:
            from .. import store as _store

            self._fresh_at[turn.thread_id] = _store.now()
        with self._tlock:
            restart = self._watch(turn.thread_id)
            out = compute(self._tconn, turn, first_question=first_question, resumed=resumed)
        out.restart_session = restart
        return out

    def backfill_session(self, thread_id: str, turn_id: str, session_id: str) -> None:
        """§7.6 a fresh session's served rows were written before the SDK named it (its
        core, basket, update and first tool reads): they take its id on the worker - one
        UPDATE, never on the turn path - so `served_core` and a forget find this session."""
        if not self.injects or not session_id:
            return
        since = self._fresh_at.pop(str(thread_id), None)
        self._q.put(
            (
                "session",
                {"thread_id": str(thread_id), "turn_id": str(turn_id),
                 "session_id": str(session_id), "since": since},
            )
        )  # fmt: skip

    def _backfill(
        self, conn: sqlite3.Connection, *, thread_id: str, turn_id: str, session_id: str,
        since: str | None,
    ) -> None:  # fmt: skip
        from .. import store as _store

        with _store.write_tx(conn, priority="background"):
            # this turn's rows, and the turn-less core rows since the session took its core
            conn.execute(
                "UPDATE served SET session_id = ? WHERE thread_id = ? AND session_id IS NULL"
                " AND via <> 'session' AND (turn_id = ? OR (turn_id IS NULL AND created_at >= ?))",
                (session_id, thread_id, turn_id, since if since is not None else "\uffff"),
            )

    def _watch(self, thread_id: str) -> bool:
        """§7.7 the cross-process watermark (< 1 ms): forget or restore events this host has
        not seen restart the sessions that served their items (a restore: every session).
        True when this thread is one of them."""
        from .. import api

        assert self._tconn is not None
        mv, last = self.mark
        ch = api.changes_since(memory_version=mv, last_event_id=last, store=self._tconn)
        self.mark = (ch.memory_version, ch.last_event_id)
        if not ch.forget_events:
            return False
        marks = ", ".join("?" for _ in ch.forget_events)
        rows = self._tconn.execute(
            f"SELECT kind, item_ids FROM events WHERE id IN ({marks})",  # noqa: S608
            list(ch.forget_events),
        ).fetchall()
        if (
            any(r[0] == "restore" for r in rows)
            or len(rows) < len(ch.forget_events)
            or any(r[1] in (None, "", "[]") for r in rows)  # a forget naming no items: all
        ):
            if self.restart_all is not None:
                self.restart_all()
            return True
        import json

        ids = sorted({i for r in rows for i in json.loads(r[1] or "[]")})
        threads: set[str] = set()
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            q = ", ".join("?" for _ in chunk)
            threads |= {
                r[0]
                for r in self._tconn.execute(
                    f"SELECT DISTINCT thread_id FROM served WHERE item_id IN ({q})",  # noqa: S608
                    chunk,
                )
            }
        if threads:
            from .. import forget

            conv = forget._W.conversation  # noqa: SLF001 - the wired store's restart path
            if conv is not None:
                conv.restart_sessions(sorted(threads), exclude_turns=[])
        return thread_id in threads

    def mid_turn(self, thread_id: str, session_id: str | None) -> str | None:
        """§7.6 mid-turn: a correction that landed while this session runs, for its next
        PostToolUse `additionalContext`. One indexed count when nothing is pending."""
        if not self.injects or not self._ready.is_set() or self._tconn is None:
            return None
        from .. import api

        # on the event loop: never wait for another session's before() - a busy lock skips
        # this step, the pending rows stay for the next step or the next call
        if not self._tlock.acquire(blocking=False):
            return None
        try:
            n = self._tconn.execute(
                "SELECT count(*) FROM pending_updates WHERE thread_id = ?", (thread_id,)
            ).fetchone()[0]
            if not n:
                return None
            return api.update_block(
                thread_id=thread_id, session_id=session_id, store=self._tconn
            ).block
        finally:
            self._tlock.release()

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until the work queued before this call is done (tests, shutdown)."""
        ev = threading.Event()
        self._q.put(("sync", {"ev": ev}))
        return ev.wait(timeout)

    def after(self, **ids: Any) -> None:
        if self.active:
            self._q.put(("after", ids))

    # ------------------------------------------------------------ shadow (§20)

    def _shadow(self, conn: sqlite3.Connection, turn: TurnCtx, first: bool, t0: float) -> None:
        """What ledger mode would inject: content-free counts and timings, logged locally.
        Runs on the worker under a shadow thread id whose served rows are removed after."""
        from .. import store as _store

        tid = SHADOW_PREFIX + turn.thread_id
        shadow_turn = turn.model_copy(update={"thread_id": tid})
        inj = compute(conn, shadow_turn, first_question=first, resumed=False)
        with _store.write_tx(conn):
            conn.execute("DELETE FROM served WHERE thread_id = ?", (tid,))
            conn.execute("DELETE FROM pending_updates WHERE thread_id = ?", (tid,))
        from ..recall import core_view

        row = {
            "turn_id": turn.turn_id,
            "origin": str(turn.origin),
            "first": first,
            "core_tokens": core_view.tokens(inj.core) if inj.core else 0,
            "update": int(bool(inj.update)),
            "basket_items": inj.basket_items,
            "read_set": len(inj.read_set),
            "queued_ms": round((time.perf_counter() - t0) * 1000 - inj.timings_ms["total"], 2),
            **{f"{k}_ms": v for k, v in inj.timings_ms.items()},
        }
        self.shadow_log.append(row)
        log.info("memory shadow: %s", row)


HOST = Host()


def before(turn: TurnCtx, *, first_question: bool) -> Injection:
    return HOST.before(turn, first_question=first_question)


def after(
    *, turn_id: str, thread_id: str, origin: str, status: str, pass_id: str | None = None
) -> None:
    """Ids only; enqueue, no model and no work on the event loop (G9)."""
    HOST.after(turn_id=turn_id, thread_id=thread_id, origin=origin, status=status, pass_id=pass_id)
