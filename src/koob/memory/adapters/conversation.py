"""The koob.db side of memory behind a protocol, so `memory/` never imports `koob/`
(spec §16.4): read a turn's rows, find turns by match keys, blank and restore rows, collect
session ids, restart sessions from a seed without excluded turns.

Spec: §6.2, §9.1b-c, §9.2, §9.5, §11. The protocol is the contract; the implementation is
wired by runtime; forget is its main caller.

`CareerConversation` implements the protocol over a koob.db connection with plain SQL
(no `koob.assistant` import). Restarting live SDK sessions needs the runtime, so it is
an injected callable. Two extras beyond the protocol, found by duck typing in forget.py:
`mask_turns` (§9.8: a turn whose only hit is X's name is masked in place) and `checkpoint`
(§9.1 stage 3: the verified `wal_checkpoint(TRUNCATE)` on koob.db).

Matching (§9.5) is shared with forget.py: a row matches when any key occurs in its text
after `match_norm` (NFKC, casefold, every non-word run collapsed to one space).
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any, Protocol

FORGOTTEN = "[забыто]"

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def match_norm(text: str) -> str:
    """NFKC, casefold, every run of non-word characters to one space (§9.5 matching)."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return " " + _NON_WORD.sub(" ", folded).strip() + " "


def compile_keys(keys: Iterable[str]) -> re.Pattern[str] | None:
    """One pattern for a set of match keys; a key matches at word starts (stems are
    prefixes: «переезж» finds «переезжает»), never inside another word."""
    parts = sorted({match_norm(k).strip() for k in keys if k and match_norm(k).strip()}, key=len)
    if not parts:
        return None
    alt = "|".join(re.escape(p) for p in reversed(parts))
    return re.compile(rf"(?<=\s)(?:{alt})")


def json_text(raw: str | None) -> str:
    """Every string leaf of a JSON value (tool rows escape Cyrillic as \\uXXXX)."""
    if not raw:
        return ""
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    out: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(value)
    return " ".join(out)


def text_matches(text: str, pattern: re.Pattern[str] | None) -> bool:
    return bool(pattern is not None and text and pattern.search(match_norm(text)))


class ConversationStore(Protocol):
    def turn_rows(self, turn_id: str) -> list[dict[str, Any]]:
        """`koob_messages` rows of one turn (text, role, ids; no tool_json content)."""
        ...

    def rows_without_turn(self, *, after_id: int, limit: int = 500) -> list[dict[str, Any]]: ...

    def find_turns(
        self, thread_ids: list[str], match_keys: list[str], *, after_turn: str | None
    ) -> list[str]:
        """§9.5: later turns in those threads whose rows match the keys."""
        ...

    def blank_turns(self, turn_ids: list[str], *, forget_id: str) -> list[dict[str, Any]]:
        """Blank to «[забыто]» with secure_delete=ON; returns the payload for the trash."""
        ...

    def restore_turns(self, payload: list[dict[str, Any]]) -> None:
        """Restore blanked rows from the trash payload («Вернуть»)."""
        ...

    def session_ids(self, thread_id: str) -> list[str]: ...

    def restart_sessions(self, thread_ids: list[str], *, exclude_turns: list[str]) -> list[str]:
        """Close after the current turn, restart from a seed without the turns, pre-warm."""
        ...


# ---------------------------------------------------------------- koob.db implementation

# Columns blanked per table, keyed by how the table finds a turn's rows (§9.2).
_MSG_COLS = ("text", "tool_json")
_TURN_COLS = ("context", "error")
_PERM_COLS = ("label", "detail", "reason", "note", "extra_json")

Restarter = Callable[[list[str], list[str]], list[str]]


def _turn_key(turn_id: str) -> int | str:
    try:
        return int(turn_id)
    except (TypeError, ValueError):
        return turn_id


def _blank_tool_json(raw: str | None) -> str | None:
    """Keep the tool's name so the island still draws the row; drop every content field."""
    if raw is None:
        return None
    try:
        name = json.loads(raw).get("name")
    except (ValueError, AttributeError):
        name = None
    return json.dumps({"name": name, "label": FORGOTTEN, "detail": "", "result": ""})


class CareerConversation:
    """`ConversationStore` over koob.db (`koob_messages`, `koob_turns`,
    `koob_permissions`, `koob_threads`) by SQL only. `restarter(thread_ids,
    exclude_turns)` is the runtime's restart path (runtime); without it no live
    session exists to restart (headless, tests)."""

    def __init__(self, conn: sqlite3.Connection, *, restarter: Restarter | None = None) -> None:
        self.conn = conn
        self.restarter = restarter
        # §9.1b: deletes and blanking overwrite freed space on this connection.
        conn.execute("PRAGMA secure_delete = ON")
        self._cols = {
            t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
            for t in ("koob_messages", "koob_turns", "koob_permissions", "koob_threads")
        }

    # ------------------------------------------------------------ reads

    def _dicts(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, tuple(params))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]

    def turn_rows(self, turn_id: str) -> list[dict[str, Any]]:
        rows = self._dicts(
            "SELECT id, thread_id, turn_id, role, text, status, created_at "
            "FROM koob_messages WHERE turn_id = ? ORDER BY id",
            (_turn_key(turn_id),),
        )
        for r in rows:
            r["thread_id"] = str(r["thread_id"])
            r["turn_id"] = str(r["turn_id"])
        return rows

    def rows_without_turn(self, *, after_id: int, limit: int = 500) -> list[dict[str, Any]]:
        return self._dicts(
            "SELECT id, thread_id, role, text, status, created_at FROM koob_messages "
            "WHERE turn_id IS NULL AND id > ? ORDER BY id LIMIT ?",
            (after_id, limit),
        )

    def _turn_texts(self, thread_ids: list[str], after_turn: str | None):
        params: list[Any] = [_turn_key(t) for t in thread_ids]
        where = f"thread_id IN ({','.join('?' * len(thread_ids))}) AND turn_id IS NOT NULL"
        if after_turn is not None:
            where += " AND turn_id > ?"
            params.append(_turn_key(after_turn))
        yield from self.conn.execute(
            f"SELECT turn_id, text, tool_json FROM koob_messages WHERE {where} ORDER BY id",
            params,
        )

    def find_turns(
        self, thread_ids: list[str], match_keys: list[str], *, after_turn: str | None
    ) -> list[str]:
        pattern = compile_keys(match_keys)
        if not thread_ids or pattern is None:
            return []
        hit: list[str] = []
        for turn_id, text, tool_json in self._turn_texts(thread_ids, after_turn):
            tid = str(turn_id)
            if tid in hit:
                continue
            if text_matches(text or "", pattern) or text_matches(json_text(tool_json), pattern):
                hit.append(tid)
        return sorted(set(hit), key=_turn_key_sort)

    def session_ids(self, thread_id: str) -> list[str]:
        row = self.conn.execute(
            "SELECT sdk_session_id FROM koob_threads WHERE id = ?", (_turn_key(thread_id),)
        ).fetchone()
        out: list[str] = []
        sid = row[0] if row else None
        if sid:
            out.append(sid)
            if sid.startswith("seed:"):
                out.append(sid.removeprefix("seed:"))
        return out

    # ------------------------------------------------------------ writes

    def _rows_of(self, table: str, cols: tuple[str, ...], where: str, key: Any):
        have = [c for c in cols if c in self._cols[table]]
        if not have:
            return [], []
        rows = self.conn.execute(
            f"SELECT id, {', '.join(have)} FROM {table} WHERE {where}", (key,)
        ).fetchall()
        return have, rows

    def _blank(
        self, turn_ids: list[str], rewrite: Callable[[str, str, Any], Any]
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for turn_id in turn_ids:
            key = _turn_key(turn_id)
            for table, cols, where in (
                ("koob_messages", _MSG_COLS, "turn_id = ?"),
                ("koob_turns", _TURN_COLS, "id = ?"),
                ("koob_permissions", _PERM_COLS, "turn_id = ?"),
            ):
                have, rows = self._rows_of(table, cols, where, key)
                for row in rows:
                    old = dict(zip(have, row[1:], strict=True))
                    new = {c: rewrite(table, c, v) for c, v in old.items()}
                    changed = {c: old[c] for c in have if new[c] != old[c]}
                    if not changed:
                        continue
                    sets = ", ".join(f"{c} = ?" for c in changed)
                    self.conn.execute(
                        f"UPDATE {table} SET {sets} WHERE id = ?",
                        [new[c] for c in changed] + [row[0]],
                    )
                    payload.append(
                        {"turn_id": str(turn_id), "table": table, "id": row[0], "cols": changed}
                    )
        self.conn.commit()
        return payload

    def blank_turns(self, turn_ids: list[str], *, forget_id: str) -> list[dict[str, Any]]:
        def rewrite(table: str, col: str, value: Any) -> Any:
            if value is None:
                return None
            if col == "tool_json":
                return _blank_tool_json(value)
            if table == "koob_messages" and col == "text":
                return FORGOTTEN
            if col == "label":
                return FORGOTTEN
            return None

        return self._blank(turn_ids, rewrite)

    def mask_turns(
        self, turn_ids: list[str], terms: list[str], *, forget_id: str
    ) -> list[dict[str, Any]]:
        """§9.8: replace each term (X's aliases and stable ids) with «[забыто]» in place."""
        pats = [re.compile(re.escape(t), re.IGNORECASE) for t in sorted(terms, key=len)[::-1] if t]

        def rewrite(table: str, col: str, value: Any) -> Any:
            if not isinstance(value, str):
                return value
            out = value
            for p in pats:
                out = p.sub(FORGOTTEN, out)
            return out

        return self._blank(turn_ids, rewrite)

    def restore_turns(self, payload: list[dict[str, Any]]) -> None:
        for item in payload:
            table = item["table"]
            if table not in self._cols:
                continue
            cols = [c for c in item["cols"] if c in self._cols[table]]
            if not cols:
                continue
            self.conn.execute(
                f"UPDATE {table} SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
                [item["cols"][c] for c in cols] + [item["id"]],
            )
        self.conn.commit()

    def restart_sessions(self, thread_ids: list[str], *, exclude_turns: list[str]) -> list[str]:
        if self.restarter is None or not thread_ids:
            return []
        return list(self.restarter(list(thread_ids), list(exclude_turns)))

    def checkpoint(self) -> bool:
        """Verified `wal_checkpoint(TRUNCATE)`: busy = 0 and log = checkpointed."""
        busy, log, done = self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return busy == 0 and log == done


def _turn_key_sort(turn_id: str) -> tuple[int, int | str]:
    key = _turn_key(turn_id)
    return (0, key) if isinstance(key, int) else (1, key)


class PerThreadConversation:
    """A `ConversationStore` whose koob.db connection belongs to the calling thread: the
    host calls forget from the event loop, the job worker and `to_thread` helpers, and a
    sqlite3 connection stays on its thread. `connect()` opens koob.db; `restarter` is the
    runtime's restart path (runtime)."""

    def __init__(
        self, connect: Callable[[], sqlite3.Connection], *, restarter: Restarter | None = None
    ) -> None:
        import threading

        self._connect = connect
        self.restarter = restarter
        self._local = threading.local()

    def _store(self) -> CareerConversation:
        s = getattr(self._local, "store", None)
        if s is None:
            s = self._local.store = CareerConversation(self._connect(), restarter=self.restarter)
        return s

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._store(), name)
