"""Tool-routed probes (README: probes, `route`).

Some questions are answered by the memory tools, not by the basket alone: «что ты помнишь про
X» by `memory_what`, «откуда это» by `memory_where_from`, «кто это» by `memory_resolve`
(assistant/tools.py `LEDGER_MEMORY_SPECS`). For such a probe the view-level check also reads what
the tool returns for the probe's question, the same way for every system under test that has
the tool. A system with no equivalent gets `None` (reported n/a), never a free pass: its
answer is then judged on its view alone, as before.

Without a model the tool calls are emulated deterministically: `memory_what` takes the
question itself as `about` (the engine strips the lead words and finds the terms);
`memory_where_from` takes the items the thread was shown before this turn (what «это» points
at) and the top items `memory_what` returned; `memory_resolve` takes the question.
"""

from __future__ import annotations

import sqlite3
from typing import Any

TOOLS = ("memory_what", "memory_where_from", "memory_resolve")
ROUTES = ("view", "tool", *TOOLS)
# Groups whose probes are answered by the memory tools when a probe names no route; applies to
# sets nobody here can read (the hidden set) as well as the dev set.
GROUP_ROUTE = {"inspect": "tool", "scout": "tool"}
WHERE_FROM_ITEMS = 3  # items per source (last shown, memory_what) a brain would ask about


def probe_route(probe: Any, group: str | None) -> str:
    return getattr(probe, "route", None) or GROUP_ROUTE.get(group or "", "view")


def route_tools(route: str) -> tuple[str, ...]:
    """The tools a route reads: `tool` = all three, a tool name = that one, `view` = none."""
    if route == "tool":
        return TOOLS
    return (route,) if route in TOOLS else ()


def _shown_before(conn: sqlite3.Connection, thread_id: str, turn_id: str) -> list[str]:
    """Items served to the thread's latest earlier turn: what "this" refers to."""
    rows = conn.execute(
        "SELECT item_id, turn_id FROM served WHERE thread_id = ? AND (turn_id IS NULL OR "
        "turn_id != ?) AND via IN ('core', 'basket', 'tool') ORDER BY id DESC",
        (thread_id, turn_id),
    ).fetchall()
    if not rows:
        return []
    last = rows[0][1]
    out: list[str] = []
    for item, t in rows:
        if t != last:
            break
        if item not in out:
            out.append(item)
    return out[:WHERE_FROM_ITEMS]


def api_tool_text(conn: sqlite3.Connection, wording: str, turn: Any, tools: tuple[str, ...]) -> str:
    """What the brain would read back from the engine's memory tools for this question."""
    from .. import api, commands
    from .sut import NotBuilt

    parts: list[str] = []
    what_ids: list[str] = []
    shown = (
        _shown_before(conn, turn.thread_id, turn.turn_id) if "memory_where_from" in tools else []
    )
    if "memory_what" in tools or "memory_where_from" in tools:
        try:
            w = commands.what_view(conn, wording, turn)
        except (NotBuilt, sqlite3.OperationalError):
            w = None
        if w is not None:
            what_ids = commands.what_ids(w)
            if "memory_what" in tools:
                parts += commands.render_what(w)
    if "memory_where_from" in tools:
        items = list(dict.fromkeys([*shown, *what_ids[:WHERE_FROM_ITEMS]]))
        for item in items:
            try:
                parts += api.where_from(item, turn=turn, store=conn).lines
            except (NotBuilt, sqlite3.OperationalError, ValueError, KeyError):
                continue
    if "memory_resolve" in tools:
        try:
            cands = api.resolve(wording, "any", turn=turn, store=conn)
        except (NotBuilt, sqlite3.OperationalError):
            cands = []
        for c in cands:
            ids = ", ".join(str(s.get("value", "")) for s in c.stable_ids if isinstance(s, dict))
            parts.append(f"{c.display_name}" + (f" ({ids})" if ids else ""))
    return "\n".join(p for p in parts if p)
