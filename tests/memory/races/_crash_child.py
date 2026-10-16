"""Crash child for the race suite: runs one memory call in its own process and, when told,
dies with `os._exit(77)` after N SQLite VM steps (a progress handler), i.e. at an arbitrary
point inside or between the call's transactions. No knowledge of the call's internals.

Usage: python _crash_child.py '<json spec>'
  spec = {"db", "home", "kill_at": int | null, "target": "api" | "commit",
          "call": name, "args": [...], "kwargs": {...}, "proposal": {...}}
Prints `RESULT {"steps": n, "result": ...}` when the call returns.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)


def main() -> None:
    spec = json.loads(sys.argv[1])
    from koob.memory import api, store
    from koob.memory.model import ChatMode, ForgetScope, Proposal, TurnCtx

    conn = store.open(Path(spec["db"]), home=Path(spec["home"]))
    kill_at = spec.get("kill_at")
    steps = 0

    def tick() -> int:
        nonlocal steps
        steps += 1
        if kill_at is not None and steps >= kill_at:
            os._exit(77)
        return 0

    kwargs = dict(spec.get("kwargs") or {})
    if "turn" in kwargs:
        kwargs["turn"] = TurnCtx.model_validate(kwargs["turn"])
    if "scope" in kwargs:
        kwargs["scope"] = ForgetScope(kwargs["scope"])
    if "mode" in kwargs:
        kwargs["mode"] = ChatMode(kwargs["mode"])
    args = list(spec.get("args") or [])
    if spec.get("mode_arg"):
        args[1] = ChatMode(args[1])

    conn.set_progress_handler(tick, 1)
    if spec["target"] == "commit":

        def set_value(ctx: Any, op: Any) -> list[str]:
            ctx.conn.execute(
                "UPDATE facts SET value_text = ?, value_norm = ? WHERE id = ?",
                (op.payload["value"], op.payload["value"], op.target),
            )
            ctx.conn.execute(
                "UPDATE records SET version = ? WHERE id = ?", (ctx.version, op.target)
            )
            return [op.target]

        result = store.commit(conn, Proposal.model_validate(spec["proposal"]), set_value)
    else:
        result = getattr(api, spec["call"])(*args, store=conn, **kwargs)
    conn.set_progress_handler(None, 1)
    print("RESULT " + json.dumps({"steps": steps, "result": _jsonable(result)}), flush=True)
    conn.close()
    os._exit(0)


if __name__ == "__main__":
    main()
