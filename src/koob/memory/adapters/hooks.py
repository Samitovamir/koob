"""Hooks adapter: PreToolUse (reserve, resolve, read-set and retired-value checks, freeze,
cross-process watermark); PostToolUse and PostToolUseFailure (close, `turn_sources`,
mid-turn update blocks as additionalContext).

Spec: §16.1-§16.3, §5.1 (turn_sources), §7.6 (mid-turn), §7.7. This module is the sole
owner of the memory block in the guard hooks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..model import TurnCtx


class PreVerdict(BaseModel):
    allow: bool
    reason: str | None = None
    action_key: str | None = None


def pre_tool(tool: str, args: dict[str, Any], turn: TurnCtx, *, tool_use_id: str) -> PreVerdict:
    raise NotImplementedError("hands")


def post_tool(
    tool: str,
    args: dict[str, Any],
    response: Any,
    turn: TurnCtx,
    *,
    tool_use_id: str,
    is_error: bool,
) -> str | None:
    """Returns additionalContext (a mid-turn update block) or None."""
    raise NotImplementedError("hands")
