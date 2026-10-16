"""Acceptance stubs for the memory spec §16.1.

Each test is `xfail(strict=True)` until its body is written and the marker removed.
Spec marker: `[T hands/test_resolve::<case>]`."""

from __future__ import annotations

from koob.memory.adapters import hooks

from .._contract import not_built, pending


@not_built("hands")
def test_two_sashas_ask(turn):
    """§16.1: two candidates above the floor: PreToolUse denies with 'ask which'."""
    hooks.pre_tool("tg_send", {}, turn, tool_use_id="t1")
    pending("hands")


@not_built("hands")
def test_stale_token_denied(turn):
    """§16.1: a token whose entity version moved is denied."""
    hooks.pre_tool("tg_send", {}, turn, tool_use_id="t1")
    pending("hands")


@not_built("hands")
def test_alias_lesson_applies(turn):
    """§16.1: an alias lesson («не путать двух Саш», "don't mix up the two Sashas") forces the
    question."""
    hooks.pre_tool("tg_send", {}, turn, tool_use_id="t1")
    pending("hands")
