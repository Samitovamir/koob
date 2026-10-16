"""Acceptance stubs for the memory spec §16.2.

Each test is `xfail(strict=True)` until its body is written and the marker removed. Spec
marker: `[T hands/test_outbox::<case>]`."""

from __future__ import annotations

from koob.memory import api

from .._contract import not_built, pending


@not_built("hands")
def test_crash_after_send_no_duplicate():
    """§16.2: a crash after the send never yields a second send."""
    api.record_action(tool="tg_send", stable_target="tg:1", origin_key="o", read_set=[])
    pending("hands")


@not_built("hands")
def test_failure_closed():
    """§16.2: PostToolUseFailure closes the action as failed."""
    api.record_action(tool="tg_send", stable_target="tg:1", origin_key="o", read_set=[])
    pending("hands")


@not_built("hands")
def test_rerun_same_key():
    """§16.2: a re-run after a crash computes the same action_key."""
    api.record_action(tool="tg_send", stable_target="tg:1", origin_key="o", read_set=[])
    pending("hands")
