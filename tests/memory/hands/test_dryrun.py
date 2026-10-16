"""Acceptance stubs for the memory spec §16.3.

Each test is `xfail(strict=True)` until its body is written and the marker is removed.
Spec marker: `[T hands/test_dryrun::<case>]`."""

from __future__ import annotations

from koob.memory import api

from .._contract import not_built, pending


@not_built("hands")
def test_old_address_planted_in_history_denied(turn):
    """§16.3: a corrected address planted in history is denied at the hand, the current value offered."""
    api.check_action("mail_send", {}, turn=turn)
    pending("hands")
