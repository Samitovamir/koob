"""Acceptance stubs for the memory spec §15.1.

Each test is `xfail(strict=True)` until its body is written and the marker removed.
Spec marker: `[T capabilities/test_c1::<case>]`."""

from __future__ import annotations

from koob.memory import api

from .._contract import not_built, pending


@not_built("assoc")
def test_same_day_trip_gets_offer(turn):
    """§15.1: a goal event enqueues an assoc job at once; a same-day trip gets one offer with grounds."""
    api.offers(turn)
    pending("assoc")
