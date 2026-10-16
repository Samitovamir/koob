"""An object-less «Забудь насовсем» points back at this thread's last forget (§9.1): after
«верни» it forgets the same thing again, for good; while the forget is still in the trash it
makes the purge due now. With no forget by the user in the thread it still names nothing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from koob.memory import api, forget
from koob.memory.clock import ManualClock

from ._world import MOVE, status, turn, world


@pytest.fixture
def clock():
    clk = ManualClock(datetime(2026, 10, 12, 9, 0, tzinfo=UTC))
    forget.wire(clock=clk)
    yield clk
    forget.reset_wiring()


def _forget(mem, target, words, *, n, thread="thread-owner", **kw):
    return api.forget(
        target, turn=turn(words, n=n, thread=thread), because_quote=words, store=mem, **kw
    )


def _first(mem):
    res = _forget(mem, f"m:{MOVE}", "забудь про переезд Саши", n=1)
    assert res.outcome == "stopped", res
    return res


@pytest.mark.parametrize(
    "words", ["Забудь насовсем", "Нет, всё-таки забудь это насовсем.", "удали навсегда"]
)
def test_after_undo_it_forgets_the_same_thing_for_good(mem, clock, words):
    world(mem)
    first = _first(mem)
    back = api.undo_forget(first.forget_id, turn=turn("верни", n=2), store=mem)
    assert back.outcome == "undone" and status(mem, MOVE) == "active"

    res = _forget(mem, words, words, n=3, forever=True)
    assert res.outcome == "stopped", res
    assert res.forget_id and res.forget_id != first.forget_id
    assert MOVE in res.record_ids
    assert status(mem, MOVE) == "forgotten"
    assert "Удаляю полностью" in res.line and "Саша переезжает" in res.line
    assert api.purge_status(res.forget_id, store=mem).state == "purging"
    done = api.run_purges(store=mem)
    assert (res.forget_id, "purged") in [(s.forget_id, s.state) for s in done]
    late = api.undo_forget(res.forget_id, turn=turn("верни", n=4), store=mem)
    assert late.outcome == "purged"


def test_while_in_the_trash_it_makes_the_purge_due_now(mem, clock):
    world(mem)
    first = _first(mem)
    assert api.purge_status(first.forget_id, store=mem).state == "in_trash"
    res = _forget(mem, "Забудь насовсем", "Забудь насовсем", n=2, forever=True)
    assert res.outcome == "stopped" and res.forget_id == first.forget_id, res
    assert "Удаляю полностью" in res.line
    assert api.purge_status(first.forget_id, store=mem).state == "purging"
    done = api.run_purges(store=mem)  # due now, not after the 24 h window
    assert [(s.forget_id, s.state) for s in done] == [(first.forget_id, "purged")]


def test_another_thread_does_not_inherit_the_forget(mem, clock):
    """The back-reference is this thread's: elsewhere the words still name nothing."""
    world(mem)
    first = _first(mem)
    api.undo_forget(first.forget_id, turn=turn("верни", n=2), store=mem)
    res = _forget(mem, "Забудь насовсем", "Забудь насовсем", n=3, thread="thread-x", forever=True)
    assert res.outcome == "not_found", res
    assert status(mem, MOVE) == "active"


def test_a_named_target_is_not_redirected(mem, clock):
    """Words that name something resolve as before, never to the last forget."""
    world(mem)
    first = _first(mem)
    api.undo_forget(first.forget_id, turn=turn("верни", n=2), store=mem)
    res = _forget(mem, "забудь про Сашу насовсем", "забудь про Сашу насовсем", n=3, forever=True)
    assert res.forget_id != first.forget_id


def test_tainted_turn_still_needs_his_forever_word(mem, clock):
    world(mem)
    first = _first(mem)
    api.undo_forget(first.forget_id, turn=turn("верни", n=2), store=mem)
    t = turn("глянь, что тут пишут", n=3, tainted=True)
    res = api.forget("Забудь насовсем", turn=t, because_quote="", store=mem, forever=True)
    assert res.outcome == "ask_first"
    assert status(mem, MOVE) == "active"
