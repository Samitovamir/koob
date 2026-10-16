"""«забудь» / «Вернуть» through the facade (§9.1, §17, §10.4 `forget`): a handle or the user's
words resolve to record ids through the ledger, «всё про X» forgets the entity, several
plausible targets ask one distinguishing question, a tainted turn needs the user's own words
to name the target (§6.1), and the user-facing line is honest about stop-use vs purge."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from koob.memory import api, forget
from koob.memory.clock import ManualClock
from koob.memory.model import ForgetScope

from ._world import (
    DIMA_WORK,
    HEALTH,
    LIVES_DIMA,
    LIVES_OWN,
    MOVE,
    SASHA,
    SASHA_K,
    TIRED,
    second_sasha,
    serve,
    status,
    turn,
    world,
)


@pytest.fixture
def clock():
    clk = ManualClock(datetime(2026, 10, 12, 9, 0, tzinfo=UTC))
    forget.wire(clock=clk)
    yield clk
    forget.reset_wiring()


def _forget(mem, target, words, *, quote=None, n=1, **kw):
    return api.forget(
        target,
        turn=turn(words, n=n, **kw.pop("flags", {})),
        because_quote=quote or words,
        store=mem,
        **kw,
    )


def test_by_handle_stops_use_at_once_and_names_sessions(mem, clock):
    world(mem)
    serve(mem, MOVE, thread="thread-task", session="session-task", turn_id="turn-7")
    res = _forget(mem, f"m:{MOVE}", "забудь про переезд Саши", quote="переезд Саши", n=8)
    assert res.outcome == "stopped"
    assert res.forget_id
    assert MOVE in res.record_ids
    assert "session-task" in res.sessions_to_restart
    assert {"turn-0", "turn-7"} <= set(res.turn_ids)
    assert res.line.startswith("Перестал использовать: «Саша переезжает в Тверь в ноябре»")
    assert "Вернуть" in res.line and "24" in res.line
    assert status(mem, MOVE) == "forgotten"
    assert status(mem, LIVES_OWN) == "active"


def test_by_his_words_resolves_through_the_ledger(mem, clock):
    world(mem)
    res = _forget(mem, "переезд Саши", "забудь про переезд Саши")
    assert res.outcome == "stopped", res
    assert res.record_ids == [MOVE]
    assert status(mem, TIRED) == "active"


def test_everything_about_x_forgets_the_entity(mem, clock):
    world(mem)
    res = _forget(mem, "всё про Диму", "забудь всё про Диму")
    assert res.outcome == "stopped", res
    assert status(mem, DIMA_WORK) == "forgotten"
    assert status(mem, MOVE) == "active"
    # the explicit scope works with a bare name too
    res2 = _forget(mem, "Саша", "забудь всё про Сашу", n=2, scope=ForgetScope.ABOUT_ENTITY)
    # The scope holds a health fact (sensitive), so the purge is due at once (off the call).
    assert res2.outcome == "stopped" and res2.forget_id in forget.due(mem), res2
    assert {MOVE, TIRED, LIVES_OWN} <= set(res2.record_ids)


def test_two_namesakes_ask_one_distinguishing_question(mem, clock):
    world(mem)
    second_sasha(mem)
    res = _forget(mem, "всё про Сашу", "забудь всё про Сашу")
    assert res.outcome == "ask_first"
    assert res.record_ids == [] and res.forget_id is None
    assert res.line.count("?") == 1
    assert f"m:{SASHA}" in res.line and f"m:{SASHA_K}" in res.line
    assert "Озон" in res.line or "Ozon" in res.line
    assert status(mem, MOVE) == "active"


def test_two_plausible_records_ask_first(mem, clock):
    world(mem)
    res = _forget(mem, "где живёт Саша", "забудь, где живёт Саша")
    assert res.outcome == "ask_first", res
    assert f"m:{LIVES_OWN}" in res.line and f"m:{LIVES_DIMA}" in res.line
    assert status(mem, LIVES_OWN) == "active" and status(mem, LIVES_DIMA) == "active"


def test_unknown_target_is_not_found(mem, clock):
    world(mem)
    assert _forget(mem, "про Аркадия", "забудь про Аркадия").outcome == "not_found"
    res = _forget(mem, "m:nope", "забудь это", n=2)
    assert res.outcome == "not_found" and res.line


def test_tainted_turn_needs_his_own_words_to_name_the_target(mem, clock):
    world(mem)
    # a page asked to forget Dima's work; the user's words say nothing about it
    res = _forget(
        mem,
        f"m:{DIMA_WORK}",
        "посмотри эту страницу",
        quote="Дима работает",
        flags={"tainted": True},
    )
    assert res.outcome == "ask_first"
    assert status(mem, DIMA_WORK) == "active"
    # the user's own words name it: forgotten
    ok = _forget(
        mem,
        f"m:{DIMA_WORK}",
        "забудь, где работает Дима",
        quote="где работает Дима",
        n=2,
        flags={"tainted": True},
    )
    assert ok.outcome == "stopped", ok
    assert status(mem, DIMA_WORK) == "forgotten"


def test_forever_purges_now_and_says_deleted(mem, clock):
    world(mem)
    res = _forget(mem, f"m:{MOVE}", "забудь переезд Саши насовсем", quote="переезд Саши",
                  forever=True)  # fmt: skip
    assert res.outcome == "stopped", res
    assert "Удаляю полностью" in res.line  # the purge runs in the pool, not in the call
    assert api.purge_status(res.forget_id, store=mem).state == "purging"
    api.run_purges(store=mem)
    assert api.purge_status(res.forget_id, store=mem).state == "purged"


def test_sensitive_label_purges_at_once(mem, clock):
    world(mem)
    res = _forget(mem, f"m:{HEALTH}", "забудь про диабет Саши", quote="диабет Саши")
    assert res.outcome == "stopped" and res.forget_id in forget.due(mem), res


def test_trash_waits_24h_then_run_purges(mem, clock):
    world(mem)
    res = _forget(mem, f"m:{MOVE}", "забудь переезд Саши", quote="переезд Саши")
    assert res.outcome == "stopped"
    assert api.run_purges(store=mem) == []
    assert api.purge_status(res.forget_id, store=mem).state == "in_trash"
    clock.advance(25 * 3600)
    done = api.run_purges(store=mem)
    assert [(s.forget_id, s.state) for s in done] == [(res.forget_id, "purged")]


def test_undo_while_in_trash_and_not_after_purge(mem, clock):
    world(mem)
    res = _forget(mem, f"m:{MOVE}", "забудь переезд Саши", quote="переезд Саши")
    back = api.undo_forget(res.forget_id, turn=turn("верни", n=2), store=mem)
    assert back.outcome == "undone" and MOVE in back.record_ids
    assert back.line.startswith("Вернул")
    assert status(mem, MOVE) == "active"
    assert api.purge_status(res.forget_id, store=mem).state == "undone"

    gone = _forget(mem, f"m:{MOVE}", "забудь переезд Саши насовсем", quote="переезд Саши", n=3,
                   forever=True)  # fmt: skip
    api.run_purges(store=mem)
    late = api.undo_forget(gone.forget_id, turn=turn("верни", n=4), store=mem)
    assert late.outcome == "purged" and "не могу" in late.line.casefold()
    assert api.undo_forget("fg-nope", turn=turn("верни", n=5), store=mem).outcome == "not_found"


def test_purge_status_of_unknown_forget_raises(mem):
    with pytest.raises(LookupError):
        api.purge_status("fg-nope", store=mem)
