"""Regressions for small follow-up items: a tainted `fix` needs the user's own correction words
(rule 4, as forget does), and forget / undo / mode commits are stamped with their turn so
another thread's commit never backs this one."""

from __future__ import annotations

from koob.memory import api
from koob.memory.model import ChatMode, TurnCtx, TurnOrigin
from koob.memory.writers import hot

from ..races._kit import TS0, handle, status_of
from ._world import sasha_world, turn_of


def _turn(words: str, *, n: int, thread: str, tainted: bool = False) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{thread}-{n}",
        thread_id=thread,
        session_id=f"session-{thread}",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        tainted=tainted,
        message_id=f"msg-{thread}-{n}",
        clock=TS0,
    )


# ---------------------------------------------------------------- tainted fix (rule 4)


def test_tainted_fix_without_his_correction_words_asks_first(mem):
    """A paste is open; the user's words only ask about it and name the target. An injected
    `fix(...)` comes from the page, not the user: `ask_first`, the record stays as it was."""
    _sasha, bday = sasha_world(mem)
    t = turn_of("что тут пишут про день рождения Саши?", n=2, tainted=True)
    r = api.fix(
        handle(bday.id),
        "у Саши день рождения 1 мая",
        turn=t,
        because_quote="день рождения Саши",
        store=mem,
    )
    assert r.outcome == "ask_first"
    assert status_of(mem, bday.id) == "active"


def test_tainted_retract_without_his_words_asks_first(mem):
    """The same for «это неправда» (retract): none of the user's words say it is wrong."""
    _sasha, bday = sasha_world(mem)
    t = turn_of("глянь, что тут про день рождения Саши", n=2, tainted=True)
    r = api.fix(
        handle(bday.id), "", turn=t, because_quote="день рождения Саши", retract=True, store=mem
    )
    assert r.outcome == "ask_first"
    assert status_of(mem, bday.id) == "active"


def test_tainted_fix_with_his_correction_words_commits(mem):
    """Control: the user's own words correct it after a paste: the fix commits."""
    _sasha, bday = sasha_world(mem)
    words = "нет, у Саши день рождения 13 мая, исправь"
    q = "у Саши день рождения 13 мая"
    r = api.fix(
        handle(bday.id), q, turn=turn_of(words, n=3, tainted=True), because_quote=q, store=mem
    )
    assert r.outcome == "committed"


# ---------------------------------------------------------------- per-turn commit stamping


def test_forget_in_thread_a_does_not_mask_a_forget_cue_in_thread_b(mem):
    """A real forget in thread A backs A's «забыл», never B's «забудь» cue."""
    _sasha, bday = sasha_world(mem)
    a = _turn("забудь день рождения Саши", n=1, thread="A")
    r = api.forget(handle(bday.id), turn=a, because_quote="день рождения Саши", store=mem)
    assert r.outcome == "stopped"
    assert "forget" in hot.turn_commits(mem, a)
    assert not hot.claims_unbacked_change(mem, a, "Забыл.")
    b = _turn("забудь мой старый номер", n=1, thread="B")
    assert hot.turn_commits(mem, b) == set()
    assert hot.claims_unbacked_change(mem, b, "Забыл.")
    fired, _ents, jid = hot.after_turn_check(mem, b, "Забыл.")
    assert fired and jid


def test_undo_and_mode_are_stamped_with_their_turn(mem):
    """«Вернуть» and a chat-mode change mark their own turn and thread only."""
    _sasha, bday = sasha_world(mem)
    a = _turn("забудь день рождения Саши", n=1, thread="A")
    fid = api.forget(handle(bday.id), turn=a, because_quote="день рождения Саши", store=mem)
    back = _turn("верни день рождения Саши", n=2, thread="A")
    assert api.undo_forget(fid.forget_id, turn=back, store=mem).outcome == "undone"
    assert "undo" in hot.turn_commits(mem, back)
    m = _turn("не запоминай ничего из чата работа", n=3, thread="A")
    api.set_chat_mode("tg:work", ChatMode.NO_DIG, turn=m, store=mem)
    assert "mode" in hot.turn_commits(mem, m)
    b = _turn("забудь мой старый номер", n=1, thread="B")
    assert hot.turn_commits(mem, b) == set()
