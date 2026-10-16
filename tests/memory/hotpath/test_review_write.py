"""Regressions for the hot writer: tainted-turn tiers, turn-scoped commit marks, one cue one
commit, and never-store on every stored quote."""

from __future__ import annotations

from pathlib import Path

from koob.memory import store
from koob.memory.model import Assertion, TurnCtx, TurnOrigin
from koob.memory.writers import hot

from ..races._kit import TS0
from ._world import fact_row, sasha_world, turn_of

SECRET = "Kx9#vQ2!mZ7pL4"


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


# ---------------------------------------------------------------- W-3


def test_tainted_remember_with_a_foreign_value_word_is_tool(mem):
    """W-3 (§6.1, G11): a page is open; the user's words name only part of the text; «Берлин»
    came from the page, so the claim is `tool`, not the user's."""
    t = turn_of("запомни, что Олег переезжает", n=2, tainted=True)
    r = hot.remember(mem, "Олег переезжает в Берлин", t, because_quote="Олег переезжает")
    assert r.outcome == "committed"
    assert fact_row(mem, r.record_ids[0])["author_tier"] == "tool"


def test_tainted_remember_foreign_assertion_value_is_tool(mem):
    """The text is the user's, but the assertion's value is the page's: `tool`."""
    t = turn_of("запомни, что Олег переезжает", n=3, tainted=True)
    a = Assertion(relation="moves_to", value="Берлин")
    r = hot.remember(mem, "Олег переезжает", t, because_quote="Олег переезжает", assertion=a)
    assert fact_row(mem, r.record_ids[0])["author_tier"] == "tool"


def test_tainted_remember_all_his_words_stays_user(mem):
    """Control: every content word is the user's: `user`."""
    t = turn_of("запомни, что Олег переезжает в Берлин", n=4, tainted=True)
    r = hot.remember(mem, "Олег переезжает в Берлин", t, because_quote="Олег переезжает в Берлин")
    assert fact_row(mem, r.record_ids[0])["author_tier"] == "user"




def _files_hold(root: Path, needle: str) -> list[str]:
    raw = needle.encode()
    return [str(p) for p in root.rglob("*") if p.is_file() and raw in p.read_bytes()]


def _columns_hold(conn, needle: str) -> list[str]:
    hits = []
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        for col in conn.execute(f"PRAGMA table_info('{table}')"):
            name = col[1]
            n = conn.execute(
                f'SELECT count(*) FROM "{table}" WHERE CAST("{name}" AS TEXT) LIKE ?',
                (f"%{needle}%",),
            ).fetchone()[0]
            if n:
                hits.append(f"{table}.{name}")
    return hits


def test_remember_never_stores_a_password_from_his_words(mem, tmp_path):
    """I-3 (G1): the brain saw the user's words redacted, so its quote does not match; the
    fallback must not store the user's raw words with the password in them."""
    words = f"запомни, что я поменял пароль от роутера: пароль: {SECRET}"
    t = turn_of(words, n=5)
    seen = "запомни, что я поменял пароль от роутера: пароль: [скрыто: похоже на пароль]"
    r = hot.remember(mem, "Денис поменял пароль от роутера", t, because_quote=seen)
    assert r.outcome == "committed"
    r2 = hot.remember(mem, "пароль от роутера", turn_of(words, n=6), because_quote=words)
    assert r2.outcome == "committed"
    assert _columns_hold(mem, SECRET) == []
    assert _files_hold(tmp_path, SECRET) == []


def test_fix_never_stores_a_password_from_his_words(mem, tmp_path):
    """The same for «исправь» (its quote and the new record)."""
    _sasha, bday = sasha_world(mem)
    words = f"исправь: у Саши день рождения 13 мая, и мой пароль: {SECRET}"
    t = turn_of(words, n=7)
    hot.fix(mem, f"m:{bday.id}", "13 May", t, because_quote=words)
    assert _columns_hold(mem, SECRET) == []
    assert _files_hold(tmp_path, SECRET) == []




def _other_thread_forget(conn) -> None:
    with store.write_tx(conn):
        conn.execute(
            "INSERT INTO events (ts, kind, memory_version, item_ids) VALUES (?, 'forget', 1, '[]')",
            (TS0,),
        )


def test_forget_in_another_thread_does_not_back_this_turns_fix(mem):
    """A forget committed in thread A never counts as thread B's correction."""
    sasha_world(mem)
    _other_thread_forget(mem)
    b = _turn("нет, Саша работает в Сбере, исправь", n=1, thread="B")
    assert hot.claims_unbacked_change(mem, b, "Исправил.")
    fired, _ents, _jid = hot.after_turn_check(mem, b, "Исправил.")
    assert fired


def test_one_commit_does_not_hide_another_cue(mem):
    """Remember ran, forget did not: the forget cue fires and «забыл» is unbacked."""
    t = _turn("запомни, что я переехал в Тверь, и забудь мой старый номер", n=1, thread="C")
    r = hot.remember(mem, "я переехал в Тверь", t, because_quote="я переехал в Тверь")
    assert r.outcome == "committed"
    reply = "Запомнил про Тверь и забыл старый номер."
    assert hot.claims_unbacked_change(mem, t, reply)
    fired, _ents, jid = hot.after_turn_check(mem, t, reply)
    assert fired and jid
    import json

    (payload,) = mem.execute("SELECT payload FROM jobs WHERE id = ?", (jid,)).fetchone()
    assert json.loads(payload)["cues"] == ["forget"]


def test_matching_commits_back_their_cues(mem):
    """Control: remember backs «запомни»; a stamped forget of this turn backs «забудь»."""
    t = _turn("запомни, что я переехал в Тверь, и забудь мой старый номер", n=2, thread="D")
    hot.remember(mem, "я переехал в Тверь", t, because_quote="я переехал в Тверь")
    hot.note_commit(mem, t, "forget", ["forget-1"])
    reply = "Запомнил про Тверь и забыл старый номер."
    assert not hot.claims_unbacked_change(mem, t, reply)
    assert hot.after_turn_check(mem, t, reply)[0] is False
