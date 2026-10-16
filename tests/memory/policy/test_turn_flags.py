"""the memory spec §5.1, §5.5, §5.6 (v0.4): the admitted text in the decision, the
adapter's send flag and the reply window keyed by the Cube's message id.
Spec marker: `[T policy/test_turn_flags::<case>]`."""

from __future__ import annotations

from koob.memory import api, policy
from koob.memory.model import AdmitWindow, AuthorTier, ChatMode

from .conftest import owner


def test_decision_carries_stored_text(db):
    conn = db.conn
    turn = owner("Запомни: в субботу у Глеба дома, код домофона спрошу потом.", turn_id="t-1")
    d = api.admit(
        "owner", chat_id=None, text=turn.owner_words, author_tier=AuthorTier.USER, turn=turn,
        store=conn,
    )  # fmt: skip
    assert d.admit and d.text == turn.owner_words

    policy.set_mode(conn, "tg:gleb", ChatMode.BANNED)
    no = api.admit("telegram", chat_id="tg:gleb", text="привет", author_tier=AuthorTier.OTHER,
                   store=conn)  # fmt: skip
    assert not no.admit and no.text is None


def test_sends_flag_overrides_lexicon():
    """`TurnCtx.sends=True` from the hooks adapter makes a send turn without send words;
    `None` keeps the lexicon; `False` never clears a lexicon hit."""
    quiet = "ага, давай так и сделаем"
    assert not policy.is_send_turn(owner(quiet))
    assert policy.is_send_turn(owner(quiet, sends=True))  # tg_send about to run
    assert not policy.is_send_turn(owner(quiet, sends=False))
    asked = "напиши Глебу, что опоздаю"
    assert policy.is_send_turn(owner(asked))  # None: the lexicon decides
    assert policy.is_send_turn(owner(asked, sends=False))  # False never clears a hit
    assert not policy.is_send_turn(owner("напомни мне про врача"))  # reminder to self: no send
    assert policy.is_send_turn(owner("напомни мне про врача", sends=True))


def _no_dig_with_send(conn, **window) -> None:
    policy.set_mode(conn, "tg:olga", ChatMode.NO_DIG)
    policy.open_window(
        conn,
        AdmitWindow(
            id="w-send",
            chat_id="tg:olga",
            sender_id="u:olga",
            after_ts="2026-10-12T10:00:00.000000Z",
            action_key="act-1",
            **window,
        ),
    )


def _reply(conn, reply_to: str | None, ts: str = "2026-10-16T08:00:00.000000Z"):
    return api.admit(
        "telegram", chat_id="tg:olga", text="да, в силе", author_tier=AuthorTier.OTHER,
        sender_id="u:olga", reply_to=reply_to, ts=ts, store=conn,
    )  # fmt: skip


def test_reply_by_cube_message_id(db):
    """A reply whose `reply_to` equals the window's `cube_message_id` is admitted on a
    `no_dig` chat; a `reply:<id>` reason still works for older rows."""
    conn = db.conn
    cols = {r[1] for r in conn.execute("PRAGMA table_info(admit_windows)")}
    if "cube_message_id" not in cols:  # seam: the column migration 0002 adds
        conn.execute("ALTER TABLE admit_windows ADD COLUMN cube_message_id TEXT")
    _no_dig_with_send(conn, reason="send", cube_message_id="tg-msg-501")
    hit = _reply(conn, "tg-msg-501")  # four days after the send: only the reply link lets it in
    assert hit.admit and hit.window_id == "w-send"
    assert not _reply(conn, "tg-msg-499").admit  # a reply to another message
    assert not _reply(conn, None).admit  # no link, outside the 24 h first-reply rule

    # a row written before migration 0002: the id lives in the reason
    conn.execute("DELETE FROM admit_windows")
    _no_dig_with_send(conn, reason="reply:tg-msg-777")
    assert _reply(conn, "tg-msg-777").admit
    assert not _reply(conn, "tg-msg-501").admit
