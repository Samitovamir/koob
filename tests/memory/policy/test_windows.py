"""Memory spec §5.6: admit windows for the "don't dig" policy.
Spec marker: `[T policy/test_windows::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import api, policy
from koob.memory.model import AdmitWindow, AuthorTier, ChatMode

SEND = "2026-10-12T10:00:00.000000Z"


@pytest.fixture
def no_dig(db):
    policy.set_mode(db.conn, "tg:olga", ChatMode.NO_DIG)
    policy.open_window(
        db.conn,
        AdmitWindow(
            id="w-send",
            chat_id="tg:olga",
            sender_id="u:olga",
            after_ts=SEND,
            reason="reply:cube-msg-77",
            action_key="ak-1",
        ),
    )
    return db


def _tg(conn, *, ts, reply_to=None, sender="u:olga", text="ок, поняла"):
    return api.admit(
        "telegram",
        chat_id="tg:olga",
        text=text,
        author_tier=AuthorTier.OTHER,
        sender_id=sender,
        reply_to=reply_to,
        ts=ts,
        store=conn,
    )


def test_reply_admitted(no_dig):
    """§5.6: a direct reply in a no_dig chat is admitted under the send's window."""
    conn = no_dig.conn
    # a reply to the Cube's message, even days later
    assert _tg(conn, ts="2026-10-15T08:00:00Z", reply_to="cube-msg-77").admit
    # the peer's first message within 24 h, without a reply link
    assert _tg(conn, ts="2026-10-12T18:00:00Z").admit
    a = policy.admit_ex(
        conn,
        source_kind="telegram",
        chat_id="tg:olga",
        text="ок",
        author_tier=AuthorTier.OTHER,
        turn=None,
        sender_id="u:olga",
        ts="2026-10-12T18:00:00Z",
    )
    assert a.window_id == "w-send"


def test_other_messages_rejected(no_dig):
    """§5.6: other messages of a no_dig chat are rejected."""
    conn = no_dig.conn
    before = _tg(conn, ts="2026-10-12T09:59:00Z")
    assert not before.admit and before.reason == "no_dig_outside_window"
    late = _tg(conn, ts="2026-10-13T10:00:01Z")
    assert not late.admit
    # only the first message within 24 h: once it is in, the next one is background
    no_dig.raw("r-first", chat="tg:olga", ts="2026-10-12T11:00:00Z", text="первое")
    assert not _tg(conn, ts="2026-10-12T12:00:00Z").admit
    # a reply to some other message is not a direct reply
    assert not _tg(conn, ts="2026-10-12T12:00:00Z", reply_to="peer-msg-5").admit
    assert no_dig.count("audit", "event = 'admit.reject.no_dig_outside_window'") == 4


def test_banned_intention_refused(db):
    """§5.6: an intention on a banned chat is refused with one line."""
    conn = db.conn
    policy.set_mode(conn, "tg:gleb", ChatMode.BANNED)
    with pytest.raises(policy.PolicyRefused) as exc:
        policy.open_window(
            conn,
            AdmitWindow(
                id="w-i",
                chat_id="tg:gleb",
                sender_id="u:gleb",
                after_ts=SEND,
                reason="intention",
                intention_id="int-1",
            ),
        )
    assert "\n" not in str(exc.value) and str(exc.value)
    assert db.count("admit_windows") == 0
    d = api.admit(
        "telegram",
        chat_id="tg:gleb",
        text="я ответил",
        author_tier=AuthorTier.OTHER,
        sender_id="u:gleb",
        ts="2026-10-12T11:00:00Z",
        store=conn,
    )
    assert not d.admit and d.reason == "chat_banned"


def test_intention_window_until_done(db):
    """§5.6: an intention window admits that sender after T until done or cancelled, into
    that matter only."""
    conn = db.conn
    policy.set_mode(conn, "tg:team", ChatMode.NO_DIG)
    db._record("int-1", "intention", tier="user", labels=0, final=True, status="active")
    conn.execute("INSERT INTO intentions (id, trigger, state) VALUES ('int-1', '{}', 'armed')")
    policy.open_window(
        conn,
        AdmitWindow(
            id="w-int",
            chat_id="tg:team",
            sender_id="u:ilya",
            after_ts=SEND,
            reason="intention",
            intention_id="int-1",
        ),
    )

    def msg(sender, ts):
        return policy.admit_ex(
            conn,
            source_kind="telegram",
            chat_id="tg:team",
            text="про резюме",
            author_tier=AuthorTier.OTHER,
            turn=None,
            sender_id=sender,
            ts=ts,
        )

    a = msg("u:ilya", "2026-10-20T10:00:00Z")
    assert a.admit and a.intention_id == "int-1"
    assert not msg("u:other", "2026-10-20T10:00:00Z").admit
    assert not msg("u:ilya", "2026-10-11T10:00:00Z").admit
    conn.execute("UPDATE intentions SET state = 'done' WHERE id = 'int-1'")
    assert not msg("u:ilya", "2026-10-21T10:00:00Z").admit
    conn.execute("UPDATE intentions SET state = 'armed' WHERE id = 'int-1'")
    assert policy.close_windows(conn, intention_id="int-1", ts="2026-10-22T00:00:00Z") == 1
    assert not msg("u:ilya", "2026-10-23T10:00:00Z").admit


def test_ban_drops_windows(no_dig):
    policy.set_mode(no_dig.conn, "tg:olga", ChatMode.BANNED)
    assert no_dig.count("admit_windows") == 0
    assert not _tg(no_dig.conn, ts="2026-10-12T18:00:00Z", reply_to="cube-msg-77").admit
