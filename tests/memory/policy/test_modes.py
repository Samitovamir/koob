"""the memory spec §5.2: chat modes.
Spec marker: `[T policy/test_modes::<case>]`.

The forget pipeline itself belongs to forget (`forget.stop_use`); here a recorder stands in for
it, so these tests pin what policy hands over. `test_ban_purges_end_to_end` runs the real
pipeline and stays xfail until that pipeline is implemented."""

from __future__ import annotations

import pytest

from koob.memory import api, forget, policy
from koob.memory.model import AdmitWindow, AuthorTier, ChatMode, ForgetScope

from .conftest import owner

GLEB_1 = "Мира, я переезжаю в Ригу в июле. Новый код от домофона VORTELL-5521, на всякий случай."
GLEB_2 = "Глеб: верни, пожалуйста, мой серый свитер. Я дома в субботу, домофон GRAVEN-8813."


@pytest.fixture
def stop_use(monkeypatch):
    calls: list[dict] = []

    def fake(conn, target_ids, *, scope, forever, conversation=None):
        calls.append(
            {
                "ids": set(target_ids),
                "scope": scope,
                "forever": forever,
                "conversation": conversation,
            }
        )
        return f"fg-{len(calls)}"

    monkeypatch.setattr(forget, "stop_use", fake)
    return calls


def _seed_gleb(db):
    db.raw("r-g1", text=GLEB_1, chat="tg:gleb", author="e-gleb", ts="2026-06-01T10:00:00Z")
    db.fact(
        "f-riga",
        subject="e-gleb",
        text_en="Gleb moves to Riga in July",
        tier="other",
        source_raw_id="r-g1",
    )
    db.fact("f-just", subject="e-gleb", text_en="Gleb leaves Moscow", tier="system")
    db.conn.execute(
        "INSERT INTO justifications (id, target_record_id, group_no, member_record_id, provenance,"
        " created_at) VALUES ('j1', 'f-just', 0, 'r-g1', 'rule', '2026-06-01T10:00:00Z')"
    )
    # a Cube turn that read the chat at the user's request
    policy.record_turn_source(db.conn, "t-2", "tg_read", "tg:gleb", requested=True)
    db.raw(
        "r-t2-user",
        text="Прочитай, что мне сегодня написал Глеб.",
        turn="t-2",
        tier="user",
        source_kind="cube_turn",
    )
    db.raw(
        "r-t2-answer",
        text="Глеб просит вернуть свитер.",
        turn="t-2",
        tier="system",
        source_kind="cube_turn",
    )
    db.raw("r-t2-tool", text=GLEB_2, turn="t-2", tier="tool", source_kind="cube_turn")
    db.raw(
        "r-t2-quote",
        text="он пишет: Мира, я переезжаю в Ригу в июле. Новый код от домофона",
        turn="t-2",
        tier="user",
        source_kind="cube_turn",
    )
    # untouched: another chat, another turn
    db.raw("r-other", text="привет", chat="tg:mira")
    db.raw("r-t3", text="Какая погода?", turn="t-3", tier="user", source_kind="cube_turn")


def test_ban_purges_memory_side(db, turn, stop_use):
    """§5.2: a ban runs the full forget pipeline on the chat's memory side; the user's chat history
    stays."""
    _seed_gleb(db)
    conn = db.conn
    change = api.set_chat_mode("tg:gleb", ChatMode.BANNED, turn=turn, store=conn)
    assert change.mode is ChatMode.BANNED and change.policy_version == 1
    assert change.forget_id == "fg-1"
    [call] = stop_use
    assert call["scope"] is ForgetScope.CHAT and call["forever"] is True
    assert call["conversation"] is None  # the ban governs memory, not the user's island history
    assert call["ids"] == {"r-g1", "r-t2-answer", "r-t2-tool", "r-t2-quote", "f-riga", "f-just"}
    # it applies before the next admission
    d = api.admit(
        "telegram", chat_id="tg:gleb", text=GLEB_2, author_tier=AuthorTier.OTHER, store=conn
    )
    assert not d.admit and d.reason == "chat_banned"
    assert db.count("events", "kind = 'mode'") == 1
    assert db.count("chat_policy", "chat_id = 'tg:gleb' AND mode = 'banned'") == 1


def test_no_dig_keeps_requested(db, turn, stop_use):
    """§5.2: «не копать» forgets background-derived records and keeps requested exchanges."""
    conn = db.conn
    policy.open_window(
        conn,
        AdmitWindow(
            id="w1",
            chat_id="tg:olga",
            after_ts="2026-10-10T10:00:00Z",
            reason="reply:cube-9",
            action_key="ak-9",
        ),
    )
    db.raw("r-bg", text="Ольга: мы с Ильёй поссорились", chat="tg:olga", ts="2026-10-09T10:00:00Z")
    db.raw("r-reply", text="Ольга: да, в пятницу удобно", chat="tg:olga", ts="2026-10-10T12:00:00Z")
    db.fact(
        "f-bg",
        subject="e-olga",
        text_en="Olga quarrelled with Ilya",
        tier="other",
        source_raw_id="r-bg",
    )
    db.fact(
        "f-reply",
        subject="e-olga",
        text_en="Olga can meet on Friday",
        tier="other",
        source_raw_id="r-reply",
    )
    policy.record_turn_source(conn, "t-dig", "tg_read", "tg:olga", requested=False)
    db.raw("r-dig-user", text="Что нового?", turn="t-dig", tier="user", source_kind="cube_turn")
    db.raw(
        "r-dig-answer",
        text="Ольга поссорилась с Ильёй",
        turn="t-dig",
        tier="system",
        source_kind="cube_turn",
    )
    policy.record_turn_source(conn, "t-asked", "tg_read", "tg:olga", requested=True)
    db.raw(
        "r-asked",
        text="Ольга согласна на пятницу",
        turn="t-asked",
        tier="system",
        source_kind="cube_turn",
    )

    change = api.set_chat_mode("tg:olga", ChatMode.NO_DIG, turn=turn, store=conn)
    assert change.mode is ChatMode.NO_DIG and change.forget_id == "fg-1"
    assert stop_use[0]["ids"] == {"r-bg", "r-dig-answer", "f-bg"}
    # from now on only windows admit
    assert not api.admit(
        "telegram",
        chat_id="tg:olga",
        text="ещё новость",
        author_tier=AuthorTier.OTHER,
        ts="2026-10-12T10:00:00Z",
        store=conn,
    ).admit


def test_no_dig_keeps_exchange_he_ordered_before(db, turn, stop_use):
    """Normal -> «не копать»: a send the user ordered earlier (`actions`) keeps its exchange: both
    sides within 24 h and the peer's first reply; the rest of the chat was background."""
    conn = db.conn
    conn.execute(
        "INSERT INTO actions (id, action_key, origin_key, seq, tool, stable_target, state,"
        " reserved_at) VALUES ('a1', 'ak-1', 'msg-1', 0, 'tg_send', 'tg:olga', 'ok',"
        " '2026-10-05T10:00:00Z')"
    )
    db.raw(
        "r-sent",
        text="Ольга, в пятницу в 7?",
        chat="tg:olga",
        tier="system",
        ts="2026-10-05T10:00:01Z",
    )
    db.raw("r-first", text="Да, давай", chat="tg:olga", ts="2026-10-05T11:00:00Z")
    db.raw(
        "r-second", text="Кстати, мы с Ильёй поссорились", chat="tg:olga", ts="2026-10-05T11:05:00Z"
    )
    db.raw("r-older", text="Старое", chat="tg:olga", ts="2026-10-01T09:00:00Z")
    api.set_chat_mode("tg:olga", ChatMode.NO_DIG, turn=turn, store=conn)
    assert stop_use[0]["ids"] == {"r-second", "r-older"}


def test_mode_change_versions_and_events(db, turn, stop_use):
    conn = db.conn
    assert policy.chat_mode(conn, "tg:new") is ChatMode.NORMAL  # new contacts are normal
    assert policy.set_mode(conn, "tg:a", ChatMode.NO_DIG) == 1
    assert policy.set_mode(conn, "tg:a", ChatMode.NO_DIG) == 1  # unchanged: no bump, no event
    assert policy.set_mode(conn, "tg:b", ChatMode.BANNED) == 2
    assert policy.set_mode(conn, "tg:a", ChatMode.NORMAL) == 3
    assert db.count("events", "kind = 'mode' AND item_ids = '[]'") == 3
    assert conn.execute("SELECT policy_version FROM meta").fetchone()[0] == 3
    # back to normal forgets nothing
    change = api.set_chat_mode("tg:b", ChatMode.NORMAL, turn=turn, store=conn)
    assert change.forget_id is None and stop_use == []


def test_speakerphone_mode_change_waits_for_his_channel(db):
    """On speakerphone a chat-mode change needs a line in the user's own channel."""
    room = owner("Куб, не копай чат с Ольгой", speaker=True)
    with pytest.raises(policy.PolicyRefused):
        api.set_chat_mode("tg:olga", ChatMode.NO_DIG, turn=room, store=db.conn)
    assert policy.chat_mode(db.conn, "tg:olga") is ChatMode.NORMAL


def test_ban_purges_end_to_end(db, turn):
    """§5.2 + §9: after a ban and the purge, no content of the chat is left in memory.db."""
    _seed_gleb(db)
    conn = db.conn
    api.set_chat_mode("tg:gleb", ChatMode.BANNED, turn=turn, store=conn)
    api.run_purges(store=conn)
    for rid in ("r-g1", "r-t2-answer", "r-t2-tool", "r-t2-quote"):
        row = conn.execute("SELECT status, text FROM raw_chunks WHERE id = ?", (rid,)).fetchone()
        assert row is None or (row["status"] == "purged" and row["text"] is None)
    for rec in ("f-riga", "f-just"):
        row = conn.execute("SELECT status FROM records WHERE id = ?", (rec,)).fetchone()
        assert row is None or row["status"] == "purged"
    assert (
        conn.execute("SELECT status FROM raw_chunks WHERE id = 'r-t2-user'").fetchone()[0]
        == "active"
    )
