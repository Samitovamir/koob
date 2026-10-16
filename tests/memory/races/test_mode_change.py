"""A chat-mode change mid-turn (spec §5.1, §5.2, §5.6, §7.4): «полный запрет» and «не копать»
apply before the next admission and the next use, even when the turn that read the chat started
before the change; the ban runs the forget pipeline on what was stored."""

from __future__ import annotations

import threading

from koob.memory import api
from koob.memory.model import AuthorTier, ChatMode, FKind

from ._kit import (
    TS0,
    WAIT,
    add_raw,
    fact,
    lineage,
    open_db,
    owner_turn,
    person,
    ref,
    run_threads,
    seed,
    status_of,
)

CHAT = "tg:-100777"
SENDER = "tg:777"


def _admit_peer(conn, text: str):
    return api.admit(
        "telegram",
        chat_id=CHAT,
        text=text,
        author_tier=AuthorTier.OTHER,
        sender_id=SENDER,
        ts=TS0,
        store=conn,
    )


def _mode_events(conn) -> int:
    return conn.execute("SELECT count(*) FROM events WHERE kind = 'mode'").fetchone()[0]


def test_ban_lands_mid_turn_and_binds_the_after_turn_admission(db_path, conn):
    """§5.1, §5.2: the turn read the chat while it was normal; the ban commits before the
    after-turn ingestion; the answer and tool rows are rejected, only the user's own words
    (sharing no shingle with the read) are kept, and the next peer message is rejected."""
    turn = owner_turn("что там пишет Дима про поездку?", n=5)
    api.record_turn_source(turn.turn_id, "telegram", CHAT, requested=True, store=conn)
    assert _admit_peer(conn, "Дима: едем в субботу, билеты у меня").admit
    before = _mode_events(conn)
    read_done, banned = threading.Event(), threading.Event()

    def turn_side():
        t = open_db(db_path)
        try:
            read_done.set()
            assert banned.wait(WAIT)
            answer = api.admit(
                "cube_answer",
                chat_id=None,
                text="Дима пишет, что едете в субботу и билеты у него.",
                author_tier=AuthorTier.SYSTEM,
                turn=turn,
                store=t,
            )
            tool = api.admit(
                "tool_result",
                chat_id=CHAT,
                text="Дима: едем в субботу, билеты у меня",
                author_tier=AuthorTier.TOOL,
                turn=turn,
                store=t,
            )
            own = api.admit(
                "owner_message",
                chat_id=None,
                text=turn.owner_words,
                author_tier=AuthorTier.USER,
                turn=turn,
                store=t,
            )
            return answer, tool, own
        finally:
            t.close()

    def ban_side():
        assert read_done.wait(WAIT)
        b = open_db(db_path)
        try:
            return api.set_chat_mode(
                CHAT, ChatMode.BANNED, turn=owner_turn("не запоминай этот чат вообще", n=6), store=b
            )
        finally:
            b.close()
            banned.set()

    (answer, tool, own), change = run_threads(turn_side, ban_side)
    assert change.mode is ChatMode.BANNED
    assert not answer.admit and answer.reason
    assert not tool.admit
    assert own.admit and own.keep_only_own_words
    assert not _admit_peer(conn, "Дима: и ещё, адрес отеля такой-то").admit
    assert _mode_events(conn) == before + 1


def test_admissions_racing_a_ban_never_pass_after_it(db_path, conn):
    """§5.2 «applies before the next admission»: peer messages are admitted back to back on
    another connection while the ban commits; none that starts after the ban returned passes,
    and the policy version moved."""
    returned, started = threading.Event(), threading.Event()
    seen: list[tuple[bool, bool]] = []

    def admitter():
        a = open_db(db_path)
        try:
            after = 0
            i = 0
            while after < 5:
                was_after = returned.is_set()
                seen.append((was_after, _admit_peer(a, f"сообщение номер {i}").admit))
                started.set()
                after += was_after
                i += 1
        finally:
            a.close()

    changes = []

    def ban():
        assert started.wait(WAIT)
        b = open_db(db_path)
        try:
            changes.append(
                api.set_chat_mode(
                    CHAT,
                    ChatMode.BANNED,
                    turn=owner_turn("полный запрет на этот чат", n=2),
                    store=b,
                )
            )
        finally:
            b.close()
            returned.set()

    run_threads(admitter, ban)
    assert seen[0] == (False, True)
    assert all(not ok for was_after, ok in seen if was_after)
    assert changes[0].policy_version >= 1


def test_ban_forgets_what_the_chat_already_put_in_memory(db_path, conn):
    """§5.2 to `banned`: the chat's raw and the records derived from it run the full forget
    pipeline; a basket prepared before the ban is refused at the next use."""
    dima = person("Дима")
    raw = add_raw(
        conn,
        "dima-hotel",
        "Дима: живём в отеле Морской, номер 214",
        chat_id=CHAT,
        tier=AuthorTier.OTHER,
        author_entity_id=dima.id,
        source_kind="telegram",
    )
    hotel = fact(
        "dima-hotel",
        dima,
        "other",
        "Morskoy hotel",
        "Dima stays at the Morskoy hotel, room 214",
        "живём в отеле Морской, номер 214",
        tier=AuthorTier.OTHER,
        speaker=dima,
        source_raw_id=raw,
    )
    seed(conn, dima, hotel, extra=lambda sql: lineage(sql, hotel.id, raw))
    basket = [ref(conn, hotel.id)]

    change = api.set_chat_mode(
        CHAT, ChatMode.BANNED, turn=owner_turn("полный запрет на чат с Димой", n=2), store=conn
    )
    assert change.forget_id
    assert status_of(conn, hotel.id) in ("forgotten", "purged")
    raw_status = conn.execute("SELECT status FROM raw_chunks WHERE id = ?", (raw,)).fetchone()[0]
    assert raw_status in ("forgotten", "purged")
    pre = api.pre_send_check(thread_id="thread-owner", read_set=basket, store=conn)
    assert not pre.ok and basket[0] in pre.stale


def test_no_dig_mid_turn_forgets_background_keeps_requested(db_path, conn):
    """§5.2 to `no_dig`: background-derived records are forgotten; the exchange the user
    asked for stays; the next background message is rejected."""
    dima = person("Дима")
    background = add_raw(
        conn,
        "bg",
        "Дима: кстати, я сменил работу",
        chat_id=CHAT,
        tier=AuthorTier.OTHER,
        author_entity_id=dima.id,
        source_kind="telegram",
    )
    asked_turn = owner_turn("спроси у Димы, во сколько встреча", n=3)
    api.record_turn_source(asked_turn.turn_id, "telegram", CHAT, requested=True, store=conn)
    requested = add_raw(
        conn,
        "asked",
        "Дима: встреча в 19:00",
        chat_id=CHAT,
        turn_id=asked_turn.turn_id,
        tier=AuthorTier.OTHER,
        author_entity_id=dima.id,
        source_kind="telegram",
    )
    job = fact(
        "dima-job",
        dima,
        "other",
        "new job",
        "Dima changed jobs",
        "я сменил работу",
        tier=AuthorTier.OTHER,
        speaker=dima,
        source_raw_id=background,
    )
    meet = fact(
        "dima-meet",
        dima,
        "other",
        "19:00",
        "The meeting with Dima is at 19:00",
        "встреча в 19:00",
        tier=AuthorTier.OTHER,
        speaker=dima,
        source_raw_id=requested,
        fkind=FKind.PLAN,
    )

    def edges(sql):
        lineage(sql, job.id, background)
        lineage(sql, meet.id, requested)

    seed(conn, dima, job, meet, extra=edges)
    change = api.set_chat_mode(
        CHAT, ChatMode.NO_DIG, turn=owner_turn("не копай чат с Димой", n=4), store=conn
    )
    assert change.mode is ChatMode.NO_DIG
    assert status_of(conn, job.id) in ("forgotten", "purged")
    assert status_of(conn, meet.id) == "active"
    assert not _admit_peer(conn, "Дима: а ещё я купил машину").admit
