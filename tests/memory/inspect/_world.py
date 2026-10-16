"""A small seeded world for the inspection commands (§17, G8), written through the real ledger
(`store.commit` + `ledger.apply_op`), the way writers commit.

Not a test module. Fictional people only: two Sashas, Dima."""

from __future__ import annotations

import hashlib
from typing import Any

from koob.memory import ledger, store
from koob.memory.model import (
    AuthorTier,
    Entity,
    EventKind,
    Fact,
    FKind,
    Impression,
    Label,
    OpKind,
    Proposal,
    ProposalOp,
    TurnCtx,
    TurnOrigin,
)

TS = "2026-10-12T09:00:00.000000Z"
TS_DIMA = "2026-10-03T18:30:00.000000Z"

SASHA = "ent-sasha"
SASHA_K = "ent-sasha-k"
DIMA = "ent-dima"

MOVE = "fact-sasha-move"
LIVES_OWN = "fact-sasha-lives-own"
LIVES_DIMA = "fact-sasha-lives-dima"
HEALTH = "fact-sasha-health"
TIRED = "imp-sasha-tired"
DIMA_WORK = "fact-dima-work"
SASHA_K_WORK = "fact-sasha-k-work"

RAW_MOVE = "raw-move"
RAW_DIMA = "raw-dima"


def turn(
    words: str,
    *,
    n: int = 1,
    thread: str = "thread-owner",
    session: str | None = "session-owner",
    **flags: Any,
) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id=thread,
        session_id=session,
        origin=TurnOrigin.OWNER,
        owner_words=words,
        message_id=f"msg-{n}",
        clock=TS,
        **flags,
    )


def add(conn, record, *, tier: AuthorTier | None = None, **payload) -> None:
    body = {"record": record.model_dump(mode="json"), **payload}
    op = ProposalOp(op=OpKind.ADD, group=record.id, payload=body)
    res = store.commit(
        conn,
        Proposal(ops=[op], author_tier=tier or record.author_tier),
        ledger.apply_op,
        event_kind=EventKind.WRITE,
    )
    assert res.ok, (record.id, res)


def person(eid: str, name: str, *, recorded: str = TS) -> Entity:
    return Entity(
        id=eid,
        etype="person",
        display_name=name,
        author_tier=AuthorTier.USER,
        recorded_at=recorded,
        labels_final=True,
    )


def fact(
    fid: str,
    subject: str,
    relation: str,
    value: str,
    text_en: str,
    quote: str | None,
    *,
    tier: AuthorTier = AuthorTier.USER,
    speaker: str | None = None,
    raw: str | None = None,
    labels: Label = Label.NONE,
    recorded: str = TS,
    fkind: FKind = FKind.STATE,
) -> Fact:
    return Fact(
        id=fid,
        subject_id=subject,
        relation=relation,
        value_text=value,
        value_norm=value.casefold(),
        text_en=text_en,
        quote_orig=quote,
        lang_orig="ru" if quote else None,
        fkind=fkind,
        author_tier=tier,
        author_entity_id=speaker,
        speaker_entity_id=speaker,
        source_raw_id=raw,
        origin_quote=None if raw else (quote or text_en),  # §3.2 F9: a root without raw
        labels=labels,
        labels_final=True,
        recorded_at=recorded,
    )


def raw(
    conn,
    rid: str,
    text: str,
    *,
    chat_id: str | None = None,
    turn_id: str | None = None,
    tier: AuthorTier = AuthorTier.USER,
    author: str | None = None,
    source_kind: str = "owner_message",
    ts: str = TS,
    labels: Label = Label.NONE,
) -> str:
    with store.write_tx(conn):
        conn.execute(
            "INSERT INTO raw_chunks (id, source_kind, chat_id, turn_id, author_tier,"
            " author_entity_id, ts, text, content_hmac, labels, labels_final, policy_version,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)",
            (
                rid,
                source_kind,
                chat_id,
                turn_id,
                tier.value,
                author,
                ts,
                text,
                hashlib.sha256(text.encode()).hexdigest(),
                int(labels),
                ts,
                ts,
            ),
        )
        row = conn.execute("SELECT fts_rowid FROM raw_chunks WHERE id = ?", (rid,)).fetchone()
        conn.execute("INSERT INTO fts_raw (rowid, text) VALUES (?, ?)", (row[0], text.casefold()))
    return rid


def serve(conn, item_id: str, *, thread: str, session: str | None, turn_id: str | None = None):
    version = conn.execute("SELECT version FROM records WHERE id = ?", (item_id,)).fetchone()[0]
    with store.write_tx(conn):
        conn.execute(
            "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id, via,"
            " created_at) VALUES (?, ?, ?, 'fact', ?, ?, 'basket', ?)",
            (thread, session, item_id, version, turn_id, TS),
        )


def status(conn, rid: str) -> str:
    return conn.execute("SELECT status FROM records WHERE id = ?", (rid,)).fetchone()[0]


def first_record(conn) -> str:
    """The very first stored record (G8: «from the first stored record»)."""
    raw(conn, RAW_MOVE, "Саша переезжает в Тверь в ноябре", chat_id="cube", turn_id="turn-0")
    add(conn, person(SASHA, "Саша"))
    add(
        conn,
        fact(
            MOVE,
            SASHA,
            "moves_to",
            "Tver",
            "Sasha is moving to Tver in November",
            "Саша переезжает в Тверь в ноябре",
            raw=RAW_MOVE,
            fkind=FKind.PLAN,
        ),
    )
    return MOVE


def world(conn) -> None:
    """Sasha (a move plan, two claims about where she lives, a health claim by Dima, an
    impression) and Dima (his work)."""
    first_record(conn)
    add(conn, person(DIMA, "Дима"), aliases=[{"alias": "Дмитрий", "kind": "name"}])
    raw(
        conn,
        RAW_DIMA,
        "Саша вообще-то живёт в Перми, и у неё диабет",
        chat_id="tg:555",
        tier=AuthorTier.OTHER,
        author=DIMA,
        source_kind="telegram",
        ts=TS_DIMA,
    )
    add(
        conn,
        fact(
            LIVES_OWN, SASHA, "lives_in", "Moscow", "Sasha lives in Moscow", "Саша живёт в Москве"
        ),
    )
    add(
        conn,
        fact(
            LIVES_DIMA,
            SASHA,
            "lives_in",
            "Perm",
            "Sasha lives in Perm",
            "Саша вообще-то живёт в Перми",
            tier=AuthorTier.OTHER,
            speaker=DIMA,
            raw=RAW_DIMA,
            recorded=TS_DIMA,
        ),
        rule_match=False,
    )
    add(
        conn,
        fact(
            HEALTH,
            SASHA,
            "has_condition",
            "diabetes",
            "Sasha has diabetes",
            "у неё диабет",
            tier=AuthorTier.OTHER,
            speaker=DIMA,
            raw=RAW_DIMA,
            labels=Label.HEALTH,
            recorded=TS_DIMA,
        ),
    )
    add(
        conn,
        Impression(
            id=TIRED,
            about_entity_id=SASHA,
            text_en="Sasha seems tired of her job",
            quote_orig="кажется, Саша устала от работы",
            origin_quote="кажется, Саша устала от работы",
            observed_at=TS,
            author_tier=AuthorTier.USER,
            recorded_at=TS,
            labels_final=True,
        ),
    )
    add(
        conn,
        fact(
            DIMA_WORK, DIMA, "works_at", "Yandex", "Dima works at Yandex", "Дима работает в Яндексе"
        ),
    )


def second_sasha(conn) -> None:
    """A namesake: Sasha K. from work, displayed as «Саша» too."""
    add(conn, person(SASHA_K, "Саша"))
    add(
        conn,
        fact(
            SASHA_K_WORK,
            SASHA_K,
            "works_at",
            "Ozon",
            "Sasha K. works at Ozon",
            "Саша К. работает в Озоне",
        ),
    )
