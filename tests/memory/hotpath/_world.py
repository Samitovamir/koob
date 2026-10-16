"""Shared seeds for the hot-path tests (§6.1): `Саша` with a confirmed alias and a birthday."""

from __future__ import annotations

from koob.memory import entities
from koob.memory.model import Alias, AliasKind, AuthorTier, TurnCtx, TurnOrigin

from ..races._kit import TS0, fact, person, seed, ulid


def sasha_world(conn):
    sasha = person("Саша")
    bday = fact(
        "bday",
        sasha,
        "birthday",
        "12 May",
        "Sasha's birthday is 12 May",
        "у Саши день рождения 12 мая",
        value_norm="05-12",
    )
    seed(conn, sasha, bday)
    entities.bind_alias(
        conn,
        Alias(
            id=ulid("alias:sasha"),
            entity_id=sasha.id,
            alias="Саша",
            alias_norm=entities.normalize_alias("Саша"),
            script="cyrl",
            kind=AliasKind.NAME,
            tier=AuthorTier.USER,
            confirmed=True,
        ),
        tier=AuthorTier.USER,
        platform=False,
    )
    return sasha, bday


def turn_of(words: str, *, n: int = 1, tainted: bool = False, origin=TurnOrigin.OWNER) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id="thread-1",
        session_id="session-1",
        origin=origin,
        owner_words=words,
        task_words=words if origin is not TurnOrigin.OWNER else None,
        tainted=tainted,
        message_id=f"msg-{n}",
        clock=TS0,
    )


def fact_row(conn, rid: str):
    return conn.execute(
        "SELECT f.subject_id, f.relation, f.value_text, f.value_norm, f.text_en, f.quote_orig,"
        " r.origin_quote, r.author_tier, r.status FROM facts f JOIN records r ON r.id = f.id"
        " WHERE f.id = ?",
        (rid,),
    ).fetchone()
