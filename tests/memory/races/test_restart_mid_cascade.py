"""The process dies mid-way and restarts (spec §6.1, §6.3, §8.4, §9.1, §9.3, §5.2, §19.1 «a
restart mid-cascade»): a child process runs one facade call and is killed at points spread
over its SQLite steps; on reopen nothing is half-applied, and what was committed either
finishes (its job resumes) or is absent (the caller re-issues it).

Crash injection is a SQLite progress handler in the child (`_crash_child.py`), so the suite
needs no knowledge of the call's internals."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from koob.memory import api
from koob.memory.model import AuthorTier, ChatMode, Decision, Domain
from koob.memory.workers import scheduler

from ._kit import (
    TS0,
    add_raw,
    crash_runs,
    fact,
    handle,
    journal_path,
    justify,
    lineage,
    open_db,
    open_jobs,
    owner_turn,
    person,
    seed,
    status_of,
    ulid,
    usable_slot,
)

CHAT = "tg:-100888"


def test_restart_mid_fix_cascade_is_never_half_applied(db_path, conn, tmp_path, clock):
    """§6.1 + §8.4: a fix on a stated premise of a decision (AND-group «12 мая и ресторан
    свободен») dies mid-way. On reopen: either nothing changed, or the fact is corrected and
    the decision is excluded, flagged or has a durable cascade job; after the scheduler
    resumes (or the user re-issues the fix) the decision is `needs_review` and the slot holds 14."""
    sasha = person("Саша")
    bday = fact(
        "bday", sasha, "birthday", "12 May", "Sasha's birthday is 12 May",
        "у Саши день рождения 12 мая", value_norm="05-12",
    )  # fmt: skip
    free = fact(
        "free", sasha, "other", "free on 12 May", "The restaurant is free on 12 May",
        "ресторан свободен 12 мая",
    )  # fmt: skip
    dinner = Decision(
        id=ulid("decision:dinner"),
        author_tier=AuthorTier.USER,
        recorded_at=TS0,
        labels_final=True,
        domain=Domain.SOCIAL,
        chosen_text="Book the dinner for Sasha's birthday on 12 May",
        decided_at=TS0,
    )
    seed(
        conn,
        sasha,
        bday,
        free,
        dinner,
        extra=lambda sql: justify(sql, dinner.id, 1, [bday.id, free.id]),
    )
    conn.close()
    turn = owner_turn("нет, у Саши день рождения 14 мая", n=2)
    call = {
        "target": "api",
        "call": "fix",
        "args": [handle(bday.id), "у Саши день рождения 14 мая"],
        "kwargs": {
            "turn": turn.model_dump(mode="json"),
            "because_quote": "у Саши день рождения 14 мая",
        },
    }

    for db, code in crash_runs(db_path, tmp_path, call):
        c = open_db(db)
        try:
            assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            fixed = status_of(c, bday.id) not in ("active", "archived")
            slot = usable_slot(c, sasha.id, "birthday")
            assert len(slot) == 1, f"slot torn ({code}): {slot}"
            d_status, d_review = c.execute(
                "SELECT status, review FROM records WHERE id = ?", (dinner.id,)
            ).fetchone()
            if not fixed:
                assert slot[0][0] == bday.id and d_status == "active"
                api.fix(
                    handle(bday.id),
                    "у Саши день рождения 14 мая",
                    turn=turn,
                    because_quote="у Саши день рождения 14 мая",
                    store=c,
                )
            else:
                assert "14" in (slot[0][1] or "")
                assert d_status == "needs_review" or d_review == 1 or open_jobs(c) > 0, (
                    "the fact changed but its dependent is neither excluded, flagged nor queued"
                )
            for _ in range(3):
                clock.advance(timedelta(hours=1).total_seconds())
                scheduler.tick(c, clock)
            assert status_of(c, dinner.id) == "needs_review"
            slot = usable_slot(c, sasha.id, "birthday")
            assert len(slot) == 1 and "14" in (slot[0][1] or "")
        finally:
            c.close()


def test_restart_mid_forget_ledger_and_journal_agree(db_path, conn, tmp_path):
    """§9.1 1a/1e, §9.3: a forget dies mid-way. On reopen the item is either untouched or
    forgotten with its `stopped` journal line (a forgotten item without a journal line would
    come back from the next backup restore); re-issuing the forget completes it."""
    sasha = person("Саша")
    raw = add_raw(conn, "move", "Саша переезжает в Тверь", chat_id=None, turn_id="turn-0")
    move = fact(
        "move", sasha, "moves_to", "Tver", "Sasha is moving to Tver",
        "Саша переезжает в Тверь", source_raw_id=raw,
    )  # fmt: skip
    seed(conn, sasha, move, extra=lambda sql: lineage(sql, move.id, raw))
    conn.close()
    turn = owner_turn("забудь про переезд Саши", n=2)
    call = {
        "target": "api",
        "call": "forget",
        "args": [handle(move.id)],
        "kwargs": {
            "turn": turn.model_dump(mode="json"),
            "because_quote": "переезд Саши",
            "scope": "item",
        },
    }

    def journaled(db: Path) -> bool:
        path = journal_path(db)
        if not path.exists():
            return False
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        return any(x["stage"] == "stopped" and move.id in x["record_ids"] for x in lines)

    for db, code in crash_runs(db_path, tmp_path, call):
        c = open_db(db)
        try:
            assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            st = status_of(c, move.id)
            assert st in ("active", "forgotten"), st
            if st == "forgotten":
                assert journaled(db), f"forgotten without a journal line (exit {code})"
            else:
                raw_st = c.execute("SELECT status FROM raw_chunks WHERE id = ?", (raw,)).fetchone()
                assert raw_st[0] == "active", "the raw was forgotten without its record"
            again = api.forget(handle(move.id), turn=turn, because_quote="переезд Саши", store=c)
            assert again.outcome in ("stopped", "not_found")
            assert status_of(c, move.id) == "forgotten" and journaled(db)
        finally:
            c.close()


def test_restart_mid_ban_leaves_no_banned_chat_in_use(db_path, conn, tmp_path):
    """§5.2: a ban dies mid-way. On reopen a chat marked `banned` has nothing usable left from
    it, or a durable job that finishes the forget; a chat still `normal` kept its memory."""
    dima = person("Дима")
    raw = add_raw(
        conn, "hotel", "Дима: живём в отеле Морской", chat_id=CHAT, tier=AuthorTier.OTHER,
        author_entity_id=dima.id, source_kind="telegram",
    )  # fmt: skip
    hotel = fact(
        "hotel", dima, "other", "Morskoy hotel", "Dima stays at the Morskoy hotel",
        "живём в отеле Морской", tier=AuthorTier.OTHER, speaker=dima, source_raw_id=raw,
    )  # fmt: skip
    seed(conn, dima, hotel, extra=lambda sql: lineage(sql, hotel.id, raw))
    conn.close()
    turn = owner_turn("полный запрет на чат с Димой", n=2)
    call = {
        "target": "api",
        "call": "set_chat_mode",
        "args": [CHAT, ChatMode.BANNED.value],
        "mode_arg": True,
        "kwargs": {"turn": turn.model_dump(mode="json")},
    }
    for db, code in crash_runs(db_path, tmp_path, call):
        c = open_db(db)
        try:
            row = c.execute("SELECT mode FROM chat_policy WHERE chat_id = ?", (CHAT,)).fetchone()
            mode = row[0] if row else "normal"
            st = status_of(c, hotel.id)
            if mode == "banned":
                assert st not in ("active", "archived") or open_jobs(c) > 0, (
                    f"banned chat still in use after a crash (exit {code})"
                )
            else:
                assert st == "active"
        finally:
            c.close()
