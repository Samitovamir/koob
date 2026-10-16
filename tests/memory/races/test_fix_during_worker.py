"""A correction while a background worker holds the old version (spec §7.5, §6.3, §7.3, §7.4,
§7.6, §16.3): after the worker finishes, the old value is gone from every channel memory
owns: usable records, valid derived items, the pre-send check of any read set that held it,
the threads that were served it, and the hands.

Spec marker: `[T races/test_fix_during_worker]`. Fixtures seed through the ledger and the
contract schema; the fix goes through `api.fix`."""

from __future__ import annotations

import threading

from koob.memory import api, entities, store
from koob.memory.model import (
    Alias,
    AliasKind,
    AuthorTier,
    Label,
    OpKind,
    Proposal,
    ProposalOp,
    ReadRef,
)

from .._contract import not_built
from ._kit import (
    TS0,
    WAIT,
    add_derived,
    add_raw,
    bare_fact,
    derived_ref,
    fact,
    handle,
    lineage,
    open_db,
    owner_turn,
    person,
    ref,
    run_threads,
    seed,
    serve,
    status_of,
    ulid,
    usable_slot,
    usable_texts,
    valid_derived_texts,
)

OLD = ("12 May", "12 мая", "05-12")
FIX_WORDS = "нет, у Саши день рождения 14 мая"


def _world(conn):
    sasha, dima = person("Саша"), person("Дима")
    raw = add_raw(conn, "bday", "у Саши день рождения 12 мая", chat_id=None, turn_id="turn-0")
    bday = fact(
        "sasha-bday",
        sasha,
        "birthday",
        "12 May",
        "Sasha's birthday is 12 May",
        "у Саши день рождения 12 мая",
        source_raw_id=raw,
        value_norm="05-12",
    )
    city = fact("dima-city", dima, "lives_in", "Tver", "Dima lives in Tver", "Дима в Твери")
    seed(conn, sasha, dima, bday, city, extra=lambda sql: lineage(sql, bday.id, raw))
    return sasha, dima, bday, city


def _touch(ctx: store.CommitCtx, op: ProposalOp) -> list[str]:
    """The worker's applier: a non-content consolidation note (importance), so a committed
    group is visible without changing what the fix changed."""
    ctx.conn.execute("UPDATE records SET importance = 0.8 WHERE id = ?", (op.target,))
    return [op.target]


def _old_value_absent(conn) -> None:
    for text in usable_texts(conn) + valid_derived_texts(conn):
        assert not any(o in text for o in OLD), text


def test_fix_during_worker(db_path, conn):
    """§7.5: the fix lands between the worker's snapshot and its commit."""
    sasha, _dima, bday, city = _world(conn)
    f0, c0 = ref(conn, bday.id), ref(conn, city.id)
    serve(conn, thread="thread-owner", session="session-owner", item=f0, turn_id="turn-0")
    serve(conn, thread="thread-task", session="session-task", item=f0, via="core")
    summary = add_derived(conn, "day", "Sasha's birthday is 12 May. Dima lives in Tver.", [f0, c0])

    snapshot_taken, fixed = threading.Event(), threading.Event()

    def worker():
        w = open_db(db_path)
        try:
            proposal = Proposal(
                ops=[
                    ProposalOp(op=OpKind.FLAG, target=bday.id, group=sasha.id, read_set=[f0]),
                    ProposalOp(op=OpKind.FLAG, target=city.id, group="dima", read_set=[c0]),
                ],
                model_id="gold-extractor",
            )
            snapshot_taken.set()
            assert fixed.wait(WAIT)
            return store.commit(w, proposal, _touch)
        finally:
            w.close()

    def hot():
        assert snapshot_taken.wait(WAIT)
        h = open_db(db_path)
        try:
            return api.fix(
                handle(bday.id),
                "у Саши день рождения 14 мая",
                turn=owner_turn(FIX_WORDS, n=2),
                because_quote="у Саши день рождения 14 мая",
                store=h,
            )
        finally:
            h.close()
            fixed.set()

    work, result = run_threads(worker, hot)

    assert result.outcome == "committed" and result.record_ids
    assert result.memory_version is not None and result.memory_version > f0.version
    # The worker's group that read the old version is rejected; its other group commits.
    assert work.committed == ["dima"] and list(work.stale) == [sasha.id]
    assert f0 in work.stale[sasha.id]
    # The target is retired, the slot holds the new value only.
    assert status_of(conn, bday.id) in ("superseded", "retracted")
    slot = usable_slot(conn, sasha.id, "birthday")
    assert len(slot) == 1 and "14" in (slot[0][1] or "")
    _old_value_absent(conn)
    # The summary that read it is invalid and excluded (§7.3).
    assert conn.execute("SELECT valid FROM derived WHERE id = ?", (summary,)).fetchone()[0] == 0
    # Any prepared prompt holding the old version is refused (§7.4) ...
    pre = api.pre_send_check(thread_id="thread-task", read_set=[f0, c0], store=conn)
    assert not pre.ok and f0 in pre.stale
    pre_summary = api.pre_send_check(
        thread_id="thread-task", read_set=[derived_ref(summary)], store=conn
    )
    assert not pre_summary.ok
    # ... and every thread it was served to owes an update block before its next call (§7.6).
    owed = {
        r[0]
        for r in conn.execute(
            "SELECT thread_id FROM pending_updates WHERE item_id = ? AND old_version = ?",
            (bday.id, f0.version),
        ).fetchall()
    }
    assert "thread-task" in owed
    gate = api.pre_send_check(thread_id="thread-task", read_set=[], store=conn)
    assert not gate.ok and gate.undelivered_updates >= 1
    kinds = [r[0] for r in conn.execute("SELECT kind FROM events ORDER BY id").fetchall()]
    assert "fix" in kinds


def test_fix_during_worker_update_block_reaches_task_session(db_path, conn):
    """§7.5/§7.6: the task session that holds the old value in its transcript gets
    `[m:<id>] outdated: now … not …` (or a restart) before its next call; after delivery its
    gate opens and its served row is at the new version."""
    _sasha, _dima, bday, _city = _world(conn)
    f0 = ref(conn, bday.id)
    serve(conn, thread="thread-task", session="session-task", item=f0, via="core")
    fixed = api.fix(
        handle(bday.id),
        "у Саши день рождения 14 мая",
        turn=owner_turn(FIX_WORDS, n=2),
        because_quote="у Саши день рождения 14 мая",
        store=conn,
    )
    assert fixed.outcome == "committed"
    upd = api.update_block(thread_id="thread-task", session_id="session-task", store=conn)
    assert upd.restart or (
        upd.block is not None
        and f"[m:{bday.id}]" in upd.block
        and "14" in upd.block
        and ("outdated" in upd.block or "retracted" in upd.block)
    )
    after = api.pre_send_check(thread_id="thread-task", read_set=[], store=conn)
    assert after.ok and after.undelivered_updates == 0


def test_what_after_fix_during_worker_shows_only_the_new_value(db_path, conn):
    """§7.5 «island and Telegram bot» read through the inspection commands: «что ты помнишь
    про Сашу» after the race carries no unmarked old value."""
    sasha, _dima, bday, _city = _world(conn)
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
    api.fix(
        handle(bday.id),
        "у Саши день рождения 14 мая",
        turn=owner_turn(FIX_WORDS, n=2),
        because_quote="у Саши день рождения 14 мая",
        store=conn,
    )
    answer = api.what("Саша", turn=owner_turn("что ты помнишь про Сашу", n=3), store=conn)
    texts = [i.text for i in answer.items + answer.competing_claims]
    assert any("14" in t for t in texts)
    assert not any(o in t for t in texts for o in OLD)


@not_built("hands")
def test_corrected_address_denied_to_hands_while_old_one_sits_in_history(db_path, conn):
    """§16.3 (covers actions): after «исправь почту Саши», a hand step carrying the old
    address — planted in history before the fix — is denied and offered the current one."""
    sasha = person("Саша")
    mail = fact(
        "sasha-mail",
        sasha,
        "has_email",
        "sasha@old.example",
        "Sasha's email is sasha@old.example",
        "почта Саши sasha@old.example",
    )
    seed(conn, sasha, mail)
    words = "исправь: почта Саши теперь sasha@new.example"
    api.fix(
        handle(mail.id),
        "почта Саши теперь sasha@new.example",
        turn=owner_turn(words, n=2),
        because_quote="почта Саши теперь sasha@new.example",
        store=conn,
    )
    verdict = api.check_action(
        "mail_send",
        {"to": "sasha@old.example", "body": "Привет!"},
        turn=owner_turn("напиши Саше привет", n=3),
        store=conn,
    )
    assert not verdict.allow
    assert verdict.current_value == "sasha@new.example"


def test_label_added_after_basket_built_drops_the_item(db_path, conn):
    """§7.1, §7.4: a worker adds a label (version-bearing) after the basket's read set was
    taken; the pre-send check drops that item; a concurrent heat bump on the other does not."""
    bare_fact(conn, "bday", "12 May")
    bare_fact(conn, "city", "Tver")
    basket = [ReadRef(id="bday", version=0), ReadRef(id="city", version=0)]
    assert api.pre_send_check(thread_id="thread-owner", read_set=basket, store=conn).ok

    def add_label(ctx: store.CommitCtx, op: ProposalOp) -> list[str]:
        ctx.conn.execute(
            "UPDATE records SET labels = labels | ?, version = ? WHERE id = ?",
            (int(Label.HEALTH), ctx.version, op.target),
        )
        return [op.target]

    flag = ProposalOp(op=OpKind.FLAG, target="city", group="dima", read_set=[basket[1]])
    assert store.commit(conn, Proposal(ops=[flag]), add_label).ok
    conn.execute(
        "INSERT INTO record_stats (record_id, uses, last_used_at, heat) VALUES ('bday', 3, ?, 2.0)",
        (TS0,),
    )
    pre = api.pre_send_check(thread_id="thread-owner", read_set=basket, store=conn)
    assert not pre.ok
    assert pre.stale == [ReadRef(id="city", version=0)]
