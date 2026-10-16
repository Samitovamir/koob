"""Adversarial review tests: forget completeness and restore (G5).

Each test covers a defect found in the earlier code (strict-xfail under `fix` until fixed);
the docstring names the finding in the design notes."""

from __future__ import annotations

from koob.memory import backup, forget, store
from koob.memory.model import ForgetScope, OpKind

from .._ledger import add, commit, entity, fact, op, open_seeded, raw
from ..forget import _seed as s

clock = s.clock  # fixture: a manual clock wired into forget


def test_retracted_value_leaves_with_the_forget(tmp_path, clock):
    """R-F1 (P0, G5): a retraction writes the old value in plain text to
    `raw_retractions`; neither stop-use nor purge touches that table, so after «забудь
    насовсем» the value survives, and the purge either reports «удалил полностью» (short
    values are not scan keys) or never finishes (long values)."""
    conn = open_seeded(tmp_path)
    raw(conn, "raw-1", "Мой адрес Рим, кв 7")
    assert add(conn, entity("e-me", "Denis")).ok
    assert add(
        conn,
        fact(
            "f-addr",
            "e-me",
            "home_address",
            "Rome",
            text="Denis lives in Rome",
            source_raw_id="raw-1",
            quote_orig="Мой адрес Рим",
        ),
    ).ok
    assert commit(conn, op(OpKind.RETRACT, target="f-addr", group="e-me")).ok
    fid = forget.stop_use(conn, ["f-addr"], scope=ForgetScope.ITEM, forever=True)
    assert forget.purge(conn, fid) is True
    left = [r[0] for r in conn.execute("SELECT retracted_value_norm FROM raw_retractions")]
    assert left == [], f"forgotten value still stored in raw_retractions: {left}"


def test_long_retracted_value_does_not_block_the_purge(tmp_path, clock):
    """R-F1 (P0, G5), the other face: a value of 6+ characters is a scan key, the full-table
    scan finds it in `raw_retractions`, and the purge retries forever («удаляю»)."""
    conn = open_seeded(tmp_path)
    raw(conn, "raw-1", "Мой адрес Ленина 5, кв 7")
    assert add(conn, entity("e-me", "Denis")).ok
    assert add(
        conn,
        fact(
            "f-addr",
            "e-me",
            "home_address",
            "Lenina 5",
            text="Denis lives at Lenina 5",
            source_raw_id="raw-1",
            quote_orig="Мой адрес Ленина 5",
        ),
    ).ok
    assert commit(conn, op(OpKind.RETRACT, target="f-addr", group="e-me")).ok
    fid = forget.stop_use(conn, ["f-addr"], scope=ForgetScope.ITEM, forever=True)
    assert forget.purge(conn, fid) is True
    assert forget.status(conn, fid) == "purged"


def test_restore_does_not_forget_records_that_only_shared_a_redacted_raw(tmp_path, clock):
    """R-F3 (P1, data loss): forgetting one fact of a two-fact message only redacts its span
    (§9.6) and keeps the other fact, but the journal lists the redacted raw's HMAC like a
    forgotten one. Restoring an older backup (or the nightly rewrite of that backup) replays
    the HMAC as an ITEM forget of the whole raw, and `_resolve_targets` takes every record
    learned from it: the fact the user never asked to forget is gone after the restore."""
    db = tmp_path / "data" / "memory" / "memory.db"
    conn = store.open(db, home=tmp_path / "home")
    s.relation(conn, "lives_in", "residence")
    s.relation(conn, "likes", "preference", "multi")
    s.entity(conn, "e-sasha", "Саша", [("Саша", "name")])
    s.entity(conn, "e-lena", "Лена", [("Лена", "name")])
    s.raw(conn, "raw-1", "Саша живёт в Берлине, а Лена любит джаз", turn="7")
    s.fact(
        conn, "f-1", "e-sasha", "lives_in", value="Berlin",
        text_en="Sasha lives in Berlin", quote="Саша живёт в Берлине", raw="raw-1",
    )  # fmt: skip
    s.fact(
        conn, "f-2", "e-lena", "likes", value="jazz",
        text_en="Lena likes jazz", quote="Лена любит джаз", raw="raw-1",
    )  # fmt: skip
    conn.commit()
    old_set = backup.make_set(conn, clock=clock)
    clock.advance(3600)
    forget.stop_use(conn, ["f-1"], scope=ForgetScope.ITEM, forever=False)
    # live: the span is redacted, the other fact stays
    assert conn.execute("SELECT status FROM records WHERE id = 'f-2'").fetchone()[0] == "active"
    conn.close()

    backup.restore(old_set, db, home=tmp_path / "home")
    conn = store.open(db, home=tmp_path / "home")
    try:
        assert conn.execute("SELECT status FROM records WHERE id = 'f-1'").fetchone()[0] in (
            "forgotten",
            "purged",
        )
        assert conn.execute("SELECT status FROM records WHERE id = 'f-2'").fetchone()[0] == (
            "active"
        ), "restore forgot a fact he never asked to forget"
    finally:
        conn.close()


def test_forget_about_x_does_not_mask_other_names_that_start_with_xs_name(mem, clock):
    """R-F4 (P1, data corruption): «забудь всё про Макса» masks X's aliases in other records
    with a plain substring replace (`_mask_text`), and picks those records with a word-start
    prefix match (`compile_keys`). A record about Максим (a different person) becomes
    «[забыто]им работает в Google»."""
    s.relation(mem, "works_at", "employment")
    s.entity(mem, "e-max", "Макс", [("Макс", "name"), ("Max", "name")])
    s.entity(mem, "e-maxim", "Максим", [("Максим", "name"), ("Maxim", "name")])
    s.fact(
        mem, "f-max", "e-max", "works_at", value="Yandex",
        text_en="Max works at Yandex", quote="Макс работает в Яндексе",
    )  # fmt: skip
    s.fact(
        mem, "f-maxim", "e-maxim", "works_at", value="Google",
        text_en="Maxim works at Google", quote="Максим работает в Google",
    )  # fmt: skip
    forget.stop_use(mem, ["e-max"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    row = mem.execute("SELECT text_en, quote_orig FROM facts WHERE id = 'f-maxim'").fetchone()
    assert tuple(row) == ("Maxim works at Google", "Максим работает в Google")


def test_undo_does_not_bring_back_a_chat_banned_since(mem, clock):
    """R-F5 (P1, G1): «Вернуть» restores trash payloads unconditionally. A message forgotten
    into the 24 h trash, then its chat put under a full ban («ничего, никогда»), comes back
    as active raw with its fact on «Вернуть»: the ban's own forget skipped it because it was
    already forgotten."""
    from koob.memory import api
    from koob.memory.model import ChatMode, TurnCtx

    s.relation(mem, "moving_to", "residence")
    s.entity(mem, "e-gleb", "Глеб", [("Глеб", "name")])
    s.raw(mem, "raw-1", "Глеб переезжает в Тверь", chat="tg:gleb", tier="other")
    s.fact(
        mem, "f-1", "e-gleb", "moving_to", value="Tver",
        text_en="Gleb moves to Tver", quote="Глеб переезжает в Тверь", raw="raw-1", tier="other",
    )  # fmt: skip
    fid = forget.stop_use(mem, ["f-1"], scope=ForgetScope.ITEM, forever=False)
    owner = TurnCtx(turn_id="t-2", thread_id="th", owner_words="полный запрет на чат с Глебом")
    api.set_chat_mode("tg:gleb", ChatMode.BANNED, turn=owner, store=mem)
    forget.undo(mem, fid)
    raw_status = mem.execute("SELECT status FROM raw_chunks WHERE id = 'raw-1'").fetchone()[0]
    assert raw_status != "active", "a banned chat's message is active memory again"


def test_undo_of_an_item_does_not_reverse_a_later_forget_all_about_x(mem, clock):
    """R-F6 (P1, G5): «забудь всё про Сашу» skips records already in the trash (its plan
    takes only live ones), so it writes no suppression for them and does not take over
    their trash. «Вернуть» on the earlier, narrower forget then restores a fact about her
    after the user asked to forget everything about her."""
    s.relation(mem, "moving_to", "residence")
    s.relation(mem, "works_at", "employment")
    s.entity(mem, "e-sasha", "Саша", [("Саша", "name")])
    s.fact(mem, "f-move", "e-sasha", "moving_to", value="Tver", text_en="Sasha moves to Tver")
    s.fact(mem, "f-job", "e-sasha", "works_at", value="Yandex", text_en="Sasha works at Yandex")
    first = forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)
    forget.stop_use(mem, ["e-sasha"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    forget.undo(mem, first)
    status = mem.execute("SELECT status FROM records WHERE id = 'f-move'").fetchone()[0]
    assert status in ("forgotten", "purged"), "a fact about Саша is back after «всё про Сашу»"


def test_purge_clears_the_source_ref_of_a_forgotten_message(mem, clock):
    """R-F7 (P2, G5): stop-use and purge blank `raw_chunks.text` but keep `source_ref`
    (a file path, page URL or subject line), and the purge scan never looks at it. A
    forgotten file «Анализы Саши — ВИЧ.pdf» stays named in memory.db forever."""
    s.raw(mem, "raw-1", "Результаты анализов: положительно")
    mem.execute(
        "UPDATE raw_chunks SET source_kind = 'file', source_ref = ? WHERE id = 'raw-1'",
        ("~/Downloads/Анализы Саши — ВИЧ.pdf",),
    )
    fid = forget.stop_use(mem, ["raw-1"], scope=ForgetScope.ITEM, forever=True)
    assert forget.purge(mem, fid) is True
    ref = mem.execute("SELECT source_ref FROM raw_chunks WHERE id = 'raw-1'").fetchone()[0]
    assert ref is None or "ВИЧ" not in ref


def test_undo_of_the_wider_forget_hands_its_claims_back(mem, clock):
    """R-F6, the other order: «Вернуть» on «всё про Сашу» gives the earlier item forget its
    trash entry back (still forgotten, suppression kept); «Вернуть» on that one then
    restores it."""
    s.relation(mem, "moving_to", "residence")
    s.relation(mem, "works_at", "employment")
    s.entity(mem, "e-sasha", "Саша", [("Саша", "name")])
    s.fact(mem, "f-move", "e-sasha", "moving_to", value="Tver", text_en="Sasha moves to Tver")
    s.fact(mem, "f-job", "e-sasha", "works_at", value="Yandex", text_en="Sasha works at Yandex")
    first = forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)
    wide = forget.stop_use(mem, ["e-sasha"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    forget.undo(mem, wide)
    status = mem.execute("SELECT status FROM records WHERE id = 'f-move'").fetchone()[0]
    assert status == "forgotten"
    assert mem.execute("SELECT status FROM records WHERE id = 'f-job'").fetchone()[0] == "active"
    held = mem.execute("SELECT count(*) FROM suppressions WHERE forget_id = ?", (first,))
    assert held.fetchone()[0] >= 1
    forget.undo(mem, first)
    assert mem.execute("SELECT status FROM records WHERE id = 'f-move'").fetchone()[0] == "active"
    text = mem.execute("SELECT text_en FROM facts WHERE id = 'f-move'").fetchone()[0]
    assert text == "Sasha moves to Tver"
