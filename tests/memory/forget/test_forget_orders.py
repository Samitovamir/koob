"""Forget and cancel orders: a forgotten or cancelled value must never show again. Covers the
user's forget in any colloquial form, a cancel that retires the plan it names, every kind minted
from the same words retired at once, the order's own text never served back; undo, «насовсем»
and a restore of an older backup keep it. Synthetic names and wordings, Russian and English."""

from __future__ import annotations

import pytest

from koob.memory import api, backup, forget, store
from koob.memory.model import TurnCtx, TurnOrigin
from koob.memory.recall import safe_text
from koob.memory.writers import hot

from . import _seed as s

clock = s.clock  # fixture


def _turn(words: str, n: int = 1) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id="thread-forget",
        session_id="session-forget",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        message_id=f"msg-{n}",
    )


def _status(mem, rid: str) -> str:
    return mem.execute("SELECT status FROM records WHERE id = ?", (rid,)).fetchone()[0]


def _plan_record(mem, rid: str, summary: str, keywords: str, quote: str, raw: str) -> None:
    """An episode (a plan) minted from `raw`, with the user's Russian words in its keywords."""
    s.record(mem, rid, "episode", origin_quote=quote, fts=(summary.casefold(), keywords, ""))
    mem.execute("INSERT INTO episodes (id, summary_en) VALUES (?, ?)", (rid, summary))
    s.edge(mem, rid, raw, "derived_from")


def _world(mem) -> None:
    """Arseny sold his boat (fact + episode from the same words, one message that also says
    he got a dog); Veronika has a meeting with the user on Friday (a plan) and works at a bakery."""
    s.relation(mem, "sold", "possession")
    s.relation(mem, "has_pet", "pet")
    s.relation(mem, "works_at", "employer")
    s.entity(mem, "e-ars", "Арсений", [("Арсений", "name"), ("Arseny", "translit")])
    s.entity(mem, "e-ver", "Вероника", [("Вероника", "name"), ("Veronika", "translit")])
    msg = "Арсений продал свою лодку. А ещё он завёл щенка."
    s.raw(mem, "raw-ars", msg)
    s.fact(
        mem, "f-boat", "e-ars", "sold", value="boat", text_en="Arseny sold his boat",
        quote="Арсений продал свою лодку", raw="raw-ars", keywords_ru="лодка продал",
    )  # fmt: skip
    # the episode's own words are English only: the user's Russian order can reach it by its quote
    _plan_record(mem, "ep-boat", "Arseny sold the boat he owned", "", msg[:25], "raw-ars")
    mem.execute(
        "INSERT INTO episode_links (episode_id, role, target_id) VALUES ('ep-boat', 'about', 'e-ars')"
    )
    s.fact(
        mem, "f-dog", "e-ars", "has_pet", value="puppy", text_en="Arseny got a puppy",
        quote="А ещё он завёл щенка", raw="raw-ars", keywords_ru="щенок завёл",
    )  # fmt: skip
    s.raw(mem, "raw-ver", "В пятницу встреча с Вероникой по смете. Вероника работает в пекарне.")
    _plan_record(
        mem, "ep-meet", "Meeting with Veronika about the estimate on Friday",
        "встреча смета пятница", "В пятницу встреча с Вероникой по смете", "raw-ver",
    )  # fmt: skip
    mem.execute(
        "INSERT INTO episode_links (episode_id, role, target_id) VALUES ('ep-meet', 'about', 'e-ver')"
    )
    s.fact(
        mem, "f-bakery", "e-ver", "works_at", value="bakery", text_en="Veronika works at a bakery",
        quote="Вероника работает в пекарне", raw="raw-ver", keywords_ru="пекарня работает",
    )  # fmt: skip
    mem.commit()


@pytest.mark.parametrize(
    "words",
    [
        "Выкинь из головы, что Арсений продал лодку.",
        "Не держи в голове, что Арсений продал лодку",
        "Вычеркни, что Арсений продал лодку",
        "Сотри из памяти про лодку Арсения",
        "stop keeping in mind that Arseny sold his boat",
        "Please get it out of your head that Arseny sold the boat",
    ],
)
def test_colloquial_forget_retires_every_kind_from_the_same_words(mem, clock, words):
    """Any colloquial form is «забудь»: the fact and the episode minted from the same words go
    at once; the other statement of that message and another person's records stay."""
    _world(mem)
    res = api.forget(words, turn=_turn(words), because_quote=words, store=mem)
    assert res.outcome == "stopped", res.line
    assert _status(mem, "f-boat") == "forgotten"
    assert _status(mem, "ep-boat") == "forgotten"
    for kept in ("f-dog", "ep-meet", "f-bakery", "e-ver"):
        assert _status(mem, kept) == "active", kept


def test_negated_forms_are_not_orders():
    for words in ("Не забудь купить хлеб", "don't forget the keys", "I forget his name",
                  "не отменяй ужин", "don't cancel the trip", "Встречу отменили вчера"):  # fmt: skip
        assert safe_text.order_kind(words) is None, words


def test_canon_keeps_the_object():
    assert safe_text.canon_order("Не держи это в голове") == "забудь это"
    assert safe_text.canon_order("stop keeping the gate code in mind") == "forget the gate code"
    assert safe_text.canon_order("Scratch that.") == "forget that."
    assert safe_text.canon_order("Отмени встречу с Олей") == "забудь встречу с Олей"


def test_cancel_retires_the_plan_it_names(mem, clock):
    """«Отмени встречу с Вероникой»: the meeting goes; her job (no plan) stays."""
    _world(mem)
    words = "Отмени встречу с Вероникой в пятницу"
    res = api.forget(words, turn=_turn(words), because_quote=words, store=mem)
    assert res.outcome == "stopped", res.line
    assert _status(mem, "ep-meet") == "forgotten"
    assert _status(mem, "f-bakery") == "active"
    assert _status(mem, "e-ver") == "active"


def test_scratch_that_takes_back_his_last_message(mem, clock):
    """An object-less "scratch that" with no forget before it is the user's last message, this
    sitting: its facts go; an older message of the user's stays."""
    _world(mem)
    now = s.now_iso(clock)
    s.raw(mem, "raw-new", "Тимофей переезжает в Тверь", turn="turn-0", ts=now)
    s.relation(mem, "moves_to", "residence")
    s.entity(mem, "e-tim", "Тимофей", [("Тимофей", "name")])
    s.fact(
        mem, "f-tver", "e-tim", "moves_to", value="Tver", text_en="Timofey moves to Tver",
        quote="Тимофей переезжает в Тверь", raw="raw-new",
    )  # fmt: skip
    mem.execute("UPDATE raw_chunks SET source_kind = 'owner' WHERE id = 'raw-new'")
    mem.commit()
    res = api.forget("Scratch that.", turn=_turn("Scratch that.", 2), because_quote="x", store=mem)
    assert res.outcome == "stopped", res.line
    assert _status(mem, "f-tver") == "forgotten"
    assert _status(mem, "f-boat") == "active"


def test_the_order_is_never_served_back():
    """`clean` (every fenced read) cuts the order's sentence for the user's own tier too; the rest
    of the message stays."""
    out = safe_text.clean("Тимофей вернулся. Про его долг выкинь из головы, неважно.", tier="user")
    assert "долг" not in out and "Тимофей вернулся." in out
    assert "bonus" not in safe_text.clean("drop what i said about the bonus", tier="user")
    assert safe_text.clean("Не забудь про сверку", tier="user") == "Не забудь про сверку"


@pytest.mark.parametrize(
    "words",
    [
        "Выкинь из головы про Арсения",
        "не держи это в голове",
        "scratch that",
        "drop it",
        "Отмени встречу с Вероникой",
    ],  # fmt: skip
)
def test_safety_net_hears_the_colloquial_forms(words):
    assert "forget" in hot.detect(_turn(words), "")


@pytest.mark.parametrize("words", ["Не забудь позвонить Веронике", "don't forget the meeting"])
def test_safety_net_ignores_reminders(words):
    assert "forget" not in hot.detect(_turn(words), "")


@pytest.mark.parametrize(
    "again", ["Ладно, вычеркни навсегда.", "OK, scratch that for good.", "drop it once and for all"]
)
def test_undo_then_forever_then_restore_keeps_every_kind_gone(tmp_path, clock, again):
    """Forget → «верни» brings both kinds back → «вычеркни навсегда» (object-less) forgets
    them again for good → a restore of the backup taken before stays forgotten."""
    paths = store.MemoryPaths(tmp_path / "data" / "memory")
    home = tmp_path / "home"
    conn = store.open(paths.db, home=home)
    _world(conn)
    old_set = backup.make_set(conn, clock=clock)
    clock.advance(3600)
    words = "Выкинь из головы, что Арсений продал лодку"
    res = api.forget(words, turn=_turn(words), because_quote=words, store=conn)
    assert res.outcome == "stopped", res.line
    back = api.undo_forget(res.forget_id, turn=_turn("верни про лодку", 2), store=conn)
    assert back.outcome == "undone"
    assert _status(conn, "f-boat") == "active" and _status(conn, "ep-boat") == "active"
    res2 = api.forget(again, turn=_turn(again, 3), because_quote=again, forever=True, store=conn)
    assert res2.outcome == "stopped", res2.line
    assert res2.forget_id in forget.due(conn)
    assert _status(conn, "f-boat") == "forgotten" and _status(conn, "ep-boat") == "forgotten"
    conn.close()
    backup.restore(old_set, paths.db, home=home)
    conn = store.open(paths.db, home=home)
    try:
        for rid in ("f-boat", "ep-boat"):
            assert _status(conn, rid) in ("forgotten", "purged"), rid
        for kept in ("f-dog", "ep-meet", "f-bakery"):
            assert _status(conn, kept) == "active", kept
    finally:
        conn.close()


def test_the_order_turn_cannot_bring_the_value_back(mem, clock):
    """The «забудь» turn is ingested after the forget and is newer than it: what its
    extraction repeats of the forgotten value is the order, not the user's new word; a plain
    new message from the user with the same value is admitted."""
    _world(mem)
    words = "Выкинь из головы, что Арсений продал лодку"
    res = api.forget(words, turn=_turn(words), because_quote=words, store=mem)
    assert res.outcome == "stopped", res.line
    clock.advance(30)
    order_ts = s.now_iso(clock)
    s.raw(mem, "raw-order", words, turn="turn-1", ts=order_ts)
    old = dict(subject_id="e-ars", relation="sold", value_norm="boat")
    assert forget.blocked_by(mem, source_ts=order_ts, author_tier="user", **old) == res.forget_id
    clock.advance(600)
    later = s.now_iso(clock)
    s.raw(mem, "raw-again", "Арсений всё-таки продал лодку соседу", turn="turn-5", ts=later)
    assert forget.blocked_by(mem, source_ts=later, author_tier="user", **old) is None
