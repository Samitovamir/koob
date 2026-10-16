"""the memory spec §9.1 (A14), §9.6: «забудь» means the fact, not one copy. A forget takes
every record holding the same proposition (subject entity + relation family + normalized value)
in any status and from any source; a different value is another fact (a successor stays).
Asking happens only when the candidates really differ.
Spec marker: `[T forget/test_duplicates::<case>]`."""

from __future__ import annotations

from koob.memory import api, forget
from koob.memory.model import ForgetScope, TurnCtx, TurnOrigin

from . import _seed as s

clock = s.clock  # fixture

RIGA_HIS = "Саша живёт в Риге"
RIGA_FRIEND = "Петя: кстати, Саша живёт в Риге"
TALLINN = "Петя: Саша переехал в Таллин"


def _status(mem, rid: str) -> str:
    return mem.execute("SELECT status FROM records WHERE id = ?", (rid,)).fetchone()[0]


def _raw_status(mem, rid: str) -> str:
    return mem.execute("SELECT status FROM raw_chunks WHERE id = ?", (rid,)).fetchone()[0]


def _turn(words: str, n: int = 1) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id="thread-owner",
        session_id="session-owner",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        message_id=f"msg-{n}",
    )


def _world(mem, *, friend_status: str = "retracted") -> None:
    """Саша lives in Riga, twice: the user's own word (f-his, active) and Петя's claim from a chat
    (f-friend). Петя later said Tallinn: f-tln superseded f-friend (a successor, another value)."""
    s.relation(mem, "lives_in", "residence")
    s.entity(mem, "e-sasha", "Саша", [("Саша", "name"), ("Sasha", "translit")])
    s.entity(mem, "e-petya", "Петя", [("Петя", "name")])
    s.raw(mem, "raw-his", RIGA_HIS, turn="t-his")
    s.raw(mem, "raw-friend", RIGA_FRIEND, chat="chat-petya", tier="other")
    s.raw(mem, "raw-tln", TALLINN, chat="chat-petya", tier="other")
    for rid, raw_id, city, ru, quote, tier, speaker in (
        ("f-his", "raw-his", "Riga", "Рига", RIGA_HIS, "user", None),
        ("f-friend", "raw-friend", "Riga", "Рига", "Саша живёт в Риге", "other", "e-petya"),
        ("f-tln", "raw-tln", "Tallinn", "Таллин", "Саша переехал в Таллин", "other", "e-petya"),
    ):
        s.fact(
            mem, rid, "e-sasha", "lives_in", value=city, text_en=f"Sasha lives in {city}",
            quote=quote, raw=raw_id, keywords_ru=f"{ru} живёт", keywords_en=city.casefold(),
            tier=tier, speaker=speaker,
        )  # fmt: skip
    mem.execute("UPDATE records SET status = ? WHERE id = 'f-friend'", (friend_status,))
    s.edge(mem, "f-tln", "f-friend", "supersedes")
    s.derived(mem, "d-friend", "Today: Sasha lives in Riga (Petya says)", [("f-friend", 1)])
    mem.commit()


def test_forever_takes_every_copy_and_purges_both(mem, clock):
    """«забудь насовсем, что Саша живёт в Риге»: the user's record and Петя's retracted twin stop
    being used at once and both purge; Tallinn (a successor, another value) stays."""
    _world(mem)
    words = "забудь насовсем, что Саша живёт в Риге"
    res = api.forget(words, turn=_turn(words), because_quote=words, forever=True, store=mem)
    assert res.outcome == "stopped", res
    assert {"f-his", "f-friend"} <= set(res.record_ids)
    assert "f-tln" not in res.record_ids
    for rid in ("f-his", "f-friend"):
        assert _status(mem, rid) == "forgotten", rid
    for raw_id in ("raw-his", "raw-friend"):
        assert _raw_status(mem, raw_id) == "forgotten", raw_id
    # a derived item built from the twin is invalidated as for the target (§9.1a)
    assert mem.execute("SELECT valid FROM derived WHERE id = 'd-friend'").fetchone()[0] == 0
    # A14: never a successor, and never one with another value
    assert _status(mem, "f-tln") == "active" and _raw_status(mem, "raw-tln") == "active"

    assert forget.due(mem) == [res.forget_id]
    assert forget.purge(mem, res.forget_id) is True
    for rid in ("f-his", "f-friend"):
        assert _status(mem, rid) == "purged", rid
    texts = [r[0] for r in mem.execute("SELECT text_en FROM facts") if r[0]]
    assert not any("Riga" in t for t in texts), texts
    assert any("Tallinn" in t for t in texts)


def test_handle_target_takes_its_twin(mem, clock):
    """A forget aimed at one copy by handle takes the other copies too (stop-use, 24 h)."""
    _world(mem)
    fid = forget.stop_use(mem, ["f-friend"], scope=ForgetScope.ITEM, forever=False)
    assert _status(mem, "f-friend") == "forgotten" and _status(mem, "f-his") == "forgotten"
    assert _status(mem, "f-tln") == "active"
    assert forget.status(mem, fid) == "in_trash"


def test_identical_candidates_are_one_target(mem, clock):
    """Two active copies of one fact are not a question: one target, both forgotten."""
    _world(mem, friend_status="active")
    words = "забудь, что Саша живёт в Риге"
    res = api.forget(words, turn=_turn(words), because_quote=words, store=mem)
    assert res.outcome == "stopped", res
    assert {"f-his", "f-friend"} <= set(res.record_ids)
    assert "2 записи" not in res.line  # one fact, said once


def test_different_candidates_still_ask(mem, clock):
    """Candidates that really differ (Riga / Tallinn on the same words) still get one question."""
    _world(mem, friend_status="active")
    mem.execute("UPDATE records SET status = 'active' WHERE id = 'f-tln'")
    mem.execute("DELETE FROM edges WHERE etype = 'supersedes'")
    mem.execute("UPDATE facts SET keywords_ru = 'Рига Таллин живёт' WHERE id = 'f-tln'")
    mem.commit()
    words = "забудь, что Саша живёт"
    res = api.forget(words, turn=_turn(words), because_quote=words, store=mem)
    assert res.outcome == "ask_first", res
    assert all(_status(mem, r) == "active" for r in ("f-his", "f-friend", "f-tln"))


def test_undo_brings_every_copy_back(mem, clock):
    """«Вернуть» restores both copies to their own statuses (active, retracted)."""
    _world(mem)
    words = "забудь, что Саша живёт в Риге"
    res = api.forget(words, turn=_turn(words), because_quote=words, store=mem)
    assert res.outcome == "stopped", res
    back = api.undo_forget(res.forget_id, turn=_turn("верни", n=2), store=mem)
    assert back.outcome != "not_found", back
    assert _status(mem, "f-his") == "active" and _status(mem, "f-friend") == "retracted"
    assert _raw_status(mem, "raw-his") == "active" and _raw_status(mem, "raw-friend") == "active"
    assert _status(mem, "f-tln") == "active"


def test_copy_in_an_earlier_trash_is_claimed(mem, clock):
    """A copy an earlier forget still holds in its trash goes with «насовсем» too: the new
    forget claims it, so the purge covers every copy."""
    _world(mem)
    first = forget.stop_use(mem, ["f-his"], scope=ForgetScope.ITEM, forever=False)
    # the user's new statement after the forget is admitted (§9.6): the same fact, a new record
    s.raw(mem, "raw-again", "Саша всё ещё живёт в Риге", turn="t-again")
    s.fact(
        mem, "f-again", "e-sasha", "lives_in", value="Riga", text_en="Sasha lives in Riga",
        quote="Саша всё ещё живёт в Риге", raw="raw-again", keywords_ru="Рига живёт",
    )  # fmt: skip
    mem.commit()
    second = forget.stop_use(mem, ["f-again"], scope=ForgetScope.ITEM, forever=True)
    assert forget.purge(mem, second) is True
    for rid in ("f-his", "f-friend", "f-again"):
        assert _status(mem, rid) == "purged", rid
    del first
