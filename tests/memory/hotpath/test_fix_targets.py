"""the memory spec §6.1 (v0.4): fix targets in history, structured hot writes.
Spec marker: `[T hotpath/test_fix_targets::<case>]`."""

from __future__ import annotations

from koob.memory import api
from koob.memory.model import OWNER_ENTITY_ID, Assertion

from ..races._kit import handle, status_of, usable_slot
from ._world import fact_row, sasha_world, turn_of


def test_superseded_target_retargets_head(mem):
    """A second «исправь» aimed at an already superseded id corrects the slot's current head;
    one usable value remains, the newest."""
    sasha, bday = sasha_world(mem)
    q1, q2 = "у Саши день рождения 14 мая", "у Саши день рождения 15 мая"
    a = api.fix(handle(bday.id), q1, turn=turn_of(f"нет, {q1}", n=2), because_quote=q1, store=mem)
    assert a.outcome == "committed" and a.line and "14" in a.line
    b = api.fix(handle(bday.id), q2, turn=turn_of(f"нет, {q2}", n=3), because_quote=q2, store=mem)
    assert b.outcome == "committed"
    assert status_of(mem, a.record_ids[0]) == "superseded"
    slot = usable_slot(mem, sasha.id, "birthday")
    assert [tuple(r) for r in slot] == [(b.record_ids[0], "15 мая")]
    assert fact_row(mem, b.record_ids[0])["value_norm"] == "05-15"
    assert b.memory_version > a.memory_version
    # the eval harness: the user's whole message as both target and quote
    msg = "нет, у Саши день рождения 16 мая"
    c = api.fix(msg, msg, turn=turn_of(msg, n=4), because_quote=msg, store=mem)
    assert c.outcome == "committed"
    assert [v for _i, v in usable_slot(mem, sasha.id, "birthday")] == ["16 мая"]


def test_forgotten_target_not_found(mem):
    """«исправь» on a forgotten or purged record → `rejected` with «это уже забыто»."""
    _sasha, bday = sasha_world(mem)
    gone = api.forget(
        handle(bday.id),
        turn=turn_of("забудь день рождения Саши", n=2),
        because_quote="день рождения Саши",
        store=mem,
    )
    assert gone.outcome == "stopped"
    q = "у Саши день рождения 14 мая"
    r = api.fix(handle(bday.id), q, turn=turn_of(q, n=3), because_quote=q, store=mem)
    assert r.outcome == "rejected" and "забыто" in (r.line or "")
    assert status_of(mem, bday.id) == "forgotten"


def test_retracted_target_is_not_resurrected(mem):
    _sasha, bday = sasha_world(mem)
    words = "это неправда про день рождения Саши"
    r = api.fix(
        handle(bday.id), "", turn=turn_of(words, n=2), because_quote=words, retract=True, store=mem
    )
    assert r.outcome == "committed" and status_of(mem, bday.id) == "retracted"
    q = "у Саши день рождения 14 мая"
    again = api.fix(handle(bday.id), q, turn=turn_of(q, n=3), because_quote=q, store=mem)
    assert again.outcome == "rejected" and "отменено" in (again.line or "")


def test_structured_remember(mem):
    """`assertion` builds a fact in the closed vocabulary, subject resolved, value normalized,
    `origin_quote` = because_quote."""
    sasha, _bday = sasha_world(mem)
    q = "Саша живёт в Твери"
    r = api.remember(
        "Sasha lives in Tver",
        turn=turn_of(f"запомни: {q}", n=2),
        because_quote=q,
        assertion=Assertion(subject="Саша", relation="lives_in", value="Тверь"),
        store=mem,
    )
    assert r.outcome == "committed" and r.memory_version
    row = fact_row(mem, r.record_ids[0])
    assert (row["subject_id"], row["relation"], row["value_norm"]) == (
        sasha.id,
        "lives_in",
        "тверь",
    )
    assert row["origin_quote"] == q and row["quote_orig"] == q and row["author_tier"] == "user"
    q2 = "мой день рождения 3 мая"
    me = api.remember(
        "My birthday is 3 May",
        turn=turn_of(f"запомни, {q2}", n=3),
        because_quote=q2,
        assertion=Assertion(subject="я", relation="birthday", value="3 мая"),
        store=mem,
    )
    row = fact_row(mem, me.record_ids[0])
    assert (row["subject_id"], row["value_norm"]) == (OWNER_ENTITY_ID, "05-03")
    odd = api.remember(
        "Likes the colour teal",
        turn=turn_of("запомни: мне нравится бирюзовый", n=4),
        because_quote="мне нравится бирюзовый",
        assertion=Assertion(relation="favourite_colour_xyz", value="teal"),
        store=mem,
    )
    assert fact_row(mem, odd.record_ids[0])["relation"] == "other"


def test_unstructured_is_other(mem):
    """Without `assertion`: one `relation='other'` fact about `self`, text_en = the text."""
    q = "я люблю места у окна"
    r = api.remember(
        "I like window seats", turn=turn_of(f"запомни: {q}"), because_quote=q, store=mem
    )
    assert r.outcome == "committed" and len(r.record_ids) == 1
    row = fact_row(mem, r.record_ids[0])
    assert (row["subject_id"], row["relation"], row["text_en"]) == (
        OWNER_ENTITY_ID,
        "other",
        "I like window seats",
    )
    assert row["origin_quote"] == q
