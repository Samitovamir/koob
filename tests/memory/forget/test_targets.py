"""the memory spec §9.1, §9.6, §9.8 (v0.4): which records a forget aimed at history
takes, platform ids after a forget, and the purge off the tool call.
Spec marker: `[T forget/test_targets::<case>]`."""

from __future__ import annotations

from koob.memory import api, forget
from koob.memory.model import ForgetScope, Label, TurnCtx, TurnOrigin

from . import _seed as s

clock = s.clock  # fixture


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


def _slot(mem) -> None:
    """A person's home, three times: Moscow ← Tver ← Berlin (the head), each from its own
    message; f-kzn supersedes f-msk, f-ber supersedes f-kzn."""
    s.relation(mem, "lives_in", "residence")
    s.entity(mem, "e-gleb", "Глеб", [("Глеб", "name"), ("Gleb", "translit")])
    for rid, raw_id, city, ru, quote in (
        ("f-msk", "raw-msk", "Moscow", "Москва", "Глеб живёт в Москве"),
        ("f-kzn", "raw-kzn", "Tver", "Тверь", "Глеб переехал в Тверь"),
        ("f-ber", "raw-ber", "Berlin", "Берлин", "Глеб теперь живёт в Берлине"),
    ):
        s.raw(mem, raw_id, quote)
        s.fact(
            mem, rid, "e-gleb", "lives_in", value=city, text_en=f"Gleb lives in {city}",
            quote=quote, raw=raw_id, keywords_ru=f"{ru} живёт", keywords_en=city.casefold(),
        )  # fmt: skip
    mem.execute("UPDATE records SET status = 'superseded' WHERE id IN ('f-msk', 'f-kzn')")
    s.edge(mem, "f-kzn", "f-msk", "supersedes")
    s.edge(mem, "f-ber", "f-kzn", "supersedes")
    mem.commit()


def test_superseded_target_forgets_chain_not_head(mem, clock):
    """«забудь» on a superseded record forgets it and its predecessors; the current head and
    its own lineage stay."""
    _slot(mem)
    res = api.forget(
        "m:f-kzn", turn=_turn("забудь, что Глеб жил в Твери"),
        because_quote="Глеб жил в Твери", store=mem,
    )  # fmt: skip
    assert res.outcome == "stopped", res
    assert {"f-kzn", "f-msk"} <= set(res.record_ids)
    assert _status(mem, "f-kzn") == "forgotten" and _status(mem, "f-msk") == "forgotten"
    assert _raw_status(mem, "raw-kzn") == "forgotten" and _raw_status(mem, "raw-msk") == (
        "forgotten"
    )
    # never a record that superseded the target
    assert "f-ber" not in res.record_ids
    assert _status(mem, "f-ber") == "active" and _raw_status(mem, "raw-ber") == "active"

    # a forgotten record is not a target any more
    again = api.forget(
        "m:f-kzn", turn=_turn("забудь Тверь", n=2), because_quote="Тверь", store=mem
    )
    assert again.outcome == "not_found"


def test_superseded_target_found_by_his_words(mem, clock):
    """§9.1 A14: history is content - the user's words find a superseded record («раньше жил в
    Москве»), and the head is not taken with it."""
    _slot(mem)
    res = api.forget(
        "Москва", turn=_turn("забудь про Москву"), because_quote="забудь про Москву", store=mem
    )
    assert res.outcome == "stopped", res
    assert res.record_ids == ["f-msk"]
    assert _status(mem, "f-kzn") == "superseded" and _status(mem, "f-ber") == "active"


def test_retracted_target_is_valid(mem, clock):
    """§9.1 A14: a retracted record («это неправда») is a valid target."""
    _slot(mem)
    mem.execute("UPDATE records SET status = 'retracted' WHERE id = 'f-msk'")
    mem.commit()
    fid = forget.stop_use(mem, ["f-msk"], scope=ForgetScope.ITEM, forever=False)
    assert fid and _status(mem, "f-msk") == "forgotten"
    assert _status(mem, "f-kzn") == "superseded"


def test_head_forget_takes_its_history(mem, clock):
    """«забудь» on the current head forgets its predecessors in the slot too (no «раньше …»)."""
    _slot(mem)
    res = api.forget(
        "m:f-ber", turn=_turn("забудь, где живёт Глеб"), because_quote="где живёт Глеб", store=mem
    )
    assert res.outcome == "stopped", res
    assert {"f-ber", "f-kzn", "f-msk"} <= set(res.record_ids)
    for rid in ("f-ber", "f-kzn", "f-msk"):
        assert _status(mem, rid) == "forgotten", rid
    for raw_id in ("raw-ber", "raw-kzn", "raw-msk"):
        assert _raw_status(mem, raw_id) == "forgotten", raw_id
    # the entity itself is not history of a fact
    assert _status(mem, "e-gleb") == "active"


def test_platform_id_after_forget_binds(tmp_path):
    """§9.8: a stable id of a forgotten person binds again from a message newer than the
    forget (platform metadata); older raw stays refused."""
    import pytest

    from koob.memory import forget
    from koob.memory.ledger import CommitRejected
    from koob.memory.model import AuthorTier, ForgetScope, StableId

    from .. import _ledger as L

    forget.reset_wiring()
    forget.use_key(None)
    conn = L.open_seeded(tmp_path)
    tg = StableId(kind="tg_user_id", value="777001", tier="platform")
    assert L.add(conn, L.entity("e-x", "Илья", stable_ids=[tg])).ok
    L.raw(conn, "r-old", "Илья: привет", tier="other", ts_="2026-10-01T09:00:00.000000Z")
    forget.stop_use(conn, ["e-x"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    later = "2099-01-01T00:00:00.000000Z"  # a message after the forget
    assert conn.execute("SELECT max(created_at) FROM suppressions").fetchone()[0] < later
    L.raw(conn, "r-new", "Илья: я снова тут", tier="other", ts_=later)

    def bind(eid: str, raw_id: str):  # code sets platform metadata (model_id None)
        ent = L.entity(eid, "Илья", tier=AuthorTier.SYSTEM, stable_ids=[tg])
        return L.add(conn, ent, tier=AuthorTier.SYSTEM, derived_from=[raw_id])

    with pytest.raises(CommitRejected) as e:
        bind("e-old", "r-old")  # the person's old message: still refused, whatever the tier
    assert e.value.reason == "suppressed"
    assert bind("e-new", "r-new").ok  # the person writes again: a new person met today
    got = conn.execute(
        "SELECT entity_id FROM entity_stable_ids WHERE kind = 'tg_user_id' AND value = '777001'"
    ).fetchall()
    assert [r[0] for r in got] == ["e-new"]
    conn.close()
    forget.reset_wiring()


def test_purge_runs_off_the_call(mem, clock, monkeypatch):
    """«насовсем»: the tool returns after stop-use (≤ 50 ms commit); the purge job runs at
    once in the pool; «удалил полностью» only from purge_status."""
    _slot(mem)

    def no_inline(*_a, **_k):
        raise AssertionError("purge ran inside the forget call")

    with monkeypatch.context() as m:
        m.setattr(forget, "purge", no_inline)
        res = api.forget(
            "m:f-ber", turn=_turn("забудь насовсем, где живёт Глеб"),
            because_quote="где живёт Глеб", forever=True, store=mem,
        )  # fmt: skip
    assert res.outcome == "stopped", res
    assert "Удаляю полностью" in res.line and "удалил полностью" not in res.line.casefold()
    assert api.purge_status(res.forget_id, store=mem).state != "purged"
    assert res.forget_id in forget.due(mem)  # due now: the pool takes it right after the call

    api.run_purges(store=mem)  # the pool's next pass
    assert api.purge_status(res.forget_id, store=mem).state == "purged"


def test_sensitive_forget_is_due_not_inline(mem, clock):
    """F11: a sensitive label makes the purge due at once, still off the call."""
    s.relation(mem, "health", "health")
    s.entity(mem, "e-gleb", "Глеб", [("Глеб", "name")])
    s.fact(
        mem, "f-h", "e-gleb", "health", value="asthma", text_en="Gleb has asthma",
        labels=int(Label.HEALTH),
    )  # fmt: skip
    mem.commit()
    res = api.forget("m:f-h", turn=_turn("забудь про астму"), because_quote="про астму", store=mem)
    assert res.outcome == "stopped" and res.forget_id in forget.due(mem)
