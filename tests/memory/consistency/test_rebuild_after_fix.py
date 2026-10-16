"""§9.6 retraction overlay (scenario B): after a fix, a rebuild that reads the old raw cannot
bring the corrected value back, except inside a history clause.

Spec marker: `[T consistency/test_rebuild_after_fix::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import ledger
from koob.memory.ledger import CommitRejected
from koob.memory.model import AuthorTier, EventKind, OpKind, ReadRef, ReadSrc

from .._ledger import add, commit, entity, fact, op, open_seeded, raw, status


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    assert add(c, entity("e-s", "Sasha")).ok
    raw(c, "r-1", "у Саши день рождения 3 мая")
    old = fact(
        "f-old",
        "e-s",
        "birthday",
        "05-03",
        text="Sasha's birthday is 3 May",
        source_raw_id="r-1",
        quote_orig="день рождения 3 мая",
    )
    assert add(c, old, tier=AuthorTier.SYSTEM, model_id="w").ok
    yield c
    c.close()


def _fix(conn):
    new = fact("f-new", "e-s", "birthday", "05-12", text="Sasha's birthday is 12 May")
    res = commit(
        conn,
        op(OpKind.RETRACT, target="f-old", group="e-s"),
        op(OpKind.ADD, group="e-s", record=new.model_dump(mode="json")),
        tier=AuthorTier.USER,
        event_kind=EventKind.FIX,
    )
    assert res.ok


def _summary(conn, content, did="d-day"):
    body = {"id": did, "dkind": "summary", "dkey": "2026-10-01", "content": content}
    refs = [ReadRef(id="r-1", version=1, src=ReadSrc.RAW)]
    return commit(
        conn, op(OpKind.ADD, group=did, read_set=refs, derived=body), tier=AuthorTier.SYSTEM
    )


def test_old_birthday_absent(conn):
    """§9.6: commit rejects rebuilt output holding a retracted value outside a history clause (scenario B)."""
    _fix(conn)
    assert status(conn, "f-old") == "retracted" and status(conn, "f-new") == "active"
    rr = conn.execute(
        "SELECT raw_id, record_id, retracted_value_norm FROM raw_retractions"
    ).fetchall()
    assert [tuple(r) for r in rr] == [("r-1", "f-old", "05-03")]

    for text in (
        "Sasha's birthday is 3 May.",
        "Sasha's birthday is May 3rd.",
        "День рождения Саши — 3 мая.",
        "Birthday on 03.05, party planned.",
    ):
        with pytest.raises(CommitRejected) as e:
            _summary(conn, text)
        assert e.value.reason == "retracted_value"

    assert _summary(conn, "Sasha's birthday is 12 May (previously recorded as 3 May).").ok
    assert _summary(conn, "Sasha's birthday is 12 May, not 3 May.", did="d-2").ok


def test_rederived_record_from_old_raw_rejected(conn):
    _fix(conn)
    again = fact(
        "f-again", "e-s", "birthday", "05-03", text="Sasha's birthday is 3 May", source_raw_id="r-1"
    )
    res = add(conn, again, tier=AuthorTier.SYSTEM, model_id="w")  # worker: dropped (F6)
    assert not res.ok and res.rejected["e-s"].startswith("retracted_value")
    with pytest.raises(CommitRejected) as e:  # a code write is rejected too
        add(conn, again, tier=AuthorTier.SYSTEM)
    assert e.value.reason == "retracted_value"
    # Other values from that raw still pass.
    ok = fact(
        "f-city", "e-s", "lives_in", "Tver", text="Sasha lives in Tver", source_raw_id="r-1"
    )
    assert add(conn, ok, tier=AuthorTier.SYSTEM, model_id="w").ok


def test_value_tokens_cover_dates_numbers_emails():
    t = ledger.value_tokens
    assert (
        t("05-03")
        == t("3 May")
        == t("May 3")
        == t("3 мая")
        == t("03.05")
        == t("2026-05-03") - {"date:2026-05-03"}
    )
    assert "email:a@b.org" in t("Mail A@B.org please")
    assert "num:42" in t("flat 42")
    assert t("+7 (912) 345-67-89") & t("89123456789")


def test_supersede_by_life_change_is_not_a_retraction(conn):
    """Moving is not "it was wrong": no overlay, the old value may still be told as history."""
    assert add(conn, fact("f-msk", "e-s", "lives_in", "Moscow")).ok
    assert add(conn, fact("f-kzn", "e-s", "lives_in", "Tver")).ok
    assert status(conn, "f-msk") == "superseded"
    assert (
        conn.execute("SELECT count(*) FROM raw_retractions WHERE record_id = 'f-msk'").fetchone()[0]
        == 0
    )
