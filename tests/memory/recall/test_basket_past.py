"""The basket's raw lane reads the fix lineage (§9.6 retraction overlay, `supersedes` edges):
an old raw quote is never presented as current. Every record learned from it overtaken: the
quote drops out; only some: the overtaken values are masked («[сменилось]»), never quoted."""

from __future__ import annotations

import pytest

from koob.memory import api
from koob.memory.model import OpKind, TurnCtx, TurnOrigin
from koob.memory.recall import lexical, vectors

from .._ledger import add, commit, fact, op, open_seeded, raw, status

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str, *, n: int = 1) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id=f"thread-{n}",
        session_id=f"session-{n}",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=CLOCK,
    )


def _raw(conn, rid: str, text: str, day: str) -> None:
    raw(conn, rid, text, ts_=f"{day}T10:00:00Z")
    lexical.index_raw(conn, rid)


def _tver_then_tver(conn, *, fix: bool = True) -> None:
    """«Я живу в Твери», then «Нет, я живу в Твери»: the second supersedes the first."""
    _raw(conn, "r-kzn", "Я живу в Твери", "2026-10-01")
    assert add(
        conn, fact("f-kzn", "self", "lives_in", "Tver", text="I live in Tver",
                   source_raw_id="r-kzn", quote_orig="Я живу в Твери"),
    ).ok  # fmt: skip
    _raw(conn, "r-tvr", "Нет, я живу в Твери", "2026-10-05")
    new = fact("f-tvr", "self", "lives_in", "Tver", text="I live in Tver",
               source_raw_id="r-tvr", quote_orig="я живу в Твери", valid_from="2026-10-05")  # fmt: skip
    r = commit(
        conn,
        op(OpKind.SUPERSEDE, target="f-kzn", group="self",
           record=new.model_dump(mode="json"), fix=fix),
    )  # fmt: skip
    assert r.ok, r
    assert status(conn, "f-kzn") == "superseded"


def _never_current(items, needle: str) -> None:
    for i in items:
        if needle in i.text:
            raise AssertionError(i.text)  # a retired value is never quoted


@pytest.mark.parametrize("fix", [True, False])
def test_overtaken_raw_quote_drops_out(conn, fix):
    """Tver corrected to Tver. The basket never presents Tver as current;
    the only record learned from the Tver quote is superseded, so the quote drops out."""
    _tver_then_tver(conn, fix=fix)
    if fix:
        assert conn.execute(
            "SELECT count(*) FROM raw_retractions WHERE raw_id = 'r-kzn'"
        ).fetchone()[0]
    for words in ("где я живу", "я живу в Твери?", "Тверь"):
        b = api.recall(t(words), store=conn)
        ids = [i.id for i in b.items]
        assert "r-kzn" not in ids and "f-kzn" not in ids, words
        _never_current(b.items, "Твер")
        assert "Твер" not in (b.block or "")
    b = api.recall(t("где я живу", n=2), store=conn)
    assert {"r-tvr", "f-tvr"} & {i.id for i in b.items}  # the raw, or the record learned from it


def test_retracted_record_drops_its_quote(conn):
    """«Это неправда» (RETRACT) overtakes the record the same way."""
    _raw(conn, "r-kzn", "Я живу в Твери", "2026-10-01")
    assert add(
        conn, fact("f-kzn", "self", "lives_in", "Tver", text="I live in Tver",
                   source_raw_id="r-kzn", quote_orig="Я живу в Твери"),
    ).ok  # fmt: skip
    assert commit(conn, op(OpKind.RETRACT, target="f-kzn", group="self")).ok
    b = api.recall(t("где я живу"), store=conn)
    assert "r-kzn" not in [i.id for i in b.items]


def test_partly_overtaken_quote_is_shown_as_past(conn):
    """One quote, two records: the city is corrected, the job still holds. The quote stays
    (the job is current) with the old city masked: it says THAT it changed."""
    _raw(conn, "r-both", "Я живу в Твери и работаю в Яндексе", "2026-10-01")
    assert add(
        conn, fact("f-kzn", "self", "lives_in", "Tver", text="I live in Tver",
                   source_raw_id="r-both", quote_orig="Я живу в Твери"),
    ).ok  # fmt: skip
    assert add(
        conn, fact("f-job", "self", "works_at", "Yandex", text="I work at Yandex",
                   source_raw_id="r-both", quote_orig="работаю в Яндексе"),
    ).ok  # fmt: skip
    _raw(conn, "r-tvr", "Нет, я живу в Твери", "2026-10-05")
    new = fact("f-tvr", "self", "lives_in", "Tver", text="I live in Tver",
               source_raw_id="r-tvr", quote_orig="я живу в Твери", valid_from="2026-10-05")  # fmt: skip
    assert commit(
        conn,
        op(OpKind.SUPERSEDE, target="f-kzn", group="self",
           record=new.model_dump(mode="json"), fix=True),
    ).ok  # fmt: skip
    b = api.recall(t("где я живу и работаю"), store=conn)
    by = {i.id: i for i in b.items}
    assert "r-both" in by
    assert by["r-both"].text.startswith(
        "с 2026-10-05 часть сменилась · [сменилось] и работаю в Яндексе"
    )
    _never_current(b.items, "Твер")


def test_raw_without_learned_records_is_shown_as_is(conn):
    """Nothing learned from the quote yet (extract not run): no lineage, shown unchanged."""
    _raw(conn, "r-new", "Я живу в Твери", "2026-10-01")
    b = api.recall(t("где я живу"), store=conn)
    by = {i.id: i for i in b.items}
    assert by["r-new"].text == "Я живу в Твери"
