"""RAW-TRUTH: the raw lane tells the truth before extraction. The user's later contrast
(«не X, а Y», "Y, not X") cuts the retired value out of an older raw and brings the correcting
raw along; the user's own word of the last hour is evidence even when no record carries it yet."""

from __future__ import annotations

import pytest

from koob.memory import api
from koob.memory.model import TurnCtx, TurnOrigin
from koob.memory.recall import lexical, vectors

from .._ledger import open_seeded, raw

CLOCK = "2026-05-20T12:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str, *, n: int = 1, clock: str = CLOCK) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}", thread_id=f"thread-{n}", session_id=f"session-{n}",
        origin=TurnOrigin.OWNER, owner_words=words, clock=clock,
    )  # fmt: skip


def _raw(conn, rid: str, text: str, ts: str, tier: str = "user") -> None:
    raw(conn, rid, text, tier=tier, ts_=ts)
    lexical.index_raw(conn, rid)


def _texts(b) -> str:
    return "\n".join([b.block or "", *(i.text for i in b.items)])


def test_later_contrast_cuts_the_old_value_and_brings_the_correction(conn):
    _raw(conn, "r-old", "Signed the lease! the studio on Larch Street, keys on 7/3, rent 1450",
         "2026-05-02T10:00:00Z")  # fmt: skip
    _raw(conn, "r-fix", "hm, landlord says it's Maple Street, not Larch Street, and keys 7/9 not 7/3",
         "2026-05-04T10:00:00Z")  # fmt: skip
    b = api.recall(t("what's the address of my new studio lease?"), store=conn)
    text = _texts(b)
    assert "Larch" not in text and "7/3" not in text
    assert "r-fix" in [i.id for i in b.items]
    assert "Maple" in text


def test_a_foreign_contrast_never_cuts_his_words(conn):
    _raw(conn, "r-his", "Signed the lease for the studio on Larch Street", "2026-05-02T10:00:00Z")
    _raw(conn, "r-them", "agent: it's Maple Street, not Larch Street", "2026-05-04T10:00:00Z",
         tier="other")  # fmt: skip
    b = api.recall(t("which street is my studio lease on?"), store=conn)
    assert "Larch" in _texts(b)


def test_an_earlier_contrast_never_cuts_a_later_value(conn):
    _raw(conn, "r-a", "the demo is on Thursday, not Friday", "2026-05-01T10:00:00Z")
    _raw(conn, "r-b", "the demo room is B-204 on Friday after all", "2026-05-03T10:00:00Z")
    b = api.recall(t("when is the demo and in which room?"), store=conn)
    assert "B-204" in _texts(b)


def test_his_word_of_the_last_hour_is_found_before_extraction(conn):
    _raw(conn, "r-now", "напомни в пятницу забрать куртку из химчистки", "2026-05-20T11:40:00Z")
    _raw(conn, "r-day", "купил новые кроссовки", "2026-05-19T09:00:00Z")
    b = api.recall(t("что я просил сделать на этой неделе?"), store=conn)
    ids = [i.id for i in b.items]
    assert "r-now" in ids
    assert "r-day" not in ids  # yesterday's word still needs its own evidence
    late = api.recall(t("что я просил сделать?", n=2, clock="2026-05-20T14:30:00.000000Z"),
                      store=conn)  # fmt: skip
    assert "r-now" not in [i.id for i in late.items]  # fresh is the last hour only


def test_a_fresh_forget_request_is_never_evidence(conn):
    _raw(conn, "r-forget", "забудь всё, что я говорил про ссору с Гошей", "2026-05-20T11:50:00Z")
    b = api.recall(t("напомни, что нового сегодня?"), store=conn)
    assert "r-forget" not in [i.id for i in b.items]
