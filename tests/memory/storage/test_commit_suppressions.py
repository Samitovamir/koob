"""The ledger's commit joined to the forget pipeline: the memory spec §9.6, §9.8 (integ).

Ops built from raw older than a forget are rejected (`ledger.CommitRejected("suppressed")`)
through `forget.blocked_by`, over every key a forget writes: the proposition, the
proposition under an alias of X, X's stable ids. The user's word from newer raw is admitted."""

from __future__ import annotations

import pytest

from koob.memory import forget
from koob.memory.ledger import CommitRejected
from koob.memory.model import AuthorTier, ForgetScope, Impression, StableId

from .. import _ledger as L
from ..forget import _seed as fs

clock = fs.clock  # fixture: forget's manual clock (2026-10-12 09:00)

SYS = AuthorTier.SYSTEM
OLD = "2026-10-01T10:00:00Z"  # before the forget
NEW = "2026-10-13T10:00:00Z"  # after it


@pytest.fixture
def conn(tmp_path, clock):
    c = L.open_seeded(tmp_path)
    yield c
    c.close()


def _worker_add(conn, record, raw_id: str):
    return L.add(conn, record.model_copy(update={"source_raw_id": raw_id}), tier=SYS, model_id="w")


def _rejected(fn) -> str:
    """The reason code: a hot or code write raises it; a worker's group lands in
    `CommitResult.rejected` (§6.3, v0.4 F6)."""
    try:
        res = fn()
    except CommitRejected as e:
        return e.reason
    assert not res.ok and res.rejected, "the commit was not rejected"
    return next(iter(res.rejected.values())).split(":")[0]


def test_worker_rederiving_a_forgotten_fact_from_old_raw_is_rejected(conn):
    """§9.6: a worker re-extracting the forgotten fact from another old message is refused;
    the user's new word after the forget is admitted; foreign text is not."""
    assert L.add(conn, L.entity("ent-ilya", "Илья")).ok
    L.raw(conn, "raw-a", "Илья переезжает в Берлин", ts_=OLD)
    L.raw(conn, "raw-b", "Илья уже почти в Берлине", ts_=OLD)
    assert _worker_add(conn, L.fact("f-1", "ent-ilya", "lives_in", "Berlin"), "raw-a").ok
    forget.stop_use(conn, ["f-1"], scope=ForgetScope.ITEM, forever=False)

    again = L.fact("f-2", "ent-ilya", "lives_in", "Berlin")
    assert _rejected(lambda: _worker_add(conn, again, "raw-b")) == "suppressed"
    # no raw at all and not the user's tier: refused too
    bare = again.model_copy(update={"author_tier": SYS})
    assert _rejected(lambda: L.add(conn, bare, tier=SYS)) == "suppressed"
    # foreign text newer than the forget is still not the user's word
    L.raw(conn, "raw-mail", "Ilya moved to Berlin, says the newsletter", tier="other", ts_=NEW)
    other = AuthorTier.OTHER
    foreign = L.fact("f-3", "ent-ilya", "lives_in", "Berlin", tier=other, source_raw_id="raw-mail")
    assert _rejected(lambda: L.add(conn, foreign, tier=other)) == "suppressed"
    # the user's own message after the forget is the user's new word
    L.raw(conn, "raw-new", "Илья всё-таки переехал в Берлин", ts_=NEW)
    assert _worker_add(conn, L.fact("f-4", "ent-ilya", "lives_in", "Berlin"), "raw-new").ok
    # another proposition about the same person was never suppressed
    assert _worker_add(conn, L.fact("f-5", "ent-ilya", "likes", "jazz"), "raw-b").ok


def test_forgotten_person_does_not_come_back_under_a_new_entity(conn):
    """§9.8: after «забудь всё про Илью» old raw cannot rebuild him: not his platform id on
    a new entity, not his propositions under a namesake's alias; another fact passes."""
    tg = StableId(kind="tg_user_id", value="777001", tier="user")
    assert L.add(conn, L.entity("ent-x", "Илья", stable_ids=[tg])).ok
    L.raw(conn, "raw-a", "Илья переезжает в Берлин", ts_=OLD)
    L.raw(conn, "raw-b", "Илья пишет из Берлина", ts_=OLD)
    assert _worker_add(conn, L.fact("f-x", "ent-x", "lives_in", "Berlin"), "raw-a").ok
    forget.stop_use(conn, ["ent-x"], scope=ForgetScope.ABOUT_ENTITY, forever=False)

    platform = StableId(kind="tg_user_id", value="777001", tier="platform")
    ghost = L.entity("ent-x2", "Илья", tier=SYS, stable_ids=[platform])
    assert _rejected(lambda: L.add(conn, ghost, tier=SYS)) == "suppressed"
    # a namesake without his ids may exist, but his old propositions do not attach to it
    assert L.add(conn, L.entity("ent-x3", "Илья")).ok
    again = L.fact("f-x3", "ent-x3", "lives_in", "Berlin")
    assert _rejected(lambda: _worker_add(conn, again, "raw-b")) == "suppressed"
    assert _worker_add(conn, L.fact("f-x4", "ent-x3", "lives_in", "Moscow"), "raw-b").ok
    # his id cannot be bound onto the namesake either
    bind = L.op(L.OpKind.UPDATE, target="ent-x3", add_stable_ids=[platform.model_dump()])
    assert _rejected(lambda: L.commit(conn, bind, tier=SYS)) == "suppressed"


def test_forgotten_impression_is_not_rebuilt(conn):
    """§9.6 for an impression: the same reading of the same person from old raw is refused."""
    assert L.add(conn, L.entity("ent-d", "Дима")).ok
    L.raw(conn, "raw-a", "Дима сегодня какой-то грустный", ts_=OLD)
    L.raw(conn, "raw-b", "Дима опять грустный", ts_=OLD)

    def impression(iid: str, raw_id: str) -> Impression:
        return Impression(
            id=iid,
            about_entity_id="ent-d",
            text_en="Dima seems sad",
            observed_at=OLD,
            author_tier=AuthorTier.USER,
            recorded_at=L.ts(),
        )

    first = impression("i-1", "raw-a")
    assert L.add(conn, first, tier=SYS, model_id="w", derived_from=["raw-a"]).ok
    forget.stop_use(conn, ["i-1"], scope=ForgetScope.ITEM, forever=False)
    again = impression("i-2", "raw-b")
    reason = _rejected(lambda: L.add(conn, again, tier=SYS, model_id="w", derived_from=["raw-b"]))
    assert reason == "suppressed"
