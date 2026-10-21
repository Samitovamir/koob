"""A forget and a correction always leave their mark, through the same `api` calls a
conversation makes: a forget said as a description of the item (not its words) still finds
it by meaning - the fact, or the user's own message when nothing was learned from it yet; a
correction that arrives before any fact holds the old value marks the user's earlier message,
so a later extraction cannot learn the old value and no raw retrieval path serves it; a
forget for good also takes the person's other facts holding the value and their nickname
holding it. Uses a synthetic persona."""

from __future__ import annotations

import pytest

from koob.memory import api, forget
from koob.memory.model import AuthorTier, TurnCtx, TurnOrigin
from koob.memory.recall import basket, core_view, vectors

from .._ledger import DayClock, add, entity, fact, open_seeded, status
from ..recall.test_raw_meaning_lane import ConceptEmbedder

CLOCK = "2026-10-12T12:00:00.000000Z"
LENA = "ent-lena"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(ConceptEmbedder())
    basket._NO_CONTRAST.clear()
    basket._NO_REVISION.clear()
    forget.reset_wiring()
    forget.use_key(None)
    forget.wire(stop_hooks=[vectors.stop_hook], clock=DayClock())
    c = open_seeded(tmp_path)
    vectors.attach(tmp_path / "vectors.db")
    yield c
    c.close()
    forget.reset_wiring()
    vectors.set_embedder(None)


def t(words: str, n: int = 9) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=CLOCK,
    )


def said(conn, n: int, text: str, day: str = "2026-10-01") -> str:
    api.ingest_turn(
        turn_id=f"turn-{n}",
        thread_id="thread-1",
        origin="owner",
        status="done",
        rows=[{"ts": f"{day}T10:00:00Z", "role": "owner", "text": text}],
        store=conn,
    )
    return conn.execute("SELECT id FROM raw_chunks WHERE turn_id = ?", (f"turn-{n}",)).fetchone()[0]


def _embed(conn, path) -> None:
    vdb = vectors.open_db(path)
    try:
        vectors.embed_pending(conn, vdb)
    finally:
        vdb.close()


def _shown(conn, ask: str, n: int = 20) -> str:
    b = api.recall(t(ask, n), store=conn)
    found = api.search(ask, turn=t(ask, n + 1), store=conn)
    return "\n".join([b.block or "", *(i.text for i in b.items), *(i.text for i in found)])


def _sups(conn) -> int:
    return conn.execute("SELECT count(*) FROM suppressions").fetchone()[0]


@pytest.mark.parametrize(
    "words",
    [
        "Forget where I leave the thing that opens my garage",
        "Забудь, что я говорил про ключи от гаража",
        "Please drop what you know about getting into my garage if I lose the keys",
    ],
)
def test_a_forget_that_describes_the_fact_finds_it_by_meaning(conn, tmp_path, words):
    rid = said(conn, 1, "Left the spare one with the lady next door, under her mat.")
    add(conn, fact("f-key", "self", "spare_key_place", "under the neighbour's mat",
                   text="Denis left the spare garage key under the neighbour's mat",
                   source_raw_id=rid, quote_orig="under her mat"))  # fmt: skip
    _embed(conn, tmp_path / "vectors.db")
    res = api.forget(words, turn=t(words), because_quote=words, store=conn)
    assert res.outcome == "stopped", res.line
    assert status(conn, "f-key") == "forgotten"
    assert _sups(conn) >= 1


def test_a_described_forget_before_extraction_takes_his_message(conn, tmp_path):
    rid = said(conn, 1, "ключи от гаража я оставил у соседки под ковриком")
    said(conn, 2, "купил новые кроссовки и шнурки к ним", "2026-10-02")
    _embed(conn, tmp_path / "vectors.db")
    words = "Forget where I said the garage keys are"
    res = api.forget(words, turn=t(words), because_quote=words, store=conn)
    assert res.outcome == "stopped", res.line
    row = conn.execute("SELECT status FROM raw_chunks WHERE id = ?", (rid,)).fetchone()
    assert row[0] != "active"
    assert "коврик" not in _shown(conn, "where are the garage keys?")


def test_a_forget_of_something_never_said_forgets_nothing(conn, tmp_path):
    said(conn, 1, "ключи от гаража я оставил у соседки под ковриком")
    _embed(conn, tmp_path / "vectors.db")
    words = "Forget my blood sugar numbers"
    res = api.forget(words, turn=t(words), because_quote=words, store=conn)
    assert res.outcome == "not_found", res.line


@pytest.mark.parametrize(
    ("first", "correction", "old"),
    [
        ("The team offsite is in Leeds next month, I booked the hotel already.",
         "Actually the team offsite is in York, not Leeds.", "Leeds"),
        ("Денис Смит ведёт семинар в Твери в среду, зал уже забронирован.",
         "Поправка: семинар не в Твери, а в Самаре.", "Твери"),
    ],
)  # fmt: skip
def test_a_correction_before_any_fact_marks_his_message_for_good(conn, first, correction, old):
    rid = said(conn, 1, first)
    api.fix(correction, correction, turn=t(correction, 2), because_quote=correction, store=conn)
    marks = conn.execute(
        "SELECT retracted_value_norm FROM raw_retractions WHERE raw_id = ?", (rid,)
    ).fetchall()
    assert marks, "no durable mark on his earlier message"
    # the user's correction is not ingested yet: no retrieval path serves the old value as current
    shown = _shown(conn, "where is the team offsite?" if "Leeds" in old else "где семинар?")
    assert old.casefold() not in shown.casefold()


def _lena(conn) -> None:
    said(conn, 1, "Lena Park streams every night, her tag is Ghostfox-88.")
    add(conn, entity(LENA, "Lena Park"))
    conn.execute(
        "INSERT INTO aliases (id, entity_id, alias, alias_norm, script, kind, tier, created_at,"
        " updated_at) VALUES ('al-lena', ?, 'Ghostfox-88', 'ghostfox-88', 'latin', 'nickname',"
        " 'user', ?, ?)",
        (LENA, CLOCK, CLOCK),
    )
    add(conn, fact("f-tag", LENA, "gamer_tag", "Ghostfox-88",
                   text="Lena Park's gamer tag is Ghostfox-88"))  # fmt: skip
    add(conn, fact("f-stream", LENA, "streams_on", "Twitch as Ghostfox-88",
                   text="Lena Park streams on Twitch as Ghostfox-88"))  # fmt: skip
    add(conn, fact("f-city", LENA, "lives_in", "Porto", text="Lena Park lives in Porto"))


def _core(conn) -> str:
    return "\n".join(core_view.render([sec]) for sec in core_view.compute(conn))


def test_a_forget_for_good_takes_sibling_facts_and_the_nickname_holding_the_value(conn):
    _lena(conn)
    assert "ghostfox" in _core(conn).casefold()
    words = "Forget Lena's gamer tag for good, насовсем"
    res = api.forget(words, turn=t(words), because_quote=words, forever=True, store=conn)
    assert res.outcome == "stopped", res.line
    assert status(conn, "f-tag") == "forgotten"
    assert status(conn, "f-stream") == "forgotten"
    assert status(conn, "f-city") == "active"
    core = _core(conn)
    assert "ghostfox" not in core.casefold()
    assert "Lena Park" in core


def test_a_suppressed_value_hides_another_persons_fact_holding_it(conn):
    _lena(conn)
    add(conn, entity("ent-max", "Max Rowe"))
    add(conn, fact("f-other", "ent-max", "plays_with", "Ghostfox-88",
                   text="Max Rowe plays online with Ghostfox-88",
                   tier=AuthorTier.OTHER))  # fmt: skip
    words = "Forget Lena's gamer tag"
    res = api.forget(words, turn=t(words), because_quote=words, store=conn)
    assert res.outcome == "stopped", res.line
    assert "f-other" in forget.hidden_by_forget(conn, ["f-other", "f-city"])
    assert "f-city" not in forget.hidden_by_forget(conn, ["f-other", "f-city"])
