"""G4 (§4.1): a fact keeps its dimensions apart, unknown stays NULL, and the record round-trips;
lineage edges and FTS rows land in the same transaction as the record (§3.2)."""

from __future__ import annotations

import pytest

from koob.memory import ledger
from koob.memory.model import (
    Applicability,
    AuthorTier,
    Decision,
    Episode,
    FKind,
    Impression,
    Polarity,
    Stability,
)

from .._ledger import add, edges, entity, fact, open_seeded, raw


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    assert add(c, entity("e-sasha", "Саша")).ok
    assert add(c, entity("e-berlin", "Berlin", etype="place")).ok
    yield c
    c.close()


def test_sasha_says_she_will_move_is_a_claim_and_a_plan(conn):
    raw(conn, "r-1", "Саша говорит, что собирается переехать в Берлин", tier="other")
    f = fact(
        "f-move",
        "e-sasha",
        "moves_to",
        "berlin",
        tier=AuthorTier.OTHER,
        fkind=FKind.PLAN,
        speaker_entity_id="e-sasha",
        author_entity_id="e-sasha",
        value_entity_id="e-berlin",
        source_raw_id="r-1",
        quote_orig="собираюсь переехать в Берлин",
        lang_orig="ru",
        text="Sasha plans to move to Berlin",
    )
    assert add(conn, f, tier=AuthorTier.SYSTEM, model_id="w").ok
    got = ledger.get_record(conn, "f-move")
    assert got.fkind is FKind.PLAN and got.author_tier is AuthorTier.OTHER
    assert got.speaker_entity_id == "e-sasha" and got.polarity is Polarity.AFFIRMED
    # Unknown stays NULL: nobody said when, how lasting, or for whom.
    assert got.valid_from is None and got.valid_to is None and got.stability is None
    assert got.applicability is None
    assert ("f-move", "e-sasha") in edges(conn, "about")
    assert ("f-move", "e-berlin") in edges(conn, "mentions")
    assert ("f-move", "r-1") in edges(conn, "derived_from")


def test_one_off_for_whom_round_trip(conn):
    f = fact(
        "f-golf",
        "u",
        "plans_to",
        "golf tickets",
        fkind=FKind.EVENT,
        stability=Stability.ONE_OFF,
        applicability=Applicability(for_whom="son"),
        valid_to="2026-10-12T23:59:59Z",
    )
    assert add(conn, f).ok
    got = ledger.get_record(conn, "f-golf")
    assert got.applicability == Applicability(for_whom="son")
    assert got.stability is Stability.ONE_OFF
    stored = conn.execute("SELECT applicability FROM facts WHERE id='f-golf'").fetchone()[0]
    assert stored == '{"for_whom":"son"}'  # only what the source said


@pytest.mark.parametrize(
    "record",
    [
        Impression(
            id="i-1",
            about_entity_id="e-sasha",
            text_en="Seemed tired",
            observed_at="2026-10-01",
            author_tier=AuthorTier.SYSTEM,
            recorded_at="2026-10-01T00:00:00Z",
        ),
        Decision(
            id="d-1",
            chosen_text="Take the train",
            matter_id="e-sasha",
            domain="travel",
            alternatives=[{"text": "fly", "reasons": ["expensive"]}],
            author_tier=AuthorTier.USER,
            recorded_at="2026-10-01T00:00:00Z",
        ),
        Episode(
            id="ep-1",
            summary_en="Dinner with Sasha",
            place="Tver",
            author_tier=AuthorTier.SYSTEM,
            recorded_at="2026-10-01T00:00:00Z",
        ),
    ],
)
def test_other_kinds_round_trip(conn, record):
    # §3.2 F9: a non-entity record needs a lineage root; the user's creating words are one
    record = record.model_copy(update={"origin_quote": "their words"})
    assert add(conn, record).ok
    got = ledger.get_record(conn, record.id)
    assert got.model_dump(exclude={"version", "created_at", "updated_at"}) == record.model_dump(
        exclude={"version", "created_at", "updated_at"}
    )
    n = conn.execute(
        "SELECT count(*) FROM fts_records WHERE rowid = (SELECT fts_rowid FROM records WHERE id = ?)",
        (record.id,),
    ).fetchone()[0]
    assert n == 1


def test_labels_inherit_from_sources(conn):
    raw(conn, "r-h", "у меня астма", labels=1)  # health
    f = fact("f-asthma", "u", "has_condition", "asthma", source_raw_id="r-h")
    assert f.labels == 0
    assert add(conn, f).ok
    assert conn.execute("SELECT labels FROM records WHERE id='f-asthma'").fetchone()[0] & 1


def test_seed_relations_is_closed_and_idempotent(conn):
    n = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
    assert ledger.seed_relations(conn) == n
    fam = dict(conn.execute("SELECT relation, family FROM relations").fetchall())
    assert fam["other"] == "other" and fam["prefers"] and fam["eats"] == fam["prefers_food"]
    card = dict(conn.execute("SELECT relation, cardinality FROM relations").fetchall())
    assert card["lives_in"] == "single" and card["has_email"] == "multi"
