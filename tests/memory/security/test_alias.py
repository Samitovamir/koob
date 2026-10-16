"""§13.3 aliases and stable ids: bindings come only from their word or platform metadata.

Spec marker: `[T security/test_alias::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import entities
from koob.memory.ledger import CommitRejected
from koob.memory.model import Alias, AliasKind, AuthorTier, OpKind, StableId

from .._ledger import add, commit, entity, op, open_seeded, version


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    sid = StableId(kind="tg_user_id", value="111", tier=AuthorTier.USER)
    assert add(c, entity("e-sasha", "Саша", stable_ids=[sid])).ok
    yield c
    c.close()


def _alias(entity_id: str, text: str, kind=AliasKind.HANDLE) -> Alias:
    return Alias(
        id=f"a-{text}",
        entity_id=entity_id,
        alias=text,
        alias_norm=entities.normalize_alias(text),
        script=entities.script_of(text),
        kind=kind,
        tier=AuthorTier.OTHER,
    )


def _stable(conn, eid):
    import json

    return json.loads(
        conn.execute("SELECT stable_ids FROM entities WHERE id = ?", (eid,)).fetchone()[0]
    )


def test_group_message_cannot_rebind(conn):
    """§13.3: «Саша — это @sasha_new» in a group is only an unconfirmed candidate."""
    before = _stable(conn, "e-sasha")
    bound = entities.bind_alias(
        conn, _alias("e-sasha", "@sasha_new"), tier=AuthorTier.OTHER, platform=False
    )
    assert bound is False
    row = conn.execute(
        "SELECT confirmed, tier FROM aliases WHERE alias_norm = '@sasha_new'"
    ).fetchone()
    assert tuple(row) == (0, "other")

    # The handle resolves only as an unconfirmed candidate: hands must ask first (§16.1).
    rows = entities.resolve_rows(conn, "@sasha_new", "person")
    assert [(r.entity_id, r.unconfirmed) for r in rows] == [("e-sasha", True)]
    assert rows[0].confidence < 0.7

    # Its stable ids did not move, and a stable-id write from that tier is refused.
    assert _stable(conn, "e-sasha") == before
    bad = StableId(kind="tg_user_id", value="999", tier=AuthorTier.USER).model_dump(mode="json")
    with pytest.raises(CommitRejected) as e:
        commit(
            conn,
            op(OpKind.UPDATE, target="e-sasha", add_stable_ids=[bad]),
            tier=AuthorTier.OTHER,
        )
    assert e.value.reason == "stable_id_tier"
    assert _stable(conn, "e-sasha") == before

    # A second «Саша» created from the group keeps the name unconfirmed; the user's Sasha leads.
    assert add(conn, entity("e-impostor", "Саша", tier=AuthorTier.OTHER)).ok
    rows = entities.resolve_rows(conn, "Саша", "person")
    assert rows[0].entity_id == "e-sasha" and not rows[0].unconfirmed
    assert [r.unconfirmed for r in rows if r.entity_id == "e-impostor"] == [True]


def test_platform_metadata_binds(conn):
    """The sender id of that person's own message binds (§13.3)."""
    assert entities.bind_alias(
        conn, _alias("e-sasha", "@sasha_real"), tier=AuthorTier.OTHER, platform=True
    )
    rows = entities.resolve_rows(conn, "@sasha_real", "person")
    assert [(r.entity_id, r.unconfirmed) for r in rows] == [("e-sasha", False)]
    sid = StableId(kind="tg_user_id", value="222", tier="platform").model_dump(mode="json")
    assert commit(
        conn, op(OpKind.UPDATE, target="e-sasha", add_stable_ids=[sid]), tier=AuthorTier.OTHER
    ).ok
    assert {s["value"] for s in _stable(conn, "e-sasha")} == {"111", "222"}


def test_platform_claim_from_a_model_is_refused(conn):
    sid = StableId(kind="tg_user_id", value="333", tier="platform").model_dump(mode="json")
    res = commit(
        conn,
        op(OpKind.UPDATE, target="e-sasha", add_stable_ids=[sid]),
        tier=AuthorTier.SYSTEM,
        model_id="worker",
    )
    assert not res.ok and res.rejected["g"].startswith("stable_id_tier")  # dropped alone (F6)


def test_his_word_binds_and_bumps_the_entity(conn):
    v = version(conn, "e-sasha")
    a = _alias("e-sasha", "Шура", kind=AliasKind.DIMINUTIVE)
    assert entities.bind_alias(conn, a, tier=AuthorTier.USER, platform=False)
    assert version(conn, "e-sasha") > v  # aliases are version-bearing for entities (§7.1)
    rows = entities.resolve_rows(conn, "Шура", "person")
    assert [(r.entity_id, r.unconfirmed) for r in rows] == [("e-sasha", False)]


def test_room_speech_waits_for_confirmation(conn):
    a = _alias("e-sasha", "Санёк", kind=AliasKind.NICKNAME)
    assert entities.bind_alias(conn, a, tier=AuthorTier.USER_ROOM, platform=False) is False


def test_stable_id_resolves_first(conn):
    rows = entities.resolve_rows(conn, "tg:111", "person")
    assert [(r.entity_id, r.confidence, r.via) for r in rows] == [("e-sasha", 1.0, "stable_id")]
    assert rows[0].token.id == "e-sasha" and rows[0].token.version == version(conn, "e-sasha")
