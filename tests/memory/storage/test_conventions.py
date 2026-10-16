"""the memory spec v0.4 conventions: edge direction (§3.2), the owner entity (§4.5), the
lineage root of every ADD (§3.2 invariants) and the one process clock (§16.4).
Spec marker: `[T storage/test_conventions::<case>]`."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from koob.memory import api, backup, entities, forget, policy, store
from koob.memory.clock import ManualClock, iso
from koob.memory.ledger import CommitRejected
from koob.memory.model import (
    OWNER_ENTITY_ID,
    AuthorTier,
    ForgetScope,
    JustificationIn,
    LinkPayload,
    OpKind,
    ProposalOp,
    payload_dict,
)

from .. import _ledger as L


def test_edges_direction(tmp_path):
    """§3.2: `from` is the newer, derived or dependent side; `to` is what it rests on,
    replaces or names."""
    conn = L.open_seeded(tmp_path)
    L.add(conn, L.entity("e-sasha", "Саша"))
    L.raw(conn, "r1", "у Саши день рождения 12 мая")
    L.add(conn, L.fact("f1", "e-sasha", "birthday", "05-12", source_raw_id="r1"))
    L.add(conn, L.fact("f2", "e-sasha", "birthday", "05-14", origin_quote="нет, 14 мая"))
    member = LinkPayload(justification=JustificationIn(member="r1", provenance="rule"))
    L.commit(conn, ProposalOp(op=OpKind.LINK, target="f2", group="g", payload=payload_dict(member)))
    assert ("f1", "r1") in L.edges(conn, "derived_from")
    assert ("f1", "e-sasha") in L.edges(conn, "about")
    assert ("f2", "f1") in L.edges(conn, "supersedes")  # by rule: single cardinality
    assert ("r1", "f2") in L.edges(conn, "supports")


T0 = datetime(2030, 3, 4, 5, 6, 7, tzinfo=UTC)


@pytest.fixture
def wired():
    clk = ManualClock(T0)
    forget.reset_wiring()
    api.wire(clock=clk)
    yield clk
    forget.reset_wiring()  # back to the system clock


def test_store_now_follows_wired_clock(tmp_path, wired):
    """§16.4: `api.wire(clock=...)` moves `store.now()`, events.ts, trash.until and freezes."""
    assert store.now() == iso(T0)
    conn = L.open_seeded(tmp_path)
    assert L.add(conn, L.entity("e-s", "Sasha")).ok
    (ts,) = conn.execute("SELECT ts FROM events ORDER BY id DESC LIMIT 1").fetchone()
    assert ts == iso(T0)
    wired.advance(60)
    assert L.add(conn, L.fact("f-1", "e-s", "moving_to", "Berlin")).ok
    created = conn.execute("SELECT created_at FROM records WHERE id = 'f-1'").fetchone()[0]
    assert created == iso(T0 + timedelta(seconds=60))
    fid = forget.stop_use(conn, ["f-1"], scope=ForgetScope.ITEM, forever=False)
    (until,) = conn.execute("SELECT until FROM trash WHERE forget_id = ?", (fid,)).fetchone()
    assert until == iso(T0 + timedelta(seconds=60) + forget.TRASH_WINDOW)
    assert policy._now() == iso(T0 + timedelta(seconds=60))
    assert backup.make_set(conn, tmp_path / "sets").name.startswith("20300304T050707")
    forget.reset_wiring()
    assert store.now()[:4] != "2030"  # unwired: the system clock again
    conn.close()


def test_self_entity_seeded(tmp_path):
    """§4.5: the first open creates the user's entity `self`; facts about the user use it."""
    path = tmp_path / "memory" / "memory.db"
    conn = store.open(path, home=tmp_path / "home", owner_name="Денис")
    row = conn.execute(
        "SELECT r.kind, r.author_tier, r.status, e.etype, e.display_name FROM records r"
        " JOIN entities e ON e.id = r.id WHERE r.id = ?",
        (OWNER_ENTITY_ID,),
    ).fetchone()
    assert tuple(row) == ("entity", "user", "active", "person", "Денис")
    assert store.memory_version(conn) == 0  # a seed, not a commit
    for word in ("меня", "мне", "я", "me", "Денис"):
        hits = [r.entity_id for r in entities.resolve_rows(conn, word, "person")]
        assert hits == [OWNER_ENTITY_ID], word
    assert L.add(conn, L.fact("f-me", OWNER_ENTITY_ID, "lives_in", "Tver")).ok
    assert ("f-me", OWNER_ENTITY_ID) in L.edges(conn, "about")
    conn.close()
    again = store.open(path, home=tmp_path / "home", owner_name="Someone else")
    assert again.execute("SELECT count(*) FROM entities").fetchone()[0] == 1  # never twice
    name = again.execute("SELECT display_name FROM entities WHERE id = 'self'").fetchone()[0]
    assert name == "Денис"
    again.close()
    plain = store.open(":memory:")
    assert plain.execute("SELECT display_name FROM entities").fetchone()[0] == "Owner"
    plain.close()


def test_add_without_root_rejected(tmp_path):
    """§3.2: an ADD of a non-entity record with no raw, no derived_from, no member and no
    origin_quote is refused (`no_lineage`)."""
    conn = L.open_seeded(tmp_path)
    assert L.add(conn, L.entity("e-s", "Sasha")).ok  # entities are roots
    bare = L.fact("f-bare", "e-s", "works_at", "Yandex").model_copy(update={"origin_quote": None})
    with pytest.raises(CommitRejected) as e:
        L.add(conn, bare)  # the user's tier, hot path: the writer hears it
    assert e.value.reason == "no_lineage"
    res = L.add(conn, bare, tier=AuthorTier.SYSTEM, model_id="w")  # a worker: dropped alone
    assert not res.ok and res.rejected["e-s"].startswith("no_lineage")
    assert conn.execute("SELECT 1 FROM records WHERE id = 'f-bare'").fetchone() is None
    # each root is enough on its own
    L.raw(conn, "r1", "Саша работает в Яндексе")
    assert L.add(conn, L.fact("f-raw", "e-s", "works_at", "Yandex", source_raw_id="r1")).ok
    by_src = bare.model_copy(update={"id": "f-src"})
    assert L.add(conn, by_src, derived_from=["r1"]).ok
    member = JustificationIn(member="r1", provenance="rule").model_dump(mode="json")
    assert L.add(conn, bare.model_copy(update={"id": "f-mem"}), justifications=[member]).ok
    quoted = bare.model_copy(update={"id": "f-q", "origin_quote": "Саша в Яндексе"})
    assert L.add(conn, quoted).ok
    conn.close()


def test_version_reservation_file(tmp_path):
    """§7.1: `version.hwm` is reserved ahead of `memory_version` and only grows."""
    conn = L.open_seeded(tmp_path)
    hwm = tmp_path / "memory" / store.HWM_FILE
    assert store.read_hwm(hwm) == 0  # nothing committed yet
    assert L.add(conn, L.entity("e-s", "Sasha")).ok
    v = store.memory_version(conn)
    assert json.loads(hwm.read_text()) == {"reserved": v + store.HWM_STEP}
    before = hwm.stat().st_mtime_ns
    for i in range(5):
        assert L.add(conn, L.entity(f"e-{i}", f"N{i}")).ok
    assert hwm.stat().st_mtime_ns == before  # one small write per 1,000 commits
    assert store.read_hwm(hwm) >= store.memory_version(conn)
    store.write_hwm(hwm, 5)  # never lowers
    assert store.read_hwm(hwm) == v + store.HWM_STEP
    conn.execute("UPDATE meta SET memory_version = ?", (v + store.HWM_STEP,))
    assert L.add(conn, L.entity("e-next", "Next")).ok  # past the reservation: raised first
    assert store.read_hwm(hwm) == v + store.HWM_STEP + 1 + store.HWM_STEP
    s = backup.make_set(conn, tmp_path / "sets")
    assert store.read_hwm(s / store.HWM_FILE) == store.read_hwm(hwm)  # in every set
    assert store.read_hwm(tmp_path / "missing.hwm") == 0
    conn.close()
    mem = store.open(":memory:")
    assert L.add(mem, L.entity("e-m", "M")).ok  # no file for ":memory:"
    mem.close()
