"""Acceptance for the memory spec §6.4: decay archives, never forgets.
Spec marker: `[T consolidate/test_decay::<case>]`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from koob.memory import api, backup, forget, store
from koob.memory.clock import ManualClock
from koob.memory.model import AuthorTier
from koob.memory.workers import nightly

from .. import _ledger as L

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


@pytest.fixture
def clock():
    clk = ManualClock(T0)
    forget.reset_wiring()
    api.wire(clock=clk)
    nightly.use_power(lambda: True)
    yield clk
    nightly.use_power(None)
    forget.reset_wiring()
    api.wire(clock=None)


def _seed(conn) -> None:
    L.add(conn, L.entity("sasha", "Sasha"))
    L.add(conn, L.entity("dima", "Dima"))
    other = {"tier": AuthorTier.OTHER, "speaker_entity_id": "dima"}
    L.add(conn, L.fact("f-stale", "sasha", "nationality", "French", **other), rule_match=False)
    L.add(conn, L.fact("f-used", "sasha", "nationality", "Swiss", **other), rule_match=False)
    L.add(conn, L.fact("f-own", "sasha", "nationality", "German"), rule_match=False)


def test_archives_not_forgets(mem, clock):
    """§6.4: decay moves stale low-value items to archived; no journal line, no backup rewrite."""
    _seed(mem)
    first = backup.make_set(mem)
    manifest = (first / backup.MANIFEST).read_text(encoding="utf-8")
    clock.advance(timedelta(days=200).total_seconds())
    mem.execute(
        "INSERT INTO record_stats (record_id, uses, last_used_at) VALUES ('f-used', 3, ?)",
        (store.now(),),
    )
    counts = api.run_nightly(store=mem)
    assert counts["archived"] == 1
    assert L.status(mem, "f-stale") == "archived"
    assert L.status(mem, "f-used") == "active"  # used lately
    assert L.status(mem, "f-own") == "active"  # the user's own words never decay
    paths = store.MemoryPaths.of(store.db_file(mem))
    assert not backup.read_journal(paths.journal)  # no journal line
    assert counts["rewritten"] == 0  # no backup rewrite
    assert (first / backup.MANIFEST).read_text(encoding="utf-8") == manifest
    # the archive is kept: the row and its text are still there
    assert mem.execute("SELECT text_en FROM facts WHERE id = 'f-stale'").fetchone()[0]


def test_decay_is_weekly(mem, clock):
    _seed(mem)
    clock.advance(timedelta(days=200).total_seconds())
    assert api.run_nightly(store=mem)["archived"] == 2  # both stale items, none used
    L.add(
        mem,
        L.fact(
            "f-new",
            "sasha",
            "nationality",
            "Dutch",
            tier=AuthorTier.OTHER,
            speaker_entity_id="dima",
        ),
        rule_match=False,
    )
    mem.execute("UPDATE records SET recorded_at = '2026-01-01T00:00:00.000000Z' WHERE id = 'f-new'")
    assert api.run_nightly(store=mem)["archived"] == 0  # the next decay is a week later
