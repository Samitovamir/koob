"""Ledger invariants: the memory spec §3.2 (store).

`store.check_invariants` is the checker; Hypothesis drives random op sequences through the
store's own write paths (commit with a reference applier, derived builds with read_deps,
served/pending rows, real forgets with purges, backup + restore replaying the journal). The
same property over the real ledger applier is the last test. Spec marker:
`[T storage/test_invariants]`."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from koob.memory import backup, forget, store
from koob.memory.clock import ManualClock, iso
from koob.memory.model import AuthorTier, ForgetScope, OpKind, ReadRef

from .. import _ledger as L
from .._store_seed import (
    T,
    add_derived,
    add_fact,
    add_op,
    add_raw,
    fix_op,
    ref,
    ref_apply,
)

OPS = ("raw", "add", "fix", "derive", "serve", "pending", "forget", "restore")
step = st.tuples(st.sampled_from(OPS), st.integers(0, 5), st.integers(0, 5), st.integers(0, 3))


class _World:
    def __init__(self, root: Path) -> None:
        self.paths = store.MemoryPaths(root / "memory")
        self.home = root / "home"
        self.clock = ManualClock(datetime(2026, 10, 12, 9, 0, tzinfo=UTC))
        forget.reset_wiring()
        forget.wire(clock=self.clock)
        self.conn = store.open(self.paths.db, home=self.home)
        self.restored = False

    def one(self, sql: str, *args):
        return self.conn.execute(sql, args).fetchone()

    def usable(self, rid: str) -> bool:
        row = self.one("SELECT status FROM records WHERE id = ?", rid)
        return row is not None and row[0] in ("active", "archived")

    def forget(self, rid: str, *, purge: bool) -> None:
        fid = forget.stop_use(self.conn, [rid], scope=ForgetScope.ITEM, forever=purge)
        if purge and not forget.purge(self.conn, fid):
            # an older set still holds it: the nightly rewrite, then the purge passes
            backup.rewrite_after_forgets(self.paths.backups)
            assert forget.purge(self.conn, fid)

    def apply(self, op: str, i: int, j: int, k: int) -> None:
        c = self.conn
        self.clock.advance(60)
        rid, wid = f"r{i}", f"w{j}"
        if op == "raw" and self.one("SELECT 1 FROM raw_chunks WHERE id = ?", f"w{i}") is None:
            with store.write_tx(c):
                add_raw(c, f"w{i}", f"message {i} word{i}", labels=(1 << k) >> 1)
        elif op == "add" and self.one("SELECT 1 FROM records WHERE id = ?", rid) is None:
            if self.one("SELECT 1 FROM raw_chunks WHERE id = ? AND status = 'active'", wid):
                assert store.commit(c, add_op(rid, f"fact {i}", raw=wid), ref_apply).ok
        elif op == "fix" and self.usable(rid):
            assert store.commit(c, fix_op(c, rid, f"fact {i} fixed {k}"), ref_apply).ok
        elif op == "derive":
            ids = sorted({f"r{i}", f"r{j}"})
            if all(self.usable(x) for x in ids):
                did = f"d{i}{j}"
                refs = [ref(c, x) for x in ids]
                labels = 0
                for x in ids:
                    labels |= self.one("SELECT labels FROM records WHERE id = ?", x)[0]
                with store.write_tx(c):
                    c.execute("DELETE FROM derived WHERE id = ?", (did,))
                    add_derived(c, did, refs, f"summary of {ids}", labels=labels)
        elif op == "serve" and self.one("SELECT 1 FROM records WHERE id = ?", rid):
            c.execute(
                "INSERT INTO served (thread_id, item_id, item_kind, version, via, created_at)"
                " VALUES (?, ?, 'fact', ?, 'basket', ?)",
                (f"t{k}", rid, ref(c, rid).version, T),
            )
        elif op == "pending" and self.usable(rid):
            if self.one("SELECT 1 FROM served WHERE thread_id = ? AND item_id = ?", f"t{k}", rid):
                v = ref(c, rid).version
                c.execute(
                    "INSERT OR IGNORE INTO pending_updates (thread_id, item_id, old_version,"
                    " new_version, created_at) VALUES (?, ?, ?, ?, ?)",
                    (f"t{k}", rid, v - 1, v, T),
                )
        elif op == "forget" and self.usable(rid):
            self.forget(rid, purge=bool(k % 2))
        elif op == "restore" and not self.restored:
            self.restored = True
            s = backup.make_set(c, clock=self.clock)
            c.close()
            backup.restore(s, self.paths.db, home=self.home)
            self.conn = store.open(self.paths.db, home=self.home)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(steps=st.lists(step, max_size=30))
def test_invariants(steps):
    """§3.2: ledger invariants hold over random op sequences (Hypothesis)."""
    root = Path(tempfile.mkdtemp(prefix="mem-inv-"))
    world = _World(root)
    try:
        for op, i, j, k in steps:
            world.apply(op, i, j, k)
            assert not world.conn.in_transaction
        assert store.check_invariants(world.conn) == []
    finally:
        world.conn.close()
        forget.reset_wiring()
        shutil.rmtree(root, ignore_errors=True)


SCRIPT = [
    ("raw", 0, 0, 1),
    ("raw", 1, 0, 2),
    ("add", 0, 0, 0),
    ("add", 1, 1, 0),
    ("add", 2, 0, 0),
    ("derive", 0, 1, 0),
    ("serve", 0, 0, 1),
    ("fix", 0, 0, 1),
    ("pending", 0, 0, 1),
    ("derive", 1, 2, 0),
    ("forget", 1, 0, 0),
    ("restore", 0, 0, 0),
    ("forget", 2, 0, 1),
    ("fix", 0, 0, 2),
    ("derive", 0, 0, 0),
    ("forget", 0, 0, 1),
]


def test_invariants_scripted_covers_every_op(tmp_path):
    """A fixed walk through every op, restore included, so the property's coverage does not
    rest on what Hypothesis happens to draw."""
    world = _World(tmp_path)
    try:
        for op, i, j, k in SCRIPT:
            world.apply(op, i, j, k)
            assert store.check_invariants(world.conn) == [], (op, i, j, k)
        c = world.conn
        assert world.restored
        assert c.execute("SELECT status FROM records WHERE id = 'r1'").fetchone()[0] == "forgotten"
        assert c.execute("SELECT status FROM records WHERE id = 'r0'").fetchone()[0] == "purged"
        # derived items that read a forgotten fact are invalid, or gone once it was purged
        valid = "SELECT count(*) FROM derived WHERE valid = 1 AND id IN ('d00', 'd01', 'd12')"
        assert c.execute(valid).fetchone()[0] == 0
    finally:
        world.conn.close()
        forget.reset_wiring()


# ---------------------------------------------------------------- the checker itself

BREAKS = {
    "empty read set": "UPDATE derived SET read_set = '[]' WHERE id = 'd1'",
    "read_deps does not mirror": "DELETE FROM read_deps WHERE item_id = 'r2'",
    "labels miss": "UPDATE raw_chunks SET labels = 1 WHERE id = 'w1'",
    "valid but reads a forgotten": "UPDATE records SET status = 'forgotten' WHERE id = 'r2'",
    "no lineage": "DELETE FROM edges WHERE from_id = 'r1'; UPDATE facts SET source_raw_id = NULL WHERE id = 'r1'",
    "justification member": (
        "INSERT INTO justifications (target_record_id, group_no, member_record_id, provenance, created_at)"
        " VALUES ('r1', 1, 'nowhere', 'stated', 't')"
    ),
    "still in fts_records": "UPDATE records SET status = 'purged' WHERE id = 'r9'",
    "still in fts_raw": "UPDATE raw_chunks SET status = 'forgotten' WHERE id = 'w1'",
    "text kept": "UPDATE raw_chunks SET status = 'forgotten' WHERE id = 'w2'",
    "served": (
        "INSERT INTO served (thread_id, item_id, item_kind, version, via, created_at)"
        " VALUES ('t', 'ghost', 'fact', 1, 'basket', 't')"
    ),
    "pending_update": (
        "INSERT INTO pending_updates (thread_id, item_id, old_version, new_version, created_at)"
        " VALUES ('t', 'ghost', 1, 2, 't')"
    ),
    "holder does not exist": (
        "INSERT INTO read_deps (holder_kind, holder_id, item_id, version) VALUES ('derived', 'dx', 'r1', 1)"
    ),
}


@pytest.mark.parametrize("violation", sorted(BREAKS))
def test_checker_catches_each_violation(mem, violation):
    """The checker is not vacuous: every §3.2 invariant has a break it reports."""
    add_raw(mem, "w1", "у Саши день рождения 12 мая")
    add_raw(mem, "w2", "Дима в Твери")
    mem.execute(
        "DELETE FROM fts_raw WHERE rowid = (SELECT fts_rowid FROM raw_chunks WHERE id = 'w2')"
    )
    add_fact(mem, "r1", "Sasha birthday 12 May", raw="w1")
    add_fact(mem, "r2", "Dima lives in Tver", raw="w2")
    add_fact(mem, "r9", "standalone", origin_quote="запомни")
    add_derived(mem, "d1", [ref(mem, "r1"), ref(mem, "r2")], "summary")
    assert store.check_invariants(mem) == []
    mem.executescript(BREAKS[violation])
    problems = store.check_invariants(mem)
    assert any(violation in p for p in problems), problems
    assert all("12 мая" not in p and "Tver" not in p for p in problems)  # ids only


def test_read_set_round_trip():
    from koob.memory.model import ReadSrc

    refs = [
        ReadRef(id="r1", version=3),
        ReadRef(id="w1", version=2, src=ReadSrc.RAW),
        ReadRef(id="d1", version=0, src=ReadSrc.DERIVED),
    ]
    text = store.dump_read_set(refs)
    assert text.startswith('[["r1", 3]')
    assert store.load_read_set(text) == refs


LEDGER_OPS = ("raw", "add", "add_from_raw", "retract", "supersede", "forget", "restore")
ledger_step = st.tuples(
    st.sampled_from(LEDGER_OPS), st.integers(0, 4), st.integers(0, 3), st.integers(0, 3)
)
VALUES = ("Berlin", "Moscow", "Tver", "Riga")


class _LedgerWorld(_World):
    """The same walk over the real applier (`ledger.apply_op`): ADD (rule match may
    supersede), SUPERSEDE, RETRACT, real forgets with purges, backup + restore."""

    def __init__(self, root: Path) -> None:
        from koob.memory import ledger

        super().__init__(root)
        self.ledger = ledger
        ledger.seed_relations(self.conn)
        assert L.add(self.conn, L.entity("ent-u", "Саша")).ok

    def _add(self, rid: str, value: str, raw: str | None) -> None:
        # the user's word on the hot path carries its quote (the lineage root, §3.2)
        quote = None if raw else f"запомни: Саша живёт в {value}"
        rec = L.fact(rid, "ent-u", "lives_in", value, source_raw_id=raw, origin_quote=quote)
        with contextlib.suppress(self.ledger.CommitRejected):
            L.add(
                self.conn,
                rec,
                tier=AuthorTier.SYSTEM if raw else None,
                model_id="w" if raw else None,
            )

    def apply(self, op: str, i: int, j: int, k: int) -> None:
        c = self.conn
        self.clock.advance(60)
        rid, wid = f"f{i}", f"w{j}"
        exists = self.one("SELECT 1 FROM records WHERE id = ?", rid) is not None
        if op == "raw" and self.one("SELECT 1 FROM raw_chunks WHERE id = ?", f"w{i}") is None:
            with store.write_tx(c):
                L.raw(c, f"w{i}", f"Саша живёт в {VALUES[k]} {i}", ts_=iso(self.clock.now()))
        elif op == "add" and not exists:
            self._add(rid, VALUES[k], None)
        elif op == "add_from_raw" and not exists:
            if self.one("SELECT 1 FROM raw_chunks WHERE id = ? AND status = 'active'", wid):
                self._add(rid, VALUES[k], wid)
        elif op == "retract" and self.usable(rid):
            assert L.commit(c, L.op(OpKind.RETRACT, target=rid, group=rid)).ok
        elif op == "supersede" and self.usable(rid) and self.usable(f"f{j}") and i != j:
            with contextlib.suppress(self.ledger.CommitRejected):
                L.commit(c, L.op(OpKind.SUPERSEDE, target=rid, by=f"f{j}", group=rid))
        elif op == "forget" and self.usable(rid):
            self.forget(rid, purge=bool(k % 2))
        elif op == "restore":
            super().apply(op, i, j, k)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(steps=st.lists(ledger_step, max_size=25))
# a retracted twin of the forgotten fact (same proposition): the forget takes every copy
# (forget._copies), so the «насовсем» purge passes with nothing left holding it
@example(steps=[("add", 0, 0, 3), ("add", 3, 0, 3), ("retract", 3, 0, 0), ("forget", 0, 0, 1)])
def test_invariants_over_ledger_ops(steps):
    """§3.2 over the real applier: the property above with `ledger.apply_op` ops (add with
    rule match, supersede, retract), real forgets and restore."""
    root = Path(tempfile.mkdtemp(prefix="mem-inv-ledger-"))
    world = _LedgerWorld(root)
    try:
        for op, i, j, k in steps:
            world.apply(op, i, j, k)
            assert not world.conn.in_transaction
            assert store.check_invariants(world.conn) == [], (op, i, j, k)
    finally:
        world.conn.close()
        forget.reset_wiring()
        shutil.rmtree(root, ignore_errors=True)
