"""Acceptance for the memory spec §18: the doctor checks and repairs; the user sees one
plain line only when memory is unusable. Spec marker: `[T maintenance/test_doctor::<case>]`."""

from __future__ import annotations

import re

from koob.memory import api, backup, doctor, store

from .. import _ledger as L

_CODE = re.compile(r"^[a-z_]+(:[a-z_0-9]+)?$")


def _job(conn, jid: str, state: str) -> None:
    conn.execute(
        "INSERT INTO jobs (id, jkind, payload, state, idempotency_key, created_at, updated_at)"
        " VALUES (?, 'extract', '{}', ?, ?, '2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z')",
        (jid, state, f"k-{jid}"),
    )


def test_doctor(mem, monkeypatch):
    """§18: the doctor checks and repairs; the user sees one plain line only when memory is unusable."""
    # a healthy store without a backup yet: a problem for the doctor, silence for the user
    rep = api.doctor(store=mem)
    assert "backup_none" in rep.problems and rep.owner_line is None
    backup.make_set(mem)
    rep = api.doctor(store=mem)
    assert rep.ok and rep.problems == [] and rep.owner_line is None
    # parked and stuck jobs: doctor lines, never the user's
    _job(mem, "j1", "parked")
    _job(mem, "j2", "needs_attention")
    rep = api.doctor(store=mem)
    assert "jobs_needs_attention:1" in rep.problems and rep.owner_line is None
    assert all(_CODE.match(p) for p in rep.problems)  # content-free codes
    # unusable: one plain line
    monkeypatch.setattr(store, "sqlite_version", lambda: (3, 40, 0))
    rep = api.doctor(store=mem)
    assert "sqlite_too_old" in rep.problems
    assert rep.owner_line == doctor.OWNER_LINE and "\n" not in rep.owner_line


def test_owner_line_only_when_unusable():
    assert doctor.owner_line([]) is None
    assert doctor.owner_line(["backup_stale", "jobs_parked:2", "junk_ratio_high"]) is None
    for code in ("corrupt", "synced_path", "sqlite_too_old", "unopenable"):
        assert doctor.owner_line([code]) == doctor.OWNER_LINE


def test_junk_ratio_alarm(mem):
    for i in range(60):  # a week where every arrival passed the junk gate
        L.raw(mem, f"r{i}", f"text {i}", ts_=store.now())
    assert "junk_ratio_high" in doctor.check(mem)
    mem.execute("UPDATE raw_chunks SET junk = 1 WHERE CAST(substr(id, 2) AS INTEGER) >= 10")
    assert "junk_ratio_high" not in doctor.check(mem)


def test_journal_replay_check(mem):
    L.add(mem, L.entity("sasha", "Sasha"))
    L.add(mem, L.fact("f1", "sasha", "nationality", "German"))
    paths = store.MemoryPaths.of(store.db_file(mem))
    paths.journal.parent.mkdir(parents=True, exist_ok=True)
    line = '{"ts": "2026-10-01T00:00:00Z", "stage": "stopped", "forget_id": "fg1", "memory_version": 1, "record_ids": ["f1"]}'
    paths.journal.write_text(line + "\n", encoding="utf-8")
    assert "journal_not_applied:1" in doctor.check(mem)
