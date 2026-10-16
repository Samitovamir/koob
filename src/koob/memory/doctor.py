"""G12 self-maintenance and health, at startup and nightly; the user sees one plain line only
when memory is unusable.

Spec: §18. The SQLite version and synced-path checks live in store.py and are reported here.

`check(conn)` reads and returns content-free problem codes (`code` or `code:N`); `repair(conn)`
does the light, safe repairs (WAL checkpoint, backup rotation, the orphan-vector sweep,
re-armed parked jobs) and returns what it did. Heavy work (sets, rewrites, purges, decay) is
the night pass's (`workers.nightly`). Recovery from a damaged file needs the live connections
closed: the host runs `backup.recover` before opening; here a damaged file is reported.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from . import clock as _clockmod
from . import store

WAL_MAX_BYTES = 64 * 1024 * 1024
BACKUP_MAX_AGE = timedelta(hours=48)
NIGHT_MAX_AGE = timedelta(days=3)
GROWTH_MAX_BYTES = 2 * 1024**3  # per 30 days (§18)
JUNK_MIN_SAMPLE = 50
JUNK_ALARM = 0.30  # share of memory-worthy arrivals over a week (§18)
SETS_KEPT = 3 + 2

# Codes after which memory cannot serve; only these reach the user, as one plain line.
UNUSABLE = frozenset({"sqlite_too_old", "synced_path", "corrupt", "unopenable"})
OWNER_LINE = "Память сейчас не работает: отвечаю без неё, пока она чинится."


def _parse(ts: object) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _open_checks(conn: sqlite3.Connection) -> list[str]:
    out = []
    try:
        store.check_sqlite()
    except store.SQLiteTooOld:
        out.append("sqlite_too_old")
    live = store.db_file(conn)
    if live is not None:
        try:
            store.check_path(live)
        except store.SyncedPathRefused:
            out.append("synced_path")
    pragmas = {"foreign_keys": 1, "secure_delete": 1, "auto_vacuum": 2}
    for name, want in pragmas.items():
        if conn.execute(f"PRAGMA {name}").fetchone()[0] != want:
            out.append(f"pragma_{name}")
    if live is not None and conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
        out.append("pragma_journal_mode")
    if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        out.append("corrupt")
    return out


def _fts_checks(conn: sqlite3.Connection) -> list[str]:
    out = []
    for table in ("fts_records", "fts_names", "fts_raw"):
        try:
            conn.execute(f"INSERT INTO {table} ({table}) VALUES ('integrity-check')")
        except sqlite3.DatabaseError:
            out.append(f"fts_damaged:{table}")
    return out


def _file_checks(conn: sqlite3.Connection, now: datetime) -> list[str]:
    from . import backup

    live = store.db_file(conn)
    if live is None:
        return []
    out = []
    paths = store.MemoryPaths.of(live)
    wal = live.with_name(live.name + "-wal")
    if wal.exists() and wal.stat().st_size > WAL_MAX_BYTES:
        out.append("wal_large")
    sets = backup.list_sets(paths.backups)
    if not sets:
        out.append("backup_none")
    else:
        if now - sets[0].created_at > BACKUP_MAX_AGE:
            out.append("backup_stale")
        if len(sets) > SETS_KEPT:
            out.append("backup_rotation_due")
        if paths.hotlog.exists():
            hot = backup.open_hotlog(paths.hotlog)
            try:
                old = [
                    o
                    for o in backup.hotlog_ops(hot)
                    if (o.memory_version or 0) and (o.memory_version or 0) <= sets[0].memory_version
                ]
            finally:
                hot.close()
            if old:
                out.append("hotlog_untruncated")
    if paths.vectors.exists():
        served = None
        try:
            vdb = sqlite3.connect(f"file:{paths.vectors}?mode=ro", uri=True)
            try:
                row = vdb.execute("SELECT value FROM vmeta WHERE key = 'serving'").fetchone()
                served = row[0] if row else None
            finally:
                vdb.close()
        except sqlite3.DatabaseError:
            out.append("vectors_unreadable")  # rebuildable: the next embed job remakes it
        meta = conn.execute("SELECT embedder_id FROM meta WHERE id = 1").fetchone()[0]
        if served and meta and served != meta:
            out.append("reembed_pending")
    out += _journal_check(conn, paths)
    return out


def _journal_check(conn: sqlite3.Connection, paths: store.MemoryPaths) -> list[str]:
    """Every forget the journal holds stopped (and not undone) is out of use in the store."""
    from . import backup
    from .model import ForgetStage

    gone: list[str] = []
    for st in backup.fold_journal(backup.read_journal(paths.journal)):
        if st.stage in (ForgetStage.STOPPED, ForgetStage.PURGED):
            gone += st.record_ids
    if not gone:
        return []
    marks = ",".join("?" * len(gone))
    n = conn.execute(
        f"SELECT count(*) FROM records WHERE id IN ({marks})"
        " AND status NOT IN ('forgotten', 'purged')",
        gone,
    ).fetchone()[0]
    return [f"journal_not_applied:{n}"] if n else []


def _growth_check(conn: sqlite3.Connection, now: datetime) -> list[str]:
    from .workers import nightly

    size = nightly.db_bytes(conn)
    for mark in nightly.marks(conn):  # oldest first: the first mark at least 25 days old
        at = _parse(mark.get("at"))
        if at is None or now - at < timedelta(days=25):
            continue
        days = max((now - at).days, 1)
        grew = size - int(mark.get("db_bytes") or 0)  # type: ignore[call-overload]
        return ["disk_growth"] if grew * 30 / days > GROWTH_MAX_BYTES else []
    return []


def _jobs_checks(conn: sqlite3.Connection, now: datetime) -> list[str]:
    from .workers import nightly

    out = []
    for state, code in (("parked", "jobs_parked"), ("needs_attention", "jobs_needs_attention")):
        n = conn.execute("SELECT count(*) FROM jobs WHERE state = ?", (state,)).fetchone()[0]
        if n:
            out.append(f"{code}:{n}")
    stuck = conn.execute(
        "SELECT count(*) FROM jobs WHERE state = 'running' AND not_before < ?", (_stamp(now),)
    ).fetchone()[0]
    if stuck:
        out.append(f"jobs_stuck:{stuck}")
    last = nightly.last_mark(conn)
    at = _parse(last.get("at")) if last else None
    if at is not None and now - at > NIGHT_MAX_AGE:
        out.append("nightly_missed")
    return out


def _junk_check(conn: sqlite3.Connection, now: datetime) -> list[str]:
    total, worthy = conn.execute(
        "SELECT count(*), coalesce(sum(junk = 0), 0) FROM raw_chunks WHERE ts >= ?",
        (_stamp(now - timedelta(days=7)),),
    ).fetchone()
    if total >= JUNK_MIN_SAMPLE and worthy / total > JUNK_ALARM:
        return ["junk_ratio_high"]
    return []


def _migrations_check(conn: sqlite3.Connection) -> list[str]:
    files = {p.name for p in store.MIGRATIONS_DIR.glob("*.sql")}
    applied = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
    return ["migrations_pending"] if files - applied else []


def check(conn: sqlite3.Connection) -> list[str]:
    """Problems, content-free (version, path, pragmas, quick_check, WAL size, backups,
    hot log, FTS optimize, orphan vectors, embedder, Keychain key, growth, junk ratio, jobs)."""
    from . import forget

    now = _clockmod.current().now()
    problems = _open_checks(conn)
    if "corrupt" in problems:
        return problems  # nothing else is trustworthy on a damaged file
    problems += _fts_checks(conn)
    problems += _migrations_check(conn)
    problems += _file_checks(conn, now)
    if not forget.has_key():
        problems.append("keychain_key_missing")
    problems += _growth_check(conn, now)
    problems += _jobs_checks(conn, now)
    problems += _junk_check(conn, now)
    return problems


def repair(conn: sqlite3.Connection) -> dict[str, int]:
    """Light, idempotent repairs; content-free counts. Never raises for one repair."""
    from . import backup
    from .workers import scheduler

    out: dict[str, int] = {}
    live = store.db_file(conn)
    if live is not None:
        try:
            busy = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()[0]
            out["wal_checkpoint_busy"] = int(busy)
        except sqlite3.DatabaseError:
            out["wal_checkpoint_busy"] = 1
        paths = store.MemoryPaths.of(live)
        try:
            out["sets_rotated"] = len(backup.rotate(paths.backups))
        except OSError:
            out["sets_rotated"] = 0
        if paths.vectors.exists():
            try:
                from .recall import vectors

                vdb = vectors.open_db(paths.vectors)
                try:
                    out["orphan_vectors"] = vectors.sweep_orphans(conn, vdb)
                finally:
                    vdb.close()
            except Exception:  # noqa: BLE001 - vectors.db is rebuildable; the embed job remakes it
                out["orphan_vectors"] = 0
    try:
        out["jobs_rearmed"] = scheduler.rearm_parked(conn)
    except Exception:  # noqa: BLE001 - parked jobs stay parked, the check reports them
        out["jobs_rearmed"] = 0
    return out


def owner_line(problems: list[str]) -> str | None:
    """One plain line, only when memory cannot serve (§18); otherwise None (silent)."""
    codes = {p.split(":", 1)[0] for p in problems}
    return OWNER_LINE if codes & UNUSABLE else None


__all__ = ["OWNER_LINE", "UNUSABLE", "check", "owner_line", "repair"]
