"""Connections, migrations, versions and read-set validation for the memory ledger.

Spec: the memory spec §3.1 (files, engine, pragmas), §3.2 (schema), §6.3 (propose/commit),
§7.1-§7.4 (versions, read sets, pre-send query).

Provides the SQLite version gate, the synced-path refusal, per-connection pragmas, the
migrations runner and `commit`, which every writer goes through. Also: the file layout
(`MemoryPaths`), write priority so a hot write never starves behind a worker (§3.1 write-lock
budget, G9), per-group lock timing and op rejection in `commit` (§6.3), the pre-send check
(§7.4, §7.6 gate), the cross-process watermark (§7.7), read-set serialisation and the reverse
index (`read_deps`, §3.2), and the §3.2 invariant checker. Backups live in backup.py.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, timedelta
from pathlib import Path
from typing import Literal

from . import clock as _clock
from .model import EventKind, Label, Proposal, ProposalOp, ReadRef, ReadSrc

MIN_SQLITE: tuple[int, int, int] = (3, 51, 3)  # WAL corruption fix (§3.1)
MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations" / "memory"
_MIGRATION_NAME = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")
_BUSY_TIMEOUT_MS = 5000


class MemoryStoreError(RuntimeError):
    """The store cannot be opened safely; the doctor reports the message (§18)."""


class SQLiteTooOld(MemoryStoreError):
    """The linked SQLite predates the WAL corruption fix (§3.1)."""


class SyncedPathRefused(MemoryStoreError):
    """The live DB would sit in a synced folder (§3.1)."""


def now() -> str:
    """The process clock (§16.4, `clock.use`) as stored in every memory table."""
    return _clock.now().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sqlite_version() -> tuple[int, int, int]:
    """The linked SQLite's version. A function so tests can stand in an old one."""
    major, minor, patch = sqlite3.sqlite_version_info[:3]
    return (major, minor, patch)


def check_sqlite() -> None:
    """Refuse SQLite < 3.51.3. `[T storage/test_open::refuses_old_sqlite]`"""
    have = sqlite_version()
    if have < MIN_SQLITE:
        need = ".".join(map(str, MIN_SQLITE))
        raise SQLiteTooOld(
            f"memory needs SQLite >= {need}, this Python links {'.'.join(map(str, have))}; "
            "run `uv python pin 3.13.15 && uv sync`"
        )


def _icloud_desktop_documents_on(home: Path) -> bool:
    """iCloud Desktop & Documents keeps its mirror folders under CloudDocs when it is on."""
    cloud = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    return (cloud / "Desktop").exists() or (cloud / "Documents").exists()


def check_path(path: Path, *, home: Path | None = None) -> None:
    """Refuse a live DB in a synced folder. `[T storage/test_open::refuses_synced_path]`"""
    home = (home or Path.home()).resolve()
    p = path.expanduser().resolve()
    if p.is_relative_to(home / "Library" / "Mobile Documents"):
        raise SyncedPathRefused(f"memory DB under iCloud Drive: {p}")
    if _icloud_desktop_documents_on(home) and any(
        p.is_relative_to(home / d) for d in ("Desktop", "Documents")
    ):
        raise SyncedPathRefused(f"memory DB under iCloud Desktop & Documents: {p}")


def connect(path: Path | str, *, journal_mode: str = "WAL") -> sqlite3.Connection:
    """One connection with the §3.1 pragmas. Every memory file goes through here;
    the hot log passes journal_mode="DELETE". Autocommit: writers open BEGIN IMMEDIATE
    themselves. `[T storage/test_open::pragmas_on_every_connection]`"""
    check_sqlite()
    conn = sqlite3.connect(str(path), autocommit=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    fresh = conn.execute("SELECT count(*) FROM sqlite_schema").fetchone()[0] == 0
    if fresh:
        # Only effective before the first table exists (§3.1: set at creation).
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
    mode = conn.execute(f"PRAGMA journal_mode = {journal_mode}").fetchone()[0]
    if str(path) != ":memory:" and mode.lower() != journal_mode.lower():
        conn.close()
        raise MemoryStoreError(f"journal_mode {journal_mode} refused (got {mode})")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA secure_delete = ON")
    return conn


def _applied(conn: sqlite3.Connection) -> set[str]:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if not exists:
        return set()
    return {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}


def migrate(conn: sqlite3.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every `NNNN_name.sql` not yet recorded, each in its own BEGIN IMMEDIATE.
    Returns the names applied. A failing migration rolls back whole."""
    done = _applied(conn)
    applied: list[str] = []
    for sql_file in sorted(directory.glob("*.sql")):
        name = sql_file.name
        if name in done:
            continue
        if not _MIGRATION_NAME.match(name):
            raise MemoryStoreError(f"bad migration name: {name}")
        body = sql_file.read_text(encoding="utf-8")
        number = int(name[:4])
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{body}\n;\n"
            f"INSERT INTO schema_migrations (name, applied_at) VALUES ('{name}', '{now()}');\n"
            f"UPDATE meta SET schema_version = max(schema_version, {number}), "
            f"updated_at = '{now()}' WHERE id = 1;\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        applied.append(name)
    return applied


OWNER_NAME: str | None = None  # `memory.owner_name` (§4.5); the host may set it before open


def open(  # noqa: A001
    path: Path | str, *, home: Path | None = None, owner_name: str | None = None
) -> sqlite3.Connection:
    """Open memory.db: SQLite gate, synced-path gate, pragmas, migrations (§3.1), the
    relation vocabulary (§4.1) and the owner entity `self` (§4.5; `owner_name`, else
    `OWNER_NAME`, else "Owner"). `path` may be ":memory:" in tests."""
    check_sqlite()
    if str(path) != ":memory:":
        p = Path(path)
        check_path(p, home=home)
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    migrate(conn)
    from .ledger import seed_owner, seed_relations  # idempotent seeds

    seed_relations(conn)
    seed_owner(conn, owner_name or OWNER_NAME)
    return conn


def memory_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT memory_version FROM meta WHERE id = 1").fetchone()[0])


# ---------------------------------------------------------------- files (§3.1)


@dataclass(frozen=True)
class MemoryPaths:
    """The §3.1 layout of one memory folder (`data/memory/` in production)."""

    root: Path

    @property
    def db(self) -> Path:
        return self.root / "memory.db"

    @property
    def vectors(self) -> Path:
        return self.root / "vectors.db"

    @property
    def hotlog(self) -> Path:
        return self.root / "hotlog.db"

    @property
    def journal(self) -> Path:
        return self.root / "journal" / "forget.jsonl"

    @property
    def backups(self) -> Path:
        return self.root / "backups"

    @classmethod
    def of(cls, db: Path | str) -> MemoryPaths:
        """The layout around a live memory.db."""
        return cls(Path(db).parent)

    @classmethod
    def default(cls) -> MemoryPaths:
        """`$KOOB_DATA_DIR/memory` (api.py: `store=None` means this file)."""
        from koob.paths import DATA_DIR

        return cls(DATA_DIR / "memory")


def db_file(conn: sqlite3.Connection) -> Path | None:
    """The main database file of a connection; None for ":memory:"."""
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main":
            return Path(row[2]) if row[2] else None
    return None


# ---------------------------------------------------------------- write priority (§3.1, G9)
#
# SQLite's write lock is not fair: a worker committing group after group starves a waiting
# writer until its busy timeout runs out (measured: a hot write behind a loop of 30 ms groups
# got "database is locked" after 5 s). So a hot writer announces itself with an intent file
# beside the DB and polls for the lock in 0.5 ms steps; a background writer looks for fresh
# intents before each group and yields until they clear. A hot write then waits at most one
# group's hold (<= 50 ms, §3.1). Intents are files so `cb` and the server see each other; one
# older than _INTENT_STALE_S (a crashed writer) is ignored and removed.

Priority = Literal["hot", "background"]
LOCK_BUDGET_MS = 50.0  # a commit group holds the write lock at most this long (§3.1)
_INTENT_STALE_S = 10.0
_YIELD_MAX_S = 2.0  # a background writer yields at most this long per group, then goes anyway
_POLL_S = 0.0005
_intent_seq = itertools.count()


def _intent_dir(db: Path) -> Path:
    return db.with_name(db.name + "-hot")


@contextmanager
def _hot_intent(db: Path | None) -> Iterator[None]:
    if db is None:
        yield
        return
    d = _intent_dir(db)
    d.mkdir(exist_ok=True)
    marker = d / f"{os.getpid()}-{threading.get_ident()}-{next(_intent_seq)}"
    marker.touch()
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def hot_waiting(db: Path | None) -> bool:
    """True while a hot writer waits for or holds the write lock on `db`."""
    if db is None:
        return False
    try:
        entries = list(os.scandir(_intent_dir(db)))
    except FileNotFoundError:
        return False
    now_s = time.time()
    fresh = False
    for e in entries:
        try:
            if now_s - e.stat().st_mtime > _INTENT_STALE_S:
                os.unlink(e.path)
            else:
                fresh = True
        except FileNotFoundError:
            continue
    return fresh


def _is_busy(exc: sqlite3.OperationalError) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    return code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or "locked" in str(exc)


def begin_write(conn: sqlite3.Connection, *, priority: Priority = "hot") -> float:
    """BEGIN IMMEDIATE with the §3.1 priority rule; returns the milliseconds waited.
    Hot: poll every 0.5 ms (no busy-handler back-off) for up to the busy timeout.
    Background: first yield to any fresh hot intent, then wait normally."""
    db = db_file(conn)
    t0 = time.perf_counter()
    if priority == "background":
        deadline = time.monotonic() + _YIELD_MAX_S
        while hot_waiting(db) and time.monotonic() < deadline:
            time.sleep(_POLL_S * 2)
        conn.execute("BEGIN IMMEDIATE")
        return (time.perf_counter() - t0) * 1000
    with _hot_intent(db):
        conn.execute("PRAGMA busy_timeout = 0")
        try:
            deadline = time.monotonic() + _BUSY_TIMEOUT_MS / 1000
            while True:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if not _is_busy(exc) or time.monotonic() >= deadline:
                        raise
                    time.sleep(_POLL_S)
        finally:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return (time.perf_counter() - t0) * 1000


@contextmanager
def write_tx(
    conn: sqlite3.Connection, *, priority: Priority = "hot"
) -> Iterator[sqlite3.Connection]:
    """One write transaction under the priority rule: COMMIT on success, ROLLBACK on any
    error. For writers that do not go through `commit` (forget, policy, jobs)."""
    begin_write(conn, priority=priority)
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------- read-set storage (§3.2, §7.2)


def dump_read_set(refs: Iterable[ReadRef]) -> str:
    """JSON for a `read_set` column: `[[id, version], ...]`, raw and derived refs carry
    their source as a third element."""
    out: list[list[object]] = []
    for ref in refs:
        item: list[object] = [ref.id, ref.version]
        if ref.src is not ReadSrc.RECORD:
            item.append(ref.src.value)
        out.append(item)
    return json.dumps(out, ensure_ascii=False)


def load_read_set(text: str | None) -> list[ReadRef]:
    refs: list[ReadRef] = []
    for item in json.loads(text or "[]"):
        if isinstance(item, dict):
            refs.append(ReadRef(**item))
        else:
            src = ReadSrc(item[2]) if len(item) > 2 else ReadSrc.RECORD
            refs.append(ReadRef(id=item[0], version=int(item[1]), src=src))
    return refs


def set_read_deps(
    conn: sqlite3.Connection, holder_kind: str, holder_id: str, refs: Iterable[ReadRef]
) -> None:
    """Mirror a holder's read set into `read_deps`, inside the holder's own transaction
    (§3.2: the reverse index mirrors every stored read set exactly)."""
    conn.execute(
        "DELETE FROM read_deps WHERE holder_kind = ? AND holder_id = ?", (holder_kind, holder_id)
    )
    conn.executemany(
        "INSERT OR REPLACE INTO read_deps (holder_kind, holder_id, item_id, version) "
        "VALUES (?, ?, ?, ?)",
        [(holder_kind, holder_id, ref.id, ref.version) for ref in refs],
    )


# ---------------------------------------------------------------- read sets (§7.2, §7.4)


def stale_refs(
    conn: sqlite3.Connection, refs: Iterable[ReadRef], *, require_usable: bool = False
) -> list[ReadRef]:
    """The refs whose item is gone or changed since it was read. Records compare
    `version` (and, for the pre-send check, usability); raw compares `rev` and requires
    `active`; derived compares its `rev` (bumped on every rebuild, migration 0002: a prompt
    holding the text built before a forget is stale after the rebuild) and requires
    `valid = 1`. One indexed query per source (§7.4)."""
    refs = list(refs)
    stale: list[ReadRef] = []
    by_src: dict[ReadSrc, list[ReadRef]] = {}
    for ref in refs:
        by_src.setdefault(ref.src, []).append(ref)
    for src, group in by_src.items():
        for chunk_start in range(0, len(group), 400):
            chunk = group[chunk_start : chunk_start + 400]
            values = ", ".join("(?, ?)" for _ in chunk)
            params = [x for ref in chunk for x in (ref.id, ref.version)]
            if src is ReadSrc.RECORD:
                usable = " OR r.status NOT IN ('active', 'archived')" if require_usable else ""
                sql = (
                    f"WITH rs(id, v) AS (VALUES {values}) SELECT rs.id, rs.v FROM rs "
                    "LEFT JOIN records r ON r.id = rs.id "
                    f"WHERE r.id IS NULL OR r.version <> rs.v{usable}"
                )
            elif src is ReadSrc.RAW:
                sql = (
                    f"WITH rs(id, v) AS (VALUES {values}) SELECT rs.id, rs.v FROM rs "
                    "LEFT JOIN raw_chunks c ON c.id = rs.id "
                    "WHERE c.id IS NULL OR c.rev <> rs.v OR c.status <> 'active'"
                )
            else:
                sql = (
                    f"WITH rs(id, v) AS (VALUES {values}) SELECT rs.id, rs.v FROM rs "
                    "LEFT JOIN derived d ON d.id = rs.id"
                    " WHERE d.id IS NULL OR d.valid = 0 OR d.rev <> rs.v"
                )
            bad = {(row[0], row[1]) for row in conn.execute(sql, params)}
            stale.extend(ref for ref in chunk if (ref.id, ref.version) in bad)
    return stale


def op_refs(op: ProposalOp) -> list[ReadRef]:
    """Every ref an op read: its `read_set` plus a derived ADD's payload read set (a payload
    read set is checked like the op's, never trusted)."""
    refs = list(op.read_set)
    derived = op.payload.get("derived") if isinstance(op.payload, dict) else None
    if isinstance(derived, dict) and derived.get("read_set"):
        refs += load_read_set(json.dumps(derived["read_set"]))
    return refs


# ---------------------------------------------------------------- version reservation (§7.1, A5)
#
# Background commits after the last backup are in neither the journal nor the hot log, so a
# restore floor from those alone may hand out a version already served. `version.hwm` beside
# memory.db holds `{"reserved": N}`: no committed version exceeds N. A commit that would pass
# N first raises it to version + 1000 (atomic rename + fsync), so one small write covers
# 1,000 commits; it only grows, and every backup set carries a copy.

HWM_FILE = "version.hwm"
HWM_STEP = 1000
_hwm_seen: dict[str, int] = {}  # resolved path -> reservation this process last read/wrote
_hwm_lock = threading.Lock()


def hwm_path(conn: sqlite3.Connection) -> Path | None:
    """`version.hwm` beside the connection's file; None for ":memory:"."""
    db = db_file(conn)
    return db.parent / HWM_FILE if db is not None else None


def read_hwm(path: Path | str | None) -> int:
    """The reservation in a `version.hwm` file; 0 when missing or unreadable."""
    if path is None:
        return 0
    try:
        return int(json.loads(Path(path).read_text(encoding="utf-8"))["reserved"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def write_hwm(path: Path, reserved: int) -> int:
    """Raise the reservation to at least `reserved` (atomic rename + fsync; never lowers).
    Returns the reservation now on disk."""
    with _hwm_lock:
        value = max(reserved, read_hwm(path))
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"reserved": value}))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            fd = None
        if fd is not None:
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _hwm_seen[str(path.resolve())] = value
        return value


def reserve_version(conn: sqlite3.Connection, version: int) -> None:
    """Make sure `version.hwm` covers `version` before it is committed (§7.1)."""
    path = hwm_path(conn)
    if path is None:
        return
    key = str(path.resolve())
    if version <= _hwm_seen.get(key, 0):
        return
    on_disk = read_hwm(path)
    if version <= on_disk:
        _hwm_seen[key] = on_disk
        return
    write_hwm(path, version + HWM_STEP)


# ---------------------------------------------------------------- job leases (§18.1, A3)
#
# A `running` job holds a lease in `not_before` (now + 5 min), renewed per group; startup
# puts a `running` job whose lease ended back to `queued` with attempts + 1, past the cap to
# `needs_attention`, and never takes a live lease. Job runners register by kind; startup
# runs the kinds that must finish before a turn (`mode_forget`, §5.2).

LEASE_S = 300.0
JOB_ATTEMPTS_CAP = 3
# G-J1: a lease also names its holder (`payload.lease_holder` = "<pid>:<boot nonce>"). A
# restart inside the 5-minute lease must not wait it out: startup starts a new boot, and a
# lease held by an earlier boot of this process or by a process that no longer exists is
# taken at once. A lease with no holder (or a live other process's) ends only by its time.
_BOOT = [secrets.token_hex(8)]
HOLD_SQL = (
    "payload = CASE WHEN json_valid(payload) THEN json_set(payload, '$.lease_holder', ?)"
    " ELSE payload END"
)
JobRunner = Callable[[sqlite3.Connection, dict[str, object]], object]
_job_runners: dict[str, tuple[JobRunner, bool]] = {}


def lease_until(seconds: float = LEASE_S) -> str:
    """The `not_before` of a job claimed (or renewed) now."""
    return (
        (_clock.now() + timedelta(seconds=seconds))
        .astimezone(UTC)
        .strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )


def lease_holder() -> str:
    """Who takes a lease now: this process and its current boot (G-J1)."""
    return f"{os.getpid()}:{_BOOT[0]}"


def new_boot() -> None:
    """`api.startup`: the memory host (re)starts in this process, so leases it took before
    belong to a run that is over."""
    _BOOT[0] = secrets.token_hex(8)


def holder_gone(payload: object) -> bool:
    """True when the job's lease holder no longer runs: an earlier boot of this process, or
    a process that is gone. No holder, or a live other process: False (its time decides)."""
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload or "{}")
        except ValueError:
            return False
    holder = payload.get("lease_holder") if isinstance(payload, dict) else None
    pid_s, _, nonce = str(holder or "").partition(":")
    if not pid_s.isdigit() or not nonce:
        return False
    pid = int(pid_s)
    if pid == os.getpid():
        return nonce != _BOOT[0]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:  # alive under another user, or unknowable: not ours to take
        return False
    return False


def renew_lease(conn: sqlite3.Connection, job_id: str) -> None:
    """Push a running job's lease on (call once per group). The caller's transaction, or
    its own short one."""
    sql = "UPDATE jobs SET not_before = ?, updated_at = ? WHERE id = ? AND state = 'running'"
    if conn.in_transaction:
        conn.execute(sql, (lease_until(), now(), job_id))
    else:
        with write_tx(conn, priority="background"):
            conn.execute(sql, (lease_until(), now(), job_id))


def register_job_runner(jkind: str, fn: JobRunner, *, at_startup: bool = False) -> None:
    """A runner for one job kind: `fn(conn, job)` with the job row as a dict (payload
    parsed). `at_startup`: `api.startup` runs its queued jobs before any turn (§18.1 step 5)."""
    _job_runners[jkind] = (fn, at_startup)


def job_runner(jkind: str) -> tuple[JobRunner, bool] | None:
    return _job_runners.get(jkind)


def startup_job_kinds() -> list[str]:
    return sorted(k for k, (_fn, at_start) in _job_runners.items() if at_start)


def claim_job(conn: sqlite3.Connection, job_id: str) -> bool:
    """Take a `queued` job: `running` with a fresh lease. False when another runner holds it
    (or it is no longer queued)."""
    with write_tx(conn):
        cur = conn.execute(
            f"UPDATE jobs SET state = 'running', not_before = ?, updated_at = ?, {HOLD_SQL}"
            " WHERE id = ? AND state = 'queued'",
            (lease_until(), now(), lease_holder(), job_id),
        )
    return cur.rowcount == 1


def _job_row(conn: sqlite3.Connection, job_id: str) -> dict[str, object]:
    row = conn.execute(
        "SELECT id, jkind, payload, state, attempts, not_before, idempotency_key FROM jobs"
        " WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return {}
    keys = ("id", "jkind", "payload", "state", "attempts", "not_before", "idempotency_key")
    job: dict[str, object] = dict(zip(keys, tuple(row), strict=True))
    try:
        job["payload"] = json.loads(str(job["payload"] or "{}"))
    except ValueError:
        job["payload"] = {}
    return job


def run_queued_jobs(conn: sqlite3.Connection, jkind: str) -> int:
    """Run every due `queued` job of one kind through its registered runner: claim, run,
    then `done` unless the runner set another state. A runner that raises puts its job back
    with attempts + 1 (past the cap: `needs_attention`); the others still run. Returns the
    jobs finished."""
    entry = _job_runners.get(jkind)
    if entry is None:
        return 0
    fn = entry[0]
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM jobs WHERE jkind = ? AND state = 'queued'"
            " AND (not_before IS NULL OR not_before <= ?) ORDER BY created_at, id",
            (jkind, now()),
        )
    ]
    done = 0
    for job_id in ids:
        if not claim_job(conn, job_id):
            continue
        try:
            fn(conn, _job_row(conn, job_id))
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            with write_tx(conn):
                conn.execute(
                    "UPDATE jobs SET state = CASE WHEN attempts + 1 > ? THEN 'needs_attention'"
                    " ELSE 'queued' END, attempts = attempts + 1, not_before = ?, updated_at = ?"
                    " WHERE id = ? AND state = 'running'",
                    (JOB_ATTEMPTS_CAP, now(), now(), job_id),
                )
            continue
        with write_tx(conn):
            conn.execute(
                "UPDATE jobs SET state = 'done', updated_at = ? WHERE id = ? AND state = 'running'",
                (now(), job_id),
            )
        done += 1
    return done


def requeue_expired_jobs(conn: sqlite3.Connection) -> tuple[int, int]:
    """§18.1 step 3: every `running` job whose lease ended (or that has none, or whose
    holder is gone, G-J1) → `queued`, attempts + 1, due now. A job's own retries
    (`payload.retries`, G-J2: a purge waiting on an old backup) are not crashes: past
    `JOB_ATTEMPTS_CAP` crashes → `needs_attention`. A live lease is left alone. Returns
    (requeued, needing attention)."""
    ts = now()
    requeued = parked = 0
    with write_tx(conn):
        rows = conn.execute(
            "SELECT id, attempts, payload, not_before FROM jobs WHERE state = 'running'"
        ).fetchall()
        for job_id, attempts, payload, lease in rows:
            if lease is not None and lease > ts and not holder_gone(payload):
                continue
            try:
                body = json.loads(payload or "{}")
            except ValueError:
                body = {}
            retries = int(body.get("retries") or 0) if isinstance(body, dict) else 0
            n = int(attempts or 0) + 1
            if n - retries > JOB_ATTEMPTS_CAP:
                state, parked = "needs_attention", parked + 1
            else:
                state, requeued = "queued", requeued + 1
            conn.execute(
                "UPDATE jobs SET state = ?, attempts = ?, not_before = ?, updated_at = ?"
                " WHERE id = ?",
                (state, n, ts, ts, job_id),
            )
    return requeued, parked


# ---------------------------------------------------------------- commit (§6.3)


class OpRejected(MemoryStoreError):
    """Raised by an op applier for a check that fails on content, not on staleness:
    a suppression or retracted value (§9.6), label monotonicity, tier or vocabulary (§6.3).
    `commit` rolls the op's group back alone and reports it; other groups still commit."""


@dataclass(frozen=True)
class CommitCtx:
    """What an op applier gets inside the transaction."""

    conn: sqlite3.Connection
    version: int  # the new memory_version; records changed now take it as `version`
    ts: str
    proposal: Proposal


@dataclass
class CommitResult:
    committed: list[str] = field(default_factory=list)  # group keys
    stale: dict[str, list[ReadRef]] = field(default_factory=dict)  # group key -> stale refs
    rejected: dict[str, str] = field(default_factory=dict)  # group key -> OpRejected reason
    versions: dict[str, int] = field(default_factory=dict)  # group key -> memory_version
    lock_ms: dict[str, float] = field(default_factory=dict)  # group key -> write-lock hold
    waited_ms: float = 0.0  # total time spent waiting for the write lock

    @property
    def ok(self) -> bool:
        return not self.stale and not self.rejected

    @property
    def over_budget(self) -> list[str]:
        """Groups that held the lock longer than §3.1 allows: the worker chunks smaller."""
        return [k for k, ms in self.lock_ms.items() if ms > LOCK_BUDGET_MS]


OpApplier = Callable[[CommitCtx, ProposalOp], Sequence[str] | None]


def commit(
    conn: sqlite3.Connection,
    proposal: Proposal,
    apply: OpApplier | None = None,
    *,
    event_kind: EventKind = EventKind.WRITE,
    priority: Priority | None = None,
) -> CommitResult:
    """Commit a proposal group by group (§6.3, G2). Each group runs under BEGIN IMMEDIATE:
    its ops' read sets are validated inside the transaction; a stale group rolls back alone
    and is reported for requeue; a clean group bumps `meta.memory_version` once, applies its
    ops through `apply` (the ledger's op handler; it returns the item ids it touched) and
    writes one content-free `events` row (§7.7). Suppressions, retracted values, label
    monotonicity, tiers and the vocabulary are the applier's checks (§6.3, ledger); it
    raises `OpRejected`: a worker proposal (`model_id` set) drops that group alone and
    reports it in `rejected`; a hot or code proposal (`model_id` None) re-raises it to its
    writer. A derived ADD's payload read set is validated with the op's. Every new version
    is reserved in `version.hwm` first (§7.1). Priority (§3.1): worker proposals
    (`model_id` set) default to "background" and yield to hot writers between groups;
    hot-path and code writes default to "hot".
    `[T races/test_commit::stale_group_rejected_others_commit]`"""
    prio: Priority = priority or ("background" if proposal.model_id else "hot")
    result = CommitResult()
    groups: dict[str, list[ProposalOp]] = {}
    for op in proposal.ops:
        groups.setdefault(op.group, []).append(op)
    for key, ops in groups.items():
        result.waited_ms += begin_write(conn, priority=prio)
        held_from = time.perf_counter()
        try:
            refs = [ref for op in ops for ref in op_refs(op)]
            bad = stale_refs(conn, refs)
            if bad:
                conn.execute("ROLLBACK")
                result.stale[key] = bad
                continue
            version = conn.execute(
                "UPDATE meta SET memory_version = memory_version + 1, updated_at = ? "
                "WHERE id = 1 RETURNING memory_version",
                (now(),),
            ).fetchone()[0]
            reserve_version(conn, version)  # before the version can be served (§7.1)
            ctx = CommitCtx(conn=conn, version=version, ts=now(), proposal=proposal)
            touched: list[str] = []
            for op in ops:
                ids = apply(ctx, op) if apply is not None else None
                touched.extend(ids if ids is not None else ([op.target] if op.target else []))
            conn.execute(
                "INSERT INTO events (ts, kind, memory_version, item_ids) VALUES (?, ?, ?, ?)",
                (ctx.ts, event_kind.value, version, json.dumps(sorted(set(touched)))),
            )
            conn.execute("COMMIT")
        except OpRejected as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if proposal.model_id is None:  # hot or code: the writer answers the user (§6.3)
                raise
            result.rejected[key] = str(exc)
            continue
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            result.lock_ms[key] = (time.perf_counter() - held_from) * 1000
        result.committed.append(key)
        result.versions[key] = version
    return result


# ---------------------------------------------------------------- pre-send check (§7.4, §7.6)


@dataclass(frozen=True)
class PreSendResult:
    """Field names match `api.PreSend`."""

    ok: bool
    stale: list[ReadRef] = field(default_factory=list)
    undelivered_updates: int = 0


def pre_send_check(
    conn: sqlite3.Connection, *, thread_id: str, read_set: Iterable[ReadRef]
) -> PreSendResult:
    """Right before a model call (§7.4): every record in the prompt's read set still has
    its version and is usable (`active`/`archived`), every raw chunk its `rev` and
    `active`, every derived item `valid`; and the thread has no undelivered pending update
    (§7.6 gate: the update block consumes its rows before the call). One indexed query per
    source present plus one primary-key count; ≤ 5 ms at 10^5 records.
    `[T consistency/test_presend::only_usable_sent, ::refuses_undelivered_updates]`
    `[T perf/test_presend::p95]`"""
    stale = stale_refs(conn, read_set, require_usable=True)
    undelivered = conn.execute(
        "SELECT count(*) FROM pending_updates WHERE thread_id = ?", (thread_id,)
    ).fetchone()[0]
    return PreSendResult(
        ok=not stale and not undelivered, stale=stale, undelivered_updates=undelivered
    )


# ---------------------------------------------------------------- watermark (§7.7)


@dataclass(frozen=True)
class ChangesResult:
    """Field names match `api.Changes`."""

    memory_version: int
    last_event_id: int
    forget_events: list[int] = field(default_factory=list)


def changes_since(
    conn: sqlite3.Connection, *, memory_version: int, last_event_id: int
) -> ChangesResult:
    """§7.7 turn-start and PreToolUse read (< 1 ms): the current version and newest event
    id; on a gap, the ids of `forget` and `restore` events the host has not seen (both make
    it re-run session restarts). Selected by `memory_version`, which never goes backwards,
    so a restore that rewound the events table still reports them."""
    mv, last = conn.execute(
        "SELECT (SELECT memory_version FROM meta WHERE id = 1),"
        " (SELECT coalesce(max(id), 0) FROM events)"
    ).fetchone()
    forget: list[int] = []
    if mv != memory_version or last != last_event_id:
        forget = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM events WHERE kind IN ('forget', 'restore') AND "
                "(memory_version > ? OR id > ?) ORDER BY id",
                (memory_version, last_event_id),
            )
        ]
    return ChangesResult(memory_version=mv, last_event_id=last, forget_events=forget)


# ---------------------------------------------------------------- invariants (§3.2)

_GONE = ("forgotten", "purged")


def _labels_of(conn: sqlite3.Connection, refs: Iterable[ReadRef]) -> int:
    acc = 0
    for ref in refs:
        table = {ReadSrc.RECORD: "records", ReadSrc.RAW: "raw_chunks"}.get(ref.src)
        if table is None:
            continue
        row = conn.execute(f"SELECT labels FROM {table} WHERE id = ?", (ref.id,)).fetchone()
        if row is not None:
            acc |= int(row[0])
    return acc


def check_invariants(conn: sqlite3.Connection) -> list[str]:
    """The §3.2 invariants; returns one line per violation (ids only, no content), empty
    when the store is sound. Used by the Hypothesis suite and the doctor (§18).

    Lineage convention: an `edges` row `(from_id = record, to_id = source, 'derived_from')`,
    or `facts.source_raw_id`. A root record (no lineage needed) is an entity or a record
    carrying the user's own creating words (`origin_quote`). Vectors live in vectors.db and
    are checked by the vectors orphan sweep, not here."""
    out: list[str] = []
    q = conn.execute

    # every derived row has a read set; read_deps mirrors it exactly; labels inherit;
    # a valid derived row reads nothing forgotten
    for row in q("SELECT id, read_set, labels, valid FROM derived"):
        did = row["id"]
        try:
            refs = load_read_set(row["read_set"])
        except (ValueError, TypeError, KeyError, IndexError):
            out.append(f"derived {did}: read_set is not a read set")
            continue
        if not refs:
            out.append(f"derived {did}: empty read set")
        deps = {
            (r[0], r[1])
            for r in q(
                "SELECT item_id, version FROM read_deps WHERE holder_kind = 'derived'"
                " AND holder_id = ?",
                (did,),
            )
        }
        if deps != {(ref.id, ref.version) for ref in refs}:
            out.append(f"derived {did}: read_deps does not mirror read_set")
        missing = _labels_of(conn, refs) & ~int(row["labels"])
        if missing:
            out.append(f"derived {did}: labels miss {Label(missing)!r} of its sources")
        if row["valid"]:
            ids = [ref.id for ref in refs]
            gone = (
                q(
                    f"SELECT count(*) FROM records WHERE status IN {_GONE} AND id IN "
                    f"({','.join('?' * len(ids))})",
                    ids,
                ).fetchone()[0]
                if ids
                else 0
            )
            if gone:
                out.append(f"derived {did}: valid but reads a forgotten item")
    for row in q("SELECT id, read_set FROM actions"):
        refs = load_read_set(row["read_set"])
        deps = {
            (r[0], r[1])
            for r in q(
                "SELECT item_id, version FROM read_deps WHERE holder_kind = 'action'"
                " AND holder_id = ?",
                (row["id"],),
            )
        }
        if deps != {(ref.id, ref.version) for ref in refs}:
            out.append(f"action {row['id']}: read_deps does not mirror read_set")
    for row in q(
        "SELECT DISTINCT holder_kind, holder_id FROM read_deps WHERE holder_kind IN"
        " ('derived', 'action') AND holder_id NOT IN (SELECT id FROM derived UNION ALL"
        " SELECT id FROM actions)"
    ):
        out.append(f"read_deps {row[0]} {row[1]}: holder does not exist")

    # the stable-id index mirrors `entities.stable_ids` exactly (migration 0002)
    want = {
        (r[0], r[1], r[2])
        for r in q(
            "SELECT json_extract(j.value, '$.kind'), json_extract(j.value, '$.value'), e.id"
            " FROM entities e, json_each(e.stable_ids) j"
        )
    }
    have = {(r[0], r[1], r[2]) for r in q("SELECT kind, value, entity_id FROM entity_stable_ids")}
    if want != have:
        out.append("entity_stable_ids does not mirror entities.stable_ids")

    # every non-root record has a lineage edge
    for row in q(
        "SELECT r.id FROM records r LEFT JOIN facts f ON f.id = r.id"
        " WHERE r.kind <> 'entity' AND r.origin_quote IS NULL AND f.source_raw_id IS NULL"
        " AND r.status NOT IN ('forgotten', 'purged')"  # a root quote goes with a forget
        " AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.from_id = r.id"
        " AND e.etype = 'derived_from')"
    ):
        out.append(f"record {row[0]}: no lineage")

    # every justification member exists
    for row in q(
        "SELECT target_record_id, member_record_id FROM justifications j"
        " WHERE NOT EXISTS (SELECT 1 FROM records WHERE id = j.member_record_id)"
        " AND NOT EXISTS (SELECT 1 FROM raw_chunks WHERE id = j.member_record_id)"
    ):
        out.append(f"record {row[0]}: justification member {row[1]} does not exist")

    # records.labels ⊇ OR of its sources' labels
    for row in q(
        "SELECT r.id, r.labels, coalesce(s.labels, c.labels, 0) AS src_labels"
        " FROM records r JOIN edges e ON e.from_id = r.id AND e.etype = 'derived_from'"
        " LEFT JOIN records s ON s.id = e.to_id LEFT JOIN raw_chunks c ON c.id = e.to_id"
    ):
        missing = int(row["src_labels"]) & ~int(row["labels"])
        if missing:
            out.append(f"record {row['id']}: labels miss {Label(missing)!r} of a source")

    # no FTS row or raw text for a forgotten or purged item
    for row in q(
        f"SELECT r.id FROM records r JOIN fts_records f ON f.rowid = r.fts_rowid"
        f" WHERE r.status IN {_GONE}"
    ):
        out.append(f"record {row[0]}: forgotten but still in fts_records")
    for row in q(
        f"SELECT a.id FROM aliases a JOIN records r ON r.id = a.entity_id"
        f" JOIN fts_names f ON f.rowid = a.fts_rowid WHERE r.status IN {_GONE}"
    ):
        out.append(f"alias {row[0]}: entity forgotten but alias still in fts_names")
    for row in q(
        "SELECT c.id FROM raw_chunks c JOIN fts_raw f ON f.rowid = c.fts_rowid"
        " WHERE c.status <> 'active'"
    ):
        out.append(f"raw {row[0]}: forgotten but still in fts_raw")
    for row in q("SELECT id FROM raw_chunks WHERE status <> 'active' AND text IS NOT NULL"):
        out.append(f"raw {row[0]}: forgotten but text kept")

    # served and pending_updates refer to existing items
    for row in q(
        "SELECT DISTINCT item_id FROM served s WHERE via <> 'session'"
        " AND NOT EXISTS (SELECT 1 FROM records WHERE id = s.item_id)"
        " AND NOT EXISTS (SELECT 1 FROM raw_chunks WHERE id = s.item_id)"
        " AND NOT EXISTS (SELECT 1 FROM derived WHERE id = s.item_id)"
    ):
        out.append(f"served {row[0]}: item does not exist")
    for row in q(
        "SELECT DISTINCT item_id FROM pending_updates p"
        " WHERE NOT EXISTS (SELECT 1 FROM records WHERE id = p.item_id)"
    ):
        out.append(f"pending_update {row[0]}: item does not exist")
    return out
