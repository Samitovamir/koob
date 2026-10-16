"""Backups, the hot log and restore with journal replay.

Spec: §3.1 (files), §9.3 (journal), §9.4 (sets, hot log, restore, rewrite after forgets),
§18 (rotation 3 daily + 2 weekly, nightly rewrite).

A backup set is a folder `backups/<UTC stamp>/` holding `memory.db` (a `VACUUM INTO` copy),
`forget.jsonl` (the deletion journal at that moment), `hotlog.db` (a snapshot) and
`manifest.json`, written last: a folder without a manifest is not a set. No vectors.

Restore (§9.4): (1) the set's memory.db replaces the live file; (2) the deletion journal
(live and the set's copy, merged) is replayed; (3) the hot log is replayed; (4) the sweep
re-ingests from the set's watermark (a callable, raw); (5) vectors.db is dropped so it is
rebuilt; (6) `memory_version` continues from max(set, journal, hot log, old live) + 1.

Journal replay is forget.py's (`forget.replay_journal`): the same pipeline as a live forget
(cascade, FTS, derived, suppressions, raw by HMAC with tombstones), keyed by the original
forget_id, idempotent. koob.db rows and SDK sessions were handled when the forget happened
and are not in memory backups, so a restore passes no conversation store unless asked.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import store
from .clock import Clock, iso
from .clock import current as current_clock
from .model import EventKind, ForgetStage, JournalLine, Proposal

if TYPE_CHECKING:
    from .adapters.conversation import ConversationStore

MANIFEST = "manifest.json"
SET_DB = "memory.db"
SET_JOURNAL = "forget.jsonl"
SET_HOTLOG = "hotlog.db"
SET_HWM = store.HWM_FILE  # the version reservation, copied into every set (§7.1, A5)
_STAMP = "%Y%m%dT%H%M%S%fZ"


def _clock(clock: Clock | None) -> Clock:
    return clock or current_clock()


# ---------------------------------------------------------------- hot log (§9.4)

_HOTLOG_DDL = """
CREATE TABLE IF NOT EXISTS hot_ops (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    op              TEXT NOT NULL,
    payload         TEXT,
    memory_version  INTEGER,
    item_ids        TEXT NOT NULL DEFAULT '[]'
);
"""


@dataclass(frozen=True)
class HotOp:
    seq: int
    ts: str
    op: str
    payload: dict[str, Any] | None  # None once a forget scrubbed it
    memory_version: int | None
    item_ids: list[str]


def open_hotlog(path: Path | str) -> sqlite3.Connection:
    """hotlog.db with the §3.1 pragmas (journal_mode DELETE) and its one table."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = store.connect(path, journal_mode="DELETE")
    conn.executescript(_HOTLOG_DDL)
    return conn


def _ids_in(payload: dict[str, Any]) -> list[str]:
    """Ids a hot op names: its own id/target/record ids and, for a proposal, each op's."""
    ids: set[str] = set()
    for key in ("id", "target", "record_id", "item_id"):
        if isinstance(payload.get(key), str):
            ids.add(payload[key])
    for key in ("record_ids", "item_ids", "target_ids"):
        if isinstance(payload.get(key), list):
            ids.update(x for x in payload[key] if isinstance(x, str))
    for op in payload.get("ops", []) or []:
        if isinstance(op, dict):
            ids.update(_ids_in(op))
            if isinstance(op.get("payload"), dict):
                ids.update(_ids_in(op["payload"]))
    return sorted(ids)


def hotlog_append(
    hotlog: sqlite3.Connection,
    op: str,
    payload: dict,
    *,
    memory_version: int | None = None,
    item_ids: Iterable[str] | None = None,
) -> int:
    """§9.4: one hot-path op (remember, fix, forget, mode change) since the last verified
    backup, with its payload. Append **after** the op's commit, with the version that
    commit produced: a backup then truncates only ops it already contains. Op "proposal"
    carries a committed `Proposal` (`model_dump(mode="json")`) and replays through
    `store.commit`. Returns the op's seq."""
    ids = sorted(set(item_ids) if item_ids is not None else set(_ids_in(payload)))
    cur = hotlog.execute(
        "INSERT INTO hot_ops (ts, op, payload, memory_version, item_ids) VALUES (?, ?, ?, ?, ?)",
        (store.now(), op, json.dumps(payload, ensure_ascii=False), memory_version, json.dumps(ids)),
    )
    return int(cur.lastrowid)


def hotlog_ops(hotlog: sqlite3.Connection, *, after_version: int | None = None) -> list[HotOp]:
    rows = hotlog.execute("SELECT * FROM hot_ops ORDER BY seq").fetchall()
    out = []
    for r in rows:
        mv = r["memory_version"]
        if after_version is not None and mv is not None and mv <= after_version:
            continue
        out.append(
            HotOp(
                seq=r["seq"],
                ts=r["ts"],
                op=r["op"],
                payload=json.loads(r["payload"]) if r["payload"] is not None else None,
                memory_version=mv,
                item_ids=json.loads(r["item_ids"]),
            )
        )
    return out


def hotlog_scrub(hotlog: sqlite3.Connection, item_ids: Iterable[str]) -> int:
    """§9.4: a forget deletes the payloads of hot ops naming its ids (forget calls this
    at stop-use). Returns the number of ops scrubbed."""
    ids = sorted(set(item_ids))
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    cur = hotlog.execute(
        "UPDATE hot_ops SET payload = NULL WHERE payload IS NOT NULL AND seq IN ("
        f"SELECT h.seq FROM hot_ops h, json_each(h.item_ids) j WHERE j.value IN ({marks}))",
        ids,
    )
    return cur.rowcount


def hotlog_truncate(hotlog: sqlite3.Connection, *, upto_seq: int, upto_version: int) -> int:
    """After a verified backup: drop ops the set already contains (seq read before the
    `VACUUM INTO`, version at or below the set's)."""
    cur = hotlog.execute(
        "DELETE FROM hot_ops WHERE seq <= ? AND coalesce(memory_version, 0) <= ?",
        (upto_seq, upto_version),
    )
    return cur.rowcount


def _hotlog_max_seq(path: Path) -> int:
    if not path.exists():
        return 0
    conn = open_hotlog(path)
    try:
        return int(conn.execute("SELECT coalesce(max(seq), 0) FROM hot_ops").fetchone()[0])
    finally:
        conn.close()


# ---------------------------------------------------------------- journal (§9.3)


def read_journal(path: Path | None) -> list[JournalLine]:
    """Every parsable line; a torn last line (a crash mid-append) is skipped."""
    from . import forget

    return forget.read_journal(path)


def _line_key(line: JournalLine) -> tuple[str, str, str]:
    return (line.forget_id, line.stage.value, line.ts)


def _merge_journals(*journals: list[JournalLine]) -> list[JournalLine]:
    seen: dict[tuple[str, str, str], JournalLine] = {}
    for lines in journals:
        for line in lines:
            seen.setdefault(_line_key(line), line)
    from . import forget

    return forget.journal_order(seen.values())


def _append_lines(path: Path, lines: Iterable[JournalLine]) -> None:
    lines = list(lines)
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.model_dump_json() + "\n")
        fh.flush()
        os.fsync(fh.fileno())


@dataclass
class ForgetState:
    """One forget as the journal ends it: the final stage and every id it named."""

    forget_id: str
    stage: ForgetStage
    memory_version: int
    record_ids: list[str] = field(default_factory=list)
    raw_hmacs: list[str] = field(default_factory=list)
    suppression_hmacs: list[str] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)


def fold_journal(lines: Iterable[JournalLine]) -> list[ForgetState]:
    """Per forget_id, in journal order: the last stage wins (stopped -> undone | purged)."""
    states: dict[str, ForgetState] = {}
    from . import forget

    for line in forget.journal_order(lines):
        st = states.get(line.forget_id)
        if st is None:
            st = states[line.forget_id] = ForgetState(
                line.forget_id, line.stage, line.memory_version
            )
        st.stage = line.stage
        st.memory_version = max(st.memory_version, line.memory_version)
        for name in ("record_ids", "raw_hmacs", "suppression_hmacs", "turn_ids"):
            have = getattr(st, name)
            have.extend(x for x in getattr(line, name) if x not in have)
    return list(states.values())


def optimize_fts(conn: sqlite3.Connection) -> None:
    """§9.2: FTS `secure-delete`, then `optimize`. secure-delete drops a deleted row's
    entries but can leave the term key in a shared segment page (measured: a forgotten
    raw's word stayed in `fts_raw_data`); optimize rewrites the segments, and the freed
    pages are zeroed by `secure_delete=ON`."""
    with store.write_tx(conn):
        for table in ("fts_records", "fts_names", "fts_raw"):
            conn.execute(f"INSERT INTO {table} ({table}) VALUES ('optimize')")


# ---------------------------------------------------------------- sets (§9.4)


@dataclass(frozen=True)
class SetInfo:
    path: Path
    created_at: datetime
    memory_version: int
    manifest: dict[str, Any]


def _stamp(dt: datetime) -> str:
    return dt.strftime(_STAMP)


def _parse_stamp(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, _STAMP).replace(tzinfo=UTC)
    except ValueError:
        return None


def _quick_check(path: Path) -> bool:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


def _read_meta(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        mv, sv, wm = conn.execute(
            "SELECT memory_version, schema_version, sweep_watermark FROM meta WHERE id = 1"
        ).fetchone()
        ev = conn.execute("SELECT coalesce(max(id), 0) FROM events").fetchone()[0]
        return {
            "memory_version": mv,
            "schema_version": sv,
            "sweep_watermark": wm,
            "last_event_id": ev,
        }
    finally:
        conn.close()


def check_live(path: Path | str) -> bool:
    """`quick_check` on the live file (doctor, §18): False means recover from a set."""
    p = Path(path)
    return p.exists() and _quick_check(p)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _vacuum_into(conn: sqlite3.Connection, dest: Path) -> None:
    if conn.in_transaction:
        raise store.MemoryStoreError("VACUUM INTO must run outside a transaction (§3.1)")
    conn.execute("VACUUM INTO ?", (str(dest),))


def _snapshot_hotlog(src: Path, dest: Path, *, scrub: Iterable[str] = ()) -> None:
    if not src.exists():
        return
    conn = open_hotlog(src)
    try:
        _vacuum_into(conn, dest)
    finally:
        conn.close()
    ids = list(scrub)
    if ids:
        snap = open_hotlog(dest)
        try:
            hotlog_scrub(snap, ids)
            snap.execute("VACUUM")
        finally:
            snap.close()


def make_set(
    conn: sqlite3.Connection,
    dest_root: Path | None = None,
    *,
    journal: Path | None = None,
    hotlog: Path | None = None,
    clock: Clock | None = None,
) -> Path:
    """`VACUUM INTO` (outside a transaction) + journal + hot log snapshot; no vectors.
    Writes into `<stamp>.partial/`, verifies the copy (`quick_check`, meta), writes the
    manifest and renames the folder; then truncates the live hot log to the ops the set
    does not contain. Paths default to the live DB's layout (`store.MemoryPaths`)."""
    live = store.db_file(conn)
    paths = store.MemoryPaths.of(live) if live else None
    if dest_root is None:
        if paths is None:
            raise store.MemoryStoreError("an in-memory store needs an explicit dest_root")
        dest_root = paths.backups
    journal = journal if journal is not None else (paths.journal if paths else None)
    hotlog = hotlog if hotlog is not None else (paths.hotlog if paths else None)
    when = _clock(clock).now()
    dest_root.mkdir(parents=True, exist_ok=True)
    name = _stamp(when)
    tmp = dest_root / f"{name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    hot_seq = _hotlog_max_seq(hotlog) if hotlog is not None else 0
    _vacuum_into(conn, tmp / SET_DB)
    lines = read_journal(journal)
    _append_lines(tmp / SET_JOURNAL, lines)
    (tmp / SET_JOURNAL).touch()
    if hotlog is not None:
        _snapshot_hotlog(hotlog, tmp / SET_HOTLOG)
    reserved = store.read_hwm(store.hwm_path(conn))
    if reserved:
        store.write_hwm(tmp / SET_HWM, reserved)
    if not _quick_check(tmp / SET_DB):
        shutil.rmtree(tmp)
        raise store.MemoryStoreError("backup copy failed quick_check")
    meta = _read_meta(tmp / SET_DB)
    manifest = {
        "created_at": iso(when),
        **meta,
        "journal_lines": len(lines),
        "hotlog_seq": hot_seq,
        "version_reserved": reserved,
        "sqlite": ".".join(map(str, store.sqlite_version())),
        "verified": True,
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    final = dest_root / name
    os.replace(tmp, final)
    _fsync_dir(dest_root)
    if hotlog is not None and hotlog.exists():
        h = open_hotlog(hotlog)
        try:
            hotlog_truncate(h, upto_seq=hot_seq, upto_version=meta["memory_version"])
        finally:
            h.close()
    return final


def list_sets(dest_root: Path) -> list[SetInfo]:
    """Verified sets, newest first. Folders without a manifest are not sets."""
    out = []
    if not dest_root.exists():
        return out
    for d in dest_root.iterdir():
        created = _parse_stamp(d.name)
        if created is None or not d.is_dir() or not (d / MANIFEST).exists():
            continue
        try:
            manifest = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not manifest.get("verified"):
            continue
        out.append(SetInfo(d, created, int(manifest.get("memory_version", 0)), manifest))
    return sorted(out, key=lambda s: s.created_at, reverse=True)


def rotate(
    dest_root: Path, *, daily: int = 3, weekly: int = 2, clock: Clock | None = None
) -> list[Path]:
    """§9.4, §18: keep the newest set of each of the last `daily` days that have one, plus
    the newest set of each of the last `weekly` ISO weeks not already kept as a daily.
    Leftovers of a crashed write or rewrite (`*.partial` over an hour old, `*.old`,
    `*.rewrite`) go too: they may hold forgotten content. Returns the folders removed."""
    sets = list_sets(dest_root)
    keep: set[Path] = set()
    days: list[str] = []
    for s in sets:
        day = s.created_at.date().isoformat()
        if day not in days:
            days.append(day)
            if len(days) <= daily:
                keep.add(s.path)
    newest_of_week: dict[tuple[int, int], Path] = {}
    for s in sets:
        newest_of_week.setdefault(tuple(s.created_at.isocalendar()[:2]), s.path)
    weeklies = 0
    for path in newest_of_week.values():  # newest week first
        if path in keep:  # this week's newest is already a daily
            continue
        if weeklies < weekly:
            keep.add(path)
            weeklies += 1
    if sets:
        keep.add(sets[0].path)  # never drop the newest verified set
    removed = [s.path for s in sets if s.path not in keep]
    now_s = _clock(clock).now().timestamp()
    if dest_root.exists():
        for d in dest_root.iterdir():
            stale_partial = d.name.endswith(".partial") and now_s - d.stat().st_mtime > 3600
            if d.is_dir() and (stale_partial or d.name.endswith((".old", ".rewrite"))):
                removed.append(d)
    for d in removed:
        shutil.rmtree(d, ignore_errors=True)
    return removed


# ---------------------------------------------------------------- restore (§9.4 (1)-(6))

HotReplayer = Callable[[sqlite3.Connection, dict[str, Any]], None]


@dataclass
class RestoreReport:
    memory_version: int
    set_version: int
    forgets_replayed: int = 0
    hot_replayed: int = 0
    hot_skipped: int = 0  # scrubbed, or naming ids a later forget took
    unreplayed: list[str] = field(default_factory=list)  # hot op names with no replayer
    swept: int | None = None
    vectors_dropped: bool = False


def _remove_db_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def _live_state(path: Path) -> dict[str, int] | None:
    if not path.exists() or not _quick_check(path):
        return None
    try:
        return _read_meta(path)
    except sqlite3.DatabaseError:
        return None


def _proposal_replayer(apply: store.OpApplier | None) -> HotReplayer:
    def run(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
        proposal = Proposal.model_validate(payload)
        # read sets are stale by construction after a restore: the op is the user's word, replayed
        ops = [op.model_copy(update={"read_set": []}) for op in proposal.ops]
        applier = apply
        if applier is None:
            from .ledger import apply_op

            applier = apply_op
        store.commit(conn, proposal.model_copy(update={"ops": ops}), applier, priority="hot")

    return run


def restore_report(
    backup_set: Path | str,
    live_path: Path | str,
    *,
    journal: Path | None = None,
    hotlog: Path | None = None,
    apply: store.OpApplier | None = None,
    replayers: dict[str, HotReplayer] | None = None,
    sweep: Callable[[sqlite3.Connection], int] | None = None,
    conversation: ConversationStore | None = None,
    home: Path | None = None,
) -> RestoreReport:
    """§9.4 restore, steps (1)-(6). Close every connection to the live file first.
    `conversation`: koob.db, only when its rows must be re-blanked too (they were blanked
    when each forget happened; a memory.db restore does not bring them back)."""
    from . import forget

    backup_set, live_path = Path(backup_set), Path(live_path)
    paths = store.MemoryPaths.of(live_path)
    journal = journal if journal is not None else paths.journal
    hotlog = hotlog if hotlog is not None else paths.hotlog
    if not (backup_set / MANIFEST).exists() or not _quick_check(backup_set / SET_DB):
        raise store.MemoryStoreError(f"not a verified backup set: {backup_set.name}")
    manifest = json.loads((backup_set / MANIFEST).read_text(encoding="utf-8"))
    set_mv = int(manifest["memory_version"])
    old = _live_state(live_path)
    hurry = _purge_started(live_path, due=True) if old is not None else []
    # read before the copy: the reservation survives a live file that is gone or unreadable
    reserved = max(
        store.read_hwm(live_path.parent / store.HWM_FILE),
        store.read_hwm(backup_set / SET_HWM),
        int(manifest.get("version_reserved") or 0),
    )

    # (1) the backup replaces the live file (stale -wal/-shm of the old file go first)
    store.check_path(live_path, home=home)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    staging = live_path.with_name(live_path.name + ".restoring")
    _remove_db_files(staging)
    shutil.copyfile(backup_set / SET_DB, staging)
    _remove_db_files(live_path)
    os.replace(staging, live_path)
    conn = store.open(live_path, home=home)
    try:
        report = RestoreReport(memory_version=0, set_version=set_mv)
        # (2) the journal, merged with the set's copy and kept whole beside the DB
        live_lines = read_journal(journal)
        lines = _merge_journals(live_lines, read_journal(backup_set / SET_JOURNAL))
        have = {_line_key(x) for x in live_lines}
        _append_lines(journal, [x for x in lines if _line_key(x) not in have])
        report.forgets_replayed = len(
            forget.replay_journal(
                conn, lines, after_version=set_mv, conversation=conversation, forever=hurry
            )
        )
        # (3) the hot log: ops newer than the set, minus what a later forget took
        hot_mv = 0
        if hotlog.exists():
            report.hot_replayed, report.hot_skipped, report.unreplayed, hot_mv = _replay_hot(
                conn, hotlog, set_mv, lines, apply, replayers
            )
        # (4) the sweep from the set's watermark
        if sweep is not None:
            report.swept = sweep(conn)
        # (5) vectors are rebuilt, never restored
        if paths.vectors.exists():
            _remove_db_files(paths.vectors)
            report.vectors_dropped = True
        # (6) versions continue from the max of everything seen, plus one
        journal_mv = max([forget.journal_max_version(journal), *(x.memory_version for x in lines)])
        # and the reservation: background commits after the set may have served versions
        # that neither the journal nor the hot log holds (§7.1, A5)
        floor = max(set_mv, journal_mv, hot_mv, (old or {}).get("memory_version", 0), reserved)
        last_event = (old or {}).get("last_event_id", 0)
        report.memory_version = _finish(conn, floor, last_event)
    finally:
        conn.close()
    return report


def _replay_hot(
    conn: sqlite3.Connection,
    hotlog: Path,
    set_mv: int,
    lines: list[JournalLine],
    apply: store.OpApplier | None,
    replayers: dict[str, HotReplayer] | None,
) -> tuple[int, int, list[str], int]:
    taken: dict[str, int] = {}  # record id -> version of the forget that took it
    for st in fold_journal(lines):
        if st.stage is not ForgetStage.UNDONE:
            for rid in st.record_ids:
                taken[rid] = max(taken.get(rid, 0), st.memory_version)
    table: dict[str, HotReplayer] = {"proposal": _proposal_replayer(apply)}
    table.update(replayers or {})
    h = open_hotlog(hotlog)
    try:
        ops = hotlog_ops(h, after_version=set_mv)
        hot_mv = int(
            h.execute("SELECT coalesce(max(memory_version), 0) FROM hot_ops").fetchone()[0]
        )
    finally:
        h.close()
    done = skipped = 0
    missing: list[str] = []
    for op in ops:
        if op.op == "forget":  # forgets replay from the journal (step 2)
            continue
        named_taken = any(
            rid in taken and (op.memory_version is None or op.memory_version <= taken[rid])
            for rid in op.item_ids
        )
        if op.payload is None or named_taken:
            skipped += 1
            continue
        fn = table.get(op.op)
        if fn is None:
            missing.append(op.op)
            continue
        fn(conn, op.payload)
        done += 1
    return done, skipped, missing, hot_mv


def _finish(conn: sqlite3.Connection, floor: int, last_event: int) -> int:
    ts = store.now()
    with store.write_tx(conn):
        version = conn.execute(
            "UPDATE meta SET memory_version = max(memory_version, ?) + 1, updated_at = ?"
            " WHERE id = 1 RETURNING memory_version",
            (floor, ts),
        ).fetchone()[0]
        store.reserve_version(conn, version)
        cur = conn.execute(
            "UPDATE sqlite_sequence SET seq = max(seq, ?) WHERE name = 'events'", (last_event,)
        )
        if cur.rowcount == 0:
            conn.execute(
                "INSERT INTO sqlite_sequence (name, seq) VALUES ('events', ?)", (last_event,)
            )
        conn.execute(
            "INSERT INTO events (ts, kind, memory_version, item_ids) VALUES (?, ?, ?, '[]')",
            (ts, EventKind.RESTORE.value, version),
        )
        conn.execute(
            "INSERT INTO audit (ts, event, record_id, actor) VALUES (?, 'restore', NULL, 'system')",
            (ts,),
        )
    # old pages of the copied file must not keep replayed-away content (§9.1 stage 3)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return int(version)


def restore(backup_set: Path | str, live_path: Path | str, **kwargs: Any) -> int:
    """§9.4 (1)-(6); returns the memory_version the store continues from (max + 1).
    Keyword arguments as `restore_report`."""
    return restore_report(backup_set, live_path, **kwargs).memory_version


def recover(live_path: Path | str, **kwargs: Any) -> RestoreReport | None:
    """Corruption recovery (§18): if the live file fails `quick_check`, restore the newest
    verified set. Returns None when the live file is sound."""
    live_path = Path(live_path)
    if check_live(live_path):
        return None
    sets = list_sets(store.MemoryPaths.of(live_path).backups)
    if not sets:
        raise store.MemoryStoreError("memory.db is damaged and there is no backup set")
    return restore_report(sets[0].path, live_path, **kwargs)


# ---------------------------------------------------------------- rewrite after forgets (§9.4)


def rewrite_after_forgets(
    dest_root: Path,
    *,
    journal: Path | None = None,
    purge: Iterable[str] | None = None,
    clock: Clock | None = None,
) -> int:
    """Nightly, one batch, on AC (the scheduler decides when): every set whose journal copy
    lacks a line of the live journal, or still holds a forget the live store has begun to
    purge, is restored to a temp file, replayed (`forget.replay_journal`), `VACUUM INTO` a
    fresh set and swapped in place, keeping its stamp. `purge`: forget ids the live store
    is purging or has due (`forget.purge_started` + `forget.due`), purged in the sets too
    so the live purge's scan of memory/ can pass; read from the live memory.db beside
    `dest_root` when not given. Returns the number rewritten."""
    from . import forget

    paths = store.MemoryPaths(dest_root.parent)
    journal = journal if journal is not None else paths.journal
    lines = read_journal(journal)
    if not lines:
        return 0
    if purge is None:
        purge = _purge_started(paths.db, due=True)
    purging = set(purge)
    keys = {_line_key(x) for x in lines}
    taken = sorted({rid for st in fold_journal(lines) for rid in st.record_ids})
    count = 0
    for s in list_sets(dest_root):
        have = {_line_key(x) for x in read_journal(s.path / SET_JOURNAL)}
        if keys <= have and not purging - set(s.manifest.get("purged", [])):
            continue
        work = dest_root / f"{s.path.name}.rewrite"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        shutil.copyfile(s.path / SET_DB, work / "work.db")
        conn = store.connect(work / "work.db")
        try:
            store.migrate(conn)
            forget.replay_journal(conn, lines, purge=purging)
            out = work / "new"
            out.mkdir()
            _vacuum_into(conn, out / SET_DB)
        finally:
            conn.close()
        _append_lines(out / SET_JOURNAL, _merge_journals(lines))
        _snapshot_hotlog(s.path / SET_HOTLOG, out / SET_HOTLOG, scrub=taken)
        if not _quick_check(out / SET_DB):
            shutil.rmtree(work)
            continue
        manifest = {
            **s.manifest,
            **_read_meta(out / SET_DB),
            "journal_lines": len(lines),
            "purged": sorted(purging | set(s.manifest.get("purged", []))),
            "rewritten_at": iso(_clock(clock).now()),
        }
        (out / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        old = dest_root / f"{s.path.name}.old"
        os.replace(s.path, old)
        os.replace(out, s.path)
        shutil.rmtree(old)
        shutil.rmtree(work)
        _fsync_dir(dest_root)
        count += 1
    return count


def _purge_started(live: Path, *, due: bool = False) -> list[str]:
    """Forgets the live store has begun to purge (with `due`: also those due now), read
    without taking the write lock. Empty when the file is missing or unreadable."""
    from . import forget

    if not live.exists():
        return []
    conn = store.connect(live)
    try:
        ids = set(forget.purge_started(conn))
        if due:
            ids |= set(forget.due(conn))
        return sorted(ids)
    except sqlite3.DatabaseError:
        return []
    finally:
        conn.close()


__all__ = [
    "ForgetState",
    "HotOp",
    "RestoreReport",
    "SetInfo",
    "check_live",
    "fold_journal",
    "hotlog_append",
    "hotlog_ops",
    "hotlog_scrub",
    "hotlog_truncate",
    "list_sets",
    "make_set",
    "optimize_fts",
    "open_hotlog",
    "read_journal",
    "recover",
    "restore",
    "restore_report",
    "rewrite_after_forgets",
    "rotate",
]
