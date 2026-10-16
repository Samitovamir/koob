"""Night pass: delta-only consolidation (merges with the survival check, supersessions,
cascades, lessons, criteria, episodes, assoc refresh), decay to archived, junk control.

Spec: §6.3 (schedule), §6.4, §15.1.

- **Delta only (G10).** A night reads what changed since the previous night: records whose
  `version` is above the watermark the last pass saved (a content-free `nightly` row in
  `jobs`, state `done`). Its ops are deltas (merge / flag), never a rewrite. Summaries are
  rebuilt by their own rebuilders from the ledger and raw (`rebuild_derived` jobs, capped per
  night, newest first; the rest stay invalid and excluded), never from the previous summary.
- **Merges (§6.4).** Duplicates of a changed fact or preference (same subject, slot,
  normalized value, polarity, kind, applicability and valid time) merge into the member whose
  text keeps every name, date and number of the others (`survival_ok`); a member whose
  details would be lost stays apart. `check_merges` is the same check for any proposal: a
  value covered by a supersede op in the same proposal is not a loss.
- **Decay (weekly).** Stale low-value items of other people, tools and the system go to
  `archived` through FLAG: no journal line, no backup rewrite; the archive is kept. The
  user's own words never decay.
- **Forgetting and upkeep.** Due purges run; on AC the daily backup set, rotation (3 + 2),
  the rewrite of sets after forgets, FTS optimize, incremental vacuum and a WAL checkpoint.
- **Hooks.** `register_step(name, fn)` adds a step (assoc refresh, lessons, criteria);
  `fn(night)` returns a count or a dict of counts. A step that raises writes one
  content-free audit line and the night goes on.
- **When.** On AC power only (`use_power` injects the probe), never on a turn path: the
  scheduler calls it; a pass that finds a hot writer waiting skips. Silent: no line to the
  user.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .. import clock as _clockmod
from .. import ledger, store
from ..model import AuthorTier, OpKind, Proposal, ProposalOp, ReadRef

NIGHT_JOB = "nightly"
REBUILD_JOB = "rebuild_derived"
DECAY_EVERY = timedelta(days=7)
DECAY_AFTER = timedelta(days=180)
LOW_VALUE = 0.3  # importance below this (or unset) counts as low value
BACKUP_EVERY = timedelta(hours=20)
MARKS_KEPT = timedelta(days=62)  # nightly rows kept: the doctor's growth history
_MERGE_KINDS = ("fact", "preference")
_DECAY_KINDS = ("fact", "preference", "impression")
_DECAY_TIERS = ("other", "tool", "system", "user_room")  # the user's own words never decay
_STOP = frozenset(
    ["the", "a", "an", "i", "he", "she", "it", "we", "you", "they", "his", "her", "its", "our", "their", "my", "your", "this", "that", "these", "those", "in", "on", "at", "to", "from", "since", "by", "of", "for", "and", "or", "but", "not", "no", "yes", "also", "then", "there", "here", "as", "is", "was", "are", "were", "be", "been", "will", "would", "has", "had", "have", "do", "does", "did", "if", "so", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
)
_NAME = re.compile(r"(?<![\w'])[A-ZА-ЯЁ][\w'-]*")


# ---------------------------------------------------------------- the survival check (§6.4)


def kept_values(text: str) -> set[str]:
    """Names (capitalized words, minus function words), dates and numbers of a text, in the
    normalized form `ledger.value_tokens` uses («3 мая», May 3rd and 05-03 meet)."""
    names = {
        "name:" + re.sub(r"['’]s$", "", w.casefold()).strip("'’-")
        for w in _NAME.findall(text or "")
        if len(w) > 1 and w.casefold() not in _STOP
    }
    return names | ledger.value_tokens(text or "")


def survival_ok(inputs: list[str], result: str, superseded: list[str]) -> bool:
    """§6.4: every name, date and number of the inputs survives or is superseded."""
    need: set[str] = set()
    for text in inputs:
        need |= kept_values(text)
    covered = kept_values(result)
    for text in superseded:
        covered |= kept_values(text)
    return need <= covered


def _text(conn: sqlite3.Connection, record_id: str) -> str | None:
    """The English text a merge keeps: a fact's text and value, else the record's quote."""
    row = conn.execute(
        "SELECT f.text_en, f.value_text, r.origin_quote FROM records r"
        " LEFT JOIN facts f ON f.id = r.id WHERE r.id = ?",
        (record_id,),
    ).fetchone()
    if row is None:
        return None
    parts = [p for p in (row[0], row[1]) if p] or [row[2] or ""]
    return " ".join(parts)


def check_merges(conn: sqlite3.Connection, proposal: Proposal) -> list[str]:
    """Groups of `proposal` whose MERGE ops would lose a name, date or number of their
    inputs. A value carried by a record a SUPERSEDE op of the same proposal retires counts as
    covered. Content-free: group keys only."""
    superseded = [
        t
        for o in proposal.ops
        if o.op is OpKind.SUPERSEDE and o.target
        for t in [_text(conn, o.target)]
        if t
    ]
    bad: list[str] = []
    for o in proposal.ops:
        if o.op is not OpKind.MERGE or not o.target:
            continue
        keep, drop = _text(conn, o.target), _text(conn, str(o.payload.get("drop") or ""))
        if keep is None or drop is None or not survival_ok([keep, drop], keep, superseded):  # noqa: SIM102
            if o.group not in bad:
                bad.append(o.group)
    return bad


# ---------------------------------------------------------------- the night


@dataclass
class Night:
    """What one pass sees: the previous night's watermark and the changes since."""

    conn: sqlite3.Connection
    now: datetime
    cap: int
    since_version: int
    upto_version: int
    changed: list[str]  # record ids with since_version < version <= upto_version
    last_decay: datetime | None
    mark: dict[str, object] = field(default_factory=dict)  # saved as the next watermark


Step = Callable[[Night], "int | dict[str, int]"]
_STEPS: dict[str, Step] = {}
_power: list[Callable[[], bool] | None] = [None]


def register_step(name: str, fn: Step) -> None:
    """Add a night step (assoc refresh; lessons, criteria, episodes). Steps run
    after the built-in consolidation, in registration order; a second call replaces."""
    _STEPS[name] = fn


def unregister_step(name: str) -> None:
    _STEPS.pop(name, None)


def use_power(probe: Callable[[], bool] | None) -> None:
    """Inject the AC probe (tests, the host); None restores `pmset`."""
    _power[0] = probe


def on_ac() -> bool:
    if _power[0] is not None:
        return bool(_power[0]())
    if sys.platform != "darwin":
        return True
    try:
        out = subprocess.run(
            ["/usr/bin/pmset", "-g", "ps"], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return "Battery Power" not in out


def last_mark(conn: sqlite3.Connection) -> dict[str, object] | None:
    row = conn.execute(
        "SELECT payload FROM jobs WHERE jkind = ? AND state = 'done'"
        " ORDER BY created_at DESC, id DESC LIMIT 1",
        (NIGHT_JOB,),
    ).fetchone()
    if row is None:
        return None
    try:
        got = json.loads(row[0])
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


def marks(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """Every kept watermark, oldest first (the doctor's growth history)."""
    out = []
    for (payload,) in conn.execute(
        "SELECT payload FROM jobs WHERE jkind = ? AND state = 'done' ORDER BY created_at, id",
        (NIGHT_JOB,),
    ):
        try:
            out.append(json.loads(payload))
        except ValueError:
            continue
    return out


def _parse(ts: object) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def db_bytes(conn: sqlite3.Connection) -> int:
    path = store.db_file(conn)
    if path is None:
        return 0
    total = 0
    for p in (path, path.with_name(path.name + "-wal")):
        try:  # noqa: SIM105
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _save_mark(conn: sqlite3.Connection, mark: dict[str, object], ts: str) -> None:
    cut = (_clockmod.current().now() - MARKS_KEPT).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    jid = ledger.new_id()
    with store.write_tx(conn, priority="background"):
        conn.execute(
            "INSERT INTO jobs (id, jkind, payload, state, attempts, idempotency_key,"
            " created_at, updated_at) VALUES (?, ?, ?, 'done', 0, ?, ?, ?)",
            (jid, NIGHT_JOB, json.dumps(mark), f"{NIGHT_JOB}:{ts}:{jid}", ts, ts),
        )
        conn.execute(
            "DELETE FROM jobs WHERE jkind = ? AND state = 'done' AND created_at < ?",
            (NIGHT_JOB, cut),
        )


def _commit(conn: sqlite3.Connection, ops: list[ProposalOp]) -> store.CommitResult | None:
    """A code proposal of the night (system tier, background priority); a content check
    that fails returns None (the group stays as it was)."""
    proposal = Proposal(ops=ops, author_tier=AuthorTier.SYSTEM)
    try:
        return store.commit(conn, proposal, ledger.apply_op, priority="background")
    except store.OpRejected:
        return None


# ---------------------------------------------------------------- built-in steps


def _dupes(conn: sqlite3.Connection, record_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT d.id, d.version, d.author_tier, df.speaker_entity_id FROM facts f"
        " JOIN records r ON r.id = f.id"
        " JOIN facts df ON df.subject_id = f.subject_id AND df.relation = f.relation"
        "  AND df.value_norm = f.value_norm AND df.polarity = f.polarity AND df.fkind = f.fkind"
        "  AND coalesce(df.applicability, '') = coalesce(f.applicability, '')"
        "  AND coalesce(df.valid_from, '') = coalesce(f.valid_from, '')"
        "  AND coalesce(df.valid_to, '') = coalesce(f.valid_to, '')"
        " JOIN records d ON d.id = df.id AND d.kind = r.kind AND d.status = 'active'"
        " WHERE f.id = ? AND r.status = 'active' AND f.value_norm IS NOT NULL"
        " ORDER BY d.recorded_at, d.id",
        (record_id,),
    ).fetchall()


def merge_step(night: Night) -> dict[str, int]:
    """§6.4 duplicate merges over the delta, each cluster its own group."""
    conn, seen = night.conn, set()
    out = {"merged": 0, "kept_apart": 0, "merge_stale": 0, "merge_rejected": 0}
    for rid in night.changed:
        if rid in seen:
            continue
        rows = _dupes(conn, rid)
        seen.update(r["id"] for r in rows)
        if len(rows) < 2:
            continue
        texts = {r["id"]: _text(conn, r["id"]) or "" for r in rows}
        best: tuple[sqlite3.Row, list[sqlite3.Row]] | None = None
        for keep in rows:  # oldest first: the oldest keeper that loses nothing wins
            drops = [
                d
                for d in rows
                if d["id"] != keep["id"]
                and ledger.may_supersede(
                    AuthorTier(keep["author_tier"]),
                    AuthorTier(d["author_tier"]),
                    same_speaker=keep["speaker_entity_id"] == d["speaker_entity_id"],
                )
                and survival_ok([texts[keep["id"]], texts[d["id"]]], texts[keep["id"]], [])
            ]
            if best is None or len(drops) > len(best[1]):
                best = (keep, drops)
        keep, drops = best if best else (rows[0], [])
        out["kept_apart"] += len(rows) - 1 - len(drops)
        if not drops:
            continue
        group = f"merge:{keep['id']}"
        refs = [ReadRef(id=r["id"], version=r["version"]) for r in [keep, *drops]]
        ops = [
            ProposalOp(
                op=OpKind.MERGE,
                target=keep["id"],
                group=group,
                read_set=refs,
                payload={"drop": d["id"]},
            )
            for d in drops
        ]
        if check_merges(conn, Proposal(ops=ops)):  # the gate any merge passes
            out["kept_apart"] += len(drops)
            continue
        res = _commit(conn, ops)
        if res is None or res.rejected:
            out["merge_rejected"] += 1
        elif res.stale:
            out["merge_stale"] += 1  # changed meanwhile: in the next night's delta
        else:
            out["merged"] += len(drops)
    return out


def decay_step(night: Night) -> dict[str, int]:
    """Weekly: stale low-value items to `archived` (FLAG), oldest use first, capped."""
    if night.last_decay is not None and night.now - night.last_decay < DECAY_EVERY:
        return {"archived": 0}
    cutoff = (night.now - DECAY_AFTER).strftime("%Y-%m-%dT%H:%M:%S")
    kinds = ",".join("?" * len(_DECAY_KINDS))
    tiers = ",".join("?" * len(_DECAY_TIERS))
    rows = night.conn.execute(
        "SELECT r.id, r.version FROM records r LEFT JOIN record_stats s ON s.record_id = r.id"
        " LEFT JOIN facts f ON f.id = r.id"
        f" WHERE r.status = 'active' AND r.review = 0 AND r.pinned_category IS NULL"
        f" AND r.kind IN ({kinds}) AND r.author_tier IN ({tiers})"
        " AND coalesce(r.importance, 0) < ? AND coalesce(f.stability, '') != 'lasting'"
        " AND coalesce(s.last_used_at, r.recorded_at) < ?"
        " ORDER BY coalesce(s.last_used_at, r.recorded_at), r.id LIMIT ?",
        (*_DECAY_KINDS, *_DECAY_TIERS, LOW_VALUE, cutoff, night.cap),
    ).fetchall()
    archived = 0
    for r in rows:
        op = ProposalOp(
            op=OpKind.FLAG,
            target=r["id"],
            group=r["id"],
            read_set=[ReadRef(id=r["id"], version=r["version"])],
            payload={"status": "archived"},
        )
        res = _commit(night.conn, [op])
        archived += bool(res is not None and res.ok)
    night.mark["decay_at"] = store.now()
    return {"archived": archived}


def rebuild_step(night: Night) -> dict[str, int]:
    """Queued `rebuild_derived` jobs, newest first, at most `cap`; the rest stay queued
    (their items stay invalid and excluded). The rebuilders read the ledger and raw only."""
    try:  # noqa: SIM105
        from ..recall import core_view  # noqa: F401 - registers the rebuild runner
    except Exception:  # noqa: BLE001 - no runner here: the jobs wait
        pass
    entry = store.job_runner(REBUILD_JOB)
    if entry is None:
        return {"rebuilt": 0}
    fn = entry[0]
    ids = [
        r[0]
        for r in night.conn.execute(
            "SELECT id FROM jobs WHERE jkind = ? AND state = 'queued'"
            " AND (not_before IS NULL OR not_before <= ?) ORDER BY created_at DESC, id DESC"
            " LIMIT ?",
            (REBUILD_JOB, store.now(), night.cap),
        )
    ]
    done = 0
    for job_id in ids:
        if not store.claim_job(night.conn, job_id):
            continue
        row = night.conn.execute(
            "SELECT id, jkind, payload, state, attempts, not_before, idempotency_key"
            " FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        job = dict(row)
        job["payload"] = json.loads(job["payload"] or "{}")
        try:
            fn(night.conn, job)
        except Exception:  # noqa: BLE001 - back to the queue, attempts + 1 (store's rule)
            if night.conn.in_transaction:
                night.conn.execute("ROLLBACK")
            with store.write_tx(night.conn, priority="background"):
                night.conn.execute(
                    "UPDATE jobs SET state = CASE WHEN attempts + 1 > ? THEN 'needs_attention'"
                    " ELSE 'queued' END, attempts = attempts + 1, updated_at = ?"
                    " WHERE id = ? AND state = 'running'",
                    (store.JOB_ATTEMPTS_CAP, store.now(), job_id),
                )
            continue
        with store.write_tx(night.conn, priority="background"):
            night.conn.execute(
                "UPDATE jobs SET state = 'done', updated_at = ? WHERE id = ? AND state = 'running'",
                (store.now(), job_id),
            )
        done += 1
    return {"rebuilt": done}


def forget_step(night: Night) -> dict[str, int]:
    """Due purges (trash window over, or a retry), through the full §9.2 scope."""
    from .. import forget

    got = forget.run_due(night.conn)
    return {"purged": sum(1 for ok in got.values() if ok)}


def upkeep_step(night: Night) -> dict[str, int]:
    """On AC (§9.4, §18): the daily set, rotation 3 + 2, the rewrite after forgets, FTS
    optimize, incremental vacuum and a WAL checkpoint. A ":memory:" store does the SQL only."""
    from .. import backup

    conn, out = night.conn, {"backups": 0, "rewritten": 0, "rotated": 0}
    live = store.db_file(conn)
    if live is not None:
        paths = store.MemoryPaths.of(live)
        sets = backup.list_sets(paths.backups)
        newest = sets[0].created_at if sets else None
        if newest is None or night.now - newest >= BACKUP_EVERY:
            backup.make_set(conn)
            out["backups"] = 1
        out["rotated"] = len(backup.rotate(paths.backups))
        out["rewritten"] = backup.rewrite_after_forgets(paths.backups, journal=paths.journal)
    backup.optimize_fts(conn)
    conn.execute("PRAGMA incremental_vacuum")
    if live is not None:
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    return out


BUILT_IN: tuple[tuple[str, Step], ...] = (
    ("merge", merge_step),
    ("rebuild", rebuild_step),
    ("decay", decay_step),
    ("forget", forget_step),
)


def run(conn: sqlite3.Connection, *, cap: int = 200, force: bool = False) -> dict[str, int]:
    """One night pass (§6.3, §6.4): delta since the last night, built-in steps, registered
    steps, upkeep on AC; then the new watermark. Counts only, never content; never raises
    for a step (one audit line instead). `force` skips the AC and busy gates (`cb`)."""
    from .scheduler import doctor_line

    if not force:
        if not on_ac():
            return {"skipped_power": 1}
        if store.hot_waiting(store.db_file(conn)):
            return {"skipped_busy": 1}
    prev = last_mark(conn) or {}
    since = int(prev.get("memory_version") or 0)  # type: ignore[arg-type]
    upto = store.memory_version(conn)
    changed = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM records WHERE version > ? AND version <= ? AND kind IN (?, ?)"
            " AND status = 'active'"
            " ORDER BY version DESC, id",
            (since, upto, *_MERGE_KINDS),
        )
    ]
    night = Night(
        conn=conn,
        now=_clockmod.current().now(),
        cap=cap,
        since_version=since,
        upto_version=upto,
        changed=changed,
        last_decay=_parse(prev.get("decay_at")),
        mark={"decay_at": prev.get("decay_at")},
    )
    counts: dict[str, int] = {"delta": len(changed)}
    steps = [*BUILT_IN, *_STEPS.items(), ("upkeep", upkeep_step)]
    for name, fn in steps:
        try:
            got = fn(night)
        except Exception:  # noqa: BLE001 - one step's failure never stops the night
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            doctor_line(conn, f"nightly_failed:{name}")
            counts[f"failed_{name}"] = 1
            continue
        if isinstance(got, dict):
            for k, v in got.items():
                counts[k] = counts.get(k, 0) + int(v)
        else:
            counts[name] = counts.get(name, 0) + int(got or 0)
    ts = store.now()
    night.mark.update(
        {"memory_version": upto, "at": ts, "db_bytes": db_bytes(conn), "counts": counts}
    )
    _save_mark(conn, night.mark, ts)
    return counts


__all__ = [
    "BUILT_IN",
    "Night",
    "check_merges",
    "kept_values",
    "last_mark",
    "marks",
    "on_ac",
    "register_step",
    "run",
    "survival_ok",
    "unregister_step",
    "use_power",
]
