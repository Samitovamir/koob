"""Scheduler injection point: quiet (10 min after a chat goes quiet), evening, night, weekly;
goal events enqueue assoc jobs at once; local-only jobs wait, then park after 7 days.

Spec: §5.4 (park, never downgrade), §6.3. Started from `Runtime.ensure_started` through an
adapter (§16.4), never from web routes.

`tick(conn, clock)` runs due jobs: the urgent `safety_net` jobs first, then `extract`
batches (and their per-group child jobs), and last `embed` (`vectors.enqueue` + the `embed`
runner) on every tick: what this tick extracted and the raw messages that arrived since the
last one get their vectors before the next turn, not one idle tick later (a busy stream of
batches would otherwise keep vectors behind). One extract run is snapshot → route →
`propose` → `store.commit`:

- **Route (§5.4).** Raw labels are finalized by code first (`policy.label`, only once the
  in-process classifier is wired, fail-closed); `policy.route` over the batch's raw
  picks the processor (cloud before local when both are allowed). No allowed processor:
  the job waits (`not_before` + 10 min) and after 7 days is `parked` with a doctor line;
  a parked job whose route allows local re-arms when a local processor appears. Never a
  downgrade.
- **Stale (§6.3).** Each stale group gets one child job (`{parent, group, attempt}`,
  key `<parent>:<group>:<attempt>`) with a fresh snapshot that keeps only that group;
  attempt 3 stale again → that job is `needs_attention` (one doctor line, no user line).
- **Rejected.** A worker proposal's content rejection drops its group alone
  (`CommitResult.rejected`, store); the scheduler never retries it and writes one
  content-free audit line `extract_rejected:<reason>`.
- **Silent.** Background learning writes no user line and no feed (§6.3); doctor lines
  are `audit` rows, which only `cb memory doctor` reads.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from .. import store
from ..clock import Clock
from ..model import Label, ReadSrc, Route

EXTRACT = "extract"  # ingest.EXTRACT_JOB
SAFETY = "safety_net"  # writers.hot.SAFETY_JOB
EMBED = "embed"
WAIT = timedelta(minutes=10)
PARK_AFTER = timedelta(days=7)
ATTEMPTS = 3  # §6.3: the first run is attempt 1; attempt 3 stale → needs_attention
_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


class Scheduler(Protocol):
    """Tests drive a deterministic scheduler; production uses a bounded worker pool."""

    def call_at(self, when: datetime, fn: Callable[[], None]) -> None: ...

    def run_due(self) -> int: ...


class ManualScheduler:
    """Runs due callbacks only when told to; ordered by time, then insertion."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._queue: list[tuple[datetime, int, Callable[[], None]]] = []
        self._n = 0

    def call_at(self, when: datetime, fn: Callable[[], None]) -> None:
        self._n += 1
        self._queue.append((when, self._n, fn))
        self._queue.sort(key=lambda x: (x[0], x[1]))

    def run_due(self) -> int:
        ran = 0
        while self._queue and self._queue[0][0] <= self._clock.now():
            _, _, fn = self._queue.pop(0)
            fn()
            ran += 1
        return ran


# ---------------------------------------------------------------- queue


def _ts(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(_FMT)


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def enqueue(
    conn: sqlite3.Connection,
    jkind: str,
    payload: dict,
    *,
    idempotency_key: str,
    not_before: str | None = None,
) -> str:
    """Queue a job once per `idempotency_key` (a second enqueue returns the first id). Joins
    the caller's transaction when there is one (ingest enqueues inside its own)."""
    from ..ledger import new_id

    ts = store.now()
    sql = (
        "INSERT OR IGNORE INTO jobs (id, jkind, payload, state, attempts, not_before,"
        " idempotency_key, created_at, updated_at) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?, ?)"
    )
    args = (new_id(), jkind, json.dumps(payload), not_before, idempotency_key, ts, ts)
    if conn.in_transaction:
        conn.execute(sql, args)
    else:
        with store.write_tx(conn, priority="background"):
            conn.execute(sql, args)
    row = conn.execute("SELECT id FROM jobs WHERE idempotency_key = ?", (idempotency_key,))
    return row.fetchone()[0]


def doctor_line(conn: sqlite3.Connection, event: str, job_id: str | None = None) -> None:
    """A content-free audit row (`cb memory doctor` reads them); never a user-facing line."""
    sql = "INSERT INTO audit (ts, event, record_id, actor) VALUES (?, ?, ?, 'worker')"
    if conn.in_transaction:
        conn.execute(sql, (store.now(), event, job_id))
    else:
        with store.write_tx(conn, priority="background"):
            conn.execute(sql, (store.now(), event, job_id))


def _set_state(
    conn: sqlite3.Connection,
    job_id: str,
    state: str,
    *,
    not_before: str | None = None,
    payload: dict | None = None,
) -> None:
    with store.write_tx(conn, priority="background"):
        conn.execute(
            "UPDATE jobs SET state = ?, not_before = ?, updated_at = ?,"
            " payload = COALESCE(?, payload) WHERE id = ?",
            (state, not_before, store.now(), json.dumps(payload) if payload else None, job_id),
        )


# ---------------------------------------------------------------- labels and routes (§5.3-§5.4)


def finalize_labels(conn: sqlite3.Connection, raw_ids: list[str]) -> int:
    """§5.3: code labels the batch (labels only grow) and sets `labels_final` — only when the
    in-process classifier is installed and warm; without it, or cold, the raw stays not final
    (local only) and the job waits. Fail-closed: a classifier that goes cold mid-batch
    leaves the rest of the batch not final."""
    from .. import policy

    if not raw_ids or not policy.classifier_ready():
        return 0
    rows = conn.execute(
        f"SELECT id, labels, text FROM raw_chunks WHERE id IN ({', '.join('?' * len(raw_ids))})"
        " AND labels_final = 0 AND status = 'active'",
        raw_ids,
    ).fetchall()
    done: list[tuple[str, Label]] = []
    for rid, cur, text in rows:
        lab = policy.refine_labels(Label(cur), policy.label(text or ""))
        if not policy.classifier_ready():
            break  # the classifier failed on this row: its labels are not final
        done.append((rid, lab))
    if not done:
        return 0
    with store.write_tx(conn, priority="background"):
        for rid, lab in done:
            # OR-ed in the write: a label another writer added after the read survives (the
            # read's `labels` may be stale by now); labels only grow
            conn.execute(
                "UPDATE raw_chunks SET labels = labels | ?, labels_final = 1, updated_at = ?"
                " WHERE id = ?",
                (int(lab), store.now(), rid),
            )
    return len(done)


def choose(conn: sqlite3.Connection, raw_refs: list) -> tuple[Any, str | None, str]:
    """(processor, route, why-not): cloud before local when both are allowed; never a route
    the table does not allow (§5.4)."""
    from .. import policy
    from . import extract

    allowed = policy.route(conn, raw_refs)
    for r in (Route.CLOUD, Route.LOCAL):
        if r in allowed:
            p = extract.processor(r.value)
            if p is not None:
                return p, r.value, ""
    if not allowed:
        return None, None, "no_route"
    return None, None, "no_local" if Route.LOCAL in allowed else "no_cloud"


def split_by_route(conn: sqlite3.Connection, raw_refs: list) -> tuple[list[list], list]:
    """Per-message routes (§5.4): a batch that no processor may take whole is cut so every
    message that may be read somewhere now is read now, and only the ones that block wait.
    Returns (parts that run now, the rest). Each part is routed by `policy.route` on its own
    members, so the union rule still holds inside every part: a part never mixes the user's
    labelled words with anyone else's text (those go apart, ours and theirs), and another
    person's labelled text is never in a cloud part. Nothing runs now: ([], all)."""
    from .. import policy
    from . import extract

    up = {r for r in (Route.CLOUD, Route.LOCAL) if extract.processor(r.value) is not None}
    each = policy.route_each(conn, raw_refs)  # one load for the batch, not one per message
    ready = [r for r in raw_refs if each.get(r.id, set()) & up]
    if not ready:
        return [], list(raw_refs)
    parts: list[list] = []
    if choose(conn, ready)[0] is not None:
        parts = [ready]
    else:  # the user's labelled words beside another person's text: each side on its own
        mine = policy.ours(conn, ready)
        for side in ([r for r in ready if r.id in mine], [r for r in ready if r.id not in mine]):
            if side and choose(conn, side)[0] is not None:
                parts.append(side)
    taken = {r.id for p in parts for r in p}
    return parts, [r for r in raw_refs if r.id not in taken]


def _run_parts(
    conn: sqlite3.Connection,
    job: dict[str, Any],
    parts: list[list],
    rest: list,
    clock: Clock,
) -> str:
    """Run the parts of a split batch now, each as its own `extract` job (its stale groups
    retry under it); `job` keeps the rest and waits on its own clock (its park age is the
    batch's). Returns `partial` while some of the batch waits, else `done`."""
    payload = dict(job.get("payload") or {})
    jkind = str(job.get("jkind") or EXTRACT)
    rest_ids = [r.id for r in rest]
    kids: list[str] = []
    with store.write_tx(conn, priority="background"):
        for part in parts:
            ids = [r.id for r in part]
            kid = {"raw_ids": ids, "turn_id": payload.get("turn_id"), "split_from": job["id"]}
            kids.append(enqueue(conn, jkind, kid, idempotency_key=f"{job['id']}:route:{ids[0]}"))
        payload["raw_ids"] = rest_ids
        conn.execute(
            "UPDATE jobs SET payload = ?, updated_at = ? WHERE id = ?",
            (json.dumps(payload), store.now(), job["id"]),
        )
    for kid in kids:
        if not store.claim_job(conn, kid):
            continue  # already run (or running) by an earlier pass over the same batch
        child = {
            "id": kid,
            "jkind": jkind,
            "payload": json.loads(
                conn.execute("SELECT payload FROM jobs WHERE id = ?", (kid,)).fetchone()[0] or "{}"
            ),
        }
        if run_batch(conn, child, clock) in ("done", "empty"):
            _set_state(conn, kid, "done")
    if not rest_ids:
        return "done"
    _, _, why = choose(conn, rest)
    _wait(conn, {**job, "payload": payload}, why or "no_route", clock)
    return "partial"


def _wait(conn: sqlite3.Connection, job: dict[str, Any], why: str, clock: Clock) -> None:
    row = conn.execute("SELECT created_at FROM jobs WHERE id = ?", (job["id"],)).fetchone()
    born = _parse(row[0] if row else None) or clock.now()
    payload = dict(job.get("payload") or {})
    payload["waiting_for"] = why
    if clock.now() - born >= PARK_AFTER:
        _set_state(conn, job["id"], "parked", payload=payload)
        doctor_line(conn, f"job_parked:{why}", job["id"])
    else:
        _set_state(conn, job["id"], "queued", not_before=_ts(clock.now() + WAIT), payload=payload)


def rearm_parked(conn: sqlite3.Connection, *, limit: int = 200) -> int:
    """§5.4: parked jobs that waited for a route or a processor go back once one is
    allowed and here: a local processor appears, or the routing table changes."""
    from .. import policy
    from ..model import ReadRef

    rows = conn.execute(
        "SELECT id, payload FROM jobs WHERE state = 'parked' AND jkind IN (?, ?)"
        " AND json_extract(payload, '$.waiting_for') IN ('no_local', 'no_cloud', 'no_route')"
        " ORDER BY updated_at LIMIT ?",
        (EXTRACT, SAFETY, limit),
    ).fetchall()
    back: list[str] = []
    for jid, raw in rows:
        payload = json.loads(raw or "{}")
        ids = [str(r) for r in payload.get("raw_ids") or []]
        if not ids:
            continue
        q = ", ".join("?" * len(ids))
        refs = [
            ReadRef(id=r[0], version=int(r[1]), src=ReadSrc.RAW)
            for r in conn.execute(f"SELECT id, rev FROM raw_chunks WHERE id IN ({q})", ids)
        ]
        if len(refs) != len(set(ids)):
            back.append(jid)  # a row is gone: the run itself decides (empty / fewer rows)
            continue
        proc, _route, _why = choose(conn, refs)
        if proc is not None or (policy.route(conn, refs) and _why != payload.get("waiting_for")):
            back.append(jid)
        elif payload.get("group") is None and split_by_route(conn, refs)[0]:
            back.append(jid)  # some of its messages may be read now on their own
    if not back:
        return 0
    with store.write_tx(conn, priority="background"):
        for jid in back:
            conn.execute(
                "UPDATE jobs SET state = 'queued', not_before = NULL, updated_at = ?"
                " WHERE id = ? AND state = 'parked'",
                (store.now(), jid),
            )
    return len(back)


# ---------------------------------------------------------------- runners


def _existing(conn: sqlite3.Connection, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    q = ", ".join("?" * len(ids))
    return {r[0] for r in conn.execute(f"SELECT id FROM records WHERE id IN ({q})", ids)}


def run_batch(conn: sqlite3.Connection, job: dict[str, Any], clock: Clock) -> str:
    """One extraction run of `job` (an `extract` batch or its child). Returns its outcome:
    done / waiting / needs_attention / empty."""
    from .. import ledger
    from . import extract

    payload = dict(job.get("payload") or {})
    raw_ids = [str(r) for r in payload.get("raw_ids") or []]
    finalize_labels(conn, raw_ids)
    snap, read_set = extract.snapshot(conn, raw_ids)
    if not snap["raws"]:
        return "empty"
    raw_refs = [r for r in read_set if r.src is ReadSrc.RAW]
    proc, route, why = choose(conn, raw_refs)
    if proc is None:
        # route per message: the clean ones are read now, only the blocking ones wait; a
        # stale group's retry keeps its batch whole (its group ids are that batch's)
        parts, rest = split_by_route(conn, raw_refs) if payload.get("group") is None else ([], [])
        if parts:
            return _run_parts(conn, job, parts, rest, clock)
        _wait(conn, job, why, clock)
        return "waiting"
    if route == Route.CLOUD.value:  # a labelled known fact never rides along to the cloud
        snap = extract.cloud_safe(snap)
        read_set = extract.read_set_of(snap)
    store.renew_lease(conn, job["id"])
    proposal = extract.propose(snap, read_set, proc)
    store.renew_lease(conn, job["id"])
    ops = proposal.ops
    if payload.get("group") is not None:
        ops = [op for op in ops if op.group == payload["group"]]
    adds = [op.payload["record"]["id"] for op in ops if "record" in op.payload]
    have = _existing(conn, [i for i in adds if i])
    ops = [op for op in ops if op.payload.get("record", {}).get("id") not in have]
    if not ops:
        return "done"
    res = store.commit(conn, proposal.model_copy(update={"ops": ops}), ledger.apply_op)
    for reason in res.rejected.values():  # content-free; never retried (F6)
        doctor_line(conn, f"extract_rejected:{reason.split(':', 1)[0]}")
    attempt = int(payload.get("attempt") or 1)
    root = str(payload.get("parent") or job["id"])
    for group in res.stale:
        if attempt >= ATTEMPTS:
            _set_state(conn, job["id"], "needs_attention")
            doctor_line(conn, "job_needs_attention:stale", job["id"])
            return "needs_attention"
        child = {
            "parent": root,
            "group": group,
            "attempt": attempt + 1,
            "raw_ids": raw_ids,
            "turn_id": payload.get("turn_id"),
        }
        enqueue(
            conn,
            str(job.get("jkind") or EXTRACT),
            child,
            idempotency_key=f"{root}:{group}:{attempt + 1}",
        )
    return "done"


def _clock() -> Clock:
    from .. import clock as clockmod

    return clockmod.current()  # the tick's scoped clock, else the wired process clock


def run_extract(conn: sqlite3.Connection, job: dict[str, Any]) -> None:
    run_batch(conn, job, _clock())


def run_safety_net(conn: sqlite3.Connection, job: dict[str, Any]) -> None:
    """§6.1 step 2: the urgent re-read of one turn. Its queued `extract` batches run now (and
    are done); a turn already extracted has nothing left. No allowed processor: `failed`,
    so the hot path says its one failure line; the batches keep waiting on their own."""
    turn = (job.get("payload") or {}).get("turn_id")
    if not turn:
        return
    rows = conn.execute(
        "SELECT id, jkind, payload, attempts FROM jobs WHERE jkind = ? AND state = 'queued'"
        " AND json_extract(payload, '$.turn_id') = ? ORDER BY created_at, id",
        (EXTRACT, turn),
    ).fetchall()
    for jid, jkind, raw, attempts in rows:
        if not store.claim_job(conn, jid):
            continue
        batch = {
            "id": jid,
            "jkind": jkind,
            "payload": json.loads(raw or "{}"),
            "attempts": attempts,
        }
        outcome = run_batch(conn, batch, _clock())
        if outcome == "waiting":
            _set_state(conn, job["id"], "failed")
            return
        if outcome not in ("needs_attention", "partial"):  # a partial batch waits for its rest
            _set_state(conn, jid, "done")


def register() -> None:
    store.register_job_runner(EXTRACT, run_extract)
    store.register_job_runner(SAFETY, run_safety_net)


register()


def tick(conn: sqlite3.Connection, clock: Clock) -> int:
    """Run ready jobs whose route has an allowed processor; park after 7 days. Leases, waits
    and parks within this tick read `clock` (`clock.scoped`); the process clock is untouched,
    so a turn in another thread keeps its own time."""
    from .. import clock as clockmod

    with clockmod.scoped(clock):
        register()
        rearm_parked(conn)
        ran = store.run_queued_jobs(conn, SAFETY) + store.run_queued_jobs(conn, EXTRACT)
        return ran + embed(conn)


def embed(conn: sqlite3.Connection) -> int:
    """Vectors for every record and raw message that changed since the last pass (one pull
    job; nothing to do costs one scan). Background only: the turn never embeds. Returns the
    jobs run (0 when no embedder is available: the work waits for the next tick)."""
    try:
        from ..recall import vectors

        vectors.enqueue(conn)
        return store.run_queued_jobs(conn, EMBED)
    except Exception:  # noqa: BLE001 - no embedder here: the work waits
        return 0
