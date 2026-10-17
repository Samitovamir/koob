"""Truth maintenance: AND/OR justification groups, well-founded support, independence by root
origins, the three outcomes and background re-derivation.

Spec: §8.1, §8.4.

- **Support (§8.1).** A conclusion with justification groups is supported iff some group's
  members are all supported: the least fixed point over the member graph, so a cycle (of
  inferred links or any other) never keeps itself alive. A member with no groups is a root:
  a usable record (`active`, `archived`) or an `active` raw chunk. A record superseded by a
  change in the world (a move, a new job, §8.4) was true when the conclusion was drawn and
  still grounds it; a refuted one (retracted, fixed, `needs_review`, forgotten) does not.
- **Roots (§8.1).** A member resolves to its original author or URL: a relayed claim to its
  named original speaker (`speaker_entity_id`), the user's own statements to the user, a raw
  to its author, its URL or (unknown author) its content hash, so a forward or a pasted quote
  is the same root as the original; a derived member to the roots of its groups.
- **Independence.** Two groups are independent iff their roots, without the user, are
  disjoint: the user's statements need no independence test, the latest wins. In an outcome
  a group that still holds but shares a root with a refuted member is no second source.
- **Outcomes (§8.4).** `false`: the record itself is contradicted (superseded or retracted,
  or a changed premise's successor rule-matches it, §8.3). `unsupported`: no group holds and
  every group broke through a `stated` or `rule` link (or rests on a refuted root):
  `needs_review`, excluded from core and basket, a re-derivation job queued. `review`: a
  premise shifted without refuting it, or only an `inferred` link broke: usable, `review`
  flag, never auto-retracted. `unchanged`: some independent group still holds.
- **Cascade.** `cascade(ctx, changed)` runs inside the caller's transaction (the hot fix,
  `writers.hot._cascade`), as system-tier ledger ops at the commit's version, and follows
  `needs_review` downstream; past `INLINE_CAP` dependents the rest is a durable job.
- **Re-derivation.** A `rederive` job re-judges one `needs_review` record and, when an
  independent group holds again, revives it through `store.commit` with its read set (a
  record changed meanwhile makes the proposal stale; the next change queues it again). The
  night step (`nightly.register_step("cascade", …)`) cascades worker supersessions since the
  last night and runs both job kinds. No model: re-deriving a conclusion's text is the
  extractor's.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from . import ledger, store
from .model import (
    OWNER_ENTITY_ID,
    AuthorTier,
    OpKind,
    Proposal,
    ProposalOp,
    Provenance,
    ReadRef,
    RecordKind,
    Status,
)
from .store import CommitCtx

Outcome = Literal["false", "unsupported", "review", "unchanged"]

REDERIVE_JOB = "rederive"
CASCADE_JOB = "cascade"
INLINE_CAP = 64  # dependents judged inside the caller's transaction; the rest is a job
NIGHT_CAP = 500

OWNER_ROOT = f"entity:{OWNER_ENTITY_ID}"
# ids the user's correction retires in the running commit: its `fix` events row is not
# written yet
_FIXED: ContextVar[frozenset[str]] = ContextVar("justify_fixed", default=frozenset())
_USABLE = (Status.ACTIVE.value, Status.ARCHIVED.value)
_OWN = (AuthorTier.USER.value, AuthorTier.USER_CONFIRMED.value)
_FACTS = (RecordKind.FACT.value, RecordKind.PREFERENCE.value)
_FIXED_FAMILIES = frozenset({"identity"})  # a new birthday means the old one was wrong
# «the old value was wrong», in the words of the record that superseded it (not «поменял»:
# a changed job is a change in the world, §8.4)
_WRONG = re.compile(
    r"(?<!\w)(?:нет[,.!—–-]|неправд\w*|не\s+правда|неверн\w*|не\s+так(?!\w)|ошиб\w+"
    r"|перепута\w*|на\s+самом\s+деле|исправ\w*|поправ\w*|не\s+[\w.-]+\s*,\s*а\s"
    r"|wrong|incorrect|mistake\w*|not\s+(?:true|right)|actually)",
    re.IGNORECASE,
)

Groups = dict[int, list[tuple[str, str]]]  # group_no -> [(member, provenance)]


# ---------------------------------------------------------------- graph reads


def _groups(conn: sqlite3.Connection, record_id: str) -> Groups:
    out: Groups = {}
    for g, m, p in conn.execute(
        "SELECT group_no, member_record_id, provenance FROM justifications"
        " WHERE target_record_id = ? ORDER BY group_no, member_record_id",
        (record_id,),
    ):
        out.setdefault(int(g), []).append((str(m), str(p)))
    return out


def _closure(conn: sqlite3.Connection, start: Iterable[str]) -> dict[str, Groups]:
    """Every node reachable from `start` through justification members, with its groups."""
    graph: dict[str, Groups] = {}
    todo = list(dict.fromkeys(start))
    while todo:
        node = todo.pop()
        if node in graph:
            continue
        graph[node] = _groups(conn, node)
        todo.extend(m for ms in graph[node].values() for m, _ in ms if m not in graph)
    return graph


def _status(conn: sqlite3.Connection, node: str) -> tuple[str, str | None] | None:
    """(source, status): `record` with its status, `raw` with its status; None when absent."""
    row = conn.execute("SELECT status FROM records WHERE id = ?", (node,)).fetchone()
    if row is not None:
        return "record", str(row[0])
    row = conn.execute("SELECT status FROM raw_chunks WHERE id = ?", (node,)).fetchone()
    if row is not None:
        return "raw", str(row[0])
    return None


def _successors(conn: sqlite3.Connection, record_id: str) -> list[str]:
    return [
        str(r[0])
        for r in conn.execute(
            "SELECT from_id FROM edges WHERE to_id = ? AND etype = 'supersedes'", (record_id,)
        )
    ]


def _shifted(conn: sqlite3.Connection, record_id: str) -> bool:
    """A superseded fact whose world changed (a move, a job, a relationship: §8.4) rather
    than one found wrong: no retraction overlay (§9.6), a family whose values change, and a
    successor that does not say the old value was wrong, and no «исправь» retired it (the
    user's fix in the running commit, or a `fix` events row at the version it was retired at)."""
    if record_id in _FIXED.get():
        return False
    row = conn.execute(
        "SELECT r.kind, r.version, rel.family FROM records r LEFT JOIN facts f ON f.id = r.id"
        " LEFT JOIN relations rel ON rel.relation = f.relation WHERE r.id = ?",
        (record_id,),
    ).fetchone()
    if row is None or row[0] not in _FACTS or row[2] in _FIXED_FAMILIES:
        return False
    if conn.execute(
        "SELECT 1 FROM raw_retractions WHERE record_id = ? LIMIT 1", (record_id,)
    ).fetchone():
        return False
    for succ in _successors(conn, record_id):
        q = conn.execute("SELECT origin_quote FROM records WHERE id = ?", (succ,)).fetchone()
        if q is not None and q[0] and _WRONG.search(str(q[0])):
            return False
    fixed = conn.execute(
        "SELECT 1 FROM events WHERE memory_version = ? AND kind = 'fix'"
        " AND instr(item_ids, ?) > 0 LIMIT 1",
        (row[1], json.dumps(record_id)),
    ).fetchone()
    return fixed is None


def _is_fix(ctx: CommitCtx) -> bool:
    """A cascade inside the user's correction commit: hot.fix commits a bare SUPERSEDE or RETRACT
    placeholder (a worker's supersede carries its record or `by`)."""
    for op in ctx.proposal.ops:
        if op.op is OpKind.RETRACT or (
            op.op is OpKind.SUPERSEDE and (op.payload.get("fix") or not op.payload)
        ):
            return True
    return False


def _grounds(conn: sqlite3.Connection, node: str) -> bool:
    """Whether the node itself may ground a conclusion (its groups aside)."""
    st = _status(conn, node)
    if st is None:
        return False
    src, status = st
    if src == "raw":
        return status == "active"
    if status in _USABLE:
        return True
    return status == Status.SUPERSEDED.value and _shifted(conn, node)


def _lfp(graph: dict[str, Groups], grounds: dict[str, bool], exempt: str | None) -> set[str]:
    """§8.1 least fixed point: start from the grounded roots, add a node once one of its
    groups is all in; a cycle with no grounded way in never enters."""
    true = {n for n, gs in graph.items() if not gs and grounds[n]}
    grew = True
    while grew:
        grew = False
        for n, gs in graph.items():
            if n in true or not gs or not (grounds[n] or n == exempt):
                continue
            if any(all(m in true for m, _ in ms) for ms in gs.values()):
                true.add(n)
                grew = True
    return true


def _support(conn: sqlite3.Connection, record_id: str) -> tuple[dict[str, Groups], set[str]]:
    graph = _closure(conn, [record_id])
    grounds = {n: _grounds(conn, n) for n in graph}
    return graph, _lfp(graph, grounds, record_id)


def supported(conn: sqlite3.Connection, record_id: str) -> bool:
    """§8.1: least fixed point; inferred cycles never keep themselves alive. A record with
    groups is judged by its groups (its own `needs_review` aside, so a re-derivation can ask);
    a root by its own status."""
    graph, true = _support(conn, record_id)
    if not graph[record_id]:
        return record_id in true
    st = _status(conn, record_id)
    if st is None or st[1] in (Status.FORGOTTEN.value, Status.PURGED.value):
        return False
    return record_id in true


# ---------------------------------------------------------------- roots and independence


def _raw_root(conn: sqlite3.Connection, raw_id: str) -> str:
    row = conn.execute(
        "SELECT author_entity_id, source_ref, content_hmac FROM raw_chunks WHERE id = ?",
        (raw_id,),
    ).fetchone()
    if row is None:
        return f"raw:{raw_id}"
    author, ref, hmac = row
    if author:
        return f"entity:{author}"
    ref = str(ref or "")
    if ref.startswith(("url:", "http://", "https://")):
        return "url:" + ref.removeprefix("url:").rstrip("/").casefold()
    return f"hmac:{hmac}" if hmac else f"raw:{raw_id}"


def _leaf_roots(conn: sqlite3.Connection, record_id: str) -> set[str]:
    row = conn.execute(
        "SELECT r.author_tier, r.author_entity_id, f.speaker_entity_id, f.source_raw_id"
        " FROM records r LEFT JOIN facts f ON f.id = r.id WHERE r.id = ?",
        (record_id,),
    ).fetchone()
    if row is None:
        return {f"rec:{record_id}"}
    tier, author, speaker, source_raw = row
    if speaker:  # a relayed claim: its named original speaker (§8.1)
        return {f"entity:{speaker}"}
    if tier in _OWN:
        return {OWNER_ROOT}
    raws = [source_raw] if source_raw else []
    raws += [
        str(r[0])
        for r in conn.execute(
            "SELECT e.to_id FROM edges e JOIN raw_chunks c ON c.id = e.to_id"
            " WHERE e.from_id = ? AND e.etype = 'derived_from'",
            (record_id,),
        )
    ]
    if raws:
        return {_raw_root(conn, r) for r in dict.fromkeys(raws)}
    return {f"entity:{author}"} if author else {f"rec:{record_id}"}


def _member_roots(conn: sqlite3.Connection, member: str, seen: set[str]) -> set[str]:
    st = _status(conn, member)
    if st is not None and st[0] == "raw":
        return {_raw_root(conn, member)}
    gs = _groups(conn, member)
    if not gs:
        return _leaf_roots(conn, member)
    if member in seen:
        return set()
    seen.add(member)
    out: set[str] = set()
    for ms in gs.values():  # every way it was derived: conservative for independence
        for m, _ in ms:
            out |= _member_roots(conn, m, seen)
    return out or _leaf_roots(conn, member)


def roots(conn: sqlite3.Connection, record_id: str, group_no: int) -> set[str]:
    """§8.1: root origins of a group; forwards and quotes resolve to the original author."""
    members = _groups(conn, record_id).get(int(group_no), [])
    out: set[str] = set()
    for m, _ in members:
        out |= _member_roots(conn, m, {record_id})
    return out


def independent(conn: sqlite3.Connection, record_id: str, groups: tuple[int, int]) -> bool:
    """§8.1: groups are independent only if their root origins are disjoint. The user's own
    statements need no independence test (the latest wins), so the user's root is left out."""
    a, b = groups
    gs = _groups(conn, record_id)
    if a == b or a not in gs or b not in gs:
        return False
    ra = roots(conn, record_id, a) - {OWNER_ROOT}
    rb = roots(conn, record_id, b) - {OWNER_ROOT}
    return not (ra & rb)


# ---------------------------------------------------------------- outcomes


def _contradicted(conn: sqlite3.Connection, record_id: str, changed: Iterable[str]) -> str | None:
    """§8.4 false: a successor of a changed premise rule-matches the record itself (§8.3)."""
    kind = conn.execute("SELECT kind FROM records WHERE id = ?", (record_id,)).fetchone()
    if kind is None or kind[0] not in _FACTS:
        return None
    for x in changed:
        for succ in _successors(conn, x):
            if record_id in ledger.rule_match(conn, succ):
                return succ
    return None


def _judge(conn: sqlite3.Connection, record_id: str, changed: set[str]) -> Outcome:
    graph, true = _support(conn, record_id)
    gs = graph[record_id]
    if not gs:
        return "unchanged"
    refuted = {m for ms in gs.values() for m, _ in ms if m not in true}
    tainted_roots: set[str] = set()
    for m in refuted:
        tainted_roots |= _member_roots(conn, m, {record_id})
    tainted_roots.discard(OWNER_ROOT)
    shifted = {m for m in changed if m in true and _status(conn, m) == ("record", "superseded")}
    clean = review = False
    hard = 0
    for ms in gs.values():
        broken = [(m, p) for m, p in ms if m in refuted]
        if not broken:
            shared = [(m, p) for m, p in ms if _member_roots(conn, m, {record_id}) & tainted_roots]
            if not shared:  # an independent group still holds
                if any(m in shifted for m, _ in ms):
                    review = True  # a premise shifted without refuting it
                else:
                    clean = True
                continue
            broken = shared  # a retelling of a refuted root is no second source
        if any(p != Provenance.INFERRED.value for _, p in broken):
            hard += 1
    if clean:
        return "unchanged"
    if review:
        return "review"
    return "unsupported" if hard == len(gs) else "review"


def outcome(conn: sqlite3.Connection, record_id: str, changed: list[str]) -> Outcome:
    """§8.4: false / unsupported / review; an inferred link leads only to review. A
    `needs_review` record is judged again by its groups (`unchanged`: it holds again)."""
    st = _status(conn, record_id)
    if st is None or st[0] != "record":
        return "unchanged"
    status = st[1]
    if status in (Status.FORGOTTEN.value, Status.PURGED.value):
        return "unchanged"
    if status in (Status.SUPERSEDED.value, Status.RETRACTED.value):
        return "false"
    if _contradicted(conn, record_id, changed) is not None:
        return "false"
    return _judge(conn, record_id, set(changed))


# ---------------------------------------------------------------- cascade


def _system(ctx: CommitCtx) -> CommitCtx:
    """The cascade is code acting inside the caller's transaction, at the newest version."""
    return CommitCtx(
        conn=ctx.conn,
        version=store.memory_version(ctx.conn),
        ts=ctx.ts,
        proposal=Proposal(ops=[], author_tier=AuthorTier.SYSTEM),
    )


def _enqueue(conn: sqlite3.Connection, jkind: str, payload: dict[str, Any], key: str) -> None:
    ts = store.now()
    conn.execute(
        "INSERT OR IGNORE INTO jobs (id, jkind, payload, state, attempts, idempotency_key,"
        " created_at, updated_at) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?)",
        (ledger.new_id(), jkind, json.dumps(payload), key, ts, ts),
    )


def _dependents(conn: sqlite3.Connection, ids: list[str]) -> list[str]:
    if not ids:
        return []
    marks = ", ".join("?" for _ in ids)
    return [
        str(r[0])
        for r in conn.execute(
            "SELECT DISTINCT j.target_record_id FROM justifications j"
            f" JOIN records r ON r.id = j.target_record_id"
            f" WHERE j.member_record_id IN ({marks}) AND r.status IN {_USABLE}"  # noqa: S608
            " ORDER BY j.target_record_id",
            ids,
        )
    ]


def _flag(ctx: CommitCtx, record_id: str, **payload: Any) -> None:
    ledger.apply_op(
        ctx, ProposalOp(op=OpKind.FLAG, target=record_id, group=record_id, payload=payload)
    )


def cascade(ctx: CommitCtx, changed: list[str]) -> dict[str, Outcome]:
    """Evaluate dependents of changed records; heavy re-derivation is queued (propose/commit).

    Runs in the caller's transaction: `unsupported` → `needs_review` (excluded at once, its own
    dependents judged next, a `rederive` job queued); `review` → the flag (not
    version-bearing; derived holders are rebuilt); `false` → superseded by the matching
    successor (a tier that may not: `review`). Inside the user's correction commit the changed
    ids were wrong, not changed in the world. Returns the dependents whose state changed."""
    token = _FIXED.set(frozenset(changed) if _is_fix(ctx) else frozenset())
    try:
        return _cascade(ctx, changed)
    finally:
        _FIXED.reset(token)


def _cascade(ctx: CommitCtx, changed: list[str]) -> dict[str, Outcome]:
    conn = ctx.conn
    sys_ctx = _system(ctx)
    done: dict[str, Outcome] = {}
    seen = set(changed)
    all_changed = list(dict.fromkeys(changed))
    frontier = list(all_changed)
    judged = 0
    while frontier:
        deps = [d for d in _dependents(conn, frontier) if d not in seen]
        nxt: list[str] = []
        for i, dep in enumerate(deps):
            if judged >= INLINE_CAP:  # the rest waits for a job, durably in this transaction
                rest = list(dict.fromkeys([*all_changed, *nxt]))  # re-judging is idempotent
                key = f"cascade:{sys_ctx.version}:{deps[i]}"
                _enqueue(conn, CASCADE_JOB, {"changed": rest}, key)
                return done
            judged += 1
            seen.add(dep)
            got = outcome(conn, dep, all_changed)
            if got == "unsupported":
                _flag(sys_ctx, dep, status=Status.NEEDS_REVIEW.value)
                _enqueue(
                    conn, REDERIVE_JOB, {"record_id": dep}, f"rederive:{dep}:{sys_ctx.version}"
                )
                nxt.append(dep)
            elif got == "false":
                succ = _contradicted(conn, dep, all_changed)
                try:
                    if succ is not None:
                        ledger.supersede(sys_ctx, dep, succ)
                        nxt.append(dep)
                except ledger.CommitRejected:  # a tier that may not supersede it
                    succ = None
                if succ is None:
                    got = "review"
            if got == "review":
                row = conn.execute("SELECT review FROM records WHERE id = ?", (dep,)).fetchone()
                if row is not None and row[0]:
                    continue  # already in review: nothing new
                _flag(sys_ctx, dep, review=True)
                ledger.invalidate(sys_ctx, [dep])
            if got != "unchanged":
                done[dep] = got
        all_changed.extend(nxt)
        frontier = nxt
    return done


# ---------------------------------------------------------------- background (jobs, night)


def _ref(conn: sqlite3.Connection, record_id: str) -> ReadRef | None:
    row = conn.execute("SELECT version FROM records WHERE id = ?", (record_id,)).fetchone()
    return ReadRef(id=record_id, version=int(row[0])) if row is not None else None


def rederive(conn: sqlite3.Connection, record_id: str) -> bool:
    """Re-judge one `needs_review` record; when an independent group holds again, propose its
    revival (with `review` when only a shifted premise or an inferred link carries it) and
    commit it against its read set; its `needs_review` dependents are queued in turn. True
    when it was revived."""
    st = _status(conn, record_id)
    ref = _ref(conn, record_id)
    if st != ("record", Status.NEEDS_REVIEW.value) or ref is None:
        return False
    got = _judge(conn, record_id, set())
    if got not in ("unchanged", "review"):
        return False
    payload: dict[str, Any] = {"status": Status.ACTIVE.value}
    if got == "review":
        payload["review"] = True
    op = ProposalOp(
        op=OpKind.FLAG, target=record_id, group=record_id, read_set=[ref], payload=payload
    )

    def apply(ctx: CommitCtx, o: ProposalOp) -> list[str]:
        touched = list(ledger.apply_op(ctx, o))
        for dep in _waiting(ctx.conn, record_id):
            _enqueue(ctx.conn, REDERIVE_JOB, {"record_id": dep}, f"rederive:{dep}:{ctx.version}")
        return touched

    res = store.commit(
        conn, Proposal(ops=[op], author_tier=AuthorTier.SYSTEM), apply, priority="background"
    )
    return bool(res.committed)


def _waiting(conn: sqlite3.Connection, record_id: str) -> list[str]:
    return [
        str(r[0])
        for r in conn.execute(
            "SELECT DISTINCT j.target_record_id FROM justifications j"
            " JOIN records r ON r.id = j.target_record_id"
            " WHERE j.member_record_id = ? AND r.status = 'needs_review'",
            (record_id,),
        )
    ]


def run_cascade(conn: sqlite3.Connection, changed: list[str]) -> dict[str, Outcome]:
    """A cascade as its own system commit (a job, the night): touched ids go in its events row."""
    ids = list(dict.fromkeys(changed))
    if not ids:
        return {}
    box: dict[str, dict[str, Outcome]] = {}

    def apply(ctx: CommitCtx, _op: ProposalOp) -> list[str]:
        box["done"] = cascade(ctx, ids)
        return list(box["done"])

    op = ProposalOp(op=OpKind.FLAG, target=ids[0], group=f"cascade:{ids[0]}")
    store.commit(
        conn, Proposal(ops=[op], author_tier=AuthorTier.SYSTEM), apply, priority="background"
    )
    return box.get("done", {})


def _run_rederive_job(conn: sqlite3.Connection, job: dict[str, Any]) -> None:
    rid = (job.get("payload") or {}).get("record_id")
    if rid:
        rederive(conn, str(rid))


def _run_cascade_job(conn: sqlite3.Connection, job: dict[str, Any]) -> None:
    changed = (job.get("payload") or {}).get("changed") or []
    run_cascade(conn, [str(x) for x in changed])


def night_step(night: Any) -> dict[str, int]:
    """§6.3 night «cascades»: retirements since the last night that have dependents (worker
    supersessions never ran the hot cascade), then the queued jobs; `needs_review` records
    whose support came back are queued for re-derivation. Counts only."""
    conn = night.conn
    cap = min(int(getattr(night, "cap", NIGHT_CAP) or NIGHT_CAP), NIGHT_CAP)
    changed = [
        str(r[0])
        for r in conn.execute(
            "SELECT r.id FROM records r WHERE r.version > ? AND r.version <= ?"
            " AND r.status IN ('superseded', 'retracted', 'needs_review', 'forgotten', 'purged')"
            " AND EXISTS (SELECT 1 FROM justifications j WHERE j.member_record_id = r.id)"
            " ORDER BY r.version, r.id LIMIT ?",
            (night.since_version, night.upto_version, cap),
        )
    ]
    flipped = len(run_cascade(conn, changed)) if changed else 0
    pending = [
        str(r[0])
        for r in conn.execute(
            "SELECT r.id FROM records r WHERE r.status = 'needs_review'"
            " AND EXISTS (SELECT 1 FROM justifications j WHERE j.target_record_id = r.id)"
            " ORDER BY r.id LIMIT ?",
            (cap,),
        )
    ]
    revived = sum(1 for rid in pending if rederive(conn, rid))
    jobs = store.run_queued_jobs(conn, CASCADE_JOB) + store.run_queued_jobs(conn, REDERIVE_JOB)
    return {"cascade_flipped": flipped, "rederived": revived, "cascade_jobs": jobs}


# ---------------------------------------------------------------- what a conclusion rests on (views)

# A conclusion travels with the facts it rests on. The views (core, basket) show a decision
# with its premises, and when a premise was corrected or changed since, say so next to it
# («no longer holds because …»), so the brain never offers the old conclusion as current.
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_UNIT = re.compile(r"[^\W\d_]{3,}")
_GONE = (Status.FORGOTTEN.value, Status.PURGED.value)
_SINCE_CAP = 2


@dataclass(frozen=True)
class Grounds:
    """What a conclusion rests on, for display. Texts are raw (renderers clean them); `refs`
    are the usable records the texts name, `deps` the retired premises (never shown)."""

    held: tuple[tuple[str, str, bool], ...] = ()  # (id, text, review flag)
    changed: tuple[tuple[str, str | None], ...] = ()  # (retired premise, its successor's text)
    broken: tuple[str, ...] = ()  # premises withdrawn or on hold, no successor to show
    since: tuple[tuple[str, str], ...] = ()  # newer facts on a premise's slot or measure
    refs: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()

    @property
    def shaken(self) -> bool:
        """A premise was corrected, withdrawn or changed since: the conclusion may not hold."""
        return bool(self.changed or self.broken or self.since)


def _rec_text(conn: sqlite3.Connection, rid: str) -> str | None:
    row = conn.execute(
        "SELECT coalesce(f.text_en, d.chosen_text, r.origin_quote) FROM records r"
        " LEFT JOIN facts f ON f.id = r.id LEFT JOIN decisions d ON d.id = r.id WHERE r.id = ?",
        (rid,),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _head(conn: sqlite3.Connection, rid: str, seen: set[str] | None = None) -> str | None:
    """The usable end of a premise's supersession chain (None: withdrawn or forgotten)."""
    seen = seen or set()
    for succ in _successors(conn, rid):
        if succ in seen:
            continue
        seen.add(succ)
        st = _status(conn, succ)
        if st == ("record", Status.ACTIVE.value):
            return _head(conn, succ, seen) or succ
        if st is not None and st[1] == Status.SUPERSEDED.value:
            got = _head(conn, succ, seen)
            if got:
                return got
    return None


def _measure(text: str | None) -> tuple[set[str], set[str]]:
    t = (text or "").casefold()
    return set(_NUM.findall(t)), set(_UNIT.findall(_NUM.sub(" ", t)))


def _newer_on_slot(conn: sqlite3.Connection, premise: str, after: str, skip: set[str]) -> list[str]:
    """Active facts recorded after the conclusion that move a premise's value: the same
    subject and relation with another value, or (a free-form relation) the same subject and
    the same measure (a shared unit word) with another number («6 тысяч» → «12 тысяч»)."""
    row = conn.execute(
        "SELECT subject_id, relation, value_norm FROM facts WHERE id = ?", (premise,)
    ).fetchone()
    if row is None or not row[0]:
        return []
    subj, rel, val = row[0], row[1] or "other", row[2]
    nums, units = _measure(val)
    out = []
    for rid, rel2, val2 in conn.execute(
        "SELECT r.id, f.relation, f.value_norm FROM facts f JOIN records r ON r.id = f.id"
        " WHERE f.subject_id = ? AND r.status = 'active' AND r.recorded_at > ? AND r.id != ?"
        " ORDER BY r.recorded_at DESC, r.id LIMIT 20",
        (subj, after, premise),
    ):
        if rid in skip or not val2 or val2 == val:
            continue
        if rel != "other" and rel2 == rel:
            out.append(str(rid))
            continue
        n2, u2 = _measure(val2)
        if nums and n2 and n2 != nums and units & u2:
            out.append(str(rid))
    return out


def grounds(
    conn: sqlite3.Connection, record_id: str, allowed: Callable[[list[str]], set[str]]
) -> Grounds:
    """The premises of `record_id` (its justification members that are records) as they stand
    now. `allowed(ids)` is the view's own fence (policy, labels, frozen): a premise or
    successor it refuses is not named; a forgotten one is never mentioned at all."""
    rows = conn.execute(
        "SELECT DISTINCT j.member_record_id, r.status, r.review FROM justifications j"
        " JOIN records r ON r.id = j.member_record_id WHERE j.target_record_id = ?"
        " AND r.kind != 'entity' ORDER BY j.group_no, j.member_record_id",
        (record_id,),
    ).fetchall()
    rows = [r for r in rows if r[1] not in _GONE]
    if not rows:
        return Grounds()
    at = conn.execute("SELECT recorded_at FROM records WHERE id = ?", (record_id,)).fetchone()
    heads = {str(m): _head(conn, str(m)) for m, st, _ in rows if st == Status.SUPERSEDED.value}
    members = [str(m) for m, _, _ in rows]
    newer = {
        m: _newer_on_slot(conn, m, str(at[0]) if at else "", set(members))
        for m, st, _ in ((str(a), b, c) for a, b, c in rows)
        if st in _USABLE
    }
    ok = allowed(
        list(dict.fromkeys([*members, *(h for h in heads.values() if h)]
                           + [x for xs in newer.values() for x in xs]))
    )  # fmt: skip
    held: list[tuple[str, str, bool]] = []
    changed: list[tuple[str, str | None]] = []
    broken: list[str] = []
    since: list[tuple[str, str]] = []
    refs: list[str] = []
    deps: list[str] = []
    for m, st, review in ((str(a), b, bool(c)) for a, b, c in rows):
        if st in _USABLE:
            text = _rec_text(conn, m) if m in ok else None
            if text:
                held.append((m, text, review))
                refs.append(m)
            for n in newer.get(m, []):
                t2 = _rec_text(conn, n) if n in ok else None
                if t2 and len(since) < _SINCE_CAP and n not in refs:
                    since.append((n, t2))
                    refs.append(n)
            continue
        deps.append(m)
        h = heads.get(m)
        if st == Status.SUPERSEDED.value and h:
            t2 = _rec_text(conn, h) if h in ok else None
            changed.append((m, t2))
            if t2:
                refs.append(h)
        elif st == Status.SUPERSEDED.value:
            changed.append((m, None))
        else:  # retracted, needs_review: withdrawn or itself on hold
            broken.append(m)
    return Grounds(
        tuple(held), tuple(changed), tuple(broken), tuple(since),
        tuple(dict.fromkeys(refs)), tuple(dict.fromkeys(deps)),
    )  # fmt: skip


_WORDS = {
    "en": {
        "because": "because", "changed": "no longer holds: a premise was corrected",
        "now": "now", "broken": "no longer holds: a premise was withdrawn",
        "since": "since then", "recheck": "re-check before relying on it",
    },
    "ru": {
        "because": "основание", "changed": "больше не держится: основание исправлено",
        "now": "теперь", "broken": "больше не держится: основание снято",
        "since": "с тех пор", "recheck": "проверить, прежде чем опираться",
    },
}  # fmt: skip


def grounds_text(g: Grounds, lang: str, short: Callable[[str], str]) -> str:
    """One tail for a conclusion's line: its premises, then what changed under it."""
    w = _WORDS["ru" if lang == "ru" else "en"]
    parts = []
    held = [f"«{short(t)}»" for _, t, _ in g.held]
    if held:
        parts.append(f"{w['because']}: {'; '.join(held)}")
    for _, new in g.changed:
        parts.append(f"{w['changed']}, {w['now']} «{short(new)}»" if new else w["changed"])
    if g.broken:
        parts.append(w["broken"])
    if g.since:
        parts.append(f"{w['since']}: {'; '.join(f'«{short(t)}»' for _, t in g.since)}")
    if g.shaken:
        parts.append(w["recheck"])
    return " · ".join(parts) if lang == "ru" else "; ".join(parts)


def install() -> None:
    """Register the job runners and the night step (idempotent; runs at import)."""
    store.register_job_runner(REDERIVE_JOB, _run_rederive_job)
    store.register_job_runner(CASCADE_JOB, _run_cascade_job)
    from .workers import nightly

    nightly.register_step("cascade", night_step)


install()
