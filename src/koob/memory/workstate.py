"""Decisions with reasons, rejected alternatives and revision conditions; work state as JSON;
resume with premise re-validation.

Spec: §14, §8 (reasons as justification groups), §12 (revision conditions are intentions),
§15.1-§15.2 (decisions carry a `domain`; an active work state is a goal event).

Shapes (versioned, `SHAPE_VERSION`; the hands keep projects in this same work state, HANDS
§3, so every reader here is tolerant and every writer keeps what it does not know):

- **Decision.** `chosen_text`; reasons as justification groups (AND within a group, OR
  across, §8.1) passed as `reasons=[[member, ...], ...]`; rejected `alternatives` with their
  reasons; `revision_conditions` as `{"v", "text", "trigger", "intention_id", "armed"}`, each
  armed as an intention whose action is `{"kind": "revisit_decision", "decision_id",
  "matter_id"}`. A revised decision (`replaces=`) supersedes the old one and cancels its
  armed revision intentions.
- **Work state.** One active `work_state` record per matter (`entities.etype = 'matter'`),
  updated in place so its id is a stable handle. Body = `WorkStateBody` JSON, never prose.
  `premises` are read refs pinned at save time: a save on a premise that already moved is
  refused (`StalePremises`), so a plan is never stored on top of an old value. Keys a newer
  writer added are kept on every re-save.
- **Resume.** Premises are re-validated through their read sets (`store.stale_refs`); what
  moved comes first, then decisions on the matter whose reasons broke or whose revision
  condition fired since the last resume. Moved premises are re-pinned to the current head
  (following `supersedes`), gone ones dropped, so the same change is told once. Forgotten
  content never appears in a line.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import ledger, store
from .model import (
    AuthorTier,
    Decision,
    Intention,
    IntentionState,
    Label,
    OpKind,
    Proposal,
    ProposalOp,
    Provenance,
    ReadRef,
    ReadSrc,
    RecordKind,
    Status,
    Trigger,
    TriggerKind,
    TurnCtx,
    WorkState,
    WorkStateBody,
)

SHAPE_VERSION = 1
REVISIT = "revisit_decision"
_GONE = (Status.FORGOTTEN.value, Status.PURGED.value)
_LIVE = (Status.ACTIVE.value, Status.ARCHIVED.value)
_TEXT_MAX = 160


class WorkError(ValueError):
    """A work-memory write the caller must fix (unknown matter, bad reason member)."""


class StalePremises(WorkError):
    """The state was built on premises that moved since they were read: re-read, re-plan."""

    def __init__(self, refs: Sequence[ReadRef]) -> None:
        self.refs = list(refs)
        super().__init__("premises moved: " + ", ".join(f"{r.id}@{r.version}" for r in refs))


@dataclass(frozen=True)
class Reason:
    """One justification member: a record id or a raw chunk id (a raw root)."""

    member: str
    provenance: Provenance = Provenance.STATED


@dataclass
class Change:
    """One «С тех пор поменялось» line and the items it names (for use-now filtering)."""

    text: str
    item_ids: list[str] = field(default_factory=list)


@dataclass
class Resumed:
    state_id: str | None
    state: WorkStateBody
    changes: list[Change] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- shapes


def load_state(text: str | None) -> tuple[WorkStateBody, dict[str, Any]]:
    """(known body, keys this version does not know). Never raises on a newer or broken shape."""
    try:
        raw = json.loads(text or "{}")
    except ValueError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    known = set(WorkStateBody.model_fields)
    extra = {k: v for k, v in raw.items() if k not in known}
    body: dict[str, Any] = {}
    for k in known & set(raw):
        try:
            WorkStateBody.model_validate({k: raw[k]})
        except ValueError:
            continue  # one bad field does not cost the rest
        body[k] = raw[k]
    return WorkStateBody.model_validate(body), extra


def dump_state(state: WorkStateBody, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    out.update(state.model_dump(mode="json"))
    return out


def _short(text: str | None, n: int = _TEXT_MAX) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


# ---------------------------------------------------------------- reads


def _matter(conn: sqlite3.Connection, matter_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT e.id, e.etype, e.display_name, r.status FROM entities e"
        " JOIN records r ON r.id = e.id WHERE e.id = ?",
        (matter_id,),
    ).fetchone()


def _check_matter(conn: sqlite3.Connection, matter_id: str) -> None:
    m = _matter(conn, matter_id)
    if m is None:
        raise WorkError(f"unknown matter: {matter_id}")
    if m["etype"] != "matter":
        raise WorkError(f"not a matter: {matter_id} is a {m['etype']}")
    if m["status"] in _GONE:
        raise WorkError(f"matter forgotten: {matter_id}")


def state_row(conn: sqlite3.Connection, matter_id: str) -> sqlite3.Row | None:
    """The matter's live work state (newest first)."""
    return conn.execute(
        "SELECT r.id, r.version, r.labels, r.author_tier, r.recorded_at, w.state,"
        " w.last_resumed_at FROM work_states w JOIN records r ON r.id = w.id"
        " WHERE w.matter_id = ? AND r.status = 'active'"
        " ORDER BY r.recorded_at DESC, r.id DESC LIMIT 1",
        (matter_id,),
    ).fetchone()


def text_of(conn: sqlite3.Connection, item_id: str, src: ReadSrc = ReadSrc.RECORD) -> str | None:
    """English text of one item, or None when it is gone (forgotten text never comes back)."""
    if src is ReadSrc.RAW:
        r = conn.execute("SELECT text, status FROM raw_chunks WHERE id = ?", (item_id,)).fetchone()
        return _short(r["text"]) if r is not None and r["status"] == "active" else None
    if src is ReadSrc.DERIVED:
        return None
    r = conn.execute(
        "SELECT r.status, coalesce(f.text_en, d.chosen_text, l.text_en, le.rule_en,"
        " e.display_name, r.origin_quote) AS t FROM records r"
        " LEFT JOIN facts f ON f.id = r.id"
        " LEFT JOIN decisions d ON d.id = r.id LEFT JOIN loops l ON l.id = r.id"
        " LEFT JOIN lessons le ON le.id = r.id LEFT JOIN entities e ON e.id = r.id"
        " WHERE r.id = ?",
        (item_id,),
    ).fetchone()
    if r is None or r["status"] in _GONE or not r["t"]:
        return None
    return _short(r["t"])


def head(conn: sqlite3.Connection, record_id: str) -> str:
    """Follow `supersedes` (new → old) forward to the newest record."""
    seen = {record_id}
    cur = record_id
    while True:
        nxt = conn.execute(
            "SELECT e.from_id FROM edges e JOIN records r ON r.id = e.from_id"
            " WHERE e.to_id = ? AND e.etype = 'supersedes' ORDER BY r.recorded_at DESC LIMIT 1",
            (cur,),
        ).fetchone()
        if nxt is None or nxt[0] in seen:
            return cur
        cur = nxt[0]
        seen.add(cur)


def _status(conn: sqlite3.Connection, record_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, version, status, review FROM records WHERE id = ?", (record_id,)
    ).fetchone()


# ---------------------------------------------------------------- writes


def _commit(
    conn: sqlite3.Connection, ops: list[ProposalOp], tier: AuthorTier, model_id: str | None
):
    try:
        res = store.commit(
            conn, Proposal(ops=ops, author_tier=tier, model_id=model_id), ledger.apply_op
        )
    except store.OpRejected as e:
        raise WorkError(str(e)) from e
    if res.stale:
        raise StalePremises([r for refs in res.stale.values() for r in refs])
    if res.rejected:
        raise WorkError("; ".join(res.rejected.values()))
    return res


def _justifications(reasons: Sequence[Sequence[str | Reason]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for no, group in enumerate(reasons, start=1):
        for m in group:
            r = m if isinstance(m, Reason) else Reason(str(m))
            out.append({"group_no": no, "member": r.member, "provenance": r.provenance.value})
    return out


def _conditions(decision: Decision) -> list[dict[str, Any]]:
    out = []
    for c in decision.revision_conditions:
        text = str(c.get("text") or (c.get("trigger") or {}).get("semantic") or "").strip()
        trig = dict(c.get("trigger") or {})
        if not trig:  # «если …» with no slot to watch: only a model can judge it (§12)
            trig = {"kind": TriggerKind.SOFT_OPPORTUNITY.value, "semantic": text}
        if not text:
            raise WorkError("a revision condition needs its text")
        try:
            Trigger.model_validate(trig)  # a bad trigger fails before anything is written
        except ValueError as e:
            raise WorkError(f"bad revision trigger: {e}") from None
        out.append(
            {
                **c,
                "v": SHAPE_VERSION,
                "text": text,
                "trigger": trig,
                "intention_id": c.get("intention_id") or ledger.new_id(),
                "armed": False,
            }
        )
    return out


def _intention(decision: Decision, cond: dict[str, Any]) -> Intention:
    return Intention(
        id=cond["intention_id"],
        trigger=Trigger.model_validate(cond["trigger"]),
        action={"kind": REVISIT, "decision_id": decision.id, "matter_id": decision.matter_id},
        author_tier=decision.author_tier,
        author_entity_id=decision.author_entity_id,
        labels=decision.labels,
        recorded_at=decision.recorded_at,
        origin_quote=decision.origin_quote,
    )


def _arm(
    conn: sqlite3.Connection, intention: Intention, turn: TurnCtx | None, model_id: str | None
) -> bool:
    """Through `api.arm_intention` (intent: banned chats, windows, read deps); until that is
    available, a plain ledger ADD of the armed intention."""
    from . import api

    try:
        res = api.arm_intention(intention, turn=turn, store=conn)
    except NotImplementedError:
        op = ProposalOp(
            op=OpKind.ADD,
            target=None,
            group=intention.id,
            # the decision is its lineage: forgetting the decision takes its revisits along
            payload={
                "record": intention.model_dump(mode="json"),
                "derived_from": [intention.action["decision_id"]] if intention.action else [],
            },
        )
        _commit(conn, [op], intention.author_tier, model_id)
        return True
    return res.outcome == "committed"


def record_decision(
    conn: sqlite3.Connection,
    decision: Decision,
    *,
    reasons: Sequence[Sequence[str | Reason]] = (),
    derived_from: Sequence[str] = (),
    read_set: Sequence[ReadRef] = (),
    replaces: str | None = None,
    turn: TurnCtx | None = None,
    model_id: str | None = None,
) -> str:
    """§14: store a decision with its reasons as justification groups, rejected alternatives
    and revision conditions (each armed as an intention). `replaces`: the decision this one
    revises (it is superseded and its armed revision intentions are cancelled). Returns the
    decision id."""
    if decision.matter_id:
        _check_matter(conn, decision.matter_id)
    conds = _conditions(decision)
    rec = decision.model_copy(
        update={
            "revision_conditions": conds,
            "decided_at": decision.decided_at or decision.recorded_at,
        }
    )
    group = decision.matter_id or decision.id
    payload: dict[str, Any] = {"record": rec.model_dump(mode="json")}
    if reasons:
        payload["justifications"] = _justifications(reasons)
    if derived_from:
        payload["derived_from"] = list(derived_from)
    ops = [ProposalOp(op=OpKind.ADD, group=group, read_set=list(read_set), payload=payload)]
    if replaces:
        old = conn.execute("SELECT kind FROM records WHERE id = ?", (replaces,)).fetchone()
        if old is None or old["kind"] != RecordKind.DECISION.value:
            raise WorkError(f"not a decision: {replaces}")
        ops.append(
            ProposalOp(
                op=OpKind.SUPERSEDE, target=replaces, group=group, payload={"by": decision.id}
            )
        )
        # a worker's revision only flags the old one for review (§8.3): its conditions stay
        for (iid,) in (
            []
            if model_id is not None
            else conn.execute(
                "SELECT i.id FROM intentions i JOIN records r ON r.id = i.id"
                " WHERE i.state = 'armed' AND r.status = 'active'"
                " AND json_extract(i.action, '$.kind') = ? AND json_extract(i.action, '$.decision_id') = ?",
                (REVISIT, replaces),
            ).fetchall()
        ):
            ops.append(
                ProposalOp(
                    op=OpKind.UPDATE,
                    target=iid,
                    group=group,
                    payload={"fields": {"state": IntentionState.CANCELLED.value}},
                )
            )
    _commit(conn, ops, decision.author_tier, model_id)
    if not conds:
        return decision.id
    armed = []
    for c in conds:
        ok = _arm(conn, _intention(rec, c), turn, model_id)
        armed.append({**c, "armed": ok, "intention_id": c["intention_id"] if ok else None})
    upd = ProposalOp(
        op=OpKind.UPDATE,
        target=decision.id,
        group=group,
        payload={"fields": {"revision_conditions": armed}},
    )
    _commit(conn, [upd], decision.author_tier, model_id)
    return decision.id


def save_state(
    conn: sqlite3.Connection,
    matter_id: str,
    state: WorkStateBody,
    *,
    author_tier: AuthorTier = AuthorTier.SYSTEM,
    origin_quote: str | None = None,
    labels: Label = Label.NONE,
    model_id: str | None = None,
    record_id: str | None = None,
    derived_from: Sequence[str] = (),
) -> str:
    """§14: the matter's work state as JSON, one live record per matter, updated in place.
    Premises are pinned: one that already moved refuses the save (`StalePremises`). Labels
    come from the writer (`labels`, from its labeler), not from the premises: one health
    premise must not hide the whole trip plan on a turn about the trip; each premise keeps
    its own fence in the resume lines. A new state needs a lineage root (§3.2 F9): the
    user's words (`origin_quote`) or `derived_from` sources (the turn's raw).
    Premises are not lineage on purpose: forgetting a hotel fact must not forget the whole
    plan, only show up at resume. Returns the work-state id."""
    _check_matter(conn, matter_id)
    premises = list(dict.fromkeys(state.premises))
    state = state.model_copy(update={"premises": premises})
    for _attempt in range(3):
        cur = state_row(conn, matter_id)
        if cur is None:
            ws = WorkState(
                id=record_id or ledger.new_id(),
                matter_id=matter_id,
                state=state,
                author_tier=author_tier,
                labels=labels,
                recorded_at=store.now(),
                origin_quote=origin_quote,
            )
            op = ProposalOp(
                op=OpKind.ADD,
                group=matter_id,
                read_set=premises,
                payload={"record": ws.model_dump(mode="json"), "derived_from": list(derived_from)},
            )
            _commit(conn, [op], author_tier, model_id)
            return ws.id
        _, extra = load_state(cur["state"])
        fields: dict[str, Any] = {"state": dump_state(state, extra)}
        new_labels = Label(int(cur["labels"])) | labels
        if int(new_labels) != int(cur["labels"]):
            fields["labels"] = int(new_labels)
        op = ProposalOp(
            op=OpKind.UPDATE,
            target=cur["id"],
            group=matter_id,
            read_set=[ReadRef(id=cur["id"], version=cur["version"]), *premises],
            payload={"fields": fields},
        )
        try:
            _commit(conn, [op], author_tier, model_id)
        except StalePremises as e:
            if any(r.id != cur["id"] for r in e.refs):
                raise StalePremises([r for r in e.refs if r.id != cur["id"]]) from None
            continue  # another writer saved this state meanwhile: re-read, keep its new keys
        return cur["id"]
    raise WorkError(f"work state of {matter_id} keeps changing; try again")


# ---------------------------------------------------------------- resume


def _premise_change(conn: sqlite3.Connection, ref: ReadRef) -> tuple[Change | None, ReadRef | None]:
    """(line, the premise to keep). A moved record is re-pinned to its head."""
    if ref.src is not ReadSrc.RECORD:
        cur = conn.execute(
            "SELECT rev FROM raw_chunks WHERE id = ? AND status = 'active'"
            if ref.src is ReadSrc.RAW
            else "SELECT rev FROM derived WHERE id = ? AND valid = 1",
            (ref.id,),
        ).fetchone()
        if cur is None:
            return Change("a source this work rested on is gone; re-check the plan"), None
        return (
            Change(
                f'a source this work rested on changed: "{text_of(conn, ref.id, ref.src) or ref.id}"',
                [ref.id],
            ),
            ReadRef(id=ref.id, version=int(cur[0]), src=ref.src),
        )
    row = _status(conn, ref.id)
    if row is None or row["status"] in _GONE:
        return Change("a premise of this work was forgotten; re-check the plan"), None
    old_text = text_of(conn, ref.id)
    st = row["status"]
    if st == Status.SUPERSEDED.value:
        h = head(conn, ref.id)
        hr = _status(conn, h)
        if h != ref.id and hr is not None and hr["status"] in _LIVE:
            new_text = text_of(conn, h)
            return (
                Change(f'[m:{h}] "{old_text}" is now "{new_text}"', [ref.id, h]),
                ReadRef(id=h, version=int(hr["version"])),
            )
        return Change(f'[m:{ref.id}] "{old_text}" no longer holds', [ref.id]), None
    if st == Status.RETRACTED.value:
        return Change(f'[m:{ref.id}] "{old_text}" turned out untrue', [ref.id]), None
    if st == Status.NEEDS_REVIEW.value:
        return (
            Change(f'[m:{ref.id}] "{old_text}" is in doubt now', [ref.id]),
            ReadRef(id=ref.id, version=int(row["version"])),
        )
    return (
        Change(f'[m:{ref.id}] changed: now "{old_text}"', [ref.id]),
        ReadRef(id=ref.id, version=int(row["version"])),
    )


def _decision_view(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for j in conn.execute(
        "SELECT group_no, member_record_id, provenance FROM justifications"
        " WHERE target_record_id = ? ORDER BY group_no, member_record_id",
        (row["id"],),
    ):
        mid = j["member_record_id"]
        src = ReadSrc.RECORD if _status(conn, mid) is not None else ReadSrc.RAW
        groups.setdefault(int(j["group_no"]), []).append(
            {"id": mid, "text": text_of(conn, mid, src), "provenance": j["provenance"]}
        )
    return {
        "id": row["id"],
        "version": int(row["version"]),
        "status": row["status"],
        "review": bool(row["review"]),
        "domain": row["domain"],
        "chosen": row["chosen_text"],
        "decided_at": row["decided_at"],
        "reasons": [groups[k] for k in sorted(groups)],
        "rejected": json.loads(row["alternatives"] or "[]"),
        "revision_conditions": [
            {k: c.get(k) for k in ("text", "intention_id", "armed")}
            for c in json.loads(row["revision_conditions"] or "[]")
            if isinstance(c, dict)
        ],
    }


def _decisions(
    conn: sqlite3.Connection, matter_id: str, since: str | None
) -> tuple[list[Change], list[dict[str, Any]]]:
    rows = conn.execute(
        "SELECT r.id, r.version, r.status, r.review, r.updated_at, r.recorded_at, d.domain,"
        " d.chosen_text, d.alternatives, d.revision_conditions, d.decided_at"
        " FROM decisions d JOIN records r ON r.id = d.id WHERE d.matter_id = ?"
        " AND r.status IN ('active', 'needs_review') ORDER BY r.recorded_at, r.id",
        (matter_id,),
    ).fetchall()
    changes: list[Change] = []
    views = []
    for r in rows:
        views.append(_decision_view(conn, r))
        moved_since = since is None or (r["updated_at"] or r["recorded_at"]) > since
        if r["status"] == Status.NEEDS_REVIEW.value and moved_since:
            changes.append(
                Change(
                    f'[m:{r["id"]}] decision "{_short(r["chosen_text"])}" lost its reasons; revisit it',
                    [r["id"]],
                )
            )
        elif r["review"] and moved_since:
            changes.append(
                Change(
                    f'[m:{r["id"]}] decision "{_short(r["chosen_text"])}" rests on something that shifted; re-check it',
                    [r["id"]],
                )
            )
        for c in json.loads(r["revision_conditions"] or "[]"):
            iid = c.get("intention_id") if isinstance(c, dict) else None
            if not iid:
                continue
            ir = conn.execute(
                "SELECT state, fired_at FROM intentions WHERE id = ?", (iid,)
            ).fetchone()
            if ir is None or ir["state"] not in ("fired", "needs_attention"):
                continue
            if since is None or (ir["fired_at"] or "") > since:
                changes.append(
                    Change(
                        f'[m:{r["id"]}] revision condition met for "{_short(r["chosen_text"])}":'
                        f' "{_short(c.get("text"))}"',
                        [r["id"], iid],
                    )
                )
    return changes, views


def _repin(
    conn: sqlite3.Connection,
    state_id: str,
    version: int,
    state: WorkStateBody,
    kept: list[ReadRef],
    extra: dict[str, Any],
) -> WorkStateBody:
    """Code maintenance, not new content: the premises move to what holds now, as a trusted
    ledger change (a version bump, so read sets on the old state go stale). A save that
    landed meanwhile pinned its own premises and wins."""
    new = state.model_copy(update={"premises": kept})
    text = json.dumps(dump_state(new, extra), ensure_ascii=False, separators=(",", ":"))

    def fn(ctx: store.CommitCtx) -> tuple[bool, list[str]]:
        row = ctx.conn.execute("SELECT version FROM records WHERE id = ?", (state_id,)).fetchone()
        if row is None or int(row[0]) != version:
            return False, []
        ctx.conn.execute("UPDATE work_states SET state = ? WHERE id = ?", (text, state_id))
        ledger.bump(ctx, [state_id])
        ledger.index(ctx.conn, state_id)
        ledger.invalidate(ctx, [state_id])
        ledger.mark_pending_updates(ctx, [state_id])
        return True, [state_id]

    if ledger.run_in_commit(conn, fn, op_kind=OpKind.UPDATE, target=state_id):
        return new
    fresh = conn.execute("SELECT state FROM work_states WHERE id = ?", (state_id,)).fetchone()
    return load_state(fresh[0] if fresh else None)[0]


def resume_full(
    conn: sqlite3.Connection,
    matter_id: str,
    *,
    allow: Callable[[set[str]], set[str]] | None = None,
) -> Resumed:
    """Re-validate, re-pin, and return everything the facade needs (lines carry item ids).
    `allow` (use-now for the turn) picks the lines this turn may see; a withheld change is
    not consumed: its premise stays pinned and it is told on a turn where it applies."""
    m = _matter(conn, matter_id)
    if m is None or m["status"] in _GONE:
        return Resumed(None, WorkStateBody())
    cur = state_row(conn, matter_id)
    if cur is None:
        changes, decisions = _decisions(conn, matter_id, None)
        ok = allow({i for c in changes for i in c.item_ids}) if allow else None
        shown = [c for c in changes if ok is None or all(i in ok for i in c.item_ids)]
        return Resumed(None, WorkStateBody(), shown, decisions)
    state, extra = load_state(cur["state"])
    since = cur["last_resumed_at"] or cur["recorded_at"]
    stale = {
        (r.id, r.version, r.src)
        for r in store.stale_refs(conn, state.premises, require_usable=True)
    }
    moved = {
        ref: _premise_change(conn, ref)
        for ref in state.premises
        if (ref.id, ref.version, ref.src) in stale
    }
    dchanges, decisions = _decisions(conn, matter_id, since)
    named = {i for line, _ in moved.values() if line for i in line.item_ids}
    named |= {i for c in dchanges for i in c.item_ids}
    ok = allow(named) if allow else None

    def seen(c: Change) -> bool:
        return ok is None or all(i in ok for i in c.item_ids)

    changes: list[Change] = []
    kept: list[ReadRef] = []
    for ref in state.premises:
        if ref not in moved:
            kept.append(ref)
            continue
        line, keep = moved[ref]
        if line is not None and not seen(line):
            kept.append(ref)  # withheld now, told later
            continue
        if line is not None:
            changes.append(line)
        if keep is not None and keep not in kept:
            kept.append(keep)
    shown = [c for c in dchanges if seen(c)]
    changes.extend(shown)
    state_id = cur["id"]
    if kept != state.premises:
        state = _repin(conn, state_id, int(cur["version"]), state, kept, extra)
    if len(shown) == len(dchanges):  # a withheld decision line keeps its «since»
        with store.write_tx(conn):
            conn.execute(
                "UPDATE work_states SET last_resumed_at = ? WHERE id = ?", (store.now(), state_id)
            )
    return Resumed(state_id, state, changes, decisions, extra)


def resume(conn: sqlite3.Connection, matter_id: str) -> tuple[list[str], WorkStateBody]:
    """(changes since, state): premises re-validated through their read sets."""
    r = resume_full(conn, matter_id)
    return [c.text for c in r.changes], r.state


# ---------------------------------------------------------------- the facade's view (use-now)


def _allowed(conn: sqlite3.Connection, ids: set[str], turn: TurnCtx) -> set[str]:
    """§5.5 for every id a view names: live records by `decide`, retired ones (superseded,
    retracted, in review) by `decide_retired`, so an old value keeps its fence."""
    from . import policy

    if not ids:
        return set()
    order = sorted(ids)
    marks = ", ".join("?" for _ in order)
    status = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT id, status FROM records WHERE id IN ({marks})",  # noqa: S608
            order,
        )
    }
    live = [i for i in order if status.get(i) in _LIVE]
    verdicts = policy.decide_many(conn, live, turn) if live else {}
    ok = {i for i in live if verdicts[i].allowed}
    for i in order:
        if i in verdicts or status.get(i) in _GONE:
            continue
        if i in status:
            if policy.decide_retired(conn, i, turn).allowed:
                ok.add(i)
        elif policy.decide(conn, i, turn).allowed:  # a raw root
            ok.add(i)
    return ok


def view(
    conn: sqlite3.Connection, matter_id: str, turn: TurnCtx
) -> tuple[list[str], dict[str, Any]]:
    """`api.resume`: changes first, then the state with the matter's decisions, every item
    through use-now. A fenced work state comes back empty; a fenced line is left out."""
    r = resume_full(conn, matter_id, allow=lambda named: _allowed(conn, named, turn))
    ids = {d["id"] for d in r.decisions}
    ids |= {m["id"] for d in r.decisions for g in d["reasons"] for m in g}
    if r.state_id:
        ids.add(r.state_id)
    ok = _allowed(conn, ids, turn)
    changed = [c.text for c in r.changes]  # already through use-now (`allow`)
    decisions = []
    for d in r.decisions:
        if d["id"] not in ok:
            continue
        reasons = [[m if m["id"] in ok else {**m, "text": None} for m in g] for g in d["reasons"]]
        decisions.append({**d, "reasons": reasons})
    state: dict[str, Any] = {}
    if r.state_id and r.state_id in ok:
        state = {"id": r.state_id, "v": SHAPE_VERSION, **dump_state(r.state, r.extra)}
    if decisions:
        state["decisions"] = decisions
    return changed, state
