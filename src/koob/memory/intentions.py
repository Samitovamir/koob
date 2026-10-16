"""G7 intentions and loops: slot-bound triggers, lifecycle, exactly-once via the processed set
and outbox-first actions; bot outbox read sets.

Spec: §12, §5.6 (windows on no_dig chats), §16.2. Built on the existing duties / waits
machinery, through adapters.

How it holds together:
- **Arming** (`arm`) writes the intention record through the ledger's ADD (its checks
  apply), mirrors `action.read_set` into `read_deps` (holder `intention`) and opens an admit
  window on every `no_dig` chat the intention watches, all in one commit. A `banned` chat
  refuses it with one line (`policy.PolicyRefused`). Arming the same id twice is a no-op.
- **Candidates** (`candidates`) are chosen by code from the event's ids alone: the sender's
  entity, the slot of the changed record (following `supersedes` edges to the slot's current
  head, never trusting the payload's value), or the date. Only `armed` intentions on usable
  records are candidates. A `semantic` clause is the one thing a bare model judges
  (`set_semantic_judge`); with no judge wired the clause counts as met, so nothing the user
  asked for is silently dropped.
- **Firing** (`fire`) is one transaction: the `intention_events` row (UNIQUE), the state
  move, the version bump and, when a value in `action.read_set` changed, the action rebuilt
  from current values. A redelivered event, a second connection racing on the same event or a
  restart mid-fire all collide on the processed set or on the state, so the fire happens once.
  The fire's `origin_key = hash(intention_id, event_key)` is what the hands' outbox (§16.2)
  reserves the send under, so a re-run after a crash yields the same action key.
- **Lifecycle:** `armed` → `fired` → `done` | `cancelled` | `needs_attention`. An intention
  whose action is only telling the user (no hands tool) or whose completion says
  `{"on": "fired"}` is `done` at fire; one with a hands action stays `fired` until `settle`
  closes it (`ok` → done, `failed`/`unknown` → needs_attention). An action whose memory value
  is gone with no successor goes to `needs_attention`: an old value is never sent.
- **Loops** (open / done / overdue / cancelled / unclear): `overdue_loops` turns every open
  loop past its due into `overdue` once and returns those ids (surfaced once); `set_loop`
  closes, re-opens or marks one unclear (True only on the transition: asked once).
- **Bot outbox** (§12, koob.db `tgbot_outbox`): `stamp_outbox` mirrors a queued message's
  read set into `read_deps` (holder `outbox`), `outbox_stale` re-validates it at delivery
  (§7.4) and `compose` rebuilds an intention's text from current values for a held row.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from . import clock as _clock
from . import ledger, policy, store
from .model import (
    USABLE,
    AdmitWindow,
    ChatMode,
    Intention,
    IntentionState,
    LoopState,
    OpKind,
    Proposal,
    ProposalOp,
    ReadRef,
    ReadSrc,
    TriggerKind,
)

ARM_BANNED_LINE = "Этот чат под полным запретом: напоминание на него не ставлю."
_USABLE = tuple(s.value for s in USABLE)
_DONE_OUTCOMES = {"ok": IntentionState.DONE, "done": IntentionState.DONE}
_ATTENTION_OUTCOMES = frozenset({"failed", "unknown"})
_SLOT = re.compile(r"\{(\d+)\}")
_ASK_VERB = re.compile(r"^\s*(?:напомни(?:те)?|напоминай(?:те)?)\s*[:,]?\s*(?:мне\s+)?", re.I)

SemanticJudge = Callable[[dict[str, Any], str, dict[str, Any]], bool]
_judge: SemanticJudge | None = None


class _Collided(store.OpRejected):
    """The event was already processed for this intention, or it is no longer armed."""


def set_semantic_judge(fn: SemanticJudge | None) -> None:
    """§12: the one bare model call, wired by the runtime; `fn(intention, source, payload)`.
    The payload carries ids only: the judge reads what it needs through fenced tools."""
    global _judge
    _judge = fn


# ---------------------------------------------------------------- small helpers


def origin_key(intention_id: str, event_key: str) -> str:
    """§12/§16.2: the origin a fire's action is reserved under; the same on every re-run."""
    return hashlib.sha256(f"{intention_id}\x1f{event_key}".encode()).hexdigest()[:32]


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.strip()
    end_of_day = len(v) == 10  # a bare date: due at the end of that day
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    dt = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return dt + timedelta(days=1, microseconds=-1) if end_of_day else dt


def _now() -> datetime:
    now = _clock.now()
    return now if now.tzinfo else now.replace(tzinfo=UTC)


def _row(conn: sqlite3.Connection, intention_id: str) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT i.id, i.trigger, i.completion, i.action, i.state, i.fired_at, i.outcome,"
        " r.status, r.origin_quote, r.author_tier FROM intentions i"
        " JOIN records r ON r.id = i.id WHERE i.id = ?",
        (intention_id,),
    ).fetchone()
    if r is None:
        return None
    out = dict(r)
    for col in ("trigger", "completion", "action"):
        out[col] = json.loads(out[col]) if out[col] else None
    return out


def _refs(action: dict[str, Any] | None) -> list[ReadRef]:
    raw = (action or {}).get("read_set") or []
    return store.load_read_set(json.dumps(raw)) if raw else []


def _dump_refs(refs: Iterable[ReadRef]) -> list[dict[str, Any]]:
    return [{"id": r.id, "version": r.version, "src": r.src.value} for r in refs]


def _norm(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def _successor(conn: sqlite3.Connection, record_id: str) -> str | None:
    r = conn.execute(
        "SELECT from_id FROM edges WHERE to_id = ? AND etype = 'supersedes' LIMIT 1",
        (record_id,),
    ).fetchone()
    return r[0] if r else None


def _head(conn: sqlite3.Connection, record_id: str) -> tuple[str, list[str]]:
    """Follow `supersedes` forward to the slot's current record; returns (head, chain)."""
    chain, cur = [record_id], record_id
    while (nxt := _successor(conn, cur)) is not None and nxt not in chain:
        chain.append(nxt)
        cur = nxt
    return cur, chain


def _value(conn: sqlite3.Connection, record_id: str) -> str | None:
    """The current text of a usable memory value: a fact's value, an entity's name."""
    r = conn.execute(
        "SELECT r.status, f.value_text, f.value_norm, f.text_en, e.display_name FROM records r"
        " LEFT JOIN facts f ON f.id = r.id LEFT JOIN entities e ON e.id = r.id WHERE r.id = ?",
        (record_id,),
    ).fetchone()
    if r is None or r["status"] not in _USABLE:
        return None
    return r["value_text"] or r["value_norm"] or r["display_name"] or r["text_en"]


# ---------------------------------------------------------------- composing from current values


def compose(conn: sqlite3.Connection, action: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    """§12 (A7): the action rebuilt from current values. Every ref in `action.read_set` is
    followed to its slot's current head; `template` (`{0}`, `{1}`… in read-set order) is
    filled from those values into `text_en`. Returns (action, ok); ok is False when a value
    is gone with no successor - the old value is never put back."""
    act = dict(action or {})
    refs = _refs(act)
    if not refs:
        return act, True
    stale = {(r.id, r.version) for r in store.stale_refs(conn, refs, require_usable=True)}
    fresh: list[ReadRef] = []
    values: list[str] = []
    for ref in refs:
        rid = ref.id
        if ref.src is ReadSrc.RECORD and (ref.id, ref.version) in stale:
            rid, _ = _head(conn, ref.id)
        if ref.src is not ReadSrc.RECORD:
            if (ref.id, ref.version) in stale:
                return act, False
            fresh.append(ref)
            values.append("")
            continue
        val = _value(conn, rid)
        if val is None:
            return act, False
        ver = conn.execute("SELECT version FROM records WHERE id = ?", (rid,)).fetchone()[0]
        fresh.append(ReadRef(id=rid, version=int(ver)))
        values.append(val)
    act["read_set"] = _dump_refs(fresh)
    template = act.get("template")
    if isinstance(template, str):
        act["text_en"] = _SLOT.sub(
            lambda m: values[int(m.group(1))] if int(m.group(1)) < len(values) else m.group(0),
            template,
        )
    return act, True


def action_of(conn: sqlite3.Connection, intention_id: str) -> dict[str, Any] | None:
    """What the executor sends for a fired intention (composed at fire)."""
    row = _row(conn, intention_id)
    return row["action"] if row else None


# ---------------------------------------------------------------- arming


def _chats(conn: sqlite3.Connection, intention: Intention) -> list[tuple[str, str | None]]:
    """The chats an intention watches or writes to, with the sender it waits for."""
    out: dict[str, str | None] = {}
    act = intention.action or {}
    for key in ("chat_id", "stable_target"):
        val = act.get(key)
        if isinstance(val, str) and re.match(r"^[a-z]+:[^:\s]+$", val):
            out.setdefault(val, None)
    eid = intention.trigger.entity_id
    if eid:
        row = conn.execute("SELECT stable_ids FROM entities WHERE id = ?", (eid,)).fetchone()
        for sid in json.loads(row[0]) if row and row[0] else []:
            if isinstance(sid, dict) and sid.get("kind") == "tg_user_id" and sid.get("value"):
                chat = f"tg:{sid['value']}"
                out[chat] = chat
    return list(out.items())


def arm(conn: sqlite3.Connection, intention: Intention) -> str:
    """Refused on a banned chat (one line); opens a window on a no_dig chat (§5.6)."""
    if conn.execute("SELECT 1 FROM intentions WHERE id = ?", (intention.id,)).fetchone():
        return intention.id
    chats = _chats(conn, intention)
    for chat, _ in chats:
        if policy.chat_mode(conn, chat) is ChatMode.BANNED:
            raise policy.PolicyRefused(ARM_BANNED_LINE)
    rec = intention.model_copy(update={"state": IntentionState.ARMED})
    refs = _refs(rec.action)
    record = rec.model_dump(mode="json", exclude_none=True)
    op = ProposalOp(op=OpKind.ADD, group=f"intention:{rec.id}", payload={"record": record})

    def apply(ctx: store.CommitCtx, o: ProposalOp) -> list[str]:
        touched = list(ledger.apply_op(ctx, o))
        if refs:
            store.set_read_deps(ctx.conn, "intention", rec.id, refs)
        after = rec.trigger.after_ts or ctx.ts
        for chat, sender in chats:
            if policy.chat_mode(ctx.conn, chat) is ChatMode.NO_DIG:
                policy.open_window(
                    ctx.conn,
                    AdmitWindow(
                        id=f"win:{rec.id}:{chat}",
                        chat_id=chat,
                        sender_id=sender,
                        after_ts=after,
                        reason="intention",
                        intention_id=rec.id,
                    ),
                )
        return touched

    try:
        res = store.commit(conn, Proposal(ops=[op], author_tier=rec.author_tier), apply)
    except policy.PolicyRefused as exc:  # the chat went banned between the check and commit
        raise policy.PolicyRefused(ARM_BANNED_LINE) from exc
    if not res.ok:
        raise ValueError("Не смог поставить напоминание — попробуй ещё раз.")
    return rec.id


# ---------------------------------------------------------------- candidates


def _armed(conn: sqlite3.Connection, kind: TriggerKind, where: str = "", params: tuple = ()):
    return conn.execute(
        "SELECT i.id, i.trigger FROM intentions i JOIN records r ON r.id = i.id"
        f" WHERE i.state = 'armed' AND r.status IN ({', '.join('?' for _ in _USABLE)})"
        f" AND json_extract(i.trigger, '$.kind') = ?{where} ORDER BY i.id",  # noqa: S608
        (*_USABLE, kind.value, *params),
    ).fetchall()


def _sender_entity(conn: sqlite3.Connection, payload: dict[str, Any]) -> str | None:
    if payload.get("sender_entity_id"):
        return str(payload["sender_entity_id"])
    m = re.match(r"^tg:(\d+)$", str(payload.get("sender_id") or ""))
    if not m:
        return None
    row = conn.execute(
        "SELECT e.id FROM entities e JOIN records r ON r.id = e.id, json_each(e.stable_ids) j"
        " WHERE json_extract(j.value, '$.kind') = 'tg_user_id'"
        " AND json_extract(j.value, '$.value') = ? AND r.status IN ('active', 'archived')",
        (m.group(1),),
    ).fetchall()
    return row[0][0] if len(row) == 1 else None


def _judged(conn: sqlite3.Connection, iid: str, trig: dict, source: str, payload: dict) -> bool:
    if not trig.get("semantic") or _judge is None:
        return True
    row = _row(conn, iid)
    try:
        return bool(_judge(row or {"id": iid, "trigger": trig}, source, payload))
    except Exception:  # noqa: BLE001 - a judge failure keeps it armed for the next event
        return False


def _message_candidates(conn: sqlite3.Connection, payload: dict[str, Any]) -> list[tuple]:
    chat = payload.get("chat_id")
    if chat and policy.chat_mode(conn, str(chat)) is ChatMode.BANNED:
        return []
    eid = _sender_entity(conn, payload)
    if eid is None:
        return []
    at = _parse(payload.get("ts")) or _now()
    rows = _armed(
        conn, TriggerKind.MESSAGE_FROM, " AND json_extract(i.trigger, '$.entity_id') = ?", (eid,)
    )
    out = []
    for r in rows:
        trig = json.loads(r["trigger"])
        after = _parse(trig.get("after_ts"))
        if after is None or after < at:
            out.append((r["id"], trig))
    return out


def _record_candidates(conn: sqlite3.Connection, payload: dict[str, Any]) -> list[tuple]:
    rid = payload.get("record_id")
    if not rid:
        return []
    head, chain = _head(conn, str(rid))
    cur = conn.execute(
        "SELECT r.status, f.subject_id, f.relation, f.value_norm, f.value_text FROM records r"
        " JOIN facts f ON f.id = r.id WHERE r.id = ?",
        (head,),
    ).fetchone()
    if cur is None or cur["status"] not in _USABLE:
        return []
    marks = ", ".join("?" for _ in chain)
    slots = {
        (s, rel)
        for s, rel in conn.execute(
            f"SELECT subject_id, relation FROM facts WHERE id IN ({marks})",  # noqa: S608
            chain,
        )
    }
    value = {_norm(cur["value_norm"]), _norm(cur["value_text"])} - {""}
    out = []
    for r in _armed(conn, TriggerKind.STATUS_CHANGE):
        trig = json.loads(r["trigger"])
        if (trig.get("subject_id"), trig.get("relation")) not in slots:
            continue
        if trig.get("to") is None or _norm(trig["to"]) in value:
            out.append((r["id"], trig))
    return out


def _date_candidates(conn: sqlite3.Connection, payload: dict[str, Any]) -> list[tuple]:
    at = _parse(payload.get("ts"))
    if at is None:
        key = str(payload.get("event_key") or "")
        at = _parse(key.removeprefix("date:")) if key.startswith("date:") else _now()
    out = []
    for r in _armed(conn, TriggerKind.DATE_REACHED):
        trig = json.loads(r["trigger"])
        due = _parse(trig.get("ts"))
        if due is not None and at is not None and due <= at:
            out.append((r["id"], trig))
    return out


def candidates(conn: sqlite3.Connection, source: str, payload: dict[str, Any]) -> list[str]:
    """Code selects candidate intentions; a bare model judges only `semantic` clauses."""
    if source in ("clock", "date"):
        found = _date_candidates(conn, payload)
    elif source == "record":
        found = _record_candidates(conn, payload)
    else:  # a messenger, by its name (§12)
        found = _message_candidates(conn, payload)
    return [iid for iid, trig in found if _judged(conn, iid, trig, source, payload)]


# ---------------------------------------------------------------- firing and lifecycle


def _hands(action: dict[str, Any] | None) -> bool:
    return bool((action or {}).get("tool"))


def _move(ctx: store.CommitCtx, iid: str, state: IntentionState, **cols: Any) -> None:
    sets = ["state = ?", *(f"{c} = ?" for c in cols)]
    ctx.conn.execute(
        f"UPDATE intentions SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
        (state.value, *cols.values(), iid),
    )
    ledger.bump(ctx, [iid])
    ledger.invalidate(ctx, [iid])
    if state in (IntentionState.DONE, IntentionState.CANCELLED):
        policy.close_windows(ctx.conn, intention_id=iid, ts=ctx.ts)


def _code_commit(conn: sqlite3.Connection, iid: str, fn: Callable[[store.CommitCtx], None]) -> bool:
    op = ProposalOp(op=OpKind.UPDATE, target=iid, group=f"intention:{iid}")

    def apply(ctx: store.CommitCtx, _op: ProposalOp) -> list[str]:
        fn(ctx)
        return [iid]

    try:
        res = store.commit(conn, Proposal(ops=[op]), apply)
    except _Collided:
        return False
    return res.ok


def fire(conn: sqlite3.Connection, intention_id: str, event_key: str) -> bool:
    """False when `intention_events` already holds (intention_id, event_key)."""

    def apply(ctx: store.CommitCtx) -> None:
        row = ctx.conn.execute(
            "SELECT i.state, i.action, i.completion, r.status FROM intentions i"
            " JOIN records r ON r.id = i.id WHERE i.id = ?",
            (intention_id,),
        ).fetchone()
        if row is None or row["state"] != IntentionState.ARMED or row["status"] not in _USABLE:
            raise _Collided("not armed")
        cur = ctx.conn.execute(
            "INSERT OR IGNORE INTO intention_events (intention_id, event_key, created_at)"
            " VALUES (?, ?, ?)",
            (intention_id, event_key, ctx.ts),
        )
        if cur.rowcount != 1:
            raise _Collided("event already processed")
        action = json.loads(row["action"]) if row["action"] else None
        completion = json.loads(row["completion"]) if row["completion"] else {}
        action, ok = compose(ctx.conn, action)
        if not ok:  # the composed text held the old value: it leaves the row, never sent
            state, outcome = IntentionState.NEEDS_ATTENTION, "value_gone"
            action = {k: v for k, v in (action or {}).items() if k != "text_en"} | {"held": True}
        elif completion.get("on") == "fired" or not _hands(action):
            state, outcome = IntentionState.DONE, "fired"
        else:
            state, outcome = IntentionState.FIRED, None
        if ok and action is not None:
            store.set_read_deps(ctx.conn, "intention", intention_id, _refs(action))
        _move(
            ctx,
            intention_id,
            state,
            fired_at=ctx.ts,
            outcome=outcome,
            action=json.dumps(action, ensure_ascii=False) if action is not None else None,
        )

    return _code_commit(conn, intention_id, apply)


def settle(conn: sqlite3.Connection, intention_id: str, outcome: str) -> bool:
    """§12: a fired intention's action closed (§16.2): `ok` → done; `failed` / `unknown` →
    needs_attention (an unknown outcome is checked, never retried blind)."""
    if outcome in _DONE_OUTCOMES:
        target = _DONE_OUTCOMES[outcome]
    elif outcome in _ATTENTION_OUTCOMES:
        target = IntentionState.NEEDS_ATTENTION
    else:
        raise ValueError(f"unknown outcome {outcome!r}")

    def apply(ctx: store.CommitCtx) -> None:
        r = ctx.conn.execute(
            "SELECT state FROM intentions WHERE id = ?", (intention_id,)
        ).fetchone()
        if r is None or r[0] not in (IntentionState.FIRED, IntentionState.NEEDS_ATTENTION):
            raise _Collided("not fired")
        _move(ctx, intention_id, target, outcome=outcome)

    return _code_commit(conn, intention_id, apply)


def cancel(conn: sqlite3.Connection, intention_id: str) -> bool:
    """«отмени напоминание»: armed or waiting → cancelled; its windows close."""

    def apply(ctx: store.CommitCtx) -> None:
        r = ctx.conn.execute(
            "SELECT state FROM intentions WHERE id = ?", (intention_id,)
        ).fetchone()
        if r is None or r[0] in (IntentionState.DONE, IntentionState.CANCELLED):
            raise _Collided("already closed")
        _move(ctx, intention_id, IntentionState.CANCELLED, outcome="cancelled")

    return _code_commit(conn, intention_id, apply)


def notice(conn: sqlite3.Connection, intention_id: str) -> str | None:
    """The one plain line the user gets when an intention fires; None when the fire only
    resumes a matter quietly (`action.resume` without `notify`, §12)."""
    row = _row(conn, intention_id)
    if row is None:
        return None
    act = row["action"] or {}
    if act.get("resume") and not act.get("notify"):
        return None
    words = row["origin_quote"] or act.get("text") or act.get("text_en") or ""
    words = _ASK_VERB.sub("", " ".join(str(words).split())).strip()
    if len(words) > 200:
        words = words[:199].rstrip() + "…"
    if row["state"] == IntentionState.NEEDS_ATTENTION:
        return f"Напоминание: {words} — нужное значение больше не действует, проверь сам."
    return f"Напоминание: {words}" if words else "Напоминание сработало."


# ---------------------------------------------------------------- loops


def _loop_commit(conn: sqlite3.Connection, ids: list[str], state: LoopState) -> list[str]:
    if not ids:
        return []
    ops = [ProposalOp(op=OpKind.UPDATE, target=i, group="loops") for i in ids]
    done: list[str] = []

    def apply(ctx: store.CommitCtx, op: ProposalOp) -> list[str]:
        n = ctx.conn.execute(
            "UPDATE loops SET state = ? WHERE id = ? AND state <> ?",
            (state.value, op.target, state.value),
        ).rowcount
        if n:
            ledger.bump(ctx, [op.target])
            ledger.invalidate(ctx, [op.target])
            done.append(op.target)
        return [op.target]

    store.commit(conn, Proposal(ops=ops), apply)
    return done


def overdue_loops(conn: sqlite3.Connection, now: str) -> list[str]:
    """§12: every `open` loop past its due becomes `overdue`; returns the newly overdue ids,
    so each is surfaced once. A bare-date due is due at the end of that day."""
    at = _parse(now) or _now()
    rows = conn.execute(
        "SELECT l.id, l.due FROM loops l JOIN records r ON r.id = l.id"
        " WHERE l.state = 'open' AND l.due IS NOT NULL"
        f" AND r.status IN ({', '.join('?' for _ in _USABLE)}) ORDER BY l.due, l.id",  # noqa: S608
        _USABLE,
    ).fetchall()
    due = [r["id"] for r in rows if (d := _parse(r["due"])) is not None and d < at]
    return _loop_commit(conn, due, LoopState.OVERDUE)


def set_loop(conn: sqlite3.Connection, loop_id: str, state: LoopState) -> bool:
    """Close (done / cancelled), mark unclear, or re-open one loop. True only on a real
    transition: the caller asks about an unclear loop (`loop_line`) on that True, so it is
    asked once, like an overdue one is surfaced once."""
    return bool(_loop_commit(conn, [loop_id], LoopState(state)))


def loop_line(conn: sqlite3.Connection, loop_id: str) -> str | None:
    r = conn.execute(
        "SELECT l.state, l.text_en, r.origin_quote FROM loops l JOIN records r ON r.id = l.id"
        " WHERE l.id = ?",
        (loop_id,),
    ).fetchone()
    if r is None:
        return None
    words = " ".join((r["origin_quote"] or r["text_en"] or "").split())
    head = {"overdue": "Просрочено", "unclear": "Неясно, сделано ли"}.get(r["state"], "Дело")
    return f"{head}: {words}"


# ---------------------------------------------------------------- bot outbox read sets (§12)


def stamp_outbox(conn: sqlite3.Connection, outbox_id: str, refs: Iterable[ReadRef]) -> str:
    """Mirror a queued bot message's read set into `read_deps` (holder `outbox`); returns the
    JSON for the outbox row's `read_set` column."""
    refs = list(refs)
    with store.write_tx(conn):
        store.set_read_deps(conn, "outbox", str(outbox_id), refs)
    return store.dump_read_set(refs)


def outbox_stale(conn: sqlite3.Connection, read_set: str | Iterable[ReadRef]) -> list[ReadRef]:
    """§7.4 at delivery: the refs that changed or left use. Non-empty → hold the row and
    re-compose it; never send it."""
    refs = store.load_read_set(read_set) if isinstance(read_set, str) else list(read_set)
    return store.stale_refs(conn, refs, require_usable=True) if refs else []


def release_outbox(conn: sqlite3.Connection, outbox_id: str) -> None:
    """The row went or was dropped: its read set leaves the reverse index."""
    with store.write_tx(conn):
        store.set_read_deps(conn, "outbox", str(outbox_id), [])
