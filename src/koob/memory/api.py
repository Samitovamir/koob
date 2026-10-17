"""The memory facade: the only entry the runtime, the adapters, the MCP tools and `cb memory`
use (memory spec §16.4, G13). Everything else in `memory/` is internal.

Every function names its spec section. Bodies raise `NotImplementedError` until they are
implemented; the engine stays off (`memory.engine: legacy`) meanwhile.

Calling conventions:
- `store` is an open memory.db connection (`store.open`); None means the default file
  `data/memory/memory.db` (opened by the store module).
- `turn` is the adapters' view of the current turn (ids, flags, the user's own words).
  Tainted-turn rules (§6.1) and "use now" (§5.5) read it; nothing reads model text as the
  user's words.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections import deque
from contextvars import ContextVar
from dataclasses import asdict
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .clock import Clock
from .model import (
    ActionState,
    Applicability,
    Assertion,
    AuthorTier,
    ChatMode,
    ForgetScope,
    Intention,
    Label,
    ReadRef,
    ReadSrc,
    RecordKind,
    Status,
    TurnCtx,
)

Conn = sqlite3.Connection


def _conn(store: Conn | None) -> Conn:
    """`store=None` opens the default memory.db (`$KOOB_DATA_DIR/memory`)."""
    if store is not None:
        return store
    from . import store as _store

    return _store.open(_store.MemoryPaths.default().db)


# ---------------------------------------------------------------- result shapes


class _R(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Item(_R):
    """One memory item as the brain sees it: `[m:<id>]`, text, date, tier (§10.2 step 6)."""

    id: str
    kind: RecordKind | Literal["raw", "derived"]
    version: int
    text: str
    status: Status = Status.ACTIVE
    author_tier: AuthorTier | None = None
    said_by: str | None = None  # who said it, shown as "according to <name>"
    date: str | None = None
    labels: Label = Label.NONE
    impression: bool = False  # shown as «впечатление» (§4.3)
    review: bool = False
    ask: list[str] | None = None  # C3 tie: distinguishing features (§15.3)


class WriteResult(_R):
    """Hot-path write outcome (§6.1). `ask_first`: a tainted turn without the user's naming words."""

    outcome: Literal["committed", "ask_first", "rejected"]
    record_ids: list[str] = Field(default_factory=list)
    memory_version: int | None = None
    dependents_to_review: int = 0
    line: str | None = None  # the one line the Cube may say («Запомнил», «Исправил: …»)


class ForgetResult(_R):
    """§9: «перестал использовать» at stop-use, «удалил полностью» when purge is verified."""

    outcome: Literal["stopped", "undone", "purged", "ask_first", "not_found"]
    forget_id: str | None = None
    record_ids: list[str] = Field(default_factory=list)
    turn_ids: list[str] = Field(default_factory=list)
    sessions_to_restart: list[str] = Field(default_factory=list)
    line: str | None = None


class PurgeStatus(_R):
    forget_id: str
    state: Literal["in_trash", "purging", "purged", "undone"]
    scope_done: list[str] = Field(default_factory=list)
    retry_at: str | None = None


class Lineage(_R):
    """«откуда это» (§17): quote, chat, date, who said it, in plain words."""

    item_id: str
    lines: list[str]
    sources: list[ReadRef] = Field(default_factory=list)


class WhatAnswer(_R):
    """«что ты помнишь про X» (§17), by kind, with dates and sources."""

    about: str
    items: list[Item] = Field(default_factory=list)
    competing_claims: list[Item] = Field(default_factory=list)


class Candidate(_R):
    """§16.1: hands act only on stable ids; the token pins the version."""

    entity_id: str
    display_name: str
    stable_ids: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float
    token: tuple[str, int]
    unconfirmed: bool = False  # alias from a lower tier: ask before hands use it (§13.3)


class ActionTicket(_R):
    """§16.2 reservation. `allow=False` carries the reason the PreToolUse hook denies with."""

    allow: bool
    action_key: str | None = None
    reason: str | None = None
    ask: str | None = None  # "ask which": the one distinguishing question


class HandVerdict(_R):
    """§16.3 pre-action check over tool arguments."""

    allow: bool
    reason: str | None = None
    current_value: str | None = None  # offered when a retired value was hit


class CoreView(_R):
    """§10.1: ~6k tokens, cache-stable section order, each with its read set."""

    text: str
    sections: dict[str, list[ReadRef]] = Field(default_factory=dict)
    tokens: int = 0


class Basket(_R):
    """§10.2: the per-turn recall block; empty when nothing passes the gate."""

    block: str | None
    items: list[Item] = Field(default_factory=list)
    read_set: list[ReadRef] = Field(default_factory=list)
    tokens: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


class UpdateBlock(_R):
    """§7.6 `<memory-update>`, ≤ 200 tokens; `restart` when it would overflow."""

    block: str | None
    item_ids: list[str] = Field(default_factory=list)
    restart: bool = False


class PreSend(_R):
    """§7.4: what to drop or rebuild right before a model call."""

    ok: bool
    stale: list[ReadRef] = Field(default_factory=list)
    undelivered_updates: int = 0


class AdmitDecision(_R):
    """§5.1: admit or reject with a content-free reason."""

    admit: bool
    reason: str | None = None
    author_tier: AuthorTier | None = None
    keep_only_own_words: bool = False
    # v0.4 §5.1: the only text arrival may store (the user's own words only in a turn that read
    # a barred chat; never-store spans replaced). None when rejected.
    text: str | None = None
    window_id: str | None = None  # the admit window that let it in (§5.6)
    intention_id: str | None = None  # an intention window scopes it to that matter only


class ModeChange(_R):
    chat_id: str
    mode: ChatMode
    policy_version: int
    forget_id: str | None = None  # the purge a ban or «не копать» started (§5.2)


class Changes(_R):
    """§7.7 cross-process watermark read."""

    memory_version: int
    last_event_id: int
    forget_events: list[int] = Field(default_factory=list)


class DoctorReport(_R):
    """§18: one plain line for the user only when memory is unusable."""

    ok: bool
    problems: list[str] = Field(default_factory=list)
    owner_line: str | None = None


class SafetyNet(_R):
    """§6.1 after-turn detector result."""

    fired: bool
    frozen_entities: list[str] = Field(default_factory=list)
    job_id: str | None = None


class Fire(_R):
    """§12: one intention firing, with its exactly-once key."""

    intention_id: str
    event_key: str
    origin_key: str


class ResumeView(_R):
    """§14: «С тех пор поменялось: …» first, then the work state."""

    matter_id: str
    changed: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


class Offer(_R):
    """§15.1: at most one per turn, with grounds."""

    text: str
    grounds: list[ReadRef]
    score: float


class ImportReport(_R):
    """§20 rollout step 2: counts only on a dry run."""

    dry_run: bool
    counts: dict[str, int] = Field(default_factory=dict)


class StartupReport(_R):
    """v0.4 §18.1: what `startup` repaired before the host serves a turn. Counts only."""

    journal_healed: int = 0  # forgets finished or journaled from a write-ahead line (§9.3)
    jobs_requeued: int = 0  # `running` jobs whose lease had ended
    jobs_needing_attention: int = 0  # requeues past the attempt cap
    actions_unknown: int = 0  # §16.2 step 4
    mode_changes_finished: int = 0  # interrupted «не копать» / ban forgets (§5.2)
    vectors_swept: int = 0


# ---------------------------------------------------------------- inspection and store tools (§10.4, §17)


def search(
    query: str,
    *,
    turn: TurnCtx,
    scope: str | None = None,
    kinds: list[RecordKind] | None = None,
    as_of: str | None = None,
    include_archived: bool = False,
    store: Conn | None = None,
) -> list[Item]:
    """MCP `search` (§10.4): ranked items with handles, records and raw messages, found by
    words and by meaning and fused; every item passes the basket's truth checks and use-now
    (§5.5). On a send turn sensitive items stay visible, marked «не отправлять без его слова»
    (do not send without the user's word). Writes `served` rows. `scope` is not used yet."""
    from . import commands

    return commands.search(
        _conn(store), query, turn=turn, kinds=kinds, as_of=as_of, include_archived=include_archived
    )


def get(item_id: str, *, turn: TurnCtx, store: Conn | None = None) -> Item | None:
    """MCP `get` (§10.4): the record with sources and links. Writes a `served` row."""
    from . import commands

    return commands.get_item(_conn(store), item_id, turn)


def where_from(item_id: str, *, turn: TurnCtx, store: Conn | None = None) -> Lineage:
    """«откуда это» (§17), MCP `where_from` (§10.4): lineage in plain words."""
    from . import commands

    return commands.lineage(_conn(store), item_id, turn)


def what(about: str, *, turn: TurnCtx, store: Conn | None = None) -> WhatAnswer:
    """«что ты помнишь про X» (§17): usable records by kind, impressions marked, competing
    claims attributed, sensitive items per §5.5. `[T inspect/test_commands]`"""
    from . import commands

    return commands.what_answer(_conn(store), about, turn)


# ---------------------------------------------------------------- hot path (§6.1)


def remember(
    text: str,
    *,
    turn: TurnCtx,
    because_quote: str,
    kind: RecordKind | None = None,
    applicability: Applicability | None = None,
    assertion: Assertion | None = None,
    store: Conn | None = None,
) -> WriteResult:
    """«запомни» (§6.1, §10.4): commits before the next answer, p95 ≤ 50 ms. In a tainted
    turn it writes a `tool`-tier claim unless the user's own words name the content;
    `because_quote` must pass §5.5. `assertion` (v0.4) is the structured part the brain
    fills; without it the record is `relation='other'` about the user (§6.1)."""
    from .writers import hot

    w = hot.remember(
        _conn(store),
        text,
        turn,
        because_quote=because_quote,
        kind=kind,
        applicability=applicability,
        assertion=assertion,
    )
    res = WriteResult(**asdict(w))
    _note_write(store, "memory_remember", None, res.outcome, res.line)
    return res


def fix(
    target: str,
    correction: str,
    *,
    turn: TurnCtx,
    because_quote: str,
    assertion: Assertion | None = None,
    retract: bool = False,
    store: Conn | None = None,
) -> WriteResult:
    """«исправь» (§6.1, §10.4, §17): retract or supersede, §8 outcomes for dependents,
    invalidate derived (§7.3), pending updates for served threads (§7.6), rebuild affected
    core sections. Tainted turn: their words must name the target, else `ask_first`.
    `target` follows the §10.4 grammar; a superseded target is retargeted to the slot's
    current head (§6.1, v0.4). `retract=True`: «это неправда», no new value."""
    from .writers import hot

    w = hot.fix(
        _conn(store),
        target,
        correction,
        turn,
        because_quote=because_quote,
        assertion=assertion,
        retract=retract,
    )
    if w.outcome != "ask_first":
        hot.mark_corrected_raws(_conn(store), turn, correction or because_quote)
    res = WriteResult(**asdict(w))
    _note_write(store, "memory_fix", target, res.outcome, res.line)
    return res


correct = fix  # alias


def after_turn_check(turn: TurnCtx, reply_text: str, *, store: Conn | None = None) -> SafetyNet:
    """§6.1 safety net: correction / forget / promise phrasing with no matching commit →
    freeze cue entities (≤ 60 s), urgent job, one failure line on failure."""
    from .writers import hot

    fired, ents, job_id = hot.after_turn_check(_conn(store), turn, reply_text)
    return SafetyNet(fired=fired, frozen_entities=ents, job_id=job_id)


def claims_unbacked_change(turn: TurnCtx, reply_text: str, *, store: Conn | None = None) -> bool:
    """§6.1 step 4: the `Stop` hook blocks once a user-turn reply that claims a memory
    change («запомнил», «исправил», «забыл», "noted") with no commit in the turn."""
    from .writers import hot

    blocked = hot.claims_unbacked_change(_conn(store), turn, reply_text)
    if blocked:
        _record(
            store, "denials", {"tool": "stop", "target": None, "reason": "claim_without_commit"}
        )
    return blocked


# ---------------------------------------------------------------- forgetting (§9)


def forget(
    target: str,
    *,
    turn: TurnCtx,
    because_quote: str,
    scope: ForgetScope = ForgetScope.ITEM,
    forever: bool = False,
    store: Conn | None = None,
) -> ForgetResult:
    """«забудь» (§9.1): stop-use at once (ledger, conversation rows, sessions, caches,
    journal), then trash, then a verified purge (24 h, at once for sensitive labels or
    «насовсем»). Tainted turn: their words must name the target.

    `target`: handles `m:<id>` (§10.4) or their words, resolved through the ledger (names via
    `entities.resolve_rows`, «всё про X» → `about_entity`, other words lexically); several
    plausible targets → `ask_first` with one distinguishing question; nothing → `not_found`."""
    from . import commands

    res = commands.forget_now(
        _conn(store), target, turn=turn, because_quote=because_quote, scope=scope, forever=forever
    )
    _note_write(store, "memory_forget", target, res.outcome, res.line)
    return res


def undo_forget(forget_id: str, *, turn: TurnCtx, store: Conn | None = None) -> ForgetResult:
    """«Вернуть» while in the trash (§9.1 stage 2): restores ledger, raw and conversation
    rows; journal `stage:"undone"`."""
    from . import commands

    return commands.undo(_conn(store), forget_id, turn=turn)


def purge_status(forget_id: str, *, store: Conn | None = None) -> PurgeStatus:
    """«удаляю» until §9.1 stage 3's checks pass, then «удалил полностью». An unknown id
    raises `forget.ForgetNotFound` (a LookupError)."""
    from . import commands

    return commands.purge_status(_conn(store), forget_id)


def run_purges(*, store: Conn | None = None) -> list[PurgeStatus]:
    """Walk due trash entries through the §9.2 scope; retry with backoff."""
    from . import commands

    return commands.run_purges(_conn(store))


# ---------------------------------------------------------------- policy (§5)


def admit(
    source_kind: str,
    *,
    chat_id: str | None,
    text: str,
    author_tier: AuthorTier,
    turn: TurnCtx | None = None,
    sender_id: str | None = None,
    reply_to: str | None = None,
    ts: str | None = None,
    store: Conn | None = None,
) -> AdmitDecision:
    """`policy.admit` (§5.1): before any write; a turn is admitted on the union of its
    `turn_sources`; never-store spans; admit windows for «не копать» (§5.6)."""
    from . import policy

    store = _conn(store)  # None: the default memory.db under $KOOB_DATA_DIR/memory
    a = policy.admit_ex(
        store,
        source_kind=source_kind,
        chat_id=chat_id,
        text=text,
        author_tier=author_tier,
        turn=turn,
        sender_id=sender_id,
        reply_to=reply_to,
        ts=ts,
    )
    return AdmitDecision(
        admit=a.admit,
        reason=a.reason,
        author_tier=author_tier if a.admit else None,
        keep_only_own_words=a.keep_only_own_words,
        text=a.text if a.admit else None,
        window_id=a.window_id,
        intention_id=a.intention_id,
    )


def record_turn_source(
    turn_id: str,
    source_kind: str,
    chat_id: str,
    *,
    requested: bool,
    store: Conn | None = None,
) -> None:
    """The hooks adapter's `turn_sources` row, from tool input and platform metadata only
    (§5.1)."""
    from . import policy

    store = _conn(store)  # None: the default memory.db under $KOOB_DATA_DIR/memory
    policy.record_turn_source(store, turn_id, source_kind, chat_id, requested=requested)


def set_chat_mode(
    chat_id: str, mode: ChatMode, *, turn: TurnCtx, store: Conn | None = None
) -> ModeChange:
    """§5.2: bumps `policy_version`, writes an `events` row; to `banned` runs the full
    forget pipeline on the chat's memory side; to `no_dig` forgets background-derived records.

    v0.4: the mode row, its `events` row and a `mode_forget` job commit in one transaction
    (`policy.set_mode`); stop-use then runs from that job in this call. A crash in between
    leaves the job for `startup`. `island` never takes a mode; in a tainted or non-owner
    turn the user's own words must name the chat and the mode."""
    from . import policy

    store = _conn(store)  # None: the default memory.db under $KOOB_DATA_DIR/memory
    if turn.speaker:
        # on speakerphone a mode change waits for the user's own channel
        raise policy.PolicyRefused("Подтверди смену режима чата у себя, не по громкой связи.")
    if chat_id.strip().casefold() == policy.ISLAND_CHAT:
        raise policy.PolicyRefused(policy.ISLAND_MODE_LINE)
    if not policy.mode_ordered(store, turn, chat_id, mode):
        raise policy.PolicyRefused(policy.MODE_ORDER_LINE)
    version = policy.set_mode(store, chat_id, mode)
    forget_ids: list[str] = []
    for (job_id,) in store.execute(
        "SELECT id FROM jobs WHERE jkind = ? AND json_extract(payload, '$.chat_id') = ?"
        " AND json_extract(payload, '$.policy_version') = ?",
        (policy.MODE_FORGET_JOB, chat_id, version),
    ).fetchall():
        forget_ids += policy.run_mode_forget(store, job_id) or []
    from .writers import hot

    hot.note_commit(store, turn, "mode", forget_ids or [f"policy-v{version}"])
    return ModeChange(
        chat_id=chat_id,
        mode=mode,
        policy_version=version,
        forget_id=forget_ids[0] if forget_ids else None,
    )


# ---------------------------------------------------------------- recall and context (§10, §7.6)


def core(*, thread_id: str, session_id: str | None = None, store: Conn | None = None) -> CoreView:
    """§10.1 core view for a process's first question; built through use-now with a
    neutral turn. Writes `served` rows."""
    from .recall import core_view

    conn = _conn(store)
    text, sections = core_view.view(conn)
    core_view.serve(
        conn,
        [r for s in sections for r in s.read_set],
        thread_id=thread_id,
        session_id=session_id,
    )
    return CoreView(
        text=text,
        sections={s.name: s.read_set for s in sections if s.lines},
        tokens=core_view.tokens(text),
    )


def core_diff(*, thread_id: str, session_id: str, store: Conn | None = None) -> UpdateBlock:
    """§10.1 resumed session with a changed core: changed lines as an update block if
    ≤ 1k tokens, else `restart`."""
    from . import store as _store
    from .recall import core_view

    conn = _conn(store)
    _text, sections = core_view.view(conn)
    served = core_view.served_core(conn, thread_id=thread_id, session_id=session_id)
    lines, held = core_view.diff_lines(conn, served, sections)
    if not lines:
        return UpdateBlock(block=None)
    block = core_view.update_block(lines)
    if core_view.tokens(block) > core_view.DIFF_TOKENS:
        return UpdateBlock(block=None, restart=True)
    ids = [ln.split("]", 1)[0].removeprefix("[m:") for ln in lines]
    # delivered: the session now holds these versions (§7.6 "served rows move")
    core_view.serve(conn, held, thread_id=thread_id, session_id=session_id)
    with _store.write_tx(conn):
        for ref in held:
            conn.execute(
                "DELETE FROM pending_updates WHERE thread_id = ? AND item_id = ?"
                " AND new_version <= ?",
                (thread_id, ref.id, ref.version),
            )
    return UpdateBlock(block=block, item_ids=list(dict.fromkeys(ids)))


def recall(turn: TurnCtx, *, store: Conn | None = None) -> Basket:
    """§10.2 basket: code only, no model and no socket, p95 ≤ 150 ms at 10^5 records."""
    from .recall import basket

    return basket.assemble(_conn(store), turn)


def update_block(
    *, thread_id: str, session_id: str | None = None, store: Conn | None = None
) -> UpdateBlock:
    """§7.6 deliver: `<memory-update>` for the thread's pending updates before its next
    model call (any origin); overflow → restart from a seed carrying the updates.

    Lines come from `core_view.diff_lines` over the pending rows (the thread holds their
    `old_version`), so forgotten, purged and frozen content is never restated. Delivery is
    the return: the thread's served rows move to the new versions (`via='update'`) and the
    pending rows it covered go, so nothing is delivered twice (`core_diff` deletes its own).
    Over 200 tokens or 10 items `restart` is set; `block` still holds every line, for the
    seed (or the current question) to carry. A pending id with no line (gone, frozen,
    an entity) is closed with a neutral line or none at all, so the gate never stays shut."""
    from . import store as _store
    from .model import ReadSrc, ServedVia
    from .recall import core_view

    conn = _conn(store)
    pend = conn.execute(
        "SELECT item_id, old_version, new_version FROM pending_updates WHERE thread_id = ?"
        " ORDER BY created_at, item_id",
        (thread_id,),
    ).fetchall()
    if not pend:
        return UpdateBlock(block=None)
    served = {str(r[0]): int(r[1]) for r in pend}
    lines, held = core_view.diff_lines(conn, served, [])
    have = {r.id for r in held}
    refs = list(held)
    for item_id, _old, new_v in pend:
        if item_id in have:
            continue
        row = conn.execute("SELECT status, version FROM records WHERE id = ?", (item_id,))
        row = row.fetchone()
        if row is None or row[0] in ("forgotten", "purged") or int(row[1]) <= int(_old):
            continue  # §7.6: a forgotten id never gets a line; its session restarts instead
        lines.append(f"[m:{item_id}] on hold: re-check before relying on it")
        refs.append(ReadRef(id=item_id, version=int(new_v), src=ReadSrc.RECORD))
    ids = [ln.split("]", 1)[0].removeprefix("[m:") for ln in lines]
    block = core_view.update_block(lines) if lines else None
    restart = bool(block) and (
        core_view.tokens(block) > UPDATE_TOKENS or len(set(ids)) > UPDATE_ITEMS
    )
    core_view.serve(conn, refs, thread_id=thread_id, session_id=session_id, via=ServedVia.UPDATE)
    with _store.write_tx(conn):
        for item_id, _old, new_v in pend:
            conn.execute(
                "DELETE FROM pending_updates WHERE thread_id = ? AND item_id = ?"
                " AND new_version <= ?",
                (thread_id, item_id, int(new_v)),
            )
    return UpdateBlock(block=block, item_ids=list(dict.fromkeys(ids)), restart=restart)


UPDATE_TOKENS = 200  # §7.6: an update block is at most 200 tokens ...
UPDATE_ITEMS = 10  # ... and 10 items; over either, the session restarts from a seed


def mark_served(
    *,
    thread_id: str,
    session_id: str | None,
    turn_id: str | None,
    items: list[ReadRef],
    via: str,
    store: Conn | None = None,
) -> None:
    """§7.6 record: a `served` row per item that reached a session. A record's kind is read
    from the ledger (an unknown id writes nothing); raw and derived ids keep their source.
    `via` is one of core, basket, tool, write, update, session."""
    if not items:
        return
    from . import store as _store
    from .recall import basket

    conn = _conn(store)
    rec_ids = list(dict.fromkeys(r.id for r in items if r.src is ReadSrc.RECORD))
    kinds: dict[str, str] = {}
    for i in range(0, len(rec_ids), 400):
        chunk = rec_ids[i : i + 400]
        marks = ", ".join("?" for _ in chunk)
        q = f"SELECT id, kind FROM records WHERE id IN ({marks})"  # noqa: S608
        kinds.update({r[0]: r[1] for r in conn.execute(q, chunk)})
    shown = [
        Item(id=r.id, kind=kinds.get(r.id) or r.src.value, version=r.version, text="")
        for r in dict.fromkeys(items)
        if r.src is not ReadSrc.RECORD or r.id in kinds
    ]
    if not shown:
        return
    if via == "basket" and turn_id is not None:  # the basket's own writer, reused
        turn = TurnCtx(turn_id=turn_id, thread_id=thread_id, session_id=session_id)
        basket._write_served(conn, turn, shown, _store.now())
        return
    sql = (
        "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id, via,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    now = _store.now()
    rows = [(thread_id, session_id, i.id, str(i.kind), i.version, turn_id, via, now) for i in shown]
    if conn.in_transaction:
        conn.executemany(sql, rows)
        return
    with _store.write_tx(conn, priority="hot"):
        conn.executemany(sql, rows)


def pre_send_check(
    *, thread_id: str, read_set: list[ReadRef], store: Conn | None = None
) -> PreSend:
    """§7.4: one indexed query over records, `derived.valid` and `raw_chunks.status`;
    refuses a prompt for a thread with undelivered updates. ≤ 5 ms at 10^5 records."""
    from . import store as _store

    res = _store.pre_send_check(_conn(store), thread_id=thread_id, read_set=read_set)
    return PreSend(**asdict(res))


def changes_since(*, memory_version: int, last_event_id: int, store: Conn | None = None) -> Changes:
    """§7.7: turn-start and PreToolUse watermark read (< 1 ms)."""
    from . import store as _store

    res = _store.changes_since(
        _conn(store), memory_version=memory_version, last_event_id=last_event_id
    )
    return Changes(**asdict(res))


# ---------------------------------------------------------------- arrival (§6.2)


def ingest_turn(
    *,
    turn_id: str,
    thread_id: str,
    origin: str,
    status: str,
    pass_id: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    store: Conn | None = None,
) -> int:
    """After-turn arrival (§6.2): reads the turn's `koob_messages` rows (through the
    conversation adapter, or `rows` in tests), admits, stores raw, enqueues. No model, no
    work on the event loop. Returns the raw chunks stored."""
    from . import ingest

    return ingest.ingest_turn(
        _conn(store), turn_id=turn_id, thread_id=thread_id, origin=origin, rows=rows
    )


def sweep(*, rows: list[dict[str, Any]] | None = None, store: Conn | None = None) -> int:
    """§6.2 rows with no turn, by `meta.sweep_watermark`."""
    from . import ingest

    return ingest.sweep(_conn(store), rows)


# ---------------------------------------------------------------- hands (§16)


def resolve(
    target_text: str, kind: str, *, turn: TurnCtx, store: Conn | None = None
) -> list[Candidate]:
    """§16.1: stable ids → aliases → model only for the ambiguous; alias lessons apply;
    unconfirmed bindings are flagged."""
    from . import entities

    return [
        Candidate(
            entity_id=r.entity_id,
            display_name=r.display_name,
            stable_ids=r.stable_ids,
            confidence=r.confidence,
            token=(r.token.id, r.token.version),
            unconfirmed=r.unconfirmed,
        )
        for r in entities.resolve_rows(_conn(store), target_text, kind)
    ]


def record_action(
    *,
    tool: str,
    stable_target: str,
    origin_key: str,
    read_set: list[ReadRef],
    turn_id: str | None = None,
    intention_id: str | None = None,
    target_entity_id: str | None = None,
    store: Conn | None = None,
) -> ActionTicket:
    """§16.2 steps 1–2, in PreToolUse: reserve `actions(action_key, pending, read_set)`;
    a key already ok / pending / unknown is denied."""
    raise NotImplementedError("hands")


def close_action(
    action_key: str,
    state: ActionState,
    *,
    summary_en: str | None = None,
    store: Conn | None = None,
) -> None:
    """§16.2 step 3/6: PostToolUse → ok, PostToolUseFailure → failed, turn end → unknown,
    `undo_sent` → undone. `summary_en` from a per-tool template, no model."""
    raise NotImplementedError("hands")


def actions_at_startup(*, store: Conn | None = None) -> int:
    """§16.2 step 4: every `pending` → `unknown`. Returns how many."""
    raise NotImplementedError("hands")


def check_action(
    tool: str, args: dict[str, Any], *, turn: TurnCtx, store: Conn | None = None
) -> HandVerdict:
    """§16.3: read sets and tokens of memory-derived values, retired values, frozen
    entities."""
    raise NotImplementedError("hands")


# ---------------------------------------------------------------- work, intentions, capabilities


def resume(matter_id: str, *, turn: TurnCtx, store: Conn | None = None) -> ResumeView:
    """§14: re-validate premises through their read sets; changes first."""
    from . import workstate

    changed, state = workstate.view(_conn(store), matter_id, turn)
    return ResumeView(matter_id=matter_id, changed=changed, state=state)


def on_event(
    source: str, event_key: str, payload: dict[str, Any], *, store: Conn | None = None
) -> list[Fire]:
    """§12: code selects candidate intentions; `intention_events` makes it exactly once.
    v0.4: `event_key` and `payload` keys per source are fixed in §12 (`model.*_event_key`)."""
    from . import intentions

    conn = _conn(store)
    fires: list[Fire] = []
    for iid in intentions.candidates(conn, source, {**payload, "event_key": event_key}):
        if not intentions.fire(conn, iid, event_key):
            continue
        fires.append(
            Fire(
                intention_id=iid,
                event_key=event_key,
                origin_key=intentions.origin_key(iid, event_key),
            )
        )
        line = intentions.notice(conn, iid)
        if line:
            _record(store, "notifications", {"tool": "intention", "text": line})
    if source in ("clock", "date"):
        now = payload.get("ts") or event_key.removeprefix("date:")
        for loop_id in intentions.overdue_loops(conn, str(now)):
            line = intentions.loop_line(conn, loop_id)
            if line:
                _record(store, "notifications", {"tool": "loop", "text": line})
    return fires


def arm_intention(
    intention: Intention, *, turn: TurnCtx | None = None, store: Conn | None = None
) -> WriteResult:
    """v0.4 §12: arm an intention (code paths: duties, «напомни» through `remember`,
    revision conditions). Refused on a banned chat with one line; opens a window on a
    `no_dig` chat (§5.6); `action.read_set` mirrored into `read_deps` (holder `intention`)."""
    from . import intentions
    from . import store as _store

    conn = _conn(store)
    try:
        iid = intentions.arm(conn, intention)
    except (_store.OpRejected, ValueError) as exc:
        res = WriteResult(outcome="rejected", line=str(exc) or "Не могу это поставить.")
    else:
        res = WriteResult(
            outcome="committed",
            record_ids=[iid],
            memory_version=_store.memory_version(conn),
            line="Напомню." if turn is not None else None,
        )
    _note_write(store, "memory_arm", None, res.outcome, res.line)
    return res


def offers(turn: TurnCtx, *, store: Conn | None = None) -> list[Offer]:
    """§15.1 C1: precomputed assoc links for active goals, ≤ 1 offer per turn."""
    raise NotImplementedError("assoc")


# ---------------------------------------------------------------- maintenance, backup, rollout


def wire(
    *,
    conversation: Any = None,
    clock: Clock | None = None,
    stop_hooks: list[Any] | None = None,
    purge_steps: dict[str, Any] | None = None,
) -> None:
    """v0.4 §16.4: the host's one wiring call (runtime from `ensure_started`; tests from
    fixtures): the koob.db `ConversationStore`, the process clock every memory module and
    `store.now()` read, stop-use cache hooks and purge steps for stores outside memory/.
    Unwired: no conversation side (forget touches memory only) and the system clock.
    `[T storage/test_conventions::store_now_follows_wired_clock]`"""
    from . import forget
    from .clock import use

    if clock is not None:
        use(clock)  # one process clock: store, forget, policy, scheduler, freezes (§16.4)
    forget.wire(
        conversation=conversation,
        clock=clock,
        stop_hooks=stop_hooks,
        purge_steps=purge_steps,
    )


def startup(*, store: Conn | None = None) -> StartupReport:
    """v0.4 §18.1: before the host serves a turn (and in `cb memory doctor --repair`): heal
    the forget journal, requeue `running` jobs whose lease ended, `pending` actions →
    `unknown`, finish interrupted mode changes, sweep orphan vectors. Idempotent.
    `[T maintenance/test_startup::running_jobs_requeued, ::live_lease_not_taken,
    ::pending_actions_unknown, ::interrupted_mode_change_finished]`"""
    from . import forget
    from . import policy as _policy
    from . import store as _store

    conn = _conn(store)
    report = StartupReport()
    _store.new_boot()  # leases this process took before now belong to a run that is over
    # (1) open checks ran in `store.open`; the orphan-vector sweep (§3.2 invariants)
    db = _store.db_file(conn)
    if db is not None and _store.MemoryPaths.of(db).vectors.exists():
        forget._sweep_vectors(conn)
        report.vectors_swept = 1
    # (2) heal the forget journal (§9.1 order, §9.3): the write-ahead replay
    heal = getattr(forget, "heal_journal", None)
    if heal is not None:
        report.journal_healed = int(heal(conn) or 0)
    # a stop-use cut after its commit blanks the island turns it names now
    report.journal_healed += forget.finish_turns(conn)
    # (3) jobs whose lease ended or whose holder is gone go back to the queue; a live lease
    # is never taken
    report.jobs_requeued, report.jobs_needing_attention = _store.requeue_expired_jobs(conn)
    # (4) §16.2 step 4: an action still `pending` after a restart may or may not have run
    with _store.write_tx(conn):
        report.actions_unknown = conn.execute(
            "UPDATE actions SET state = 'unknown', closed_at = ? WHERE state = 'pending'",
            (_store.now(),),
        ).rowcount
    # (5) kinds that must finish before any turn (`mode_forget`, §5.2) run now
    # `mode_forget` claims and leases its own jobs (policy.run_mode_forget), so it is not in the
    # store registry; registered startup kinds run through the store
    report.mode_changes_finished = _policy.finish_mode_forgets(conn) + sum(
        _store.run_queued_jobs(conn, jkind) for jkind in _store.startup_job_kinds()
    )
    # (6) §5.3 the in-process label classifier: installed once, cheap (no model load and
    # no prototype build here; both start in the background on first use)
    from . import labeler

    labeler.install()
    return report


def run_jobs(*, limit: int | None = None, store: Conn | None = None) -> int:
    """v0.4 §6.3: run due jobs once (`workers.scheduler.tick` with the wired clock): stale
    groups requeue alone, at most 3 attempts, then `needs_attention`. Returns jobs run.
    `limit` is accepted for the facade; one tick runs every due job."""
    from . import clock as _clockmod
    from .workers import scheduler

    del limit
    from . import labeler

    conn = _conn(store)
    # §5.3: rows whose classifier pass ran cold, workers' records, a restore; a busy store
    # skips it (the next run sweeps; the rows stay not final)
    with contextlib.suppress(sqlite3.Error):
        labeler.finalize_pending(conn)
    return scheduler.tick(conn, _clockmod._current)  # noqa: SLF001 - the wired clock


def doctor(*, store: Conn | None = None) -> DoctorReport:
    """§18 self-maintenance at startup and nightly."""
    from . import doctor as _doctor
    from . import store as _store

    try:
        conn = _conn(store)
    except _store.SQLiteTooOld:
        problems = ["sqlite_too_old"]
    except _store.SyncedPathRefused:
        problems = ["synced_path"]
    except sqlite3.DatabaseError:
        problems = ["corrupt"]
    except (OSError, _store.MemoryStoreError):
        problems = ["unopenable"]
    else:
        try:
            _doctor.repair(conn)
            problems = _doctor.check(conn)
        except sqlite3.DatabaseError:
            problems = ["corrupt"]
    return DoctorReport(ok=not problems, problems=problems, owner_line=_doctor.owner_line(problems))


def run_nightly(*, store: Conn | None = None) -> dict[str, int]:
    """§6.3 night pass and §6.4 consolidation: delta ops only."""
    from . import episodes  # noqa: F401 - registers the episodes and summaries night step
    from .workers import nightly

    return nightly.run(_conn(store))


def backup(*, store: Conn | None = None) -> str:
    """§9.4: `VACUUM INTO` set with the journal and a hot log snapshot; no vectors."""
    from . import backup as _backup

    return str(_backup.make_set(_conn(store)))


def restore(backup_set: str, *, store: Conn | None = None) -> int:
    """§9.4: backup, then journal replay through the full forget pipeline, hot log replay,
    sweep from the backup's watermark; returns the new memory_version (max + 1).
    The caller closes its live connections first; `store` names the live file."""
    from . import backup as _backup
    from . import store as _store

    live = _store.db_file(store) if store is not None else _store.MemoryPaths.default().db
    if live is None:
        raise _store.MemoryStoreError("restore needs a file-backed store")
    return _backup.restore(Path(backup_set), live)


def import_obsidian(vault: str, *, dry_run: bool = True, store: Conn | None = None) -> ImportReport:
    """§20 rollout step 2: one-way, skips `Koob/.history/`, replays legacy forgets."""
    from . import store as _store
    from .importers import obsidian

    events = obsidian.wired_events()  # `koob_memory_events` through the wired koob.db
    if store is not None:
        return ImportReport(
            dry_run=dry_run,
            counts=obsidian.import_vault(store, Path(vault), dry_run=dry_run, events=events),
        )
    # the default store: a dry run only reads it (read-only, and never creates it)
    path = _store.MemoryPaths.default().db
    conn = obsidian.open_readonly(path) if dry_run else _store.open(path)
    try:
        counts = obsidian.import_vault(conn, Path(vault), dry_run=dry_run, events=events)
    finally:
        if conn is not None:
            conn.close()
    return ImportReport(dry_run=dry_run, counts=counts)


def shadow_mirror(op: Literal["write", "forget"], args: dict[str, Any], *, turn: TurnCtx) -> None:
    """§20 rollout step 1: legacy `memory_write` / `memory_forget` mirror here in shadow mode,
    so fixes and forgets reach the ledger.

    On the default store (opened and closed here). Never raises and never slows the legacy
    tool beyond one hot commit: a failure is one log line without content.
    - `forget`: the records imported from the note's `slug` (I-8), else the user's own words
      this turn are the target (the §10.4 grammar: a name → everything about it, other words →
      the best lexical match; several → nothing, the safety net asks); their words are the
      quote either way. No words from the user (a duty, a task): nothing.
    - `write` (`MemoryWriteIn`'s fields): «нет, …» in their words with a ledger target → a fix
      of that target; otherwise a remember of the note's title and body. A note from a page,
      a file or a chat (`source`) is written as from a tainted turn (§6.1: a tool-tier claim
      unless their words name it)."""
    import logging

    log = logging.getLogger(__name__)
    try:
        conn = _conn(None)
    except Exception as exc:  # noqa: BLE001 - shadow must never touch the legacy path
        log.warning("memory shadow: store not opened (%s)", type(exc).__name__)
        return
    quiet = _QUIET.set(True)  # a mirror of the legacy tool: nothing here reaches the user
    try:
        _shadow(conn, op, args or {}, turn)
    except Exception as exc:  # noqa: BLE001 - the same
        log.warning("memory shadow: %s not mirrored (%s)", op, type(exc).__name__)
    finally:
        _QUIET.reset(quiet)
        conn.close()


_SHADOW_TEXT_MAX = 600
_FOREIGN_SOURCES = ("web", "file", "telegram")
_IMPORT_SOURCE = "import:obsidian"  # §20 step 2: `raw_chunks.source_kind`


def _slug_records(conn: Conn, slug: str) -> list[str]:
    """The active records imported from the legacy note `slug` (§20 step 2): their source
    raw chunk is `import:obsidian` with `source_ref` `obsidian:<slug>`, the bare slug, or
    the note's path (`…/<slug>.md`). None imported (or no slug): an empty list."""
    slug = slug.strip()
    if not slug or "/" in slug:
        return []
    rows = conn.execute(
        "SELECT DISTINCT f.id FROM facts f JOIN raw_chunks c ON c.id = f.source_raw_id"
        " JOIN records r ON r.id = f.id"
        " WHERE c.source_kind = ? AND r.status = 'active'"
        " AND (c.source_ref = ? OR c.source_ref = ? OR substr(c.source_ref, -?) = ?)"
        " ORDER BY f.id",
        (_IMPORT_SOURCE, f"obsidian:{slug}", slug, len(slug) + 4, f"/{slug}.md"),
    )
    return [r[0] for r in rows]


def _shadow(conn: Conn, op: str, args: dict[str, Any], turn: TurnCtx) -> None:
    from . import policy
    from .writers import hot

    words = policy.turn_words(turn).strip()  # the user's words, or the task's order (§1)
    if op == "forget":
        if not words:
            return
        # I-8: the legacy forget names one note; the ledger forgets only what was imported
        # from that note. The user's words are the target only when no record maps to the slug.
        ids = _slug_records(conn, str(args.get("slug") or ""))
        target = " ".join(f"m:{i}" for i in ids) if ids else words
        forget(target, turn=turn, because_quote=words, store=conn)
        return
    if op != "write":
        return
    title = " ".join(str(args.get("title") or "").split())
    body = " ".join(str(args.get("body") or "").split())
    text = f"{title}: {body}" if title and body else (title or body)
    if len(text) > _SHADOW_TEXT_MAX:
        text = text[:_SHADOW_TEXT_MAX].rsplit(" ", 1)[0] + "…"
    if not text:
        return
    if str(args.get("source") or "").split(":", 1)[0] in _FOREIGN_SOURCES:
        turn = turn.model_copy(update={"tainted": True})
    if words and "correction" in hot.detect(turn, ""):
        target = hot.resolve_target(conn, words, words, None)
        if target.id is not None:
            fix(words, text, turn=turn, because_quote=words, store=conn)
            return
        if target.ask:
            return  # two plausible targets: the safety net asks the user (§6.1)
    remember(text, turn=turn, because_quote=words or text, store=conn)


# ---------------------------------------------------------------- what the engine recorded
#
# Per memory.db and process-local: the deny reasons and the one-line notices the engine
# produced on the facade - the Stop hook's claim check (`claims_unbacked_change`), writes,
# fixes and forgets that asked first or were refused, and the line each of them gave the user.
# Kept in RAM only (a bounded ring): a notice may quote content, and a forget must not have
# a log on disk to chase. The hands guard's own denials are not in here yet.

RECORDED_MAX = 500
_recorded_logs: dict[str, dict[str, deque[dict[str, Any]]]] = {}
_recorded_lock = threading.Lock()
_QUIET: ContextVar[bool] = ContextVar("memory_api_quiet", default=False)
_ASK_REASON = {"ask_first": "ask_which", "rejected": "rejected", "not_found": "not_found"}


def _log_key(store: Conn | None) -> str:
    from . import store as _store

    if store is None:
        return "default"
    db = _store.db_file(store)
    return str(db) if db is not None else f"memory:{id(store)}"


def _record(store: Conn | None, kind: str, entry: dict[str, Any]) -> None:
    if _QUIET.get():
        return
    from . import clock as _clockmod

    at = _clockmod.now()
    at = at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
    with _recorded_lock:
        logs = _recorded_logs.setdefault(_log_key(store), {})
        logs.setdefault(kind, deque(maxlen=RECORDED_MAX)).append({**entry, "at": at})


def _note_write(
    store: Conn | None, tool: str, target: str | None, outcome: str, line: str | None
) -> None:
    reason = _ASK_REASON.get(outcome)
    if reason is not None:
        _record(store, "denials", {"tool": tool, "target": target, "reason": reason})
    if line:
        _record(store, "notifications", {"tool": tool, "text": line})


def _recorded(store: Conn | None, kind: str) -> list[dict[str, Any]]:
    with _recorded_lock:
        return [dict(e) for e in _recorded_logs.get(_log_key(store), {}).get(kind, ())]


def recorded_denials(*, store: Conn | None = None) -> list[dict[str, Any]]:
    """What the engine refused on this store, oldest first: `{tool, target, reason, at}` -
    `stop`/`claim_without_commit` (the Stop hook, §6.1 step 4), `memory_remember` /
    `memory_fix` / `memory_forget` with `ask_which` (asked the user first), `rejected` or
    `not_found`. `at` is UTC from the wired clock. Process-local (RAM)."""
    return _recorded(store, "denials")


def recorded_notifications(*, store: Conn | None = None) -> list[dict[str, Any]]:
    """The one-line notices the engine gave the user on this store, oldest first: `{tool, text,
    at}` («Запомнил.», «Исправил: …», the forget lines). Process-local (RAM)."""
    return _recorded(store, "notifications")
