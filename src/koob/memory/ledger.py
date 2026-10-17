"""The ledger's writes: assert, supersede, retract, flag, needs_review; lineage edges; author
tiers; version bumps of version-bearing columns; invalidation and pending updates.

Spec: §3.2, §4, §6.3 (commit checks), §7.1-§7.3, §7.6 (mark), §8.2-§8.3, §9.6 (commit
rejections), §13. `apply_op` is the applier `store.commit` calls.

Op payloads (`ProposalOp.payload`; the op's `read_set` is validated by `store.commit`):

- ADD     {"record": {...record fields, "kind"}, "aliases": [{"alias", "kind"}],
           "platform": bool, "derived_from": [ids], "mentions": [entity ids],
           "justifications": [{"group_no", "member", "provenance"}],
           "model_matches": [ids], "rule_match": bool = True}
          or {"derived": {"id", "dkind", "dkey", "content", "read_set"?, "built_at"?}}; a derived
          item's read set defaults to the op's.
- UPDATE  target, {"fields": {column: value}, "add_aliases": [{"alias", "kind"}],
           "remove_aliases": [alias], "add_stable_ids": [StableId], "platform": bool}
- SUPERSEDE target=old, {"by": new_id} or {"record": {...}}; "fix": true when the old value
           was wrong (writes the retraction overlay, §9.6)
- RETRACT target, {"raw_ids": [...]?}  ("that's not true": the user's tiers only)
- FLAG    target, {"review": bool} and/or {"status": "needs_review" | "archived" | "active"}
- LINK    {"justification": {"target", "group_no", "member", "provenance"}}
          | {"alias": {"alias", "kind", "entity_id"?}, "platform": bool} (target = entity)
          | {"from_id", "to_id", "etype"}
- MERGE   target=keep, {"drop": id}
- SPLIT   target=entity, {"record_ids": [...], "display_name"?, "new_id"?}
- FORGET  target: the ledger part of stop-use (status, FTS, invalidation); trash, journal and
          the rest of §9.1 are forget.py's.

Value conventions: `value_norm` of a date is "YYYY-MM-DD" or "MM-DD"; emails lower-case;
phones digits. A broken check raises `CommitRejected`: `store.commit` rolls the group back.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from . import entities, store
from .model import (
    OWNER_ENTITY_ID,
    AliasKind,
    AnyRecord,
    AuthorTier,
    Criterion,
    Decision,
    EdgeType,
    Entity,
    Episode,
    EventKind,
    Fact,
    Impression,
    Intention,
    Label,
    Lesson,
    Loop,
    OpKind,
    Preference,
    Proposal,
    ProposalOp,
    Provenance,
    ReadRef,
    ReadSrc,
    RecordHeader,
    RecordKind,
    Relation,
    StableId,
    Status,
    WorkState,
    tier_rank,
)
from .store import CommitCtx

RELATIONS_YAML = Path(__file__).resolve().parents[3] / "config" / "memory" / "relations.yaml"

USER_TIERS: frozenset[AuthorTier] = frozenset({AuthorTier.USER, AuthorTier.USER_CONFIRMED})
CONTACT_VALUE_TYPES: frozenset[str] = frozenset(
    {"email", "phone", "handle", "address", "account_id"}
)
_USABLE = ("active", "archived")
_GONE = (Status.FORGOTTEN, Status.PURGED)
_RETIRED = (Status.SUPERSEDED, Status.RETRACTED, Status.FORGOTTEN, Status.PURGED)


class CommitRejected(store.OpRejected):
    """An op broke a commit check (§6.3, §9.6, §13); the group rolls back. `reason` is a
    content-free code: tier, labels, suppressed, retracted_value, stable_id_tier, vocabulary,
    unusable_source, member_missing, stated_without_quote, stated_not_user, not_found,
    bad_op, unknown_field, no_lineage. A subclass of `store.OpRejected`, so
    `store.commit` drops a worker's group alone and re-raises it to a hot or code writer."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------- ids and tables

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id() -> str:
    """A ULID: 48 bits of milliseconds, 80 random bits, Crockford base32."""
    n = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[n & 31])
        n >>= 5
    return "".join(reversed(out))


_KIND_TABLE: dict[RecordKind, str] = {
    RecordKind.ENTITY: "entities",
    RecordKind.FACT: "facts",
    RecordKind.PREFERENCE: "facts",
    RecordKind.IMPRESSION: "impressions",
    RecordKind.DECISION: "decisions",
    RecordKind.WORK_STATE: "work_states",
    RecordKind.INTENTION: "intentions",
    RecordKind.LOOP: "loops",
    RecordKind.LESSON: "lessons",
    RecordKind.CRITERION: "criteria",
    RecordKind.EPISODE: "episodes",
}
_MODEL: dict[RecordKind, type[RecordHeader]] = {
    RecordKind.ENTITY: Entity,
    RecordKind.FACT: Fact,
    RecordKind.PREFERENCE: Preference,
    RecordKind.IMPRESSION: Impression,
    RecordKind.DECISION: Decision,
    RecordKind.WORK_STATE: WorkState,
    RecordKind.INTENTION: Intention,
    RecordKind.LOOP: Loop,
    RecordKind.LESSON: Lesson,
    RecordKind.CRITERION: Criterion,
    RecordKind.EPISODE: Episode,
}
_HEADER: tuple[str, ...] = tuple(RecordHeader.model_fields)
_JSON_COLS: dict[str, frozenset[str]] = {
    "entities": frozenset({"stable_ids"}),
    "facts": frozenset({"applicability"}),
    "decisions": frozenset({"alternatives", "revision_conditions"}),
    "work_states": frozenset({"state"}),
    "intentions": frozenset({"trigger", "completion", "action"}),
    "lessons": frozenset({"applicability"}),
    "criteria": frozenset({"applicability", "evidence", "exceptions"}),
}
# §7.1: version-bearing are status, labels, author_tier and the kind row's content. Usage
# counters and timestamps of a kind row are stats, like heat.
_BEARING_HEADER = frozenset({"status", "labels", "author_tier"})
_NON_BEARING_KIND: dict[str, frozenset[str]] = {
    "lessons": frozenset({"helpful", "harmful"}),
    "work_states": frozenset({"last_resumed_at"}),
}
_UPDATABLE_HEADER = frozenset(
    {
        "labels",
        "labels_final",
        "author_tier",
        "author_entity_id",
        "importance",
        "confidence",
        "review",
        "scope_key",
        "pinned_category",
        "origin_quote",
    }
)
_TEXT_FIELDS = (
    "display_name",
    "value_text",
    "value_norm",
    "text_en",
    "keywords_en",
    "keywords_ru",
    "chosen_text",
    "rule_en",
    "summary_en",
    "outcome",
    "place",
)


def _field_col(field: str) -> str:
    return "status" if field == "criterion_status" else field


def _kind_fields(kind: RecordKind) -> list[str]:
    return [f for f in _MODEL[kind].model_fields if f not in _HEADER]


def _kind_cols(kind: RecordKind) -> set[str]:
    return {_field_col(f) for f in _kind_fields(kind)}


def _sql(table: str, col: str, value: Any) -> Any:
    if value is None:
        return None
    if col in _JSON_COLS.get(table, ()):
        if col == "applicability" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if v is not None}
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    return value


def _record_rows(record: AnyRecord) -> tuple[dict[str, Any], str, dict[str, Any]]:
    data = record.model_dump(mode="json")
    header = {k: data[k] for k in _HEADER}
    header["labels"] = int(record.labels)
    header["review"] = int(bool(header["review"]))
    header["labels_final"] = int(bool(header["labels_final"]))
    table = _KIND_TABLE[record.kind]
    kind = {_field_col(f): _sql(table, _field_col(f), data[f]) for f in _kind_fields(record.kind)}
    return header, table, kind


def parse_record(data: dict[str, Any], *, ts: str | None = None) -> AnyRecord:
    """A record from an op payload; a missing id becomes a ULID, recorded_at the commit time."""
    d = dict(data)
    d.setdefault("id", new_id())
    if not d.get("recorded_at"):
        d["recorded_at"] = ts or store.now()
    kind = RecordKind(d.get("kind") or RecordKind.FACT)
    d["kind"] = kind
    try:
        return _MODEL[kind].model_validate(d)  # type: ignore[return-value]
    except ValueError as e:
        raise CommitRejected("bad_op", f"record {d['id']}: {type(e).__name__}") from e


def get_record(conn: sqlite3.Connection, record_id: str) -> AnyRecord | None:
    """Header plus kind row as its model (reads any status)."""
    h = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if h is None:
        return None
    kind = RecordKind(h["kind"])
    table = _KIND_TABLE[kind]
    k = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (record_id,)).fetchone()  # noqa: S608
    if k is None:
        return None
    data: dict[str, Any] = {c: h[c] for c in _HEADER}
    data["review"] = bool(h["review"])
    data["labels_final"] = bool(h["labels_final"])
    data["labels"] = Label(h["labels"])
    for f in _kind_fields(kind):
        col = _field_col(f)
        v = k[col]
        if v is not None and col in _JSON_COLS.get(table, ()):
            v = json.loads(v)
        if v is not None or _MODEL[kind].model_fields[f].is_required() is False:
            data[f] = v
    data = {f: v for f, v in data.items() if v is not None or f in ("value_text",)}
    return _MODEL[kind].model_validate(data)  # type: ignore[return-value]


# ---------------------------------------------------------------- the owner entity (§4.5)

OWNER_ALIASES = ("я", "меня", "мне", "мной", "me", "myself")


def seed_owner(conn: sqlite3.Connection, owner_name: str | None = None) -> bool:
    """§4.5: the owner entity `self` (person, tier user) with the user's name forms and «я /
    меня / мне / me» as aliases, created when missing. Not a memory commit: no version bump,
    no event (like the vocabulary). Returns True when it was created now."""
    from .model import OWNER_ENTITY_ID

    if conn.execute("SELECT 1 FROM records WHERE id = ?", (OWNER_ENTITY_ID,)).fetchone():
        return False
    name = (owner_name or "").strip() or "Owner"
    own = not conn.in_transaction
    if own:
        conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM records WHERE id = ?", (OWNER_ENTITY_ID,)).fetchone():
            if own:
                conn.execute("ROLLBACK")
            return False
        ts = store.now()
        ctx = CommitCtx(
            conn=conn,
            version=store.memory_version(conn),
            ts=ts,
            proposal=Proposal(ops=[], author_tier=AuthorTier.USER),
        )
        record = Entity(
            id=OWNER_ENTITY_ID,
            etype="person",
            display_name=name,
            author_tier=AuthorTier.USER,
            recorded_at=ts,
        )
        aliases = [{"alias": a, "kind": AliasKind.NICKNAME.value} for a in OWNER_ALIASES]
        _insert(ctx, record, aliases=aliases)
        if own:
            conn.execute("COMMIT")
    except BaseException:
        if own and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return True


# ---------------------------------------------------------------- vocabulary (§4.1)


def seed_relations(conn: sqlite3.Connection, path: Path | str | None = None) -> int:
    """Load the closed relation vocabulary into `relations` (upsert; idempotent). Returns the
    number of relations in the file. Not a memory commit: no version, no event."""
    p = Path(path) if path is not None else RELATIONS_YAML
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows: list[Relation] = []
    seen: set[str] = set()
    for family, rels in (data.get("families") or {}).items():
        for rel, spec in (rels or {}).items():
            if rel in seen:
                raise ValueError(f"relations.yaml: {rel} is listed twice")
            seen.add(rel)
            rows.append(Relation(relation=rel, family=family, **spec))
    for required in ("other", "prefers"):
        if required not in seen:
            raise ValueError(f"relations.yaml: `{required}` is required")
    want = {(r.relation, r.family, r.cardinality.value, r.value_type) for r in rows}
    have = {
        tuple(r)
        for r in conn.execute("SELECT relation, family, cardinality, value_type FROM relations")
    }
    if want <= have:
        return len(rows)  # already current: opening a store never waits for the write lock
    own = not conn.in_transaction
    if own:
        conn.execute("BEGIN IMMEDIATE")
    try:
        conn.executemany(
            "INSERT INTO relations (relation, family, cardinality, value_type) VALUES (?, ?, ?, ?)"
            " ON CONFLICT (relation) DO UPDATE SET family = excluded.family,"
            " cardinality = excluded.cardinality, value_type = excluded.value_type",
            [(r.relation, r.family, r.cardinality.value, r.value_type) for r in rows],
        )
        if own:
            conn.execute("COMMIT")
    except BaseException:
        if own and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return len(rows)


def _relation(conn: sqlite3.Connection, relation: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT relation, family, cardinality, value_type FROM relations WHERE relation = ?",
        (relation,),
    ).fetchone()


# ---------------------------------------------------------------- tiers (§8.2, §13)


def may_supersede(new_tier: AuthorTier, old_tier: AuthorTier, *, same_speaker: bool) -> bool:
    """§8.2 tier order: user = user_confirmed > user_room > other about themselves > other >
    tool > system. Superseded only by a higher tier, or at equal tier by the same speaker;
    user and user_confirmed are one speaker. (Callers that know "about themselves" compare
    `tier_rank(..., about_self=True)` themselves; see `_may_supersede_rows`.)"""
    if new_tier in USER_TIERS and old_tier in USER_TIERS:
        return True
    rn, ro = tier_rank(new_tier), tier_rank(old_tier)
    return rn > ro or (rn == ro and same_speaker)


def _speaker(row: dict[str, Any]) -> str | None:
    tier = AuthorTier(row["author_tier"])
    if tier in USER_TIERS:
        return "user"
    if tier is AuthorTier.SYSTEM:
        return "system"
    who = row.get("speaker_entity_id") or row.get("author_entity_id")
    if who is None and tier is AuthorTier.USER_ROOM:
        return "room"
    return who


def _rank_row(row: dict[str, Any]) -> int:
    tier = AuthorTier(row["author_tier"])
    who = row.get("speaker_entity_id") or row.get("author_entity_id")
    about_self = tier is AuthorTier.OTHER and who is not None and who == row.get("subject")
    return tier_rank(tier, about_self=about_self)


def _may_supersede_rows(new: dict[str, Any], old: dict[str, Any]) -> bool:
    if (
        AuthorTier(new["author_tier"]) in USER_TIERS
        and AuthorTier(old["author_tier"]) in USER_TIERS
    ):
        return True
    rn, ro = _rank_row(new), _rank_row(old)
    if rn != ro:
        return rn > ro
    sn, so = _speaker(new), _speaker(old)
    return sn is not None and sn == so


def _hdr(conn: sqlite3.Connection, record_id: str) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT r.id, r.kind, r.status, r.author_tier, r.author_entity_id, r.labels,"
        " r.recorded_at, r.origin_quote, f.speaker_entity_id, f.quote_orig,"
        " coalesce(f.subject_id, i.about_entity_id) AS subject"
        " FROM records r LEFT JOIN facts f ON f.id = r.id"
        " LEFT JOIN impressions i ON i.id = r.id WHERE r.id = ?",
        (record_id,),
    ).fetchone()
    return dict(r) if r is not None else None


def _ceiling(proposal_tier: AuthorTier, source_tiers: Iterable[AuthorTier]) -> AuthorTier:
    """§6.2 step 4 / §13.2: tiers propagate lowest first; a worker (system) carries its
    sources' tier, never more."""
    tiers = list(source_tiers)
    if not tiers:
        return proposal_tier
    lowest = min(tiers, key=tier_rank)
    if proposal_tier is AuthorTier.SYSTEM:
        return lowest
    return min((proposal_tier, lowest), key=tier_rank)


# ---------------------------------------------------------------- lineage, FTS, versions


def add_edge(ctx: CommitCtx, from_id: str, to_id: str, etype: EdgeType) -> None:
    ctx.conn.execute(
        "INSERT OR IGNORE INTO edges (from_id, to_id, etype, created_at) VALUES (?, ?, ?, ?)",
        (from_id, to_id, EdgeType(etype).value, ctx.ts),
    )


def bump(ctx: CommitCtx, record_ids: Iterable[str]) -> None:
    """A version-bearing change: the records take this commit's version (§7.1)."""
    ids = list(dict.fromkeys(record_ids))
    for i in range(0, len(ids), 400):
        chunk = ids[i : i + 400]
        marks = ", ".join("?" for _ in chunk)
        ctx.conn.execute(
            f"UPDATE records SET version = ?, updated_at = ? WHERE id IN ({marks})",  # noqa: S608
            [ctx.version, ctx.ts, *chunk],
        )


def _fts_parts(conn: sqlite3.Connection, record_id: str, kind: RecordKind) -> tuple[str, str, str]:
    rec = get_record(conn, record_id)
    if rec is None:
        return "", "", ""
    texts: list[str] = []
    for f in _TEXT_FIELDS:
        if f in ("keywords_en", "keywords_ru", "value_norm"):
            continue
        v = getattr(rec, f, None)
        if isinstance(v, str) and v:
            texts.append(v)
    if isinstance(rec, WorkState):
        texts += [x for x in (rec.state.goal, rec.state.direction, rec.state.next_step) if x]
    if isinstance(rec, Intention) and rec.trigger.semantic:
        texts.append(rec.trigger.semantic)
    keywords = [
        x
        for x in (
            getattr(rec, "keywords_ru", None),
            getattr(rec, "keywords_en", None),
            getattr(rec, "quote_orig", None),
        )
        if x
    ]
    if isinstance(rec, Intention):
        # "did you promise me something?" must reach the promise itself, not only its trigger.
        act = rec.action or {}
        texts += [v for v in (act.get("text_en"), act.get("text")) if isinstance(v, str) and v]
        keywords += [v for v in (act.get("quote"), rec.origin_quote) if isinstance(v, str) and v]
    aliases = ""
    if kind is RecordKind.ENTITY:
        aliases = " ".join(
            r[0]
            for r in conn.execute("SELECT alias FROM aliases WHERE entity_id = ?", (record_id,))
        )
    return " ".join(texts).casefold(), " ".join(keywords).casefold(), aliases.casefold()


def _unindex(conn: sqlite3.Connection, record_id: str) -> None:
    conn.execute(
        "DELETE FROM fts_records WHERE rowid = (SELECT fts_rowid FROM records WHERE id = ?)",
        (record_id,),
    )
    conn.execute(
        "DELETE FROM fts_names WHERE rowid IN (SELECT fts_rowid FROM aliases WHERE entity_id = ?)",
        (record_id,),
    )


def index(conn: sqlite3.Connection, record_id: str) -> None:
    """FTS rows of a record, in the caller's transaction (§3.2). Stemming is done by
    `lexical.index_record`; until it is built, rows hold case-folded text."""
    row = conn.execute(
        "SELECT kind, status, fts_rowid FROM records WHERE id = ?", (record_id,)
    ).fetchone()
    if row is None:
        return
    if Status(row["status"]) in _GONE:
        _unindex(conn, record_id)
        return
    from .recall import lexical

    try:
        lexical.index_record(conn, record_id)
        return
    except NotImplementedError:  # lexical indexing unavailable: plain case-folded rows
        pass
    kind = RecordKind(row["kind"])
    text, keywords, aliases = _fts_parts(conn, record_id, kind)
    conn.execute("DELETE FROM fts_records WHERE rowid = ?", (row["fts_rowid"],))
    conn.execute(
        "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, ?, ?)",
        (row["fts_rowid"], text, keywords, aliases),
    )
    if kind is RecordKind.ENTITY:
        for a in conn.execute(
            "SELECT fts_rowid, alias FROM aliases WHERE entity_id = ?", (record_id,)
        ).fetchall():
            conn.execute("DELETE FROM fts_names WHERE rowid = ?", (a[0],))
            conn.execute("INSERT INTO fts_names (rowid, name) VALUES (?, ?)", (a[0], a[1]))


def touch_entity(ctx: CommitCtx, entity_id: str, *, tokens: Sequence[str]) -> None:
    """An entity's name side changed (aliases, display name, stable ids): bump, re-index, and
    invalidate only derived text holding a changed token (§7.3), then mark (§7.6)."""
    bump(ctx, [entity_id])
    index(ctx.conn, entity_id)
    invalidate(ctx, [entity_id], semantic=False, tokens=tokens)
    mark_pending_updates(ctx, [entity_id])


# ---------------------------------------------------------------- insert (§3.2, §4)


def _lineage(ctx: CommitCtx, record: AnyRecord) -> None:
    if isinstance(record, Fact):
        add_edge(ctx, record.id, record.subject_id, EdgeType.ABOUT)
        if record.value_entity_id:
            add_edge(ctx, record.id, record.value_entity_id, EdgeType.MENTIONS)
        if record.source_raw_id:
            add_edge(ctx, record.id, record.source_raw_id, EdgeType.DERIVED_FROM)
    elif isinstance(record, Impression):
        add_edge(ctx, record.id, record.about_entity_id, EdgeType.ABOUT)
    elif isinstance(record, (Decision, WorkState)) and record.matter_id:
        add_edge(ctx, record.id, record.matter_id, EdgeType.ABOUT)
    elif isinstance(record, Intention) and record.trigger.entity_id:
        add_edge(ctx, record.id, record.trigger.entity_id, EdgeType.MENTIONS)
    elif isinstance(record, Loop) and record.owner not in ("user", "cube"):
        add_edge(ctx, record.id, record.owner, EdgeType.ABOUT)


def _insert(
    ctx: CommitCtx,
    record: AnyRecord,
    *,
    platform: bool = False,
    aliases: Sequence[dict[str, Any]] = (),
) -> str:
    header, table, kind = _record_rows(record)
    header.update(version=ctx.version, created_at=ctx.ts, updated_at=ctx.ts)
    cols = list(header)
    ctx.conn.execute(
        f"INSERT INTO records ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",  # noqa: S608
        [header[c] for c in cols],
    )
    kind["id"] = record.id
    kcols = list(kind)
    ctx.conn.execute(
        f"INSERT INTO {table} ({', '.join(kcols)}) VALUES ({', '.join('?' for _ in kcols)})",  # noqa: S608
        [kind[c] for c in kcols],
    )
    if isinstance(record, Entity):
        confirmed = record.author_tier in USER_TIERS or platform
        names = [{"alias": record.display_name, "kind": AliasKind.NAME.value}, *aliases]
        for a in names:
            entities.put_alias(
                ctx.conn,
                ctx.ts,
                record.id,
                a["alias"],
                AliasKind(a.get("kind") or "name"),
                tier=record.author_tier,
                confirmed=confirmed,
                with_translit=True,
            )
    _lineage(ctx, record)
    index(ctx.conn, record.id)
    return record.id


def insert_record(ctx: CommitCtx, record: AnyRecord) -> str:
    """Header plus kind row, version = ctx.version, FTS rows, lineage edges. A primitive: the
    commit checks live in `apply_op` (ADD)."""
    return _insert(ctx, record)


# ---------------------------------------------------------------- status (§4.2)


def set_status(ctx: CommitCtx, record_id: str, status: Status) -> None:
    """A version-bearing change (§7.1). Leaving use drops FTS rows for forgotten and purged
    items; derived readers go invalid (§7.3); served threads get pending updates, except for
    forgotten ids, which never get an update block (§7.6)."""
    status = Status(status)
    row = ctx.conn.execute("SELECT status FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        raise CommitRejected("not_found", record_id)
    old = Status(row["status"])
    if old is status:
        return
    if old is Status.PURGED:
        raise CommitRejected("bad_op", "purged is final")
    retired = ctx.ts if status in _RETIRED else None
    ctx.conn.execute(
        "UPDATE records SET status = ?, version = ?, retired_at = ?, updated_at = ? WHERE id = ?",
        (status.value, ctx.version, retired, ctx.ts, record_id),
    )
    if status in _GONE:
        _unindex(ctx.conn, record_id)
    elif old in _GONE:
        index(ctx.conn, record_id)
    invalidate(ctx, [record_id])
    if status not in _GONE:
        mark_pending_updates(ctx, [record_id])


def _set_review(ctx: CommitCtx, record_id: str, on: bool) -> None:
    """`review` is not version-bearing (§4.2, §7.1)."""
    n = ctx.conn.execute(
        "UPDATE records SET review = ?, updated_at = ? WHERE id = ?", (int(on), ctx.ts, record_id)
    ).rowcount
    if n == 0:
        raise CommitRejected("not_found", record_id)


# ---------------------------------------------------------------- supersession (§8.2, §8.3)

_CUE = re.compile(
    r"(?<!\w)(теперь|больше\s+не|уже\s+не|сменил\w*|поменял\w*|перешл\w*|перешёл|перешел|вместо"
    r"|now|no\s+longer|any\s*more|switched|changed|instead\s+of|replaced)(?!\w)",
    re.IGNORECASE,
)


def _fact_rows(conn: sqlite3.Connection, where: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT r.id, r.kind, r.status, r.author_tier, r.author_entity_id, r.recorded_at,"
        " r.origin_quote, f.subject_id AS subject, f.relation, f.value_norm, f.value_entity_id,"
        " f.fkind, f.polarity, f.valid_from, f.stability, f.applicability, f.quote_orig,"
        " f.speaker_entity_id, f.value_text, rel.family, rel.cardinality,"
        " (SELECT c.ts FROM raw_chunks c WHERE c.id = f.source_raw_id) AS raw_ts"
        " FROM facts f JOIN records r ON r.id = f.id"
        " LEFT JOIN relations rel ON rel.relation = f.relation WHERE " + where,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def _plan_class(fkind: str) -> str:
    return fkind if fkind in ("plan", "condition") else "fact"


def _scope(applicability: str | None) -> dict[str, Any]:
    d = json.loads(applicability) if applicability else {}
    return {k: v for k, v in d.items() if v is not None and k not in ("general", "unstated")}


def _value_key(row: dict[str, Any]) -> str | None:
    if row["value_norm"] is not None:
        return str(row["value_norm"])
    if row["value_entity_id"] is not None:
        return "e:" + row["value_entity_id"]
    return None


def _said_at(row: dict[str, Any]) -> Any:
    """When the user said it: the raw message it rests on, else when it was recorded (a hot
    write is recorded as the user speaks). Parsed, so «Z» and «+05:00» stamps compare as instants."""
    from datetime import UTC, datetime

    ts = str(row.get("raw_ts") or row["recorded_at"])
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def _said_before(a: dict[str, Any], c: dict[str, Any]) -> bool:
    """A was said before C. A late extraction of an early message is still the early word."""
    sa, sc = _said_at(a), _said_at(c)
    if type(sa) is not type(sc):
        return a["recorded_at"] < c["recorded_at"]
    return sa < sc


def _older(a: dict[str, Any], c: dict[str, Any]) -> bool:
    if a["valid_from"] and c["valid_from"]:
        return a["valid_from"] < c["valid_from"]
    return _said_before(a, c)


def _matches(a: dict[str, Any], c: dict[str, Any]) -> bool:
    """§8.3: does assertion A restate C's slot so that A supersedes C?"""
    if c["stability"] == "one_off" or _plan_class(a["fkind"]) != _plan_class(c["fkind"]):
        return False
    if _older(a, c) or not _may_supersede_rows(a, c):
        return False
    lapse = _lapses(a, c)
    sa, sc = _scope(a["applicability"]), _scope(c["applicability"])
    if sa != sc and not (lapse and not sa and not {"for_whom", "audience"} & set(sc)):
        return False  # the user's general «больше не» covers the user's own scoped instances, never others'
    if lapse:
        return True
    va, vc = _value_key(a), _value_key(c)
    if va is None or vc is None:
        return False
    if va == vc:
        return a["polarity"] != c["polarity"]
    # A different value replaces only within one relation, both affirmed: a negation says
    # nothing about other values ("I no longer eat meat" leaves fish alone).
    if a["relation"] != c["relation"] or a["polarity"] != "affirmed" or c["polarity"] != "affirmed":
        return False
    if a["cardinality"] == "single":
        return True
    quote = a["quote_orig"] or a["origin_quote"] or ""
    if not _CUE.search(quote):
        return False
    # A partial change supersedes only the part that changed: a multi slot whose new word
    # names what it replaces («вместо тенниса», "not X but Y") retires only that value, and an
    # added one («теперь ещё и бассейн», "now I also swim") retires none.
    sides = _replaced_sides(quote)
    if sides:
        return any(holds_value(x, _value_text(c)) or _named_by(x, _value_text(c)) for x in sides)
    return not _ADDED.search(quote)


# «больше не» / "no longer" ends a habit or a taste: it retires the earlier affirmed fact of
# the same slot family whose value it names in other words («кинзу» for «кинза»), and the
# relation may be its opposite («разлюбил» vs «любит»).
_LAPSE = re.compile(
    r"(?<!\w)(?:больше\s+не|уже\s+не|более\s+не|перестал\w*|разлюбил\w*|бросил\w*|завязал\w*"
    r"|отказал\w*\s+от|no\s+longer|any\s*more|stopped|quit|gave\s+up|giving\s+up|not\s+any\s+longer)"
    r"(?!\w)",
    re.IGNORECASE,
)
_ADDED = re.compile(
    r"(?<!\w)(?:ещ[её]|тоже|также|вдобавок|кроме\s+того|also|too|as\s+well|in\s+addition)(?!\w)",
    re.IGNORECASE,
)
_INSTEAD = (
    re.compile(r"(?<!\w)вместо\s+(?P<old>[^,;:.!?\n]+?)(?=\s*[,;:.!?\n]|\s+[-—–]\s|\s*$)", re.I),
    re.compile(
        r"(?<!\w)instead\s+of\s+(?P<old>[^,;:.!?\n]+?)(?=\s*[,;:.!?\n]|\s+[-—–]\s|\s*$)", re.I
    ),
)


def _replaced_sides(quote: str) -> list[str]:
    out = retired_sides(quote)
    for pat in _INSTEAD:
        out += [" ".join(m.group("old").split()[:3]) for m in pat.finditer(quote)]
    return [x for x in dict.fromkeys(out) if x]


def _stems(text: str | None) -> set[str]:
    words = _VALUE_WORD.findall((text or "").casefold().replace("ё", "е"))
    return {w[: max(4, len(w) - 2)] for w in words if not w.isdigit()}


def _met(x: str, ys: set[str]) -> bool:
    return any(x.startswith(y) or y.startswith(x) for y in ys)


def _named_by(side: str, value: str) -> bool:
    """The replaced side names the value: every stem of the value is in the side's words
    («вместо бокса по средам» names «бокс»)."""
    sv, ss = _stems(value), _stems(side)
    return bool(sv and ss) and all(_met(y, ss) for y in sv)


def _lapses(a: dict[str, Any], c: dict[str, Any]) -> bool:
    """A, a «больше не X» / "no longer X", ends C's affirmed X: same relation (A negated) or
    another relation of the slot family (A affirmed: «разлюбил» against «любит»), and A's value
    names C's (the same word stems both ways). A rival value («больше не ем мясо» vs fish) never
    lapses."""
    if c["polarity"] != "affirmed" or a["family"] is None or a["family"] != c.get("family"):
        return False
    if a["polarity"] == "affirmed" and a["relation"] == c["relation"]:
        return False  # the same relation affirmed: the value rule above decides
    quote = a["quote_orig"] or a["origin_quote"] or ""
    if not _LAPSE.search(quote):
        return False
    sa, sc = _stems(_value_text(a)), _stems(_value_text(c))
    # the same value, not a part of it: «больше не ем красное мясо» leaves «ест мясо»
    return bool(sa and sc) and all(_met(x, sc) for x in sa) and all(_met(y, sa) for y in sc)


def rule_match(conn: sqlite3.Connection, record_id: str) -> list[str]:
    """§8.3 assertion matching by rule: the ids a new assertion supersedes. Same subject, same
    relation family; single with a new value, the same value with opposite polarity, or multi
    with a new value and a replacement cue in A's quote; A not one_off; A's tier may supersede.
    Plans and conditions meet only their own kind; a different scope coexists."""
    found = _fact_rows(conn, "f.id = ?", (record_id,))
    if not found:
        return []
    a = found[0]
    if a["relation"] == "other" and a["status"] in _USABLE and a["stability"] != "one_off":
        return _same_measure(conn, a)
    if (
        a["status"] not in _USABLE
        or a["family"] is None
        or a["relation"] == "other"
        or a["stability"] == "one_off"
    ):
        return []
    rows = _fact_rows(
        conn,
        f"f.subject_id = ? AND rel.family = ? AND f.id <> ? AND f.relation <> 'other'"
        f" AND r.status IN {_USABLE} ORDER BY r.recorded_at, r.id",
        (a["subject"], a["family"], record_id),
    )
    return [c["id"] for c in rows if _matches(a, c)]


# An `other` fact has no relation family, so §8.3 never sees two of them as one slot.
# A number with the same unit on the same subject («6 тысяч в месяц» → «12 тысяч с января»,
# "40 minutes" → "25 minutes") is one measure: the later value replaces the earlier, unless
# the later words state both as current (one message, or the old number in its quote) or a
# period tells them apart (в месяц / в год, per day / per week).
_MEASURE = re.compile(
    r"(?<![\w.,])(?P<n>\d+(?:[.,]\d+)?)\s*(?P<u>[^\W\d_]+|[$€₽£¥%])"
    r"|(?P<c>[$€₽£¥])\s*(?P<m>\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_PER = re.compile(
    r"(?<!\w)(?:в|за|per|a|an|each|/)\s*(?P<p>месяц|год|день|сутки|недел|час|month|year|day"
    r"|week|hour|night|ноч)",
    re.IGNORECASE,
)


def _measure(text: str | None) -> tuple[str, str, str] | None:
    """(number, unit stem, period) of the first amount in a value, else None."""
    t = (text or "").casefold().replace("ё", "е")
    m = _MEASURE.search(t)
    if m is None:
        return None
    num, unit = (m["n"], m["u"]) if m["n"] else (m["m"], m["c"])
    per = _PER.search(t)
    return num.replace(",", "."), unit[:4], per["p"][:4] if per else ""


_WHEN_WORDS = frozenset((
    "январ", "феврал", "март", "апрел", "мая", "май", "июн", "июл", "август", "сентя",
    "октяб", "ноябр", "декаб", "janua", "febru", "march", "april", "june", "july", "augus",
    "septe", "octob", "novem", "decem", "with", "from", "since", "until", "now", "теперь",
))  # fmt: skip


def _measure_words(text: str | None) -> set[str]:
    """What an amount is OF: the value's words beyond its number, unit, period and when."""
    m = _measure(text)
    skip = {m[1], m[2]} if m else set()
    return {
        w
        for w in _value_words(text)
        if len(w) >= 4
        and not any(w.startswith(x) or x.startswith(w) for x in skip if x)
        and not any(w.startswith(x[:4]) for x in _WHEN_WORDS)
    }


def _same_measure(conn: sqlite3.Connection, a: dict[str, Any]) -> list[str]:
    """The earlier `other` facts of A's subject whose amount A replaces (see `_MEASURE`).
    Never the user's own `other` facts (sleep, work hours and rent share units), never
    across speakers A may not supersede, never a negation or a scoped instance."""
    from .model import OWNER_ENTITY_ID

    ma = _measure(_value_text(a))
    if ma is None or a["subject"] in (None, OWNER_ENTITY_ID) or a["polarity"] != "affirmed":
        return []
    quote = a["quote_orig"] or a["origin_quote"] or ""
    own_raws = _source_raws(conn, a["id"])
    out: list[str] = []
    rows = _fact_rows(
        conn,
        f"f.subject_id = ? AND f.relation = 'other' AND f.id <> ? AND r.status IN {_USABLE}"
        " ORDER BY r.recorded_at, r.id",
        (a["subject"], a["id"]),
    )
    for c in rows:
        mc = _measure(_value_text(c))
        if mc is None or mc[1] != ma[1] or mc[0] == ma[0] or c["polarity"] != "affirmed":
            continue
        if mc[2] and ma[2] and mc[2] != ma[2]:
            continue  # «6 тысяч в месяц» and «60 тысяч в год»: two measures
        wa, wc = _measure_words(_value_text(a)), _measure_words(_value_text(c))
        if wa and wc and not wa & wc:
            continue  # «до школы 10 минут» and «до метро 5 минут»: two distances
        if c["stability"] == "one_off" or _plan_class(a["fkind"]) != _plan_class(c["fkind"]):
            continue
        if _scope(a["applicability"]) != _scope(c["applicability"]):
            continue
        if own_raws & _source_raws(conn, c["id"]) or holds_value(mc[0], quote):
            continue  # both stated as current
        if _said_before(c, a) and _may_supersede_rows(a, c):
            out.append(c["id"])
    return out


def supersede(ctx: CommitCtx, old_id: str, new_id: str) -> None:
    """§8.2-§8.3: only by the same speaker or a higher tier; writes a `supersedes` edge."""
    old, new = _hdr(ctx.conn, old_id), _hdr(ctx.conn, new_id)
    if old is None or new is None:
        raise CommitRejected("not_found", old_id if old is None else new_id)
    if old["status"] not in _USABLE:
        raise CommitRejected("bad_op", f"{old_id} is not usable")
    if not _may_supersede_rows(new, old):
        raise CommitRejected("tier", f"{new_id} may not supersede {old_id}")
    set_status(ctx, old_id, Status.SUPERSEDED)
    add_edge(ctx, new_id, old_id, EdgeType.SUPERSEDES)
    _retire_value(ctx, old_id)
    if new["author_tier"] in _HIS_TIERS:
        for cid in carriers(ctx.conn, old_id, exclude=(new_id,)):
            c = _hdr(ctx.conn, cid)
            if c is not None and c["status"] in _USABLE and _may_supersede_rows(new, c):
                set_status(ctx, cid, Status.SUPERSEDED)
                add_edge(ctx, new_id, cid, EdgeType.SUPERSEDES)
                _retire_value(ctx, cid)
    _carry_moved_day(ctx, old_id, new_id)


# ---------------------------------------------------------------- said order
# A correction and the value it corrects may be committed in either order: the hot fix finds
# nothing when the target is not extracted yet, and a late extraction of an early message
# lands after the correction. What decides is the order the user SAID them in.

_V = r"(?:[^,;:.!?\n]|(?<=\d)[:./](?=\d))"  # a clause; a time, date or decimal stays whole
_END = r"(?=\s+(?:and|but|because|since|so|и|но|потому)\s|\s+[-—–]\s|\s*[,;.!?\n]|:(?!\d)|\s*$)"
_RETIRING = (
    re.compile(rf"(?<!\w)не\s+(?P<old>{_V}+?)\s*[,—–-]?\s+а\s+(?:{_V}+)", re.IGNORECASE),
    re.compile(rf"(?:{_V}+?)\s*,\s*а\s+не\s+(?P<old>{_V}+?){_END}", re.IGNORECASE),
    re.compile(rf"(?<!\w)not\s+(?P<old>{_V}+?)\s*,?\s+but\s+(?:{_V}+)", re.IGNORECASE),
    # "Denver, not Boulder", "15:40, not 14:20", "6/8 not 6/1": a value, then «not» + the old
    re.compile(rf"(?P<new>[^\s,;!?]+)\s*,?\s+not\s+(?P<old>{_V}+?){_END}", re.IGNORECASE),
)
_AUX = frozenset((
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "will", "would", "can",
    "could", "should", "shall", "may", "might", "must", "have", "has", "had", "it's", "isn't",
    "that's", "there's", "i'm", "you're", "he's", "she's", "we're", "they're", "why",
))  # fmt: skip
_OLD_END = re.compile(
    r"\s+[-—–]\s|\s+(?:and|but|because|since|so|as|they|it|и|потому|так\s+как|они|она|он)\s",
    re.IGNORECASE,
)


def retired_sides(text: str | None) -> list[str]:
    """The old values a correction in these words retires: «не X, а Y», «Y, а не X», "not X
    but Y", "Y, not X" give X, trimmed to its own clause (at most four words)."""
    out: list[str] = []
    for pat in _RETIRING:
        for m in pat.finditer(text or ""):
            if "new" in pat.groupindex and m.group("new").casefold() in _AUX:
                continue  # "it is not ready": a negated verb, not a replaced value
            old = _OLD_END.split(" " + m.group("old") + " ", maxsplit=1)[0]
            words = old.split()[:4]
            if words:
                out.append(" ".join(words).strip(" -—–"))
    return list(dict.fromkeys(o for o in out if o))


def _value_text(row: dict[str, Any]) -> str:
    return " ".join(str(x) for x in (row.get("value_text"), row.get("value_norm")) if x)


_ORDINAL = re.compile(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)(?!\w)", re.IGNORECASE)


def holds_value(old: str, text: str) -> bool:
    """Does `text` hold the value `old` names: its dates, numbers and contacts when it has
    any (all of them), else its content words (all of them, by stem)."""
    toks = value_tokens(_ORDINAL.sub(r"\1", old))
    if toks:
        have = value_tokens(text)
        # a bare day («не 6-го», "not the 13th") names the day of a date it holds
        days = {t.rsplit("-", 1)[-1].lstrip("0") for t in have if t.startswith("date:")}
        return all(t in have or (t.startswith("num:") and t[4:] in days) for t in toks)
    stems = _value_words(old)
    return bool(stems) and stems <= _value_words(text)


def _contrast_hits(new: dict[str, Any], old: dict[str, Any]) -> bool:
    """NEW's own words retire by contrast the value OLD holds (same subject, said earlier)."""
    if AuthorTier(new["author_tier"]) not in USER_TIERS:
        return False
    sides = retired_sides(new.get("quote_orig")) or retired_sides(new.get("origin_quote"))
    have = _value_text(old)
    return any(holds_value(s, have) for s in sides)


def _overtakes(new: dict[str, Any], old: dict[str, Any]) -> bool:
    """NEW, said after OLD, replaces OLD's value: the §8.3 rule, or the user's contrast naming it."""
    if not _said_before(old, new) or not _may_supersede_rows(new, old):
        return False
    if _contrast_hits(new, old):
        return True
    rule = (
        new["family"] is not None
        and new["family"] == old["family"]
        and "other" not in (new["relation"], old["relation"])
        and new["stability"] != "one_off"
    )
    return rule and _matches(new, old)


def _same_value(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["polarity"] != b["polarity"]:
        return False
    ka, kb = _value_key(a), _value_key(b)
    if ka is not None and ka == kb:
        return True
    ta, tb = (a.get("value_text") or "").casefold().strip(), (b.get("value_text") or "").casefold()
    if ta and ta == tb.strip():
        return True
    va = value_tokens(a.get("value_text") or "")
    return bool(va) and va == value_tokens(b.get("value_text") or "")


def _usable_head(conn: sqlite3.Connection, rid: str) -> str | None:
    seen, cur = {rid}, rid
    for _ in range(64):
        row = conn.execute("SELECT status FROM records WHERE id = ?", (cur,)).fetchone()
        if row is not None and row[0] in _USABLE:
            return cur
        nxt = conn.execute(
            "SELECT e.from_id FROM edges e JOIN records r ON r.id = e.from_id"
            " WHERE e.to_id = ? AND e.etype = 'supersedes' ORDER BY r.version DESC LIMIT 1",
            (cur,),
        ).fetchone()
        if nxt is None or nxt[0] in seen:
            return None
        cur = nxt[0]
        seen.add(cur)
    return None


def _overtaken_by(conn: sqlite3.Connection, a: dict[str, Any]) -> str | None:
    """A record said after A that already retires A's value: a later statement of the slot
    (rule or the user's contrast), or the newer record that superseded an earlier copy of the
    same value when that newer record was said after A."""
    peers = _fact_rows(
        conn,
        "f.subject_id = ? AND f.id <> ? AND r.kind IN ('fact', 'preference')"
        " ORDER BY r.recorded_at DESC, r.id DESC",
        (a["subject"], a["id"]),
    )
    for c in peers:
        if c["status"] in _USABLE and _overtakes(c, a):
            return str(c["id"])
    for r in peers:
        if r["status"] != Status.SUPERSEDED.value or not _same_value(a, r):
            continue
        for (by,) in conn.execute(
            "SELECT from_id FROM edges WHERE to_id = ? AND etype = 'supersedes'", (r["id"],)
        ):
            n = _fact_rows(conn, "f.id = ?", (by,))
            if not n or not _said_before(a, n[0]):
                continue
            hd = _usable_head(conn, str(by))
            h = _fact_rows(conn, "f.id = ?", (hd,)) if hd else []
            if h and not _same_value(a, h[0]) and _may_supersede_rows(h[0], a):
                return str(hd)
    return None


def settle_order(ctx: CommitCtx, record_id: str) -> list[str]:
    """A newly committed fact meets the slot in the order the user said things. The user's
    contrast («не X, а Y», "Y, not X") retires the same subject's earlier-said facts holding
    X; and a fact a later-said record already overtook lands as history (superseded and
    linked, its validity closed), never as a current value. Returns the touched ids."""
    found = _fact_rows(ctx.conn, "f.id = ?", (record_id,))
    if not found or found[0]["status"] not in _USABLE:
        return []
    a = found[0]
    touched: list[str] = []
    if AuthorTier(a["author_tier"]) in USER_TIERS and (
        retired_sides(a["quote_orig"]) or retired_sides(a["origin_quote"])
    ):
        for c in _fact_rows(
            ctx.conn,
            f"f.subject_id = ? AND f.id <> ? AND r.status IN {_USABLE}"
            " AND r.kind IN ('fact', 'preference')",
            (a["subject"], record_id),
        ):
            if _said_before(c, a) and _may_supersede_rows(a, c) and _contrast_hits(a, c):
                supersede(ctx, str(c["id"]), record_id)
                touched.append(str(c["id"]))
    by = _overtaken_by(ctx.conn, a)
    if by is not None:
        supersede(ctx, record_id, by)
        touched.append(by)
    return touched


# ---------------------------------------------------------------- value index

_HIS_TIERS = (AuthorTier.USER.value, AuthorTier.USER_CONFIRMED.value)
_VALUE_WORD = re.compile(r"[^\W_]{4,}", re.UNICODE)


def _value_words(text: str | None) -> set[str]:
    return {w[:5] for w in _VALUE_WORD.findall((text or "").casefold().replace("ё", "е"))}


def _source_raws(conn: sqlite3.Connection, rid: str) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT e.to_id FROM edges e JOIN raw_chunks c ON c.id = e.to_id"
            " WHERE e.from_id = ? AND e.etype = 'derived_from'",
            (rid,),
        )
    } | {
        r[0]
        for r in conn.execute(
            "SELECT source_raw_id FROM facts WHERE id = ? AND source_raw_id IS NOT NULL", (rid,)
        )
    }


def _others_scope(applicability: str | None) -> dict[str, Any]:
    return {k: v for k, v in _scope(applicability).items() if k in ("for_whom", "audience")}


def carriers(conn: sqlite3.Connection, rid: str, *, exclude: Sequence[str] = ()) -> list[str]:
    """Every other usable record that carries the value fact `rid` holds, so one correction
    retires them all (guarantees 3 and 5): the episodes built from it (as seed or
    cause), and its copies - the user's own facts or preferences of the same subject and
    polarity with the same normalized value, or any from the same message with an overlapping
    value. A rival value, a negated or one-off override, another subject or another source's
    retelling is never a carrier."""
    row = conn.execute(
        "SELECT f.subject_id, f.relation, f.value_norm, f.value_text, f.polarity, f.stability,"
        " f.applicability FROM facts f WHERE f.id = ?",
        (rid,),
    ).fetchone()
    if row is None:
        return []
    subject, relation, vnorm, vtext, polarity, stability, appl = row
    other = _others_scope(appl)
    skip = {rid, *exclude}
    out: list[str] = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT e.from_id FROM edges e JOIN records r ON r.id = e.from_id"
            " WHERE e.to_id = ? AND e.etype = 'derived_from' AND r.kind = 'episode'"
            f" AND r.status IN {_USABLE}",
            (rid,),
        )
        if r[0] not in skip
    ]
    words = _value_words(vtext)
    raws = _source_raws(conn, rid)
    for cid, crel, cnorm, ctext, cpol, cstab, ctier, cappl in conn.execute(
        "SELECT f.id, f.relation, f.value_norm, f.value_text, f.polarity, f.stability,"
        " r.author_tier, f.applicability FROM facts f JOIN records r ON r.id = f.id"
        " WHERE f.subject_id = ?"
        f" AND r.kind IN ('fact', 'preference') AND r.status IN {_USABLE}",
        (subject,),
    ):
        if cid in skip or cpol != polarity or (cstab == "one_off") != (stability == "one_off"):
            continue
        if _others_scope(cappl) != other:
            continue  # what the user does for someone else is not a copy of the user's own
        same = crel == relation and (
            (vnorm is not None and cnorm == vnorm)
            or (bool(vtext) and (ctext or "").casefold().strip() == vtext.casefold().strip())
        )
        shared = bool(raws & _source_raws(conn, cid))
        if not same and words and _value_words(ctext) & words:
            same = shared
        # the user's own copy, or one from the same message; another source's retelling stays its
        # own record (§8.4: an independent source is judged by the cascade, not retired here)
        if same and (ctier in _HIS_TIERS or shared):
            out.append(cid)
    return list(dict.fromkeys(out))


def _retire_value(ctx: CommitCtx, record_id: str) -> None:
    """§9.6 retired values: an hmac of a superseded or retracted contact-like value (§16.3)."""
    row = ctx.conn.execute(
        "SELECT f.value_norm, rel.value_type FROM facts f JOIN relations rel"
        " ON rel.relation = f.relation WHERE f.id = ?",
        (record_id,),
    ).fetchone()
    if row is None or row[0] is None or row[1] not in CONTACT_VALUE_TYPES:
        return
    from . import ingest

    try:
        h = ingest.content_hmac(str(row[0]))
    except NotImplementedError:  # keyed HMAC not available yet; nothing to write
        return
    ctx.conn.execute(
        "INSERT OR IGNORE INTO retired_values (hmac, record_id, created_at) VALUES (?, ?, ?)",
        (h, record_id, ctx.ts),
    )


# ---------------------------------------------------------------- invalidation (§7.3, §7.6)


def _queue_rebuild(ctx: CommitCtx, derived_id: str, dkind: str, dkey: str | None) -> None:
    ctx.conn.execute(
        "INSERT OR IGNORE INTO jobs (id, jkind, payload, state, attempts, idempotency_key,"
        " created_at, updated_at) VALUES (?, 'rebuild_derived', ?, 'queued', 0, ?, ?, ?)",
        (
            new_id(),
            json.dumps({"derived_id": derived_id, "dkind": dkind, "dkey": dkey}),
            f"rebuild:{derived_id}:{ctx.version}",
            ctx.ts,
            ctx.ts,
        ),
    )


def invalidate(
    ctx: CommitCtx,
    item_ids: Sequence[str],
    *,
    semantic: bool = True,
    tokens: Sequence[str] = (),
) -> list[str]:
    """§7.3: derived holders of the items become invalid and are queued; a non-semantic
    entity change invalidates only text holding a changed token (`tokens`; when none are
    given, the entity's current names). Derived items reading an invalidated one follow.
    Justification dependents are §8's (justify.cascade); open proposals fail at commit."""
    ids = list(dict.fromkeys(item_ids))
    if not ids:
        return []
    toks = [t.casefold().strip() for t in tokens if t and t.strip()]
    if not semantic and not toks:
        marks = ", ".join("?" for _ in ids)
        toks = [
            r[0].casefold()
            for r in ctx.conn.execute(
                f"SELECT alias FROM aliases WHERE entity_id IN ({marks})",
                ids,  # noqa: S608
            )
        ]
    patterns = [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)") for t in toks]
    out: list[str] = []
    seen: set[str] = set()
    frontier, first = ids, True
    while frontier:
        marks = ", ".join("?" for _ in frontier)
        holders = ctx.conn.execute(
            "SELECT DISTINCT d.id, d.dkind, d.dkey, d.content FROM read_deps rd"
            " JOIN derived d ON d.id = rd.holder_id"
            f" WHERE rd.holder_kind = 'derived' AND rd.item_id IN ({marks}) AND d.valid = 1",  # noqa: S608
            frontier,
        ).fetchall()
        nxt: list[str] = []
        for h in holders:
            if h["id"] in seen:
                continue
            seen.add(h["id"])
            if first and not semantic:
                text = (h["content"] or "").casefold()
                if not any(p.search(text) for p in patterns):
                    continue
            ctx.conn.execute(
                "UPDATE derived SET valid = 0, updated_at = ? WHERE id = ?", (ctx.ts, h["id"])
            )
            _queue_rebuild(ctx, h["id"], h["dkind"], h["dkey"])
            out.append(h["id"])
            nxt.append(h["id"])
        frontier, first = nxt, False
    return out


def mark_pending_updates(ctx: CommitCtx, item_ids: Sequence[str]) -> int:
    """§7.6 mark: a `pending_updates` row per thread that served a changed id. The row keeps
    the oldest version the thread holds and moves `new_version` forward. Forgotten and purged
    ids get none."""
    n = 0
    for item in dict.fromkeys(item_ids):
        r = ctx.conn.execute("SELECT version, status FROM records WHERE id = ?", (item,)).fetchone()
        if r is None or Status(r["status"]) in _GONE:
            continue
        threads = ctx.conn.execute(
            "SELECT thread_id, max(version) FROM served WHERE item_id = ? GROUP BY thread_id",
            (item,),
        ).fetchall()
        for thread_id, served_v in threads:
            if served_v >= r["version"]:
                continue
            ctx.conn.execute(
                "INSERT INTO pending_updates (thread_id, item_id, old_version, new_version,"
                " created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT (thread_id, item_id)"
                " DO UPDATE SET new_version = excluded.new_version",
                (thread_id, item, served_v, r["version"], ctx.ts),
            )
            n += 1
    return n


# ---------------------------------------------------------------- retraction overlay (§9.6)

_MONTHS: dict[str, int] = {}
for _i, _names in enumerate(
    (
        "january jan январь января",
        "february feb февраль февраля",
        "march mar март марта",
        "april apr апрель апреля",
        "may май мая",
        "june jun июнь июня",
        "july jul июль июля",
        "august aug август августа",
        "september sep sept сентябрь сентября",
        "october oct октябрь октября",
        "november nov ноябрь ноября",
        "december dec декабрь декабря",
    ),
    start=1,
):
    for _n in _names.split():
        _MONTHS[_n] = _i
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_HANDLE = re.compile(r"(?<![\w.@])@([a-z0-9_]{3,})")
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_MMDD = re.compile(r"(?<![\d-])-{0,2}(\d{2})-(\d{2})(?![\d-])")
_DOTTED = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})(?:\.(\d{4}|\d{2}))?(?![\d.]*\d)")
_DAY_MONTH = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th|-?го|-?е)?(?:\s+of)?\s+({_MONTH_RE})(?!\w)(?:,?\s+(\d{{4}}))?"
)
# A number followed by a magnitude, currency or percent is an amount, not a day:
# «с января 12 тысяч», "since January 12k".
_AMOUNT_AHEAD = (
    r"(?!\s*(?:тыс|млн|млрд|руб|₽|долл|евро|бакс|thousand|million|billion|k(?!\w)|m(?!\w)|%|\$|€"
    r"|usd|eur|rub|dollars?|bucks|euros?|pounds?))"
)
_MONTH_DAY = re.compile(
    rf"(?<!\w)({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?!\d){_AMOUNT_AHEAD}(?:,?\s+(\d{{4}}))?"
)
_PHONE = re.compile(r"\+?\d[\d\s()\-]{5,}\d")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?!\w)")
_HISTORY = re.compile(
    r"(?<!\w)(previously|formerly|earlier|used\s+to|wrongly|mistakenly|corrected|not|no\s+longer"
    r"|instead\s+of|outdated|раньше|ранее|прежде|ошибочно|исправлен\w*|не|вместо|устарел\w*)(?!\w)",
    re.IGNORECASE,
)
_CLAUSE = re.compile(r"(?<!\d)[.,](?!\d)|(?<!\d):(?!\d)|[;()\[\]\n—–!?]|\s-\s")


def _date_tokens(month: int, day: int, year: str | None) -> set[str]:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return set()
    out = {f"date:{month:02d}-{day:02d}"}
    if year and len(year) == 4:
        out.add(f"date:{year}-{month:02d}-{day:02d}")
    return out


def value_tokens(text: str) -> set[str]:
    """Normalized dates, numbers, emails, phones and handles in a text: `date:MM-DD` (plus
    `date:YYYY-MM-DD`), `num:N`, `email:x`, `tel:<last 10 digits>`, `handle:x`. The same
    function reads retracted values and rebuilt output, so «3 мая», "May 3rd", 03.05 and
    05-03 meet (§9.6)."""
    s = text.casefold()
    toks: set[str] = set()

    def take(pattern: re.Pattern[str], fn: Callable[[re.Match[str]], set[str]]) -> None:
        nonlocal s
        for m in list(pattern.finditer(s)):
            got = fn(m)
            if got:
                toks.update(got)
                s = s[: m.start()] + " " * (m.end() - m.start()) + s[m.end() :]

    take(_EMAIL, lambda m: {"email:" + m.group()})
    take(_HANDLE, lambda m: {"handle:" + m.group(1)})
    take(_ISO, lambda m: _date_tokens(int(m[2]), int(m[3]), m[1]))
    take(_DAY_MONTH, lambda m: _date_tokens(_MONTHS[m[2]], int(m[1]), m[3]))
    take(_MONTH_DAY, lambda m: _date_tokens(_MONTHS[m[1]], int(m[2]), m[3]))
    take(_DOTTED, lambda m: _date_tokens(int(m[2]), int(m[1]), m[3]))
    take(_MMDD, lambda m: _date_tokens(int(m[1]), int(m[2]), None))

    def phone(m: re.Match[str]) -> set[str]:
        digits = re.sub(r"\D", "", m.group())
        return {"tel:" + digits[-10:]} if len(digits) >= 7 else set()

    take(_PHONE, phone)
    take(_NUMBER, lambda m: {"num:" + (m.group().replace(",", ".").lstrip("0") or "0")})
    return toks


# ------------------------------------------------------- a moved day reaches what rests on it
# A dated record whose day moves (a superseding record holds another day) carries the
# records that rest on it: the ones it justifies, the ones derived from it, and the ones
# whose own words give a reason naming its subject («чтобы Денис успел», "so Dana can make
# it", "because Lev is back"). A window taken from the moved day moves with it; a window
# the dependent states in its own words stays, and the core says the premise moved. A
# record that is only dated near the old day, with no such link, is not touched.

_WEEKDAY_WORDS = tuple(
    (re.compile(p, re.IGNORECASE), days)
    for p, days in (
        (r"(?<!\w)(?:monday|понедельник\w*)(?!\w)", (0,)),
        (r"(?<!\w)(?:tuesday|вторник\w*)(?!\w)", (1,)),
        (r"(?<!\w)(?:wednesday|сред[аеуы])(?!\w)", (2,)),
        (r"(?<!\w)(?:thursday|четверг\w*)(?!\w)", (3,)),
        (r"(?<!\w)(?:friday|пятниц\w*)(?!\w)", (4,)),
        (r"(?<!\w)(?:saturday|суббот\w*)(?!\w)", (5,)),
        (r"(?<!\w)(?:sunday|воскресень\w*)(?!\w)", (6,)),
        (r"(?<!\w)(?:weekend|выходн\w+)(?!\w)", (5, 6)),
    )
)
_BARE_DAY = re.compile(r"(?<![\d:./])(\d{1,2})(?:-?го|-?е|st|nd|rd|th)(?!\w)", re.IGNORECASE)
_RANGE_TO = r"\s*(?:[-–—]|по|to|through|thru|until|till)\s*"
_DAY_RANGE = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th|-?го|-?е)?{_RANGE_TO}(\d{{1,2}})(?:st|nd|rd|th|-?го|-?е)?"
    rf"(?:\s+of)?\s+({_MONTH_RE})(?!\w)"
    rf"|(?<!\w)({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?{_RANGE_TO}(\d{{1,2}})(?!\d)",
    re.IGNORECASE,
)
# a reason or purpose clause: what follows ties the record to someone else's day
_REASON_LEAD = re.compile(
    r"(?<!\w)(?:чтобы|чтоб|потому\s+что|так\s+как|из-за|ради|раз|пока|когда|как\s+только"
    r"|после\s+того,?\s+как|к\s+(?:приезду|возвращению|приходу|прилёту|прилету)"
    r"|so(?:\s+that)?|in\s+order\s+(?:for|that)|because|since|as\s+soon\s+as|once|when"
    r"|after|until|till|while|due\s+to|given\s+that)(?!\w)",
    re.IGNORECASE,
)
_CLAUSE_STOP = re.compile(r"[.;!?\n]")
_NAME_TOKEN = re.compile(r"[^\W\d_][\w'’-]*")
_DAY_ROW_SQL = (
    "SELECT f.id, f.subject_id, f.value_text, f.value_norm, f.text_en, f.quote_orig,"
    " f.valid_from, f.valid_to, r.status, r.recorded_at, r.retired_at, r.origin_quote"
    " FROM facts f JOIN records r ON r.id = f.id WHERE f.id = ?"
)


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _mk_day(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _without_sides(text: str) -> str:
    """The text with the old side of the user's «не X, а Y» / "not X but Y" blanked out."""
    for side in retired_sides(text):
        text = text.replace(side, " ")
    return text


def said_days(
    texts: Iterable[Any], year: int, *, month: int | None = None, near: date | None = None
) -> set[date]:
    """The days these words name: dates and day ranges («9–10 мая», "May 9-10"); with a
    `month`, a bare day («10-го», "the 13th"); with `near` and no day found, a lone weekday
    (the one nearest to `near`)."""
    out: set[date] = set()
    weekdays: set[int] = set()
    for x in texts:
        if not isinstance(x, str) or not x.strip():
            continue
        s = x.casefold()
        for m in _DAY_RANGE.finditer(s):
            a, b, mon = (m[1], m[2], m[3]) if m[1] else (m[5], m[6], m[4])
            lo, hi = int(a), int(b)
            if lo < hi <= lo + 14:
                out |= {d for k in range(lo, hi + 1) if (d := _mk_day(year, _MONTHS[mon], k))}
        toks = value_tokens(s)
        full = {t[-5:]: int(t[5:9]) for t in toks if t.startswith("date:") and len(t) == 15}
        for t in toks:
            if t.startswith("date:"):
                md = t[-5:]
                if d := _mk_day(full.get(md, year), int(md[:2]), int(md[3:])):
                    out.add(d)
        if month is not None:
            for m in _BARE_DAY.finditer(s):
                if d := _mk_day(year, month, int(m[1])):
                    out.add(d)
        weekdays |= {w for rx, ds in _WEEKDAY_WORDS if rx.search(s) for w in ds}
    if not out and near is not None and len(weekdays) == 1:
        (w,) = weekdays
        ahead, back = (w - near.weekday()) % 7, (near.weekday() - w) % 7
        out.add(near + timedelta(days=ahead if ahead <= back else -back))
    return out


def _named_weekdays(*texts: Any) -> set[int]:
    return {
        w
        for x in texts
        if isinstance(x, str)
        for rx, ds in _WEEKDAY_WORDS
        if rx.search(x)
        for w in ds
    }


def _bounds(row: sqlite3.Row, closed: Iterable[str] = ()) -> set[date]:
    """A record's own bounds; an end a fix set when it retired the record (on the day of
    the retirement or of the correcting record) is not a day the user named."""
    shut = {str(row["retired_at"] or "")[:10], *closed}
    out = {d for d in (_day(row["valid_from"]),) if d}
    end = _day(row["valid_to"])
    if end is not None and end.isoformat() not in shut:
        out.add(end)
    return out


def _record_days(
    row: sqlite3.Row, *, anchor: date | None = None, closed: Iterable[str] = ()
) -> set[date]:
    """The days a record holds. An old record: its value's and its words', else its bounds
    (a fix may have closed them). With an `anchor` (the corrected record's day), the new
    record: its value's, else its words', else its bounds; a bare day or a lone weekday
    resolves next to the anchor and the retired side of the user's contrast is not read."""
    bounds = _bounds(row, closed)
    base = anchor or min(bounds, default=None) or _day(row["recorded_at"])
    if base is None:
        return set()
    month = anchor.month if anchor else None

    def read(*texts: Any) -> set[date]:
        got = [_without_sides(t) if anchor else t for t in texts if isinstance(t, str) and t]
        return said_days(got, base.year, month=month, near=anchor)

    vals = read(row["value_text"], row["value_norm"])
    words = read(row["quote_orig"], row["text_en"], row["origin_quote"])
    if anchor is None:
        return (vals | words) or bounds
    return vals or words or bounds


def _pair_days(old: set[date], new: set[date]) -> dict[date, date]:
    """Old day -> new day: in order when both hold as many days (a shifted window), else
    each new day takes the nearest old day the new record no longer holds."""
    if not old or not new or old == new:
        return {}
    if len(old) == len(new):
        return {o: n for o, n in zip(sorted(old), sorted(new), strict=True) if o != n}
    left = sorted(old - new)
    out: dict[date, date] = {}
    for n in sorted(new - old):
        if left:
            o = min(left, key=lambda d: (abs((d - n).days), d))
            left.remove(o)
            out[o] = n
    return out


def day_move(conn: sqlite3.Connection, old_id: str, new_id: str) -> dict[date, date]:
    """The days `old_id` held that `new_id` moved, old day -> new day; empty when the
    correction kept the day (a time-only fix, another value of an undated slot)."""
    o = conn.execute(_DAY_ROW_SQL, (old_id,)).fetchone()
    n = conn.execute(_DAY_ROW_SQL, (new_id,)).fetchone()
    if o is None or n is None:
        return {}
    closed = [
        str(r[0])[:10]
        for r in conn.execute(
            "SELECT r.recorded_at FROM edges e JOIN records r ON r.id = e.from_id"
            " WHERE e.to_id = ? AND e.etype = 'supersedes'",
            (old_id,),
        )
    ]
    old = _record_days(o, closed=closed)
    if not old:
        return {}
    return _pair_days(old, _record_days(n, anchor=min(old)))


def reason_names(conn: sqlite3.Connection, texts: Iterable[Any]) -> set[str]:
    """The people (and other named entities) a reason or purpose clause of these words
    names: «…, чтобы Денис тоже успел», "so Dana can make it", "because Lev is back"."""
    keys: set[str] = set()
    for x in texts:
        if not isinstance(x, str):
            continue
        for m in _REASON_LEAD.finditer(x):
            tail = x[m.end() :]
            stop = _CLAUSE_STOP.search(tail)
            for tok in _NAME_TOKEN.findall(tail[: stop.start()] if stop else tail):
                tok = re.sub(r"['’]s$", "", tok)
                key = entities.normalize_alias(tok)
                if len(key) >= 2:
                    keys |= {key, *entities.name_bases(tok)}
                    keys |= {entities.normalize_alias(t) for t in entities.transliterations(tok)}
    if not keys:
        return set()
    marks = ", ".join("?" for _ in keys)
    rows = conn.execute(
        f"SELECT DISTINCT entity_id FROM aliases WHERE alias_norm IN ({marks})",  # noqa: S608
        sorted(keys),
    )
    return {r[0] for r in rows} - {OWNER_ENTITY_ID}


def _stated(row: sqlite3.Row, day: date) -> bool:
    """The record's own words name this day (a date, a range, a bare day or its weekday)."""
    words = (row["text_en"], row["quote_orig"], row["value_text"], row["value_norm"])
    return day in said_days(words, day.year, month=day.month) or (
        day.weekday() in _named_weekdays(*words)
    )


def _touches(row: sqlite3.Row, moved: dict[date, date]) -> bool:
    """A moved day is in the record's window: a bound sits on it, a closed window holds it,
    or its own words name it. An open-ended state («since 2024») holds no one day."""
    lo, hi = _day(row["valid_from"]), _day(row["valid_to"])
    if lo is None and hi is None:
        return False
    return any(
        d in (lo, hi) or (lo is not None and hi is not None and lo <= d <= hi) or _stated(row, d)
        for d in moved
    )


def _redated(row: sqlite3.Row, moved: dict[date, date]) -> dict[str, str]:
    """New bounds for a window taken from a moved day: each bound their words do not state
    moves by the shift of the moved day it sat on. Empty when no unstated bound sat on one
    or the window would turn inside out."""
    cols = {c: _day(row[c]) for c in ("valid_from", "valid_to")}
    free = {c: d for c, d in cols.items() if d is not None and not _stated(row, d)}
    hit = [d for d in free.values() if d in moved]
    if not hit:
        return {}
    shift = moved[hit[0]] - hit[0]
    out = {c: (moved.get(d) or d + shift) for c, d in free.items()}
    lo, hi = out.get("valid_from", cols["valid_from"]), out.get("valid_to", cols["valid_to"])
    if lo is not None and hi is not None and lo > hi:
        return {}
    return {c: d.isoformat() + str(row[c])[10:] for c, d in out.items()}


def _resting_on(conn: sqlite3.Connection, old_id: str, moved: dict[date, date]) -> list[str]:
    """Facts that rest on `old_id`: it justifies them, they are derived from it, or a reason
    clause of their own words names its subject and their window holds a moved day."""
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT target_record_id FROM justifications WHERE member_record_id = ?"
            " UNION SELECT from_id FROM edges WHERE to_id = ? AND etype = 'derived_from'",
            (old_id, old_id),
        )
    ]
    row = conn.execute("SELECT subject_id FROM facts WHERE id = ?", (old_id,)).fetchone()
    subj = row[0] if row is not None else None
    if subj and subj != OWNER_ENTITY_ID:
        lo, hi = min(moved).isoformat(), max(moved).isoformat()
        for r in conn.execute(
            "SELECT f.id, f.text_en, f.quote_orig FROM facts f JOIN records r ON r.id = f.id"
            " WHERE r.status IN ('active', 'archived') AND f.id != ?"
            " AND (f.valid_from IS NOT NULL OR f.valid_to IS NOT NULL)"
            " AND substr(coalesce(f.valid_from, '0000-00-00'), 1, 10) <= ?"
            " AND substr(coalesce(f.valid_to, '9999-12-31'), 1, 10) >= ?",
            (old_id, hi, lo),
        ):
            if subj in reason_names(conn, (r[1], r[2])):
                ids.append(r[0])
    return list(dict.fromkeys(ids))


def _carry_moved_day(ctx: CommitCtx, old_id: str, new_id: str) -> None:
    """The user's correction moved `old_id`'s day: a window taken from it moves to the new
    day; the records resting on it are rebuilt and re-served either way."""
    moved = day_move(ctx.conn, old_id, new_id)
    if not moved:
        return
    redated: list[str] = []
    reached: list[str] = []
    for rid in _resting_on(ctx.conn, old_id, moved):
        row = ctx.conn.execute(_DAY_ROW_SQL, (rid,)).fetchone()
        if rid == new_id or row is None or row["status"] not in _USABLE:
            continue
        if not _touches(row, moved):
            continue
        reached.append(rid)
        upd = _redated(row, moved)
        if upd:
            sets = ", ".join(f"{c} = ?" for c in upd)
            ctx.conn.execute(f"UPDATE facts SET {sets} WHERE id = ?", [*upd.values(), rid])  # noqa: S608
            redated.append(rid)
    if redated:
        bump(ctx, redated)
    if reached:
        invalidate(ctx, reached)
        mark_pending_updates(ctx, reached)


def premise_moves(
    conn: sqlite3.Connection, rid: str
) -> tuple[dict[str, str | None], list[tuple[str | None, str]]]:
    """How a dated record reads after a day it rests on moved: (bounds, notes). `bounds`
    maps a bound to the day to show instead (a window taken from the moved day) or to None
    (a day their words state that no longer holds for the premise); `notes` are (subject,
    new day) of each moved premise."""
    row = conn.execute(_DAY_ROW_SQL, (rid,)).fetchone()
    if row is None or not (row["valid_from"] or row["valid_to"]):
        return {}, []
    prem = [
        r[0]
        for r in conn.execute(
            "SELECT member_record_id FROM justifications WHERE target_record_id = ?"
            " UNION SELECT to_id FROM edges WHERE from_id = ? AND etype = 'derived_from'",
            (rid, rid),
        )
    ]
    named = reason_names(conn, (row["text_en"], row["quote_orig"]))
    if named:
        marks = ", ".join("?" for _ in named)
        prem += [
            r[0]
            for r in conn.execute(
                "SELECT f.id FROM facts f JOIN records r ON r.id = f.id WHERE r.status ="
                f" 'superseded' AND f.subject_id IN ({marks}) AND f.id != ?",  # noqa: S608
                [*sorted(named), rid],
            )
        ]
    bounds: dict[str, str | None] = {}
    notes: list[tuple[str | None, str]] = []
    for pid in dict.fromkeys(prem):
        p = conn.execute(_DAY_ROW_SQL, (pid,)).fetchone()
        if p is None or p["status"] != Status.SUPERSEDED.value:
            continue
        head = _usable_head(conn, pid)
        moved = day_move(conn, pid, head) if head and head != pid else {}
        if not moved or not _touches(row, moved):
            continue
        for col, d in _redated(row, moved).items():
            bounds[col] = d
        for col in ("valid_from", "valid_to"):
            d = _day(row[col])
            if d in moved and col not in bounds and _stated(row, d):
                bounds[col] = None
        note = (p["subject_id"], max(moved.values()).isoformat())
        if note not in notes:
            notes.append(note)
    return bounds, notes


def check_retracted(
    conn: sqlite3.Connection, texts: Iterable[str], *, raws: Iterable[str], records: Iterable[str]
) -> None:
    """§9.6: output read from a corrected raw (or a retracted record) may not hold a retracted
    normalized value outside a history clause. Raises CommitRejected("retracted_value")."""
    raws, records = list(dict.fromkeys(raws)), list(dict.fromkeys(records))
    if not raws and not records:
        return
    rv: list[str] = []
    for col, ids in (("raw_id", raws), ("record_id", records)):
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            marks = ", ".join("?" for _ in chunk)
            rv += [
                r[0]
                for r in conn.execute(
                    f"SELECT retracted_value_norm FROM raw_retractions WHERE {col} IN ({marks})"  # noqa: S608
                    " AND record_id NOT LIKE 'fix:%'",  # a correction's own mark: read-side only
                    chunk,
                )
            ]
    if not rv:
        return
    structured: set[str] = set()
    plain: list[re.Pattern[str]] = []
    for v in dict.fromkeys(rv):
        got = value_tokens(v)
        if got:
            structured |= got
        elif len(v.strip()) >= 4:
            plain.append(re.compile(rf"(?<!\w){re.escape(v.strip().casefold())}(?!\w)"))
    for text in texts:
        for clause in _CLAUSE.split(text or ""):
            if not clause.strip() or _HISTORY.search(clause):
                continue
            if value_tokens(clause) & structured or any(p.search(clause.casefold()) for p in plain):
                raise CommitRejected("retracted_value")


def _write_retractions(
    ctx: CommitCtx, record_id: str, raw_ids: Sequence[str] | None = None
) -> None:
    row = ctx.conn.execute(
        "SELECT value_norm, source_raw_id FROM facts WHERE id = ?", (record_id,)
    ).fetchone()
    if row is None or row["value_norm"] is None:
        return
    raws = list(raw_ids or [])
    if not raws:
        if row["source_raw_id"]:
            raws.append(row["source_raw_id"])
        raws += [
            r[0]
            for r in ctx.conn.execute(
                "SELECT e.to_id FROM edges e JOIN raw_chunks c ON c.id = e.to_id"
                " WHERE e.from_id = ? AND e.etype = 'derived_from'",
                (record_id,),
            )
        ]
    for raw_id in dict.fromkeys(raws):
        ctx.conn.execute(
            "INSERT OR IGNORE INTO raw_retractions (raw_id, record_id, retracted_value_norm,"
            " created_at) VALUES (?, ?, ?, ?)",
            (raw_id, record_id, str(row["value_norm"]), ctx.ts),
        )


# ---------------------------------------------------------------- commit checks (§6.3, §13)


def _raw_rows(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, sqlite3.Row]:
    out = {}
    for i in dict.fromkeys(ids):
        r = conn.execute(
            "SELECT id, status, author_tier, labels, ts FROM raw_chunks WHERE id = ?", (i,)
        ).fetchone()
        if r is not None:
            out[i] = r
    return out


def _check_stable_ids(ctx: CommitCtx, sids: Iterable[StableId | dict[str, Any]]) -> list[StableId]:
    """§13.3: stable ids come only from their word or platform metadata (set by code, never by a
    model proposal)."""
    out = []
    ptier = ctx.proposal.author_tier
    for s in sids:
        sid = s if isinstance(s, StableId) else StableId.model_validate(s)
        if sid.tier == "platform":
            ok = ctx.proposal.model_id is None
        else:
            ok = AuthorTier(sid.tier) in USER_TIERS and ptier in USER_TIERS
        if not ok:
            raise CommitRejected("stable_id_tier", sid.kind)
        out.append(sid)
    return out


def _check_suppressed(
    ctx: CommitCtx,
    record: AnyRecord,
    family: str | None,
    raw_ts: str | None,
    *,
    stable_ids: Iterable[StableId] = (),
    tier: AuthorTier | None = None,
) -> None:
    """§9.6, §9.8: no re-derivation of what the user had forgotten. `forget.blocked_by` decides:
    the record's proposition (`forget.proposition`), the same proposition under each alias
    of its subject entity, and stable ids. Only their word from a source newer than the forget
    is admitted; an op with no source raw counts as their word now (`ctx.ts`), so it passes
    only at a user tier (§9.6)."""
    conn = ctx.conn
    if conn.execute("SELECT 1 FROM suppressions LIMIT 1").fetchone() is None:
        return
    from . import forget

    row = record.model_dump(mode="json")
    prop = forget.proposition(conn, record.kind.value, row)
    sids = [(sid.kind, sid.value) for sid in stable_ids]
    if prop is None and not sids:
        return
    # §9.8: a platform id (the sender id of that person's own message) from a
    # message newer than the forget binds, as a new person met today; from older raw, or with
    # no raw to date it, the suppression refuses it whatever the tier.
    platform = [(sid.kind, sid.value) for sid in stable_ids if sid.tier == "platform"]
    if platform and raw_ts is not None:
        newer = forget.blocked_by(
            conn, source_ts=raw_ts, author_tier=AuthorTier.USER, stable_ids=platform
        )
        if newer is None:
            sids = [k for k in sids if k not in platform]
            if prop is None and not sids:
                return
    keys: dict[str, Any] = {"stable_ids": sids}
    if prop is not None:
        subject, fam, val = prop
        if family is not None:
            fam = family
        aliases = [
            r[0]
            for r in conn.execute("SELECT alias_norm FROM aliases WHERE entity_id = ?", (subject,))
        ]
        keys.update(subject_id=subject, family=fam, value_norm=val, alias_norms=aliases)
    hit = forget.blocked_by(
        conn,
        source_ts=raw_ts if raw_ts is not None else ctx.ts,
        author_tier=tier if tier is not None else record.author_tier,
        **keys,
    )
    if hit is not None:
        raise CommitRejected("suppressed")


def _record_texts(record: AnyRecord) -> list[str]:
    """Texts a commit scans for retracted values. Quotes are verbatim evidence: not scanned."""
    out = [getattr(record, f) for f in _TEXT_FIELDS if isinstance(getattr(record, f, None), str)]
    if isinstance(record, WorkState):
        out += [x for x in (record.state.goal, record.state.direction, record.state.next_step) if x]
    return out


def _check_record(ctx: CommitCtx, op: ProposalOp, record: AnyRecord) -> AnyRecord:
    """Every ADD check: source usability, tier ceiling and kind rules (G11), vocabulary,
    label inheritance, suppressions and the retraction overlay. Returns the record to store."""
    conn = ctx.conn
    p = op.payload
    derived_from = list(p.get("derived_from") or [])
    members = [j["member"] for j in p.get("justifications") or []]
    src_raw = getattr(record, "source_raw_id", None)
    raw_ids = (
        ([src_raw] if src_raw else [])
        + derived_from
        + [r.id for r in op.read_set if r.src is ReadSrc.RAW]
    )
    raws = _raw_rows(conn, raw_ids + members)
    for rid in [src_raw, *derived_from]:
        if rid in raws and raws[rid]["status"] != "active":
            raise CommitRejected("unusable_source", rid)
    source_recs = [_hdr(conn, i) for i in derived_from if i not in raws]
    source_recs = [s for s in source_recs if s is not None]
    for s in source_recs:
        if s["status"] not in _USABLE:
            raise CommitRejected("unusable_source", s["id"])

    # Lineage root (§3.2): entities and records with the user's creating words are roots;
    # any other record names a raw, a derived_from source or a justification member.
    if not isinstance(record, Entity) and not (
        src_raw or derived_from or members or (record.origin_quote or "").strip()
    ):
        raise CommitRejected("no_lineage", record.id)

    # Tier (§6.2 step 4, §13): lowest first; a worker never mints the user's tier from nothing.
    # Every raw chunk the op read counts, cited or not (G11): a forwarded mail read beside
    # the user's message caps what the worker may store at the mail's tier.
    read_raw = [r.id for r in op.read_set if r.src is ReadSrc.RAW]
    src_tiers = [
        AuthorTier(raws[i]["author_tier"])
        for i in dict.fromkeys([src_raw, *derived_from, *read_raw])
        if i in raws
    ]
    src_tiers += [AuthorTier(s["author_tier"]) for s in source_recs]
    ceiling = _ceiling(ctx.proposal.author_tier, src_tiers)
    if tier_rank(record.author_tier) > tier_rank(ceiling):
        raise CommitRejected("tier", f"{record.id} above its sources")
    if isinstance(record, Lesson) and record.author_tier not in USER_TIERS:
        raise CommitRejected("tier", "lessons come only from their word")
    if isinstance(record, Criterion) and record.author_tier not in (*USER_TIERS, AuthorTier.SYSTEM):
        raise CommitRejected("tier", "criteria never from foreign text")
    if isinstance(record, Entity) and record.stable_ids:
        _check_stable_ids(ctx, record.stable_ids)
    raw_ts = raws[src_raw]["ts"] if src_raw in raws else None
    if raw_ts is None:  # the newest raw it was built from (§9.6: older than the forget?)
        raw_ts = max((raws[i]["ts"] for i in derived_from if i in raws), default=None)
    if p.get("platform") and ctx.proposal.model_id is not None:
        raise CommitRejected("stable_id_tier", "platform metadata is set by code")

    update: dict[str, Any] = {}
    # Labels: the record carries at least the OR of its sources' (§5.3, G1).
    labels = int(record.labels)
    for r in raws.values():
        labels |= int(r["labels"])
    for s in source_recs:
        labels |= int(s["labels"])
    for m in members:
        if m not in raws:
            h = _hdr(conn, m)
            if h is not None:
                labels |= int(h["labels"])
    if labels != int(record.labels):
        update["labels"] = Label(labels)

    if isinstance(record, Fact):
        if conn.execute("SELECT 1 FROM relations LIMIT 1").fetchone() is None:
            raise CommitRejected("vocabulary", "relations not seeded")
        rel = _relation(conn, record.relation)
        if rel is None:  # §4.1: a statement that fits none is stored as `other`
            update["relation"] = "other"
            rel = _relation(conn, "other")
        _check_suppressed(ctx, record, rel["family"] if rel else "other", raw_ts)
    else:
        sids = record.stable_ids if isinstance(record, Entity) else []
        _check_suppressed(ctx, record, None, raw_ts, stable_ids=sids)
    if update:
        record = record.model_copy(update=update)

    check_retracted(
        conn,
        _record_texts(record),
        raws=[i for i in raw_ids if i],
        records=[r.id for r in op.read_set if r.src is ReadSrc.RECORD]
        + [s["id"] for s in source_recs],
    )
    return record


def _add_member(ctx: CommitCtx, target_id: str, j: dict[str, Any], *, bump_target: bool) -> None:
    """One justification member (§8.1). A member change is a content change of the target."""
    conn = ctx.conn
    member, prov, group_no = j["member"], Provenance(j["provenance"]), int(j["group_no"])
    target = _hdr(conn, target_id)
    if target is None:
        raise CommitRejected("not_found", target_id)
    mrec = _hdr(conn, member)
    mraw = None if mrec is not None else _raw_rows(conn, [member]).get(member)
    if mrec is None and mraw is None:
        raise CommitRejected("member_missing", member)
    if prov is Provenance.STATED:
        mtier = AuthorTier((mrec or dict(mraw))["author_tier"])  # type: ignore[arg-type]
        if mtier not in USER_TIERS:
            raise CommitRejected("stated_not_user", member)
        if not (target.get("quote_orig") or target.get("origin_quote")):
            raise CommitRejected("stated_without_quote", target_id)
    conn.execute(
        "INSERT OR IGNORE INTO justifications (id, target_record_id, group_no, member_record_id,"
        " provenance, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (new_id(), target_id, group_no, member, prov.value, ctx.ts),
    )
    add_edge(ctx, member, target_id, EdgeType.SUPPORTS)
    mlabels = int((mrec or dict(mraw))["labels"])  # type: ignore[arg-type]
    if bump_target:
        conn.execute("UPDATE records SET labels = labels | ? WHERE id = ?", (mlabels, target_id))
        bump(ctx, [target_id])
        invalidate(ctx, [target_id])
        mark_pending_updates(ctx, [target_id])


# ---------------------------------------------------------------- op handlers


def _add_derived(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    conn = ctx.conn
    d = dict(op.payload["derived"])
    did = d.get("id") or new_id()
    refs = (
        [ReadRef.model_validate(r) for r in d["read_set"]]
        if d.get("read_set")
        else list(op.read_set)
    )
    if not refs:
        raise CommitRejected("bad_op", "a derived item needs a read set (§7.2)")
    labels = int(d.get("labels") or 0)
    recs, raws = [], []
    for ref in refs:
        if ref.src is ReadSrc.RECORD:
            r = conn.execute(
                "SELECT status, labels FROM records WHERE id = ?", (ref.id,)
            ).fetchone()
            ok = r is not None and r["status"] in _USABLE
            recs.append(ref.id)
        elif ref.src is ReadSrc.RAW:
            r = conn.execute(
                "SELECT status, labels FROM raw_chunks WHERE id = ?", (ref.id,)
            ).fetchone()
            ok = r is not None and r["status"] == "active"
            raws.append(ref.id)
        else:
            r = conn.execute("SELECT valid, labels FROM derived WHERE id = ?", (ref.id,)).fetchone()
            ok = r is not None and r["valid"] == 1
        if not ok:
            raise CommitRejected("unusable_source", ref.id)  # §6.4: built from usable items only
        labels |= int(r["labels"])
    check_retracted(conn, [d["content"]], raws=raws, records=recs)
    existed = conn.execute("SELECT 1 FROM derived WHERE id = ?", (did,)).fetchone() is not None
    read_set = [
        [r.id, r.version] if r.src is ReadSrc.RECORD else [r.id, r.version, r.src.value]
        for r in refs
    ]
    conn.execute(
        "INSERT INTO derived (id, dkind, dkey, content, read_set, labels, built_at, valid,"
        " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)"
        " ON CONFLICT (id) DO UPDATE SET dkind = excluded.dkind, dkey = excluded.dkey,"
        " content = excluded.content, read_set = excluded.read_set, labels = excluded.labels,"
        " built_at = excluded.built_at, valid = 1, rev = derived.rev + 1,"
        " updated_at = excluded.updated_at",
        (
            did,
            d["dkind"],
            d.get("dkey"),
            d["content"],
            json.dumps(read_set),
            labels,
            d.get("built_at") or ctx.ts,
            ctx.ts,
            ctx.ts,
        ),
    )
    conn.execute("DELETE FROM read_deps WHERE holder_kind = 'derived' AND holder_id = ?", (did,))
    conn.executemany(
        "INSERT OR REPLACE INTO read_deps (holder_kind, holder_id, item_id, version)"
        " VALUES ('derived', ?, ?, ?)",
        [(did, r.id, r.version) for r in refs],
    )
    if existed:
        invalidate(ctx, [did])
    return [did]


def _group_peers(ctx: CommitCtx, op: ProposalOp) -> tuple[list[ProposalOp], set[str]]:
    """The record ADDs of `op`'s commit group before it (ops) and after it (record ids).
    A group commits as one (§6.3), so its order must not matter: a fact whose reason
    names a fact listed after it waits for that member's own ADD."""
    ops = ctx.proposal.ops
    i = next((k for k, o in enumerate(ops) if o is op), None)
    if i is None:  # a trusted path applying an op outside its proposal: no peers
        return [], set()

    def peer(o: ProposalOp) -> bool:
        return o.group == op.group and OpKind(o.op) is OpKind.ADD and "record" in o.payload

    later = {str(o.payload["record"].get("id")) for o in ops[i + 1 :] if peer(o)}
    return [o for o in ops[:i] if peer(o)], later


def _add_deferred_members(ctx: CommitCtx, member_id: str, earlier: list[ProposalOp]) -> None:
    """The justifications of earlier ADDs in the group that wait for `member_id` (new in this
    op): the same checks as any member (`_add_member`), and the target takes the member's
    labels, as `_check_record` gives a target the labels of members that already exist."""
    for prev in earlier:
        target = prev.payload["record"].get("id")
        for j in prev.payload.get("justifications") or []:
            if j["member"] != member_id or target is None:
                continue
            _add_member(ctx, target, j, bump_target=False)
            ctx.conn.execute(
                "UPDATE records SET labels = labels | (SELECT labels FROM records WHERE id = ?)"
                " WHERE id = ?",
                (member_id, target),
            )


def _add(ctx: CommitCtx, op: ProposalOp, *, rule: bool | None = None) -> list[str]:
    p = op.payload
    if "derived" in p:
        return _add_derived(ctx, op)
    if "record" not in p:
        raise CommitRejected("bad_op", "add needs a record or a derived item")
    record = _check_record(ctx, op, parse_record(p["record"], ts=ctx.ts))
    earlier, later = _group_peers(ctx, op)
    is_new = _hdr(ctx.conn, record.id) is None
    _insert(ctx, record, platform=bool(p.get("platform")), aliases=p.get("aliases") or [])
    for d in p.get("derived_from") or []:
        add_edge(ctx, record.id, d, EdgeType.DERIVED_FROM)
    for m in p.get("mentions") or []:
        add_edge(ctx, record.id, m, EdgeType.MENTIONS)
    for j in p.get("justifications") or []:
        if j["member"] in later and _hdr(ctx.conn, j["member"]) is None:
            continue  # its own op adds it later in this group, then this justification
        _add_member(ctx, record.id, j, bump_target=False)
    if is_new:
        _add_deferred_members(ctx, record.id, earlier)
    touched = [record.id]
    if isinstance(record, Fact) and (p.get("rule_match", True) if rule is None else rule):
        for old in rule_match(ctx.conn, record.id):
            supersede(ctx, old, record.id)
            touched.append(old)
        touched += [t for t in settle_order(ctx, record.id) if t not in touched]
    for mid in p.get("model_matches") or []:
        if mid not in touched:  # §8.3: a model match on non-identical keys gives review
            _set_review(ctx, mid, True)
            touched.append(mid)
    return touched


def _may_edit(ptier: AuthorTier, record_tier: AuthorTier) -> bool:
    return ptier in USER_TIERS or tier_rank(ptier) >= tier_rank(record_tier)


def _update(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    conn = ctx.conn
    target = op.target or ""
    h = conn.execute("SELECT * FROM records WHERE id = ?", (target,)).fetchone()
    if h is None:
        raise CommitRejected("not_found", target)
    kind = RecordKind(h["kind"])
    table = _KIND_TABLE[kind]
    p = op.payload
    ptier = ctx.proposal.author_tier
    rtier = AuthorTier(h["author_tier"])
    fields = dict(p.get("fields") or {})
    kcols = _kind_cols(kind)
    head_upd: dict[str, Any] = {}
    kind_upd: dict[str, Any] = {}
    for col, v in fields.items():
        if col in _UPDATABLE_HEADER:
            head_upd[col] = v
        elif col in kcols and col != "id":
            kind_upd[col] = _sql(table, col, v)
        else:
            raise CommitRejected("unknown_field", col)
    cur = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (target,)).fetchone()  # noqa: S608
    kind_upd = {c: v for c, v in kind_upd.items() if cur[c] != v}
    for col in list(head_upd):
        v = head_upd[col]
        v = int(v) if isinstance(v, bool) else v
        v = int(v) if col == "labels" else v
        if h[col] == v:
            del head_upd[col]
        else:
            head_upd[col] = v

    content = {c for c in kind_upd if c not in _NON_BEARING_KIND.get(table, ())}
    if (
        "labels" in head_upd
        and head_upd["labels"] & h["labels"] != h["labels"]
        and ptier not in USER_TIERS
    ):
        raise CommitRejected("labels", "only his «исправь» removes a label (§5.3)")
    if "author_tier" in head_upd:
        new_tier = AuthorTier(head_upd["author_tier"])
        limit = rtier if ptier is AuthorTier.SYSTEM else ptier
        if tier_rank(new_tier) > tier_rank(limit):
            raise CommitRejected("tier", "author tier above the proposal's")
    edits_content = bool(content) or "author_tier" in head_upd
    benign = set(head_upd) - {"author_tier"}
    if edits_content and not _may_edit(ptier, rtier):
        raise CommitRejected("tier", f"{ptier.value} may not edit {rtier.value}")
    foreign = ptier in (AuthorTier.TOOL, AuthorTier.OTHER)
    if (benign or kind_upd) and not edits_content and foreign and not _may_edit(ptier, rtier):
        raise CommitRejected("tier", "foreign text may not tune his records")

    if kind in (RecordKind.FACT, RecordKind.PREFERENCE) and content:
        merged = get_record(conn, target)
        upd = {}
        for c, v in kind_upd.items():
            upd[c] = json.loads(v) if (v is not None and c in _JSON_COLS.get(table, ())) else v
        merged = merged.model_copy(update=upd)  # type: ignore[union-attr]
        if "relation" in kind_upd and _relation(conn, kind_upd["relation"]) is None:
            kind_upd["relation"] = "other"
        rel = _relation(conn, kind_upd.get("relation", cur["relation"]))
        _check_suppressed(ctx, merged, rel["family"] if rel else "other", None)  # type: ignore[arg-type]
        check_retracted(
            conn,
            _record_texts(merged),  # type: ignore[arg-type]
            raws=[x for x in [cur["source_raw_id"]] if x]
            + [r.id for r in op.read_set if r.src is ReadSrc.RAW],
            records=[r.id for r in op.read_set if r.src is ReadSrc.RECORD],
        )

    touched_entity_tokens: list[str] = []
    if kind_upd:
        sets = ", ".join(f"{c} = ?" for c in kind_upd)
        conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*kind_upd.values(), target])  # noqa: S608
    if head_upd:
        sets = ", ".join(f"{c} = ?" for c in head_upd)
        conn.execute(
            f"UPDATE records SET {sets}, updated_at = ? WHERE id = ?",  # noqa: S608
            [*head_upd.values(), ctx.ts, target],
        )
    bearing = bool(content) or bool(set(head_upd) & _BEARING_HEADER)

    if kind is RecordKind.ENTITY:
        if "display_name" in kind_upd:
            touched_entity_tokens += [cur["display_name"], kind_upd["display_name"]]
            entities.put_alias(
                conn,
                ctx.ts,
                target,
                kind_upd["display_name"],
                AliasKind.NAME,
                tier=ptier,
                confirmed=ptier in USER_TIERS,
                with_translit=True,
            )
        platform = bool(p.get("platform"))
        if platform and ctx.proposal.model_id is not None:
            raise CommitRejected("stable_id_tier", "platform metadata is set by code")
        for a in p.get("add_aliases") or []:
            if entities.bind_in(
                ctx,
                target,
                a["alias"],
                AliasKind(a.get("kind") or "nickname"),
                tier=ptier,
                platform=platform,
                touch=False,
            ):
                touched_entity_tokens.append(a["alias"])
        for text in p.get("remove_aliases") or []:
            if ptier not in USER_TIERS:
                raise CommitRejected("tier", "only their word removes an alias")
            norm = entities.normalize_alias(text)
            gone = conn.execute(
                "SELECT fts_rowid FROM aliases WHERE entity_id = ? AND alias_norm = ?",
                (target, norm),
            ).fetchall()
            for g in gone:
                conn.execute("DELETE FROM fts_names WHERE rowid = ?", (g[0],))
            if gone:
                conn.execute(
                    "DELETE FROM aliases WHERE entity_id = ? AND alias_norm = ?", (target, norm)
                )
                touched_entity_tokens.append(text)
        if p.get("add_stable_ids"):
            new = _check_stable_ids(ctx, p["add_stable_ids"])
            # §9.8: a forgotten person's ids do not come back onto another entity
            _check_suppressed(ctx, get_record(conn, target), None, None, stable_ids=new, tier=ptier)
            have = json.loads(
                conn.execute("SELECT stable_ids FROM entities WHERE id = ?", (target,)).fetchone()[
                    0
                ]
            )
            keys = {(s["kind"], s["value"]) for s in have}
            added = [s.model_dump(mode="json") for s in new if (s.kind, s.value) not in keys]
            if added:
                conn.execute(
                    "UPDATE entities SET stable_ids = ? WHERE id = ?",
                    (json.dumps(have + added, ensure_ascii=False, separators=(",", ":")), target),
                )
                touched_entity_tokens += [s["value"] for s in added]
        semantic_cols = content - {"display_name", "stable_ids"}
        if touched_entity_tokens and not semantic_cols and not (set(head_upd) & _BEARING_HEADER):
            touch_entity(ctx, target, tokens=touched_entity_tokens)
            return [target]
        if touched_entity_tokens:
            bearing = True

    if not bearing and not head_upd and not kind_upd:
        return []
    if bearing:
        bump(ctx, [target])
    if kind_upd or touched_entity_tokens:
        index(conn, target)
    if bearing:
        invalidate(ctx, [target])
        mark_pending_updates(ctx, [target])
    return [target]


def _supersede_op(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    old = op.target
    p = op.payload
    if not old:
        raise CommitRejected("bad_op", "supersede needs a target")
    touched: list[str] = []
    if "record" in p:
        sub = op.model_copy(update={"op": OpKind.ADD})
        touched = _add(ctx, sub, rule=False)
        new = touched[0]
    else:
        new = p.get("by")
    if not new:
        raise CommitRejected("bad_op", "supersede needs `by` or a record")
    if ctx.proposal.model_id is not None and old not in rule_match(ctx.conn, new):
        _set_review(ctx, old, True)  # §8.3: a model's match on non-identical keys → review
        return [*touched, old]
    supersede(ctx, old, new)
    if p.get("fix"):
        _write_retractions(ctx, old)
    return [*touched, new, old]


def _retract(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    target = op.target or ""
    h = _hdr(ctx.conn, target)
    if h is None:
        raise CommitRejected("not_found", target)
    ptier = ctx.proposal.author_tier
    if ptier not in (*USER_TIERS, AuthorTier.USER_ROOM) or not may_supersede(
        ptier, AuthorTier(h["author_tier"]), same_speaker=True
    ):
        raise CommitRejected("tier", "a retraction is their word")
    set_status(ctx, target, Status.RETRACTED)
    _write_retractions(ctx, target, op.payload.get("raw_ids"))
    _retire_value(ctx, target)
    return [target]


def _flag(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    target = op.target or ""
    p = op.payload
    ptier = ctx.proposal.author_tier
    if ptier not in (*USER_TIERS, AuthorTier.USER_ROOM, AuthorTier.SYSTEM):
        raise CommitRejected("tier", "foreign text may not flag records")
    if "review" in p:
        _set_review(ctx, target, bool(p["review"]))
    if "status" in p:
        st = Status(p["status"])
        cur = ctx.conn.execute("SELECT status FROM records WHERE id = ?", (target,)).fetchone()
        if cur is None:
            raise CommitRejected("not_found", target)
        allowed = {Status.NEEDS_REVIEW, Status.ARCHIVED, Status.ACTIVE}
        if st not in allowed or ptier is AuthorTier.USER_ROOM:
            raise CommitRejected("bad_op", f"flag cannot set {st.value}")
        if (
            st is Status.ACTIVE
            and cur[0] not in ("needs_review", "archived")
            and ptier not in USER_TIERS
        ):
            raise CommitRejected("tier", "only their word revives a retired record")
        set_status(ctx, target, st)
    return [target]


def _link(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    p = op.payload
    if "justification" in p:
        j = dict(p["justification"])
        target = j.get("target") or op.target
        if not target:
            raise CommitRejected("bad_op", "justification needs a target")
        if not _may_edit(
            ctx.proposal.author_tier, AuthorTier(_hdr(ctx.conn, target)["author_tier"])
        ):  # type: ignore[index]
            raise CommitRejected("tier", "justification of a higher-tier record")
        _add_member(ctx, target, j, bump_target=True)
        return [target]
    if "alias" in p:
        a = dict(p["alias"])
        entity_id = a.get("entity_id") or op.target
        platform = bool(p.get("platform"))
        if platform and ctx.proposal.model_id is not None:
            raise CommitRejected("stable_id_tier", "platform metadata is set by code")
        changed = entities.bind_in(
            ctx,
            entity_id,
            a["alias"],
            AliasKind(a.get("kind") or "nickname"),
            tier=ctx.proposal.author_tier,
            platform=platform,
        )
        return [entity_id] if changed else []
    if {"from_id", "to_id", "etype"} <= set(p):
        add_edge(ctx, p["from_id"], p["to_id"], EdgeType(p["etype"]))
        return []
    raise CommitRejected("bad_op", "link needs a justification, an alias or an edge")


def _shared_stable_id(conn: sqlite3.Connection, a: str, b: str) -> bool:
    def keys(eid: str) -> set[tuple[str, str]]:
        r = conn.execute("SELECT stable_ids FROM entities WHERE id = ?", (eid,)).fetchone()
        return {(s["kind"], s["value"]) for s in json.loads(r[0])} if r else set()

    return bool(keys(a) & keys(b))


def _merge(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    keep, drop = op.target or "", op.payload.get("drop") or ""
    hk, hd = _hdr(ctx.conn, keep), _hdr(ctx.conn, drop)
    if hk is None or hd is None:
        raise CommitRejected("not_found", keep if hk is None else drop)
    if hk["kind"] != hd["kind"]:
        raise CommitRejected("bad_op", "merge needs two records of one kind")
    ptier = ctx.proposal.author_tier
    if hk["kind"] == RecordKind.ENTITY.value:
        ok = ptier in USER_TIERS or (
            ptier is AuthorTier.SYSTEM and _shared_stable_id(ctx.conn, keep, drop)
        )
        if not ok:
            raise CommitRejected("tier", "people merge on their word or a shared stable id")
        return entities.merge_in(ctx, keep, drop)
    if not (ptier in USER_TIERS or ptier is AuthorTier.SYSTEM) or not _may_supersede_rows(hk, hd):
        raise CommitRejected("tier", "merge")
    ctx.conn.execute("UPDATE records SET labels = labels | ? WHERE id = ?", (hd["labels"], keep))
    if int(hd["labels"]) | int(hk["labels"]) != int(hk["labels"]):
        bump(ctx, [keep])
        invalidate(ctx, [keep])
        mark_pending_updates(ctx, [keep])
    set_status(ctx, drop, Status.SUPERSEDED)
    add_edge(ctx, keep, drop, EdgeType.SUPERSEDES)
    return [keep, drop]


def _split(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    if ctx.proposal.author_tier not in USER_TIERS:
        raise CommitRejected("tier", "a split is his correction")
    p = op.payload
    _, touched = entities.split_in(
        ctx,
        op.target or "",
        list(p.get("record_ids") or []),
        display_name=p.get("display_name"),
        new_id=p.get("new_id"),
    )
    return touched


def _forget(ctx: CommitCtx, op: ProposalOp) -> list[str]:
    """Refused: a status flip here would keep the text, quote, FTS row and raw with no
    trash, journal line or purge job. Every forget goes through `forget.stop_use`
    (api.forget)."""
    if ctx.proposal.author_tier not in USER_TIERS:
        raise CommitRejected("tier", "«забудь» is their word")
    raise CommitRejected("bad_op", "a forget goes through forget.stop_use")


_HANDLERS: dict[OpKind, Callable[[CommitCtx, ProposalOp], list[str]]] = {
    OpKind.ADD: _add,
    OpKind.UPDATE: _update,
    OpKind.SUPERSEDE: _supersede_op,
    OpKind.RETRACT: _retract,
    OpKind.FLAG: _flag,
    OpKind.LINK: _link,
    OpKind.MERGE: _merge,
    OpKind.SPLIT: _split,
    OpKind.FORGET: _forget,
}


def apply_op(ctx: CommitCtx, op: ProposalOp) -> Sequence[str]:
    """Apply one delta op inside `store.commit`'s transaction. Checks suppressions,
    retracted values, label monotonicity, tiers and the relation vocabulary (§6.3);
    returns the item ids touched (for the events row and pending updates)."""
    return _HANDLERS[OpKind(op.op)](ctx, op)


# ---------------------------------------------------------------- trusted code paths


def run_in_commit(
    conn: sqlite3.Connection,
    fn: Callable[[CommitCtx], tuple[Any, list[str]]],
    *,
    op_kind: OpKind,
    target: str,
    event_kind: EventKind = EventKind.WRITE,
    author_tier: AuthorTier = AuthorTier.SYSTEM,
) -> Any:
    """Run a trusted ledger change (`fn(ctx) -> (result, touched ids)`) as its own commit, or
    inside the caller's open transaction at the current version."""
    if conn.in_transaction:
        ctx = CommitCtx(
            conn=conn,
            version=store.memory_version(conn),
            ts=store.now(),
            proposal=Proposal(ops=[], author_tier=author_tier),
        )
        return fn(ctx)[0]
    box: dict[str, Any] = {}

    def applier(ctx: CommitCtx, _op: ProposalOp) -> list[str]:
        box["result"], touched = fn(ctx)
        return touched

    proposal = Proposal(
        ops=[ProposalOp(op=op_kind, target=target, group=target)], author_tier=author_tier
    )
    store.commit(conn, proposal, applier, event_kind=event_kind)
    return box.get("result")
