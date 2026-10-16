"""Plain-SQL seeding for the store tests (store, backup, pre-send, perf).

The ledger's real applier (ledger) and forget pipeline (forget) live elsewhere; these helpers
write rows the way §3.2 describes them, so the store's guarantees can be tested on their own.
`ref_apply` is a reference op applier for `store.commit` (tests only)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from koob.memory import store
from koob.memory.model import (
    ForgetStage,
    JournalLine,
    OpKind,
    Proposal,
    ProposalOp,
    ReadRef,
    ReadSrc,
)

T = "2026-10-12T09:00:00.000000Z"


def hmac_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def add_raw(conn, rid: str, text: str, *, labels: int = 0, status: str = "active") -> int:
    conn.execute(
        "INSERT INTO raw_chunks (id, status, source_kind, author_tier, ts, text, content_hmac,"
        " labels, policy_version, created_at, updated_at)"
        " VALUES (?, ?, 'chat', 'user', ?, ?, ?, ?, 0, ?, ?)",
        (rid, status, T, text, hmac_of(text), labels, T, T),
    )
    fts = conn.execute("SELECT fts_rowid FROM raw_chunks WHERE id = ?", (rid,)).fetchone()[0]
    if status == "active":
        conn.execute("INSERT INTO fts_raw (rowid, text) VALUES (?, ?)", (fts, text))
    return fts


def add_fact(
    conn,
    rid: str,
    text: str,
    *,
    raw: str | None = None,
    version: int = 1,
    status: str = "active",
    labels: int = 0,
    quote: str | None = None,
    origin_quote: str | None = None,
) -> int:
    conn.execute(
        "INSERT INTO records (id, kind, version, status, labels, author_tier, recorded_at,"
        " origin_quote, created_at, updated_at) VALUES (?, 'fact', ?, ?, ?, 'user', ?, ?, ?, ?)",
        (rid, version, status, labels, T, origin_quote, T, T),
    )
    conn.execute(
        "INSERT INTO facts (id, subject_id, relation, value_text, text_en, quote_orig, fkind,"
        " source_raw_id) VALUES (?, 'e-sasha', 'birthday', ?, ?, ?, 'state', ?)",
        (rid, text, text, quote, raw),
    )
    if raw is not None:
        conn.execute(
            "INSERT INTO edges (from_id, to_id, etype, created_at) VALUES (?, ?, 'derived_from', ?)",
            (rid, raw, T),
        )
    fts = conn.execute("SELECT fts_rowid FROM records WHERE id = ?", (rid,)).fetchone()[0]
    if status in ("active", "archived"):
        conn.execute(
            "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, '', '')",
            (fts, text),
        )
    return fts


def add_derived(conn, did: str, refs: list[ReadRef], content: str, *, labels: int = 0) -> None:
    conn.execute(
        "INSERT INTO derived (id, dkind, dkey, content, read_set, labels, built_at, valid,"
        " created_at, updated_at) VALUES (?, 'summary', 'day', ?, ?, ?, ?, 1, ?, ?)",
        (did, content, store.dump_read_set(refs), labels, T, T, T),
    )
    store.set_read_deps(conn, "derived", did, refs)


def fact_text(conn, rid: str) -> str | None:
    row = conn.execute("SELECT text_en FROM facts WHERE id = ?", (rid,)).fetchone()
    return row[0] if row else None


def status(conn, rid: str) -> str | None:
    row = conn.execute("SELECT status FROM records WHERE id = ?", (rid,)).fetchone()
    return row[0] if row else None


def version(conn, rid: str) -> int:
    return conn.execute("SELECT version FROM records WHERE id = ?", (rid,)).fetchone()[0]


def ref(conn, rid: str) -> ReadRef:
    return ReadRef(id=rid, version=version(conn, rid))


def raw_ref(conn, rid: str) -> ReadRef:
    rev = conn.execute("SELECT rev FROM raw_chunks WHERE id = ?", (rid,)).fetchone()[0]
    return ReadRef(id=rid, version=rev, src=ReadSrc.RAW)


# ---------------------------------------------------------------- a reference applier


def ref_apply(ctx: store.CommitCtx, op: ProposalOp):
    """ADD: payload {id, text, raw?, labels?}; UPDATE: payload {text}; both invalidate
    the derived holders of what they change (§7.3) and keep labels ⊇ the raw's (§5.3)."""
    c = ctx.conn
    if op.op is OpKind.ADD:
        p = op.payload
        labels = int(p.get("labels", 0))
        if p.get("raw"):
            labels |= c.execute(
                "SELECT labels FROM raw_chunks WHERE id = ?", (p["raw"],)
            ).fetchone()[0]
        add_fact(c, p["id"], p["text"], raw=p.get("raw"), version=ctx.version, labels=labels)
        return [p["id"]]
    if op.op is OpKind.UPDATE:
        rid = op.target
        c.execute(
            "UPDATE facts SET text_en = ?, value_text = ? WHERE id = ?",
            (op.payload["text"], op.payload["text"], rid),
        )
        c.execute(
            "UPDATE records SET version = ?, updated_at = ? WHERE id = ?",
            (ctx.version, ctx.ts, rid),
        )
        fts = c.execute("SELECT fts_rowid FROM records WHERE id = ?", (rid,)).fetchone()[0]
        c.execute("DELETE FROM fts_records WHERE rowid = ?", (fts,))
        c.execute(
            "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, '', '')",
            (fts, op.payload["text"]),
        )
        c.execute(
            "UPDATE derived SET valid = 0 WHERE id IN (SELECT holder_id FROM read_deps"
            " WHERE holder_kind = 'derived' AND item_id = ?)",
            (rid,),
        )
        return [rid]
    raise AssertionError(f"reference applier has no {op.op}")


def add_op(rid: str, text: str, *, raw: str | None = None, group: str | None = None) -> Proposal:
    return Proposal(
        ops=[
            ProposalOp(
                op=OpKind.ADD, group=group or rid, payload={"id": rid, "text": text, "raw": raw}
            )
        ]
    )


def fix_op(conn, rid: str, text: str) -> Proposal:
    return Proposal(
        ops=[
            ProposalOp(
                op=OpKind.UPDATE,
                target=rid,
                group=rid,
                read_set=[ref(conn, rid)],
                payload={"text": text},
            )
        ]
    )


# ---------------------------------------------------------------- the forget side


def journal_line(
    forget_id: str,
    stage: ForgetStage,
    *,
    record_ids: Iterable[str] = (),
    raw_hmacs: Iterable[str] = (),
    memory_version: int,
    ts: str = T,
) -> JournalLine:
    return JournalLine(
        ts=ts,
        stage=stage,
        forget_id=forget_id,
        record_ids=list(record_ids),
        raw_hmacs=list(raw_hmacs),
        memory_version=memory_version,
    )


def append(journal: Path, *lines: JournalLine) -> None:
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.model_dump_json() + "\n")


def files_containing(root: Path, needle: str) -> list[str]:
    """Every file under root whose bytes contain needle (UTF-8), for the G5 byte scan."""
    hits = []
    data = needle.encode()
    for p in sorted(root.rglob("*")):
        if p.is_file() and data in p.read_bytes():
            hits.append(str(p.relative_to(root)))
    return hits


def dump(obj) -> str:
    return json.dumps(obj, sort_keys=True)
