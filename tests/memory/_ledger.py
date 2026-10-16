"""Helpers for the ledger tests: a seeded store and short record builders.

Not a test module. Every write goes through `store.commit` with `ledger.apply_op`, the way
writers and workers commit (§6.3)."""

from __future__ import annotations

import itertools
from typing import Any

from koob.memory import ledger, store
from koob.memory.model import (
    AuthorTier,
    Entity,
    EventKind,
    Fact,
    FKind,
    OpKind,
    Polarity,
    Preference,
    Proposal,
    ProposalOp,
    ReadRef,
)

_clock = itertools.count(1)


def ts() -> str:
    """A strictly increasing recorded_at, so «later» is unambiguous in tests."""
    n = next(_clock)
    return f"2026-10-12T{n // 3600 % 24:02d}:{n // 60 % 60:02d}:{n % 60:02d}.000000Z"


def open_seeded(tmp_path):
    conn = store.open(tmp_path / "memory" / "memory.db", home=tmp_path / "home")
    ledger.seed_relations(conn)
    return conn


def op(
    kind: OpKind,
    *,
    target: str | None = None,
    group: str = "g",
    read_set: list[ReadRef] | None = None,
    **payload: Any,
) -> ProposalOp:
    return ProposalOp(op=kind, target=target, group=group, read_set=read_set or [], payload=payload)


def commit(
    conn,
    *ops: ProposalOp,
    tier: AuthorTier = AuthorTier.USER,
    model_id: str | None = None,
    event_kind: EventKind = EventKind.WRITE,
) -> store.CommitResult:
    proposal = Proposal(ops=list(ops), author_tier=tier, model_id=model_id)
    return store.commit(conn, proposal, ledger.apply_op, event_kind=event_kind)


def add(conn, record, *, tier: AuthorTier | None = None, model_id: str | None = None, **payload):
    """Commit one ADD op for `record`; the proposal tier defaults to the record's tier."""
    body = {"record": record.model_dump(mode="json"), **payload}
    return commit(
        conn,
        op(OpKind.ADD, group=getattr(record, "subject_id", record.id), **body),
        tier=tier or record.author_tier,
        model_id=model_id,
    )


def entity(eid: str, name: str, *, tier=AuthorTier.USER, etype="person", **kw) -> Entity:
    return Entity(id=eid, etype=etype, display_name=name, author_tier=tier, recorded_at=ts(), **kw)


def fact(
    fid: str,
    subject: str,
    relation: str,
    value: str | None,
    *,
    tier=AuthorTier.USER,
    fkind=FKind.STATE,
    polarity=Polarity.AFFIRMED,
    text: str | None = None,
    **kw,
) -> Fact:
    if not kw.get("source_raw_id"):  # §3.2 F9: a root needs the user's creating words
        kw.setdefault("origin_quote", text or f"{subject} {relation} {value}")
    return Fact(
        id=fid,
        subject_id=subject,
        relation=relation,
        value_text=value,
        value_norm=value.casefold() if value is not None else None,
        text_en=text or f"{subject} {relation} {value}",
        fkind=fkind,
        polarity=polarity,
        author_tier=tier,
        recorded_at=ts(),
        **kw,
    )


def pref(pid: str, subject: str, relation: str, value: str, **kw) -> Preference:
    kw.setdefault("applicability", {"general": True})
    kw.setdefault("fkind", FKind.OPINION)
    base = fact(pid, subject, relation, value, **kw)
    return Preference(**{**base.model_dump(), "kind": "preference"})


def row(conn, rid: str):
    return conn.execute(
        "SELECT r.status, r.version, r.review, r.labels, r.author_tier FROM records r WHERE r.id = ?",
        (rid,),
    ).fetchone()


def status(conn, rid: str) -> str:
    return row(conn, rid)["status"]


def version(conn, rid: str) -> int:
    return row(conn, rid)["version"]


def edges(conn, etype: str) -> set[tuple[str, str]]:
    return {
        (r[0], r[1])
        for r in conn.execute("SELECT from_id, to_id FROM edges WHERE etype = ?", (etype,))
    }


def raw(conn, rid: str, text: str, *, tier="user", ts_: str = "2026-10-01T10:00:00Z", labels=0):
    conn.execute(
        "INSERT INTO raw_chunks (id, source_kind, author_tier, ts, text, content_hmac, labels,"
        " policy_version, created_at, updated_at) VALUES (?, 'chat', ?, ?, ?, ?, ?, 0, ?, ?)",
        (rid, tier, ts_, text, f"h-{rid}", labels, ts_, ts_),
    )


def served(conn, thread: str, item: str, ver: int, via: str = "basket"):
    conn.execute(
        "INSERT INTO served (thread_id, item_id, item_kind, version, via, created_at)"
        " VALUES (?, ?, 'fact', ?, ?, 't')",
        (thread, item, ver, via),
    )
