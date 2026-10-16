"""Helpers for the race and failure suite (memory spec §19.1).

Written from the spec against the public facade (`memory.api`) and the public signatures of
`store`, `backup`, `ledger.insert_record` and `intentions.arm`; never from the implementations.

Seeding goes through `store.commit` + `ledger.insert_record` (records) or plain SQL on the
contract schema (`migrations/memory/0001_core.sql`) for rows that have no writer in the
facade yet: raw chunks, served rows, derived rows with their `read_deps`, justifications.
Threads each open their own connection; ordering is forced with barriers and events, never
with long sleeps.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from koob.memory import ledger, store
from koob.memory.clock import iso
from koob.memory.model import (
    AnyRecord,
    AuthorTier,
    Entity,
    Fact,
    FKind,
    OpKind,
    Proposal,
    ProposalOp,
    ReadRef,
    ReadSrc,
    StableId,
    TurnCtx,
    TurnOrigin,
)

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
TS0 = iso(T0)
WAIT = 5.0  # seconds a barrier or event may wait before the test fails (never slept)

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(name: str) -> str:
    """A deterministic ULID-shaped id (26 Crockford chars) for a test name."""
    digest = int.from_bytes(hashlib.sha256(name.encode()).digest()[:16], "big")
    body = []
    for _ in range(22):
        digest, rem = divmod(digest, 32)
        body.append(_CROCKFORD[rem])
    return "01JR" + "".join(body)


def handle(record_id: str) -> str:
    """The brain's handle for an item (§10.2 step 6, §7.6 update format)."""
    return f"m:{record_id}"


# ---------------------------------------------------------------- connections


def open_db(path: Path) -> Any:
    """A fresh connection to an existing memory.db (migrations are idempotent)."""
    return store.open(path, home=path.parent.parent / "home")


def journal_path(db_path: Path) -> Path:
    """§3.1: `data/memory/journal/forget.jsonl` beside memory.db."""
    return db_path.parent / "journal" / "forget.jsonl"


# ---------------------------------------------------------------- turns


def owner_turn(
    words: str,
    *,
    n: int = 1,
    thread: str = "thread-owner",
    session: str = "session-owner",
    ts: str = TS0,
    **flags: Any,
) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id=thread,
        session_id=session,
        origin=TurnOrigin.OWNER,
        owner_words=words,
        message_id=f"msg-{n}",
        clock=ts,
        **flags,
    )


# ---------------------------------------------------------------- records


def person(name: str, *, tg_user_id: str | None = None) -> Entity:
    stable = [StableId(kind="tg_user_id", value=tg_user_id, tier="platform")] if tg_user_id else []
    return Entity(
        id=ulid(f"entity:{name}"),
        author_tier=AuthorTier.USER,
        recorded_at=TS0,
        labels_final=True,
        etype="person",
        display_name=name,
        stable_ids=stable,
    )


def fact(
    key: str,
    subject: Entity,
    relation: str,
    value: str,
    text_en: str,
    quote: str,
    *,
    tier: AuthorTier = AuthorTier.USER,
    speaker: Entity | None = None,
    source_raw_id: str | None = None,
    value_norm: str | None = None,
    fkind: FKind = FKind.STATE,
) -> Fact:
    return Fact(
        id=ulid(f"fact:{key}"),
        author_tier=tier,
        author_entity_id=speaker.id if speaker else None,
        recorded_at=TS0,
        labels_final=True,
        subject_id=subject.id,
        relation=relation,
        value_text=value,
        value_norm=value_norm or value.casefold(),
        text_en=text_en,
        quote_orig=quote,
        lang_orig="ru",
        fkind=fkind,
        speaker_entity_id=speaker.id if speaker else None,
        source_raw_id=source_raw_id,
    )


def seed(
    conn: Any,
    *records: AnyRecord,
    extra: Callable[[Any], None] | None = None,
    tier: AuthorTier = AuthorTier.USER,
) -> int:
    """Insert records through the ledger in one commit; `extra(sql_conn)` runs inside the same
    transaction (justifications, lineage edges). Returns the memory_version after it."""
    by_id = {r.id: r for r in records}
    ops = [ProposalOp(op=OpKind.ADD, group="seed", payload={"seed_id": r.id}) for r in records]
    done: list[bool] = []

    def apply(ctx: store.CommitCtx, op: ProposalOp) -> list[str]:
        rid = ledger.insert_record(ctx, by_id[op.payload["seed_id"]])
        if extra is not None and op is ops[-1] and not done:
            extra(ctx.conn)
            done.append(True)
        return [rid]

    res = store.commit(conn, Proposal(ops=ops, author_tier=tier), apply)
    assert res.ok, res.stale
    return store.memory_version(conn)


def version_of(conn: Any, record_id: str) -> int:
    return conn.execute("SELECT version FROM records WHERE id = ?", (record_id,)).fetchone()[0]


def status_of(conn: Any, record_id: str) -> str:
    return conn.execute("SELECT status FROM records WHERE id = ?", (record_id,)).fetchone()[0]


def ref(conn: Any, record_id: str) -> ReadRef:
    return ReadRef(id=record_id, version=version_of(conn, record_id))


def usable_slot(conn: Any, subject_id: str, relation: str) -> list[tuple[str, str | None]]:
    """(id, value_text) of the usable facts in a slot."""
    return conn.execute(
        "SELECT f.id, f.value_text FROM facts f JOIN records r ON r.id = f.id"
        " WHERE f.subject_id = ? AND f.relation = ? AND r.status IN ('active', 'archived')"
        " ORDER BY r.recorded_at",
        (subject_id, relation),
    ).fetchall()


def usable_texts(conn: Any) -> list[str]:
    """Every text column of every usable fact (what any channel could build from)."""
    rows = conn.execute(
        "SELECT f.text_en, coalesce(f.value_text, ''), coalesce(f.quote_orig, '')"
        " FROM facts f JOIN records r ON r.id = f.id WHERE r.status IN ('active', 'archived')"
    ).fetchall()
    return [" | ".join(map(str, row)) for row in rows]


def valid_derived_texts(conn: Any) -> list[str]:
    return [r[0] for r in conn.execute("SELECT content FROM derived WHERE valid = 1").fetchall()]


# ---------------------------------------------------------------- contract-schema rows


def bare_fact(conn: Any, rid: str, value: str, *, subject: str = "subj") -> None:
    """A minimal usable fact at version 0, straight on the contract schema, for tests that
    exercise only the store (no ledger needed)."""
    conn.execute(
        "INSERT INTO records (id, kind, version, author_tier, labels_final, recorded_at,"
        " created_at, updated_at) VALUES (?, 'fact', 0, 'user', 1, ?, ?, ?)",
        (rid, TS0, TS0, TS0),
    )
    conn.execute(
        "INSERT INTO facts (id, subject_id, relation, value_text, value_norm, text_en, fkind)"
        " VALUES (?, ?, 'other', ?, ?, ?, 'state')",
        (rid, subject, value, value.casefold(), f"value {value}"),
    )


def add_raw(
    conn: Any,
    key: str,
    text: str,
    *,
    chat_id: str | None,
    turn_id: str | None = None,
    tier: AuthorTier = AuthorTier.USER,
    author_entity_id: str | None = None,
    source_kind: str = "owner_message",
    ts: str = TS0,
) -> str:
    """One admitted raw chunk (§6.2 step 3) with its contentless FTS row."""
    rid = ulid(f"raw:{key}")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO raw_chunks (id, source_kind, chat_id, turn_id, author_tier, author_entity_id,"
        " ts, text, content_hmac, labels_final, policy_version, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)",
        (
            rid,
            source_kind,
            chat_id,
            turn_id,
            tier.value,
            author_entity_id,
            ts,
            text,
            hashlib.sha256(text.encode()).hexdigest(),
            ts,
            ts,
        ),
    )
    rowid = conn.execute("SELECT fts_rowid FROM raw_chunks WHERE id = ?", (rid,)).fetchone()[0]
    conn.execute("INSERT INTO fts_raw (rowid, text) VALUES (?, ?)", (rowid, text.casefold()))
    conn.execute("COMMIT")
    return rid


def lineage(sql: Any, record_id: str, raw_id: str) -> None:
    """`record derived_from raw` (§3.2 edges); idempotent if the ledger already wrote it."""
    sql.execute(
        "INSERT OR IGNORE INTO edges (from_id, to_id, etype, created_at)"
        " VALUES (?, ?, 'derived_from', ?)",
        (record_id, raw_id, TS0),
    )


def justify(sql: Any, target_id: str, group_no: int, members: Iterable[str]) -> None:
    """§8.1 member rows of one AND-group, provenance `stated`."""
    for m in members:
        sql.execute(
            "INSERT INTO justifications (id, target_record_id, group_no, member_record_id,"
            " provenance, created_at) VALUES (?, ?, ?, ?, 'stated', ?)",
            (ulid(f"just:{target_id}:{group_no}:{m}"), target_id, group_no, m, TS0),
        )


def serve(
    conn: Any,
    *,
    thread: str,
    session: str | None,
    item: ReadRef,
    via: str = "basket",
    turn_id: str | None = None,
    kind: str = "fact",
) -> None:
    """§7.6 record: the item reached a session."""
    conn.execute(
        "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id, via,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (thread, session, item.id, kind, item.version, turn_id, via, TS0),
    )


def add_derived(
    conn: Any, key: str, content: str, read_set: list[ReadRef], *, dkind: str = "summary"
) -> str:
    """A derived item with its read set and the mirroring `read_deps` rows (§3.2 invariant)."""
    did = ulid(f"derived:{key}")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO derived (id, dkind, dkey, content, read_set, built_at, valid, created_at,"
        " updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (
            did,
            dkind,
            "2026-10-12" if dkind == "summary" else key,
            content,
            json.dumps([[r.id, r.version] for r in read_set]),
            TS0,
            TS0,
            TS0,
        ),
    )
    conn.executemany(
        "INSERT INTO read_deps (holder_kind, holder_id, item_id, version) VALUES ('derived', ?, ?, ?)",
        [(did, r.id, r.version) for r in read_set],
    )
    conn.execute("COMMIT")
    return did


def derived_ref(did: str) -> ReadRef:
    return ReadRef(id=did, version=0, src=ReadSrc.DERIVED)


# ---------------------------------------------------------------- threads


def run_threads(*fns: Callable[[], Any], timeout: float = 20.0) -> list[Any]:
    """Run each fn in its own thread; re-raise the first failure; return results in order."""
    results: list[Any] = [None] * len(fns)
    errors: list[BaseException] = []

    def wrap(i: int, fn: Callable[[], Any]) -> None:
        try:
            results[i] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=wrap, args=(i, fn), daemon=True) for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
        assert not t.is_alive(), "a race thread hung (deadlock?)"
    if errors:
        raise errors[0]
    return results


# ---------------------------------------------------------------- byte scan (§19.5 method)


def encodings(token: str) -> list[bytes]:
    escaped = "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in token)
    forms = {token.encode(), token.encode("utf-16-le"), escaped.encode()}
    return sorted(forms)


def scan(root: Path, token: str, *, skip: Iterable[str] = ()) -> list[str]:
    """Files under `root` (minus `skip` dir names) whose bytes hold the token in any form."""
    hits = []
    skipped = set(skip)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or skipped & set(path.relative_to(root).parts):
            continue
        data = path.read_bytes()
        low = data.lower()
        for form in encodings(token):
            if form in data or form.lower() in low:
                hits.append(str(path.relative_to(root)))
                break
    return hits


# ---------------------------------------------------------------- crash harness


CHILD = Path(__file__).with_name("_crash_child.py")


def run_child(spec: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
    """Run one facade or store call in a fresh process that may die mid-way (os._exit)."""
    env = dict(os.environ)
    env.pop("KOOB_DATA_DIR", None)
    proc = subprocess.run(
        [sys.executable, str(CHILD), json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    out = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            out = json.loads(line[len("RESULT ") :])
    if proc.returncode not in (0, 77):
        raise AssertionError(f"child failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return proc.returncode, out


def kill_points(total: int) -> list[int]:
    """Spread crash points over a call of `total` SQLite VM steps, ends included."""
    pts = {1, total // 8, total // 4, total // 2, (3 * total) // 4, total - 40, total - 2}
    return sorted(p for p in pts if 0 < p < total)


def crash_runs(db_path: Path, tmp_path: Path, call: dict) -> list[tuple[Path, int]]:
    """Copy the seeded data dir per run; a dry run counts steps; each kill run gets a copy.
    Returns (db path, exit code) for the dry run first, then every kill point."""
    pristine = tmp_path / "pristine"
    shutil.copytree(db_path.parent.parent, pristine)
    runs = []

    def attempt(kill_at: int | None, name: str):
        root = tmp_path / name
        shutil.copytree(pristine, root)
        db = root / "memory" / "memory.db"
        spec = {"db": str(db), "home": str(root / "home"), "kill_at": kill_at, **call}
        code, out = run_child(spec)
        return db, code, out

    db, code, out = attempt(None, "dry")
    assert code == 0 and out is not None
    runs.append((db, code))
    points = kill_points(out["steps"])
    assert len(points) >= 4
    for k in points:
        db, code, _ = attempt(k, f"kill-{k}")
        runs.append((db, code))
    assert sum(code == 77 for _, code in runs) >= 3
    return runs


def open_jobs(c) -> int:
    return c.execute("SELECT count(*) FROM jobs WHERE state IN ('queued', 'running')").fetchone()[0]
