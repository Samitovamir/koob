"""Plain-SQL seeding for the forget tests (ledger bodies are seeded elsewhere), a real
koob.db built from `migrations/`, and an independent byte scanner (§19.5)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from koob.memory import forget
from koob.memory.adapters.conversation import CareerConversation
from koob.memory.clock import ManualClock, iso

ROOT = Path(__file__).resolve().parents[3]
T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
OLD = "2026-10-01T09:00:00.000000Z"  # before every forget in these tests


@pytest.fixture
def clock():
    """A manual clock wired into forget; wiring and key are reset after the test."""
    clk = ManualClock(T0)
    forget.reset_wiring()
    forget.use_key(None)
    forget.wire(clock=clk)
    yield clk
    forget.reset_wiring()
    forget.use_key(None)


@pytest.fixture
def career(tmp_path):
    """koob.db with the real migrations, as the runtime has it."""
    path = tmp_path / "career" / "koob.db"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    for sql in sorted((ROOT / "migrations").glob("*.sql")):
        conn.executescript(sql.read_text(encoding="utf-8"))
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def conv(career, clock):
    store = CareerConversation(career)
    forget.wire(conversation=store)
    return store


# ---------------------------------------------------------------- memory.db rows


def relation(conn, name: str, family: str, cardinality: str = "single") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO relations (relation, family, cardinality, value_type) "
        "VALUES (?, ?, ?, 'text')",
        (name, family, cardinality),
    )


def record(
    conn,
    rid: str,
    kind: str,
    *,
    tier: str = "user",
    labels: int = 0,
    author: str | None = None,
    origin_quote: str | None = None,
    fts: tuple[str, str, str] | None = None,
) -> None:
    conn.execute(
        "INSERT INTO records (id, kind, version, status, labels, author_tier, author_entity_id, "
        "recorded_at, origin_quote, created_at, updated_at) "
        "VALUES (?, ?, 1, 'active', ?, ?, ?, ?, ?, ?, ?)",
        (rid, kind, labels, tier, author, OLD, origin_quote, OLD, OLD),
    )
    if fts is not None:
        rowid = conn.execute("SELECT fts_rowid FROM records WHERE id = ?", (rid,)).fetchone()[0]
        conn.execute(
            "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, ?, ?)",
            (rowid, *fts),
        )


def entity(
    conn,
    eid: str,
    name: str,
    aliases: Iterable[tuple[str, str]] = (),
    stable_ids: Iterable[tuple[str, str]] = (),
    etype: str = "person",
) -> None:
    record(conn, eid, "entity", fts=(name.casefold(), "", " ".join(a for a, _ in aliases)))
    conn.execute(
        "INSERT INTO entities (id, etype, display_name, stable_ids) VALUES (?, ?, ?, ?)",
        (
            eid,
            etype,
            name,
            json.dumps(
                [{"kind": k, "value": v, "tier": "platform"} for k, v in stable_ids],
                ensure_ascii=False,
            ),
        ),
    )
    for i, (alias, akind) in enumerate(aliases):
        aid = f"{eid}-a{i}"
        conn.execute(
            "INSERT INTO aliases (id, entity_id, alias, alias_norm, script, kind, tier, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'user', ?, ?)",
            (
                aid,
                eid,
                alias,
                alias.casefold(),
                "cyrl" if alias.isascii() is False else "latn",
                akind,
                OLD,
                OLD,
            ),
        )
        rowid = conn.execute("SELECT fts_rowid FROM aliases WHERE id = ?", (aid,)).fetchone()[0]
        conn.execute("INSERT INTO fts_names (rowid, name) VALUES (?, ?)", (rowid, alias))


def fact(
    conn,
    rid: str,
    subject: str,
    rel: str,
    *,
    value: str | None,
    text_en: str,
    quote: str | None = None,
    raw: str | None = None,
    value_entity: str | None = None,
    speaker: str | None = None,
    labels: int = 0,
    keywords_ru: str | None = None,
    keywords_en: str | None = None,
    tier: str = "user",
) -> None:
    record(
        conn,
        rid,
        "fact",
        tier=tier,
        labels=labels,
        fts=(text_en.casefold(), f"{keywords_ru or ''} {keywords_en or ''}", ""),
    )
    conn.execute(
        "INSERT INTO facts (id, subject_id, relation, value_text, value_norm, value_entity_id, "
        "text_en, keywords_ru, keywords_en, quote_orig, lang_orig, fkind, speaker_entity_id, "
        "source_raw_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ru', 'state', ?, ?)",
        (
            rid,
            subject,
            rel,
            value,
            (value or "").casefold() or None,
            value_entity,
            text_en,
            keywords_ru,
            keywords_en,
            quote,
            speaker,
            raw,
        ),
    )
    if raw:
        edge(conn, rid, raw, "derived_from")


def impression(conn, rid: str, about: str, text_en: str, quote: str | None = None) -> None:
    record(conn, rid, "impression", fts=(text_en.casefold(), "", ""))
    conn.execute(
        "INSERT INTO impressions (id, about_entity_id, text_en, quote_orig, observed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (rid, about, text_en, quote, OLD),
    )


def episode(
    conn,
    rid: str,
    summary: str,
    *,
    links: Iterable[tuple[str, str]] = (),
    derived_from: Iterable[str] = (),
) -> None:
    record(conn, rid, "episode", tier="system", fts=(summary.casefold(), "", ""))
    conn.execute("INSERT INTO episodes (id, summary_en) VALUES (?, ?)", (rid, summary))
    for role, target in links:
        conn.execute(
            "INSERT INTO episode_links (episode_id, role, target_id) VALUES (?, ?, ?)",
            (rid, role, target),
        )
    for src in derived_from:
        edge(conn, rid, src, "derived_from")


def raw(
    conn,
    rid: str,
    text: str,
    *,
    turn: str | None = None,
    chat: str | None = None,
    ts: str = OLD,
    tier: str = "user",
    labels: int = 0,
) -> None:
    conn.execute(
        "INSERT INTO raw_chunks (id, status, rev, source_kind, chat_id, turn_id, author_tier, ts, "
        "text, content_hmac, labels, policy_version, created_at, updated_at) "
        "VALUES (?, 'active', 1, 'island', ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (
            rid,
            chat,
            turn,
            tier,
            ts,
            text,
            forget.keyed_hmac("raw", text.casefold()),
            labels,
            OLD,
            OLD,
        ),
    )
    rowid = conn.execute("SELECT fts_rowid FROM raw_chunks WHERE id = ?", (rid,)).fetchone()[0]
    conn.execute("INSERT INTO fts_raw (rowid, text) VALUES (?, ?)", (rowid, text.casefold()))


def edge(conn, src: str, dst: str, etype: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO edges (from_id, to_id, etype, created_at) VALUES (?, ?, ?, ?)",
        (src, dst, etype, OLD),
    )


def derived(conn, did: str, content: str, reads: Iterable[tuple[str, int]]) -> None:
    reads = list(reads)
    conn.execute(
        "INSERT INTO derived (id, dkind, dkey, content, read_set, built_at, valid, created_at, "
        "updated_at) VALUES (?, 'summary', '2026-10-01', ?, ?, ?, 1, ?, ?)",
        (did, content, json.dumps(reads), OLD, OLD, OLD),
    )
    for item, version in reads:
        conn.execute(
            "INSERT INTO read_deps (holder_kind, holder_id, item_id, version) "
            "VALUES ('derived', ?, ?, ?)",
            (did, item, version),
        )


def served(conn, thread: str, item: str, turn: str | None, session: str = "s-1") -> None:
    conn.execute(
        "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id, via, "
        "created_at) VALUES (?, ?, ?, 'fact', 1, ?, 'basket', ?)",
        (thread, session, item, turn, OLD),
    )


# ---------------------------------------------------------------- koob.db rows


def thread(career, title: str = "Новый разговор", session: str | None = "sdk-1") -> str:
    cur = career.execute(
        "INSERT INTO koob_threads (title, sdk_session_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (title, session, OLD, OLD),
    )
    career.commit()
    return str(cur.lastrowid)


def turn(career, thread_id: str, *rows: tuple[str, str], tool: dict | None = None) -> str:
    cur = career.execute(
        "INSERT INTO koob_turns (thread_id, status, context, started_at) "
        "VALUES (?, 'done', ?, ?)",
        (int(thread_id), "page: /koob", OLD),
    )
    tid = cur.lastrowid
    for role, text in rows:
        career.execute(
            "INSERT INTO koob_messages (thread_id, turn_id, role, text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (int(thread_id), tid, role, text, OLD),
        )
    if tool is not None:
        career.execute(
            "INSERT INTO koob_messages (thread_id, turn_id, role, text, tool_json, created_at) "
            "VALUES (?, ?, 'tool', '', ?, ?)",
            (int(thread_id), tid, json.dumps(tool), OLD),
        )
    career.commit()
    return str(tid)


def texts(career, turn_id: str) -> list[str]:
    return [
        r[0]
        for r in career.execute(
            "SELECT text FROM koob_messages WHERE turn_id = ? AND role <> 'tool' ORDER BY id",
            (int(turn_id),),
        )
    ]


# ---------------------------------------------------------------- independent scanner


def encodings(token: str) -> list[bytes]:
    """§19.5: UTF-8, JSON \\uXXXX escapes (both hex cases), UTF-16LE."""
    out = {token.encode("utf-8"), token.encode("utf-16-le")}
    esc = json.dumps(token)[1:-1]
    out.add(esc.encode("ascii"))
    out.add(esc.upper().replace("\\U", "\\u").encode("ascii"))
    return [b for b in out if b]


def hits(paths: Iterable[Path], tokens: Iterable[str]) -> list[tuple[str, str]]:
    found = []
    for p in paths:
        if not p.is_file():
            continue
        data = p.read_bytes()
        for tok in tokens:
            if any(enc in data for enc in encodings(tok)):
                found.append((p.name, tok))
    return found


def files_under(*roots: Path) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        out.extend(p for p in sorted(root.rglob("*")) if p.is_file())
    return out


def now_iso(clk: ManualClock) -> str:
    return iso(clk.now())
