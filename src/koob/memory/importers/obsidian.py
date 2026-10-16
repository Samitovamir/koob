"""One-way import of the legacy Obsidian memory into English records; skips
`Koob/.history/`; replays `koob_memory_events` forgets and the ledger journal. Tested on
a synthetic vault fixture; the real vault only through `cb memory import-obsidian --dry-run`.

Spec: §20 rollout step 2, §6.2 (arrival, `import:obsidian`), §9.3 (journal lines).

What is read: only the `Koob/` folder of the vault (the rest of the vault is the user's own
notes and is never opened). Inside it:

- `notes/<slug>.md`: one note, cut at paragraphs into chunks of at most `CHUNK_MAX`
  characters, each «<title>: <text>»; `source_ref` `obsidian:<slug>` (what the shadow
  mirror's `_slug_records` looks up). A note's `## source` section (a raw quote of somebody
  else's text) is its own chunk at tier `tool`.
- `Главное.md`: one chunk per `## ` section; `obsidian:главное`.
- `journal/YYYY-MM-DD.md`: one chunk per entry line, timed by the line; `obsidian:journal/<date>`.
- `sources/**.md`: digests of the user's files, tier `tool`; `obsidian:sources/<path>`.
- Skipped and counted: `.history/` (old versions, forgotten notes among them), `MEMORY.md`
  (an index of the notes), `skills/` (behaviour, kept by its own module), anything else,
  symlinks and oversized files.

Tiers (§6.2 step 4, §13.1): a note is the Cube's restatement, never the user's words: `system`;
a tainted note or one whose `source` is a page, a file or a chat: `tool`, with the quarantine
check (§6.2 step 6). Nothing is ever `user`.

Arrival is §6.2's, by code: admission (§5.1: a secret in a non-own text refuses the chunk),
the ledger journal and forgotten raw (a forgotten import never comes back), content-hash
dedupe, junk gate, labels (lexicon plus the private-area folders of the legacy store), then
raw chunks with `source_kind` `import:obsidian` and one `extract` job per note or journal
day. The extraction worker makes the English records, with `source_raw_id` and
`derived_from` lineage to these chunks. Idempotent: a chunk already imported (same
`source_ref` and content HMAC, in any status) is skipped, so a re-run adds nothing.

Legacy forgets (`koob_memory_events`, action `forgotten`, not undone): the chunks imported
from that slug whose text is no longer in the note and which date from before the forget go
through the forget pipeline (§9.1 stop-use, scope `item`, `target_ids` = the chunks; the
journal line carries both, §9.3). Memory side only: the legacy forget already ran in the
conversation store.

A dry run writes nothing (no audit line, no row, no file) and returns counts only; it never
returns note text.
"""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..model import OWNER_ENTITY_ID, AuthorTier, ForgetScope, Label, RawChunk

SOURCE_KIND = "import:obsidian"
REF_PREFIX = "obsidian:"
KOOB_DIR = "Koob"
HISTORY = ".history"
MAIN_FILE = "Главное.md"
MAIN_SLUG = "главное"
INDEX_FILE = "MEMORY.md"
CHUNK_MAX = 1500  # characters of note text per chunk (the extractor reads one batch per note)
MAX_FILE = 256_000  # bytes; a bigger file is not a legacy note (caps were 8 000 characters)
_FOREIGN_SOURCES = ("web", "file", "telegram")
# the legacy store's private folders (koob/memory.PRIVATE_AREAS) and their §5.3 labels
_PRIVATE_LABELS = {
    "id-immigration": Label.IDENTITY_DOCS,
    "finance": Label.MONEY,
    "medical": Label.HEALTH,
}
_SOURCE_HEADING = re.compile(r"^##\s+source\s*$", re.M | re.I)
_NEXT_HEADING = re.compile(r"^#{1,2}\s", re.M)
_JOURNAL_LINE = re.compile(r"^\s*[-*]\s*(\d{1,2}):(\d{2})\s*[—–-]\s*(.+?)\s*$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)

COUNT_KEYS = (
    "files",  # Markdown files read under Koob/
    "notes",
    "main_sections",
    "journal_entries",
    "source_digests",
    "tainted_notes",
    "history_skipped",
    "index_skipped",
    "skills_skipped",
    "other_skipped",
    "chunks",  # candidate raw chunks
    "new",  # stored (or, on a dry run, would be stored)
    "already_imported",
    "duplicate",  # the same text already in the store from another source
    "forgotten",  # refused by the deletion journal or a forgotten raw
    "refused",  # admission (§5.1): a secret in a non-own text, a never-store source
    "junk",
    "quarantined",
    "jobs",  # extraction jobs enqueued
    "legacy_forgets",
    "replayed",  # imported chunks forgotten by a legacy forget replay
    "records",  # active records whose lineage reaches an imported chunk
)


class NoKoobFolder(ValueError):
    """No `Koob/` folder where the vault was named."""


@dataclass(frozen=True)
class Unit:
    """One chunk to import. `group` is the extraction batch (a note, a journal day)."""

    ref: str
    text: str
    tier: AuthorTier
    ts: str
    labels: Label = Label.NONE
    group: str = ""


@dataclass
class Plan:
    units: list[Unit] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNT_KEYS, 0))


# ---------------------------------------------------------------- the vault


def koob_root(vault: Path) -> Path:
    """`<vault>/Koob`, or the Koob folder itself. Never the bare vault root: the rest of
    the vault is the user's own notes."""
    vault = Path(vault).expanduser()
    sub = vault / KOOB_DIR
    if sub.is_dir() and not sub.is_symlink():
        return sub
    if vault.is_dir() and (
        vault.name == KOOB_DIR
        or ((vault / INDEX_FILE).is_file() and (vault / MAIN_FILE).is_file())
    ):
        return vault
    raise NoKoobFolder("no Koob folder in the vault")


def _stamp(day: str | None, hh: int = 0, mm: int = 0) -> str | None:
    if not day or not _DATE.match(day):
        return None
    try:
        d = datetime.strptime(day, "%Y-%m-%d").replace(hour=hh, minute=mm, tzinfo=UTC)
    except ValueError:
        return None
    return d.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _front(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), m.group(2)


def _ts_of(meta: dict[str, Any], path: Path) -> str:
    for key in ("updated", "created"):
        v = meta.get(key)
        s = _stamp(str(v)[:10]) if v is not None else None
        if s:
            return s
    return _mtime(path)


def _tainted(meta: dict[str, Any]) -> bool:
    return meta.get("tainted") is True or str(meta.get("tainted", "")).lower() == "true"


def _labels_of_source(source: str) -> Label:
    s = source.casefold()
    out = Label.NONE
    for area, lbl in _PRIVATE_LABELS.items():
        if re.search(rf"(?:^|[/\s:]){re.escape(area)}(?:$|[/\s])", s):
            out |= lbl
    return out


def _split_source(body: str) -> tuple[str, str]:
    """(body without its `## source` sections, the quoted text of those sections)."""
    quotes: list[str] = []
    rest = body
    while True:
        m = _SOURCE_HEADING.search(rest)
        if not m:
            break
        after = rest[m.end() :]
        nxt = _NEXT_HEADING.search(after)
        section = after[: nxt.start()] if nxt else after
        quotes.append(
            "\n".join(re.sub(r"^\s*>\s?", "", ln) for ln in section.strip().splitlines()).strip()
        )
        rest = rest[: m.start()] + (after[nxt.start() :] if nxt else "")
    return rest.strip(), "\n\n".join(q for q in quotes if q)


def _pieces(text: str, limit: int = CHUNK_MAX) -> list[str]:
    """Paragraphs packed up to `limit` characters; a longer paragraph is cut at lines, then
    at words."""
    out: list[str] = []
    cur = ""

    def push(part: str) -> None:
        nonlocal cur
        if not cur:
            cur = part
        elif len(cur) + 2 + len(part) <= limit:
            cur = f"{cur}\n\n{part}"
        else:
            out.append(cur)
            cur = part

    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            push(para)
            continue
        line_buf = ""
        for ln in para.splitlines():
            while len(ln) > limit:
                cut = ln.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                if line_buf:
                    push(line_buf)
                    line_buf = ""
                push(ln[:cut].strip())
                ln = ln[cut:].strip()
            if line_buf and len(line_buf) + 1 + len(ln) > limit:
                push(line_buf)
                line_buf = ln
            else:
                line_buf = f"{line_buf}\n{ln}" if line_buf else ln
        if line_buf:
            push(line_buf)
    if cur:
        out.append(cur)
    return out


def _title(meta: dict[str, Any], body: str, fallback: str) -> tuple[str, str]:
    """(title, body without a leading `# title` line)."""
    title = " ".join(str(meta.get("title") or "").split())
    lines = body.lstrip().splitlines()
    if lines and lines[0].startswith("# "):
        head = lines[0][2:].strip()
        if not title or head == title:
            title = title or head
            body = "\n".join(lines[1:])
    return (title or fallback), body.strip()


def _note(plan: Plan, path: Path, slug: str, text: str) -> None:
    meta, body = _front(text)
    title, body = _title(meta, body, slug)
    body, quote = _split_source(body)
    source = str(meta.get("source") or "")
    tainted = _tainted(meta)
    foreign = tainted or source.split(":", 1)[0].strip().casefold() in _FOREIGN_SOURCES
    tier = AuthorTier.TOOL if foreign else AuthorTier.SYSTEM
    labels = _labels_of_source(source)
    ts = _ts_of(meta, path)
    ref = f"{REF_PREFIX}{slug}"
    plan.counts["notes"] += 1
    plan.counts["tainted_notes"] += int(tainted)
    pieces = _pieces(body) or ([title] if title != slug else [])
    for piece in pieces:
        text_ = piece if piece.startswith(title) else f"{title}: {piece}"
        plan.units.append(Unit(ref, text_, tier, ts, labels, group=ref))
    for piece in _pieces(quote):
        plan.units.append(Unit(ref, piece, AuthorTier.TOOL, ts, labels, group=ref))


def _main(plan: Plan, path: Path, text: str) -> None:
    meta, body = _front(text)
    ts = _ts_of(meta, path)
    ref = f"{REF_PREFIX}{MAIN_SLUG}"
    sections: list[tuple[str, list[str]]] = [("", [])]
    for ln in body.splitlines():
        if ln.startswith("## "):
            sections.append((ln[3:].strip(), []))
        elif ln.startswith("# "):
            continue
        elif ln.strip():
            sections[-1][1].append(ln.rstrip())
    for head, lines in sections:
        if not lines:
            continue
        content = "\n".join(lines)
        for piece in _pieces(content):
            plan.counts["main_sections"] += 1
            text_ = f"Главное — {head}: {piece}" if head else f"Главное: {piece}"
            plan.units.append(Unit(ref, text_, AuthorTier.SYSTEM, ts, Label.NONE, group=ref))


def _journal(plan: Plan, path: Path, text: str) -> None:
    day = path.stem
    _meta, body = _front(text)
    ref = f"{REF_PREFIX}journal/{day}"
    base = _stamp(day) or _mtime(path)
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = _JOURNAL_LINE.match(s)
        if m:
            hh, mm, entry = int(m.group(1)), int(m.group(2)), m.group(3)
            ts = _stamp(day, hh, mm) if hh < 24 and mm < 60 else None
        else:
            entry, ts = s.lstrip("-* ").strip(), None
        if not entry:
            continue
        plan.counts["journal_entries"] += 1
        for piece in _pieces(entry):
            plan.units.append(Unit(ref, piece, AuthorTier.SYSTEM, ts or base, Label.NONE, ref))


def _source(plan: Plan, path: Path, rel: str, text: str) -> None:
    meta, body = _front(text)
    name = rel.removesuffix(".md")
    title, body = _title(meta, body, Path(name).name)
    ref = f"{REF_PREFIX}{name}"
    labels = _labels_of_source(str(meta.get("source") or "")) | _labels_of_source(rel)
    plan.counts["source_digests"] += 1
    for piece in _pieces(body):
        text_ = piece if piece.startswith(title) else f"{title}: {piece}"
        plan.units.append(Unit(ref, text_, AuthorTier.TOOL, _ts_of(meta, path), labels, ref))


def scan(root: Path) -> Plan:
    """Walk `Koob/` without following links; never descends into `.history/`."""
    plan = Plan()
    c = plan.counts
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        rel_dir = here.relative_to(root)
        for d in list(dirnames):
            if d.startswith("."):
                dirnames.remove(d)
                if d == HISTORY and rel_dir == Path("."):
                    c["history_skipped"] += sum(len(f) for _, _, f in os.walk(here / d))
        dirnames.sort()
        top = rel_dir.parts[0] if rel_dir.parts else ""
        for name in sorted(filenames):
            path = here / name
            rel = (rel_dir / name).as_posix()
            if top == "skills":
                c["skills_skipped"] += 1
                continue
            if name.startswith(".") or not name.endswith(".md") or path.is_symlink():
                c["other_skipped"] += 1
                continue
            plain = unicodedata.normalize("NFC", name)  # a file system may hand back NFD
            rel = unicodedata.normalize("NFC", rel)
            if top == "" and plain == INDEX_FILE:
                c["index_skipped"] += 1
                continue
            try:
                if path.stat().st_size > MAX_FILE:
                    c["other_skipped"] += 1
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                c["other_skipped"] += 1
                continue
            if top == "" and plain == MAIN_FILE:
                _main(plan, path, text)
            elif top == "notes" and len(rel_dir.parts) == 1:
                _note(plan, path, plain.removesuffix(".md").casefold(), text)
            elif top == "journal" and len(rel_dir.parts) == 1 and _DATE.match(path.stem):
                _journal(plan, path, text)
            elif top == "sources":
                _source(plan, path, rel, text)
            else:
                c["other_skipped"] += 1
                continue
            c["files"] += 1
    c["chunks"] = len(plan.units)
    return plan


# ---------------------------------------------------------------- legacy forgets


def legacy_events(career: sqlite3.Connection | None) -> list[dict[str, Any]]:
    """`koob_memory_events` forgets that were not undone, oldest first; [] when there is
    no koob.db or no such table."""
    if career is None:
        return []
    try:
        rows = career.execute(
            "SELECT slug, created_at FROM koob_memory_events"
            " WHERE action = 'forgotten' AND undone_at IS NULL AND slug != ''"
            " ORDER BY created_at, id"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [{"slug": str(r[0]), "created_at": str(r[1])} for r in rows]


def wired_events() -> list[dict[str, Any]]:
    """The legacy forgets through the wired conversation store (its koob.db connection),
    duck-typed: memory/ never imports the koob side."""
    from .. import forget

    conv = forget._W.conversation
    try:
        conn = getattr(conv, "conn", None) if conv is not None else None
    except Exception:  # noqa: BLE001 - a store without a connection: no events
        conn = None
    return legacy_events(conn if isinstance(conn, sqlite3.Connection) else None)


def _when(stamp: str) -> datetime | None:
    s = (stamp or "").strip().replace(" ", "T")
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        d = None
        if _DATE.match(s[:10]):
            d = datetime.strptime(s[:10], "%Y-%m-%d")
    if d is None:
        return None
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def _replay_targets(
    conn: sqlite3.Connection, plan: Plan, events: list[dict[str, Any]]
) -> list[str]:
    """Active imported chunks of a forgotten slug whose text the note no longer holds and
    which are not newer than the forget."""
    from ..ingest import content_hmac, normalize

    current: dict[str, set[str]] = {}
    for u in plan.units:
        current.setdefault(u.ref, set()).add(content_hmac(normalize(u.text)))
    out: list[str] = []
    for ev in events:
        slug = ev["slug"].strip().casefold()
        at = _when(ev["created_at"])
        if not slug or "/" in slug or at is None:
            continue
        ref = f"{REF_PREFIX}{slug}"
        rows = conn.execute(
            "SELECT id, content_hmac, ts FROM raw_chunks WHERE source_kind = ? AND source_ref = ?"
            " AND status = 'active' ORDER BY id",
            (SOURCE_KIND, ref),
        ).fetchall()
        for rid, h, ts in rows:
            when = _when(str(ts))
            if h in current.get(ref, set()) or rid in out:
                continue
            if when is None or when <= at:
                out.append(str(rid))
    return out


# ---------------------------------------------------------------- arrival


def _already(conn: sqlite3.Connection, ref: str, h: str) -> bool:
    q = "SELECT 1 FROM raw_chunks WHERE source_kind = ? AND source_ref = ? AND content_hmac = ?"
    return conn.execute(q, (SOURCE_KIND, ref, h)).fetchone() is not None


def _classify(conn: sqlite3.Connection | None, u: Unit) -> tuple[str, str | None]:
    """(outcome, hmac) for one unit, reading only: new / already_imported / forgotten /
    duplicate / refused."""
    from .. import policy
    from ..ingest import SHORT_EXEMPT, _active_copy, content_hmac, forgotten_match, normalize

    if u.tier not in policy.OWN_TIERS and policy.never_store_spans(u.text):
        return "refused", None
    if conn is None:
        return "new", None
    n = normalize(u.text)
    h = content_hmac(n)
    if _already(conn, u.ref, h):
        return "already_imported", h
    if forgotten_match(conn, h, n):
        return "forgotten", h
    if len(n) >= SHORT_EXEMPT and _active_copy(conn, h):
        return "duplicate", h
    return "new", h


def _flags(u: Unit) -> tuple[bool, bool, Label]:
    from .. import policy
    from ..ingest import is_junk, quarantine_spans

    junk = is_junk(u.text)
    quarantined = (
        not junk
        and u.tier not in policy.OWN_TIERS
        and u.tier is not AuthorTier.SYSTEM
        and bool(quarantine_spans(u.text))
    )
    labels = Label.NONE if junk else (policy.label_topics(policy.words(u.text)) | u.labels)
    return junk, quarantined, labels


def _records(conn: sqlite3.Connection | None) -> int:
    if conn is None:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(DISTINCT r.id) FROM records r WHERE r.status = 'active' AND ("
            " r.id IN (SELECT f.id FROM facts f JOIN raw_chunks c ON c.id = f.source_raw_id"
            "          WHERE c.source_kind = ?)"
            " OR r.id IN (SELECT e.from_id FROM edges e JOIN raw_chunks c ON c.id = e.to_id"
            "             WHERE e.etype = 'derived_from' AND c.source_kind = ?))",
            (SOURCE_KIND, SOURCE_KIND),
        ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0])


def import_vault(
    conn: sqlite3.Connection | None,
    vault: Path,
    *,
    dry_run: bool = True,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Import `vault`'s Koob folder into the store on `conn`; counts only. On a dry run
    `conn` may be None (no store yet) or read-only, and nothing is written."""
    plan = scan(koob_root(Path(vault)))
    c = plan.counts
    events = list(events or [])
    c["legacy_forgets"] = len(events)
    if dry_run:
        groups: set[str] = set()
        for u in plan.units:
            outcome, _ = _classify(conn, u)
            c[outcome] += 1
            if outcome == "new":
                junk, quarantined, _ = _flags(u)
                c["junk"] += int(junk)
                c["quarantined"] += int(quarantined)
                if not junk and not quarantined:
                    groups.add(u.group)
        c["jobs"] = len(groups)
        if conn is not None and events:
            c["replayed"] = len(_replay_targets(conn, plan, events))
        c["records"] = _records(conn)
        return dict(c)
    if conn is None:
        raise ValueError("an import needs a store")
    _apply(conn, plan, events)
    c["records"] = _records(conn)
    return dict(c)


def _apply(conn: sqlite3.Connection, plan: Plan, events: list[dict[str, Any]]) -> None:
    from .. import forget, labeler, policy
    from .. import store as _store
    from ..ingest import _enqueue, store_raw
    from ..ledger import new_id

    c = plan.counts
    # legacy forgets first: what the user had the old memory forget is gone before anything new
    targets = _replay_targets(conn, plan, events) if events else []
    if targets:
        forget._stop_use(conn, targets, scope=ForgetScope.ITEM, forever=True, conversation=None)
        c["replayed"] = len(targets)
    pv = conn.execute("SELECT policy_version FROM meta WHERE id = 1").fetchone()
    policy_version = int(pv[0]) if pv else 0
    process: list[str] = []
    with _store.write_tx(conn, priority="background"):
        batches: dict[str, list[str]] = {}
        for u in plan.units:
            outcome, h = _classify(conn, u)
            if outcome != "new" or h is None:
                c[outcome] += 1
                continue
            a = policy.admit_ex(
                conn,
                source_kind=SOURCE_KIND,
                chat_id=None,
                text=u.text,
                author_tier=u.tier,
                turn=None,
                ts=u.ts,
            )
            if not a.admit or not a.text or a.text != u.text:
                c["refused"] += 1  # a redacted text is not the user's note any more: not imported
                continue
            junk, quarantined, labels = _flags(u)
            chunk = RawChunk(
                id=new_id(),
                source_kind=SOURCE_KIND,
                source_ref=u.ref,
                chat_id=None,
                turn_id=None,
                author_tier=u.tier,
                author_entity_id=OWNER_ENTITY_ID if u.tier in policy.OWN_TIERS else None,
                ts=u.ts,
                text=u.text,
                content_hmac=h,
                labels=labels,
                labels_final=False,
                policy_version=policy_version,
                quarantined=quarantined,
                junk=junk,
            )
            rid = store_raw(conn, chunk)
            if rid is None:
                c["forgotten"] += 1
                continue
            c["new"] += 1
            c["junk"] += int(junk)
            c["quarantined"] += int(quarantined)
            if not junk and not quarantined:
                batches.setdefault(u.group, []).append(rid)
        for ids in batches.values():
            _enqueue(conn, ids, None)
            process += ids
        c["jobs"] = len(batches)
    if process and not conn.in_transaction:
        labeler.finalize_later(conn, raw_ids=process)


def open_readonly(path: Path) -> sqlite3.Connection | None:
    """The default store for a dry run: read-only, or None when there is none yet (a dry run
    never creates it)."""
    if not Path(path).is_file():
        return None
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn
