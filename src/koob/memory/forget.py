"""Forgetting: stop-use, trash, verified purge, deletion journal, suppressions, raw
following its records, match keys for affected turns.

Spec: §9.1-§9.3, §9.5, §9.6, §19.5. The koob.db side goes through
`adapters/conversation.py` (a protocol), never through `koob.assistant`.

Shape of a forget (§9.1):
1. **Stop use** (`stop_use`): one `store.commit` (event `forget`) moves the content of the
   target records, the records derived from them, the derived items reading them and the raw
   of their lineage into `trash`, blanks it in place, drops FTS rows, writes suppressions and
   a `forget_purge` job; then the journal line (`stage:"stopped"`), vectors, the affected
   turns in koob.db, session restarts and cache hooks.
2. **Trash** (`undo`): «Вернуть» restores everything from the trash, journal `undone`.
3. **Purge** (`purge`): walks the §9.2 scope; «удалил полностью» only after a verified
   checkpoint, the orphan-vector sweep, a clean token scan of memory files (backups
   included) and every wired step; otherwise retries with backoff.

Wiring (`wire`): the conversation store (koob.db), the clock, cache hooks for stop-use
(§9.1d) and purge steps for stores outside memory/ (SDK files, logs, career backups, vault),
registered by the runtime. Tests wire fakes; nothing here imports `koob.assistant`.

The HMAC key lives in the Keychain (`keychain_service()`); under pytest a fixed test key is
used and the Keychain is never read.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac as _hmac
import json
import os
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import backup, store
from . import clock as _clockmod
from .adapters.conversation import (
    FORGOTTEN,
    ConversationStore,
    compile_keys,
    match_norm,
    text_matches,
)
from .clock import Clock, iso
from .model import (
    OWNER_ENTITY_ID,
    SENSITIVE_DEFAULT,
    AuthorTier,
    ChatMode,
    EventKind,
    ForgetScope,
    ForgetStage,
    JournalLine,
    OpKind,
    Proposal,
    ProposalOp,
)

TRASH_WINDOW = timedelta(hours=24)
PURGE_JOB = "forget_purge"
BACKOFF_S: tuple[int, ...] = (60, 300, 1800, 3600, 6 * 3600)
REMATCH_WINDOW_S = 600
_STOP_SCOPE = ("ledger", "fts", "raw", "derived", "suppressions", "vectors")
_MAX_SCAN_KEYS = 200

# ---------------------------------------------------------------- tables

_KIND_TABLE: dict[str, str] = {
    "entity": "entities",
    "fact": "facts",
    "preference": "facts",
    "impression": "impressions",
    "decision": "decisions",
    "work_state": "work_states",
    "intention": "intentions",
    "loop": "loops",
    "lesson": "lessons",
    "episode": "episodes",
    "criterion": "criteria",
}

# Content columns blanked at stop-use (value written in place); ids and vocabulary stay.
_BLANK: dict[str, dict[str, Any]] = {
    "entities": {"display_name": FORGOTTEN, "stable_ids": "[]"},
    "facts": {
        "value_text": None,
        "value_norm": None,
        "text_en": FORGOTTEN,
        "keywords_ru": None,
        "keywords_en": None,
        "quote_orig": None,
        "applicability": None,
    },
    "impressions": {"text_en": FORGOTTEN, "quote_orig": None},
    "decisions": {"chosen_text": FORGOTTEN, "alternatives": "[]", "revision_conditions": "[]"},
    "work_states": {"state": "{}"},
    "intentions": {"trigger": "{}", "completion": None, "action": None, "outcome": None},
    "loops": {"text_en": FORGOTTEN},
    "lessons": {"rule_en": FORGOTTEN, "applicability": None},
    "criteria": {"text_en": FORGOTTEN, "applicability": None, "evidence": "[]", "exceptions": "[]"},
    "episodes": {"summary_en": FORGOTTEN, "place": None, "outcome": None, "event_type": None},
}

# Free-text columns where «забудь всё про X» masks X's names (§9.8).
_MASK_COLS: dict[str, tuple[str, ...]] = {
    "facts": ("text_en", "quote_orig", "value_text", "keywords_ru", "keywords_en"),
    "impressions": ("text_en", "quote_orig"),
    "decisions": ("chosen_text", "alternatives", "revision_conditions"),
    "work_states": ("state",),
    "intentions": ("trigger", "completion", "action", "outcome"),
    "loops": ("text_en",),
    "lessons": ("rule_en",),
    "criteria": ("text_en", "exceptions"),
    "episodes": ("summary_en", "place", "outcome"),
}

# The main text of a kind row: suppression value and scan strings for non-fact kinds.
_MAIN_TEXT: dict[str, str] = {
    "impressions": "text_en",
    "decisions": "chosen_text",
    "loops": "text_en",
    "lessons": "rule_en",
    "criteria": "text_en",
    "episodes": "summary_en",
}

_USER_TIERS = frozenset({AuthorTier.USER.value, AuthorTier.USER_CONFIRMED.value})


class ForgetError(RuntimeError):
    """A forget operation cannot run (unknown id, already purged, ...)."""


class ForgetNotFound(ForgetError, LookupError):
    """No usable target matched (`api.forget` → outcome `not_found`)."""


# ---------------------------------------------------------------- wiring


@dataclass
class StopEvent:
    """What the cache hooks (§9.1d) get: ids only."""

    forget_id: str
    record_ids: list[str]
    raw_ids: list[str]
    turn_ids: list[str]
    thread_ids: list[str]


@dataclass
class PurgeCtx:
    """What a wired purge step gets. `keys` are content (scan strings), RAM only."""

    conn: sqlite3.Connection
    forget_id: str
    record_ids: list[str]
    raw_ids: list[str]
    turn_ids: list[str]
    thread_ids: list[str]
    keys: list[str]


PurgeStep = Callable[[PurgeCtx], bool]
StopHook = Callable[[StopEvent], None]


@dataclass
class _Wiring:
    conversation: ConversationStore | None = None
    stop_hooks: list[StopHook] = field(default_factory=list)
    purge_steps: dict[str, PurgeStep] = field(default_factory=dict)


_W = _Wiring()


def wire(
    *,
    conversation: ConversationStore | None = None,
    clock: Clock | None = None,
    stop_hooks: Iterable[StopHook] | None = None,
    purge_steps: dict[str, PurgeStep] | None = None,
) -> None:
    """Attach the runtime's pieces (runtime): the koob.db store, cache hooks run at
    stop-use (`rt.context_sent`, in-RAM vectors, the legacy recall index) and purge steps
    for stores outside memory/ (each returns True when its part is verified clean)."""
    if conversation is not None:
        _W.conversation = conversation
    if clock is not None:
        _clockmod.use(clock)  # the one process clock (§16.4)
    if stop_hooks is not None:
        _W.stop_hooks = list(stop_hooks)
    if purge_steps is not None:
        _W.purge_steps = dict(purge_steps)


def reset_wiring() -> None:
    global _W
    _W = _Wiring()
    _clockmod.use(None)


def _now() -> datetime:
    return _clockmod.now()


def _ts(dt: datetime | None = None) -> str:
    return iso(dt or _now())


def _new_id(prefix: str) -> str:
    return f"{prefix}{int(time.time() * 1000):012x}{secrets.token_hex(6)}"


# ---------------------------------------------------------------- HMAC key (§9.3)

TEST_KEY = b"koob-memory-test-hmac-key"
KEYCHAIN_ACCOUNT = "koob"
_key_override: bytes | None = None
_key_cache: bytes | None = None


def keychain_service() -> str:
    dev = ".devdata" in os.environ.get("KOOB_DATA_DIR", "")
    return "koob.memory-hmac" + (".dev" if dev else "")


def use_key(key: bytes | None) -> None:
    """Pin the key for this process (tests, tools); None returns to the default lookup."""
    global _key_override, _key_cache
    _key_override = key
    _key_cache = None


def has_key() -> bool:
    """For the doctor (§18): a new Mac has no key; ids replay without it."""
    if _key_override is not None or "PYTEST_CURRENT_TEST" in os.environ:
        return True
    import keyring

    return bool(keyring.get_password(keychain_service(), KEYCHAIN_ACCOUNT))


def hmac_key() -> bytes:
    """The Keychain key, created on first use. Under pytest: the fixed test key."""
    global _key_cache
    if _key_override is not None:
        return _key_override
    if "PYTEST_CURRENT_TEST" in os.environ:
        return TEST_KEY
    if _key_cache is None:
        import keyring

        service = keychain_service()
        value = keyring.get_password(service, KEYCHAIN_ACCOUNT)
        if not value:
            value = secrets.token_hex(32)
            keyring.set_password(service, KEYCHAIN_ACCOUNT, value)
        _key_cache = bytes.fromhex(value)
    return _key_cache


def keyed_hmac(*parts: str) -> str:
    """HMAC-SHA256 over the parts, unit-separated; hex."""
    msg = "\x1f".join(parts).encode("utf-8")
    return _hmac.new(hmac_key(), msg, hashlib.sha256).hexdigest()


def suppression_hmac(subject_id: str, family: str, value_norm: str) -> str:
    """§9.6: hmac(subject_id, family, value_norm) with the Keychain key."""
    return keyed_hmac("sup", subject_id, family, value_norm or "")


def stable_id_hmac(kind: str, value: str) -> str:
    """§9.8: a forgotten entity's stable id."""
    return keyed_hmac("stable", kind, match_norm(value).strip())


def alias_hmac(alias_norm: str, family: str, value_norm: str) -> str:
    """§9.8: an alias of a forgotten entity with a forgotten proposition."""
    return keyed_hmac("alias", alias_norm, family, value_norm or "")


# ---------------------------------------------------------------- paths (§3.1)


def memory_dir(conn: sqlite3.Connection) -> Path | None:
    """The folder of memory.db, or None for an in-memory store."""
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main":
            return Path(row[2]).parent if row[2] else None
    return None


def journal_path(conn: sqlite3.Connection) -> Path | None:
    root = memory_dir(conn)
    return root / "journal" / "forget.jsonl" if root else None


# ---------------------------------------------------------------- journal (§9.3)

_JOURNAL_ID = re.compile(r"^[A-Za-z0-9:_.\-]{1,160}$")


def append_journal(journal: Path, line: JournalLine) -> None:
    """§9.3: append-only, no content. Every string must look like an id or an HMAC, so a
    caller cannot slip text in."""
    for name in ("record_ids", "raw_hmacs", "suppression_hmacs", "turn_ids", "scope_done"):
        for value in getattr(line, name):
            if not _JOURNAL_ID.match(value):
                raise ValueError(f"journal {name} holds ids and HMACs only")
    if not _JOURNAL_ID.match(line.forget_id):
        raise ValueError("journal forget_id must be an id")
    journal.parent.mkdir(parents=True, exist_ok=True)
    data = (line.model_dump_json() + "\n").encode("utf-8")
    fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_journal(journal: Path | None) -> list[JournalLine]:
    """Every parsable line, in file order; a torn last line (a crash mid-append) is skipped."""
    if journal is None or not journal.exists():
        return []
    out: list[JournalLine] = []
    for raw in journal.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            out.append(JournalLine.model_validate_json(raw))
        except ValueError:
            continue
    return out


STAGE_RANK = {
    ForgetStage.STOPPING: -1,
    ForgetStage.STOPPED: 0,
    ForgetStage.UNDONE: 1,
    ForgetStage.PURGED: 2,
}


def journal_order(lines: Iterable[JournalLine]) -> list[JournalLine]:
    """Lines in the order they happened: by ts, then stopped < undone < purged. A journal
    merged from a backup set's copy (restore) is not in file order."""
    return sorted(lines, key=lambda x: (x.ts, STAGE_RANK[x.stage]))


def journal_max_version(journal: Path) -> int:
    """§7.1/§9.4: a restore continues from max(backup, journal, hot log) + 1."""
    return max((line.memory_version for line in read_journal(journal)), default=0)


# ---------------------------------------------------------------- stems and match keys (§9.5)

try:  # PyStemmer (Snowball), the `memory` extra; a light suffix stripper otherwise.
    import Stemmer as _pystemmer  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on the environment
    _pystemmer = None

_STEMMERS: dict[str, Any] = {}
_CYR = re.compile(r"[а-яё]")
_LAT = re.compile(r"[a-z]")
# fmt: off
_RU_ENDINGS = sorted((
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ешь", "ете", "ает", "яет", "ует",
    "ают", "яют", "ют", "ут", "ет", "ит", "ат", "ят", "ия", "ие", "ий", "ый", "ой", "ая", "яя",
    "ое", "ее", "ов", "ев", "ей", "ам", "ям", "ах", "ях", "ом", "ем", "ою", "ею", "ью", "а", "я",
    "о", "е", "ы", "и", "у", "ю", "ь", "й",
), key=len, reverse=True)
_EN_ENDINGS = ("ing", "ed", "es", "ly", "s")
_STOP = frozenset((
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "was", "were", "are",
    "his", "her", "him", "she", "they", "them", "their", "will", "would", "about", "into", "what",
    "when", "where", "which", "who", "whom", "also", "just", "very", "more", "most", "some",
    "это", "как", "что", "так", "там", "тут", "где", "когда", "который", "которая", "которые",
    "его", "она", "они", "оно", "мне", "меня", "тебя", "тебе", "себя", "свой", "своя", "быть",
    "было", "была", "были", "будет", "есть", "нет", "для", "или", "про", "при", "без", "над",
    "под", "после", "перед", "через", "между", "только", "уже", "ещё", "еще", "очень",
))
# fmt: on


def _stem(word: str) -> str:
    w = word.casefold()
    lang = "russian" if _CYR.search(w) else "english" if _LAT.search(w) else None
    if lang is None:
        return w
    if _pystemmer is not None:
        st = _STEMMERS.get(lang)
        if st is None:
            st = _STEMMERS[lang] = _pystemmer.Stemmer(lang)
        return st.stemWord(w)
    for end in _RU_ENDINGS if lang == "russian" else _EN_ENDINGS:
        if w.endswith(end) and len(w) - len(end) >= 3:
            return w[: -len(end)]
    return w


def _stems(text: str | None) -> list[str]:
    out: list[str] = []
    for tok in match_norm(text or "").split():
        if tok in _STOP or tok.isdigit():
            continue
        s = _stem(tok)
        if len(s) >= 4:
            out.append(s)
    return out


def _shingles(text: str | None, n: int = 4) -> list[str]:
    toks = match_norm(text or "").split()
    if not toks:
        return []
    if len(toks) <= n:
        return [" ".join(toks)] if len(" ".join(toks)) >= 4 else []
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def _aliases_of(conn: sqlite3.Connection, entity_ids: Iterable[str | None]) -> list[str]:
    ids = [e for e in entity_ids if e]
    if not ids:
        return []
    rows = conn.execute(
        f"SELECT alias FROM aliases WHERE entity_id IN ({_qs(ids)})", ids
    ).fetchall()
    names = [r[0] for r in rows]
    names += [
        r[0]
        for r in conn.execute(
            f"SELECT display_name FROM entities WHERE id IN ({_qs(ids)})", ids
        ).fetchall()
        if r[0] and r[0] != FORGOTTEN
    ]
    return [n for n in names if len(match_norm(n).strip()) >= 2]


def _keys_from_rows(
    conn: sqlite3.Connection, rows: list[tuple[str, str, dict[str, Any], list[str]]]
) -> list[str]:
    """Match keys from (kind_table, record_id, kind row, own aliases) tuples, live or from
    the trash: subject aliases, value phrase and stems in RU and EN, 4-word shingles of the
    quotes (§9.5)."""
    keys: set[str] = set()
    for table, _rid, row, own_aliases in rows:
        keys.update(own_aliases)
        if table == "facts":
            keys.update(_aliases_of(conn, [row.get("subject_id"), row.get("value_entity_id")]))
            value = row.get("value_text")
            if value and len(match_norm(value).strip()) >= 4:
                keys.add(value)
            for col in ("value_text", "keywords_ru", "keywords_en"):
                keys.update(_stems(row.get(col)))
            keys.update(_shingles(row.get("quote_orig")))
        elif table == "impressions":
            keys.update(_aliases_of(conn, [row.get("about_entity_id")]))
            keys.update(_shingles(row.get("quote_orig")))
        elif table == "entities":
            if row.get("display_name") and row["display_name"] != FORGOTTEN:
                keys.add(row["display_name"])
        else:
            col = _MAIN_TEXT.get(table)
            if col:
                keys.update(_shingles(row.get(col)))
        keys.update(_shingles(row.get("_origin_quote")))
    return sorted(k for k in keys if k and len(match_norm(k).strip()) >= 2)


def _live_rows(
    conn: sqlite3.Connection, record_ids: Iterable[str]
) -> list[tuple[str, str, dict[str, Any], list[str]]]:
    out = []
    for rid in record_ids:
        head = conn.execute(
            "SELECT kind, origin_quote FROM records WHERE id = ?", (rid,)
        ).fetchone()
        if head is None:
            continue
        table = _KIND_TABLE[head[0]]
        row = _row(conn, table, rid) or {}
        row["_origin_quote"] = head[1]
        own = [r[0] for r in conn.execute("SELECT alias FROM aliases WHERE entity_id = ?", (rid,))]
        out.append((table, rid, row, own))
    return out


def match_keys(conn: sqlite3.Connection, record_ids: list[str]) -> list[str]:
    """§9.5: aliases, value stems in RU and EN, 4-word shingles of quote_orig."""
    return _keys_from_rows(conn, _live_rows(conn, record_ids))


# ---------------------------------------------------------------- small SQL helpers


def _qs(items: Iterable[Any]) -> str:
    return ",".join("?" * len(list(items)))


def _row(conn: sqlite3.Connection, table: str, rid: str) -> dict[str, Any] | None:
    cur = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (rid,))
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], tuple(row), strict=True))


def _ids_in(conn: sqlite3.Connection, sql: str, ids: Iterable[str]) -> set[str]:
    ids = list(ids)
    out: set[str] = set()
    for i in range(0, len(ids), 400):
        chunk = ids[i : i + 400]
        out.update(r[0] for r in conn.execute(sql.format(qs=_qs(chunk)), chunk))
    return out


def _family(conn: sqlite3.Connection, relation: str | None) -> str:
    if not relation:
        return ""
    row = conn.execute("SELECT family FROM relations WHERE relation = ?", (relation,)).fetchone()
    return row[0] if row else relation


def _value_norm(row: dict[str, Any]) -> str:
    if row.get("value_norm"):
        return str(row["value_norm"])
    if row.get("value_entity_id"):
        return str(row["value_entity_id"])
    return match_norm(row.get("value_text") or "").strip()


def proposition(
    conn: sqlite3.Connection, kind: str, row: dict[str, Any]
) -> tuple[str, str, str] | None:
    """(subject, family, value_norm) of a kind row: what a forget suppresses and what
    ledger's commit checks through `blocked_by` (§9.6), from one definition. Facts: the
    subject, the relation's family, the normalized value; impressions: the entity it is
    about; decisions, loops, lessons, criteria, episodes: the kind and its main text.
    None for kinds without a proposition (entities: their stable ids, work states...)."""
    table = _KIND_TABLE.get(kind)
    if table == "facts":
        return (
            str(row.get("subject_id") or ""),
            _family(conn, row.get("relation")),
            _value_norm(row),
        )
    if table == "impressions":
        val = match_norm(row.get("text_en") or "").strip()
        return str(row.get("about_entity_id") or ""), "impression", val
    if table in _MAIN_TEXT:
        return kind, "", match_norm(row.get(_MAIN_TEXT[table]) or "").strip()
    return None


def _locate(text: str, quote: str | None) -> tuple[int, int] | None:
    """The span of `quote` in `text`: exact, then case- and whitespace-insensitive."""
    if not quote or not text:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    words = [re.escape(w) for w in quote.split()]
    if not words:
        return None
    m = re.search(r"\s+".join(words), text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


# Case endings a Russian name takes («Макс» → «Макса», «Саша» → «Сашей»); «им» is left out
# on purpose: «Макс» + «им» is another name.
_NAME_ENDINGS = "|".join(
    sorted(
        ("а", "я", "у", "ю", "ом", "ем", "ём", "е", "ы", "и", "ой", "ей", "ою", "ею", "ам",
         "ям", "ами", "ями", "ах", "ях", "ов", "ев", "ь", "й"),
        key=len, reverse=True,
    )
)  # fmt: skip
_STEM_LAST = frozenset("аяьйоеё")


def _name_alts(terms: Iterable[str]) -> str:
    """One alternation for X's names and stable ids as whole words: a Cyrillic name with
    its case endings (the final vowel swapped too: «Саша» → «Сашу»), anything else as is."""
    alts: set[str] = set()
    for term in terms:
        if not term:
            continue
        t = re.escape(term)
        if _CYR.search(term[-1:].casefold()):
            alts.add(rf"{t}(?:{_NAME_ENDINGS})?")
            if term[-1].casefold() in _STEM_LAST and len(term) >= 3:
                alts.add(rf"{re.escape(term[:-1])}(?:{_NAME_ENDINGS})")
        else:
            alts.add(t)
    return "|".join(sorted(alts, key=len, reverse=True))


def _name_pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    """Whole words only: «Макс» never takes «Максим», «Ира» never «Ирина»."""
    alts = _name_alts(terms)
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE) if alts else None


def _mask_text(value: Any, terms: list[str]) -> Any:
    if not isinstance(value, str) or not terms:
        return value
    pattern = _name_pattern(terms)
    return pattern.sub(FORGOTTEN, value) if pattern is not None else value


def _stem_for_fts(text: str) -> str:
    """The FTS form of a text: lex's stemmer when built, ours otherwise."""
    try:
        from .recall import lexical

        return lexical.stem(text)
    except NotImplementedError:
        return " ".join(_stem(t) for t in match_norm(text).split())


# ---------------------------------------------------------------- the plan (§9.1 stage 1a)


@dataclass
class _Plan:
    forget_id: str
    scope: ForgetScope
    records: list[str] = field(default_factory=list)
    masked: list[str] = field(default_factory=list)
    raw_forgotten: list[str] = field(default_factory=list)
    raw_redacted: list[str] = field(default_factory=list)
    raw_masked: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)
    derived_masked: list[str] = field(default_factory=list)
    raw_hmacs: list[str] = field(default_factory=list)
    suppressions: list[str] = field(default_factory=list)
    labels: int = 0
    keys: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)  # X's names and stable ids (§9.8)
    xs: list[str] = field(default_factory=list)  # entity ids forgotten whole (§9.8)
    redact_quotes: dict[str, list[str]] = field(default_factory=dict)  # RAM only
    turns: list[str] = field(default_factory=list)
    threads: list[str] = field(default_factory=list)
    session_threads: list[str] = field(default_factory=list)
    first_turn: dict[str, str] = field(default_factory=dict)
    until: str = ""
    version: int = 0
    targets: list[str] = field(default_factory=list)  # what the user named (journal, §9.3 F1)
    claimed: list[str] = field(default_factory=list)  # in scope, held by an earlier forget
    claimed_sups: list[str] = field(default_factory=list)


def _lineage_raw(conn: sqlite3.Connection, rid: str) -> set[str]:
    cand = {
        r[0]
        for r in conn.execute(
            "SELECT source_raw_id FROM facts WHERE id = ? AND source_raw_id IS NOT NULL", (rid,)
        )
    }
    cand |= {
        r[0]
        for r in conn.execute(
            "SELECT to_id FROM edges WHERE from_id = ? AND etype IN ('derived_from', 'supports') "
            "UNION SELECT from_id FROM edges WHERE to_id = ? AND etype = 'supports' "
            "UNION SELECT member_record_id FROM justifications WHERE target_record_id = ?",
            (rid, rid, rid),
        )
    }
    return _ids_in(conn, "SELECT id FROM raw_chunks WHERE id IN ({qs})", cand) if cand else set()


def _raw_users(conn: sqlite3.Connection, raw_id: str) -> set[str]:
    users = {r[0] for r in conn.execute("SELECT id FROM facts WHERE source_raw_id = ?", (raw_id,))}
    users |= {
        r[0]
        for r in conn.execute(
            "SELECT from_id FROM edges WHERE to_id = ? AND etype IN ('derived_from', 'supports') "
            "UNION SELECT to_id FROM edges WHERE from_id = ? AND etype = 'supports' "
            "UNION SELECT target_record_id FROM justifications WHERE member_record_id = ?",
            (raw_id, raw_id, raw_id),
        )
    }
    return _ids_in(conn, "SELECT id FROM records WHERE id IN ({qs})", users) if users else set()


def _minted_only(
    conn: sqlite3.Connection, ids: Iterable[str], recs: set[str], raws: set[str]
) -> set[str]:
    """: the entities (with their aliases) and impressions that exist only because
    of messages this forget takes: every message they came from is forgotten, and no usable
    record outside the forget names the entity. The user, an entity with stable ids (a
    contact) and anything with another source stay."""
    out: set[str] = set()
    for rid in ids:
        row = conn.execute("SELECT kind FROM records WHERE id = ?", (rid,)).fetchone()
        if row is None or row[0] not in ("entity", "impression") or rid == OWNER_ENTITY_ID:
            continue
        src = _lineage_raw(conn, rid)
        if not src or not src <= raws:
            continue
        if row[0] == "entity":
            sid = conn.execute("SELECT stable_ids FROM entities WHERE id = ?", (rid,)).fetchone()
            if sid is None or (sid[0] or "[]") != "[]":
                continue
            named = _usable(conn, _names_entity(conn, rid))
            if named - recs:
                continue
        out.add(rid)
    return out


def _names_entity(conn: sqlite3.Connection, eid: str) -> set[str]:
    """Records that name an entity: as subject, speaker or value, an impression about it, or
    an episode linked to it."""
    ids = {
        r[0]
        for r in conn.execute(
            "SELECT id FROM facts WHERE ? IN (subject_id, speaker_entity_id, value_entity_id)"
            " UNION SELECT id FROM impressions WHERE about_entity_id = ?"
            " UNION SELECT episode_id FROM episode_links WHERE target_id = ?",
            (eid, eid, eid),
        )
    }
    return ids


def _usable(conn: sqlite3.Connection, ids: Iterable[str]) -> set[str]:
    return _ids_in(
        conn, "SELECT id FROM records WHERE id IN ({qs}) AND status IN ('active', 'archived')", ids
    )


def _history(conn: sqlite3.Connection, recs: set[str]) -> set[str]:
    """§9.1 (v0.4, A14): every record these superseded, transitively along `supersedes`
    edges (new -> old): the slot's history goes with its target. Never a successor."""
    out = set(recs)
    frontier = set(recs)
    while frontier:
        nxt = _ids_in(
            conn, "SELECT to_id FROM edges WHERE from_id IN ({qs}) AND etype = 'supersedes'",
            frontier,
        )  # fmt: skip
        frontier = nxt - out
        out |= frontier
    return out


def _fact_key(conn: sqlite3.Connection, rid: str) -> tuple[str, tuple[str, str, str]] | None:
    """(kind table, proposition) of a record still holding its content; None for kinds
    without a proposition or one with an empty subject or value (never a wildcard)."""
    row = conn.execute("SELECT kind, status FROM records WHERE id = ?", (rid,)).fetchone()
    table = _KIND_TABLE.get(row[0]) if row and row[1] not in ("forgotten", "purged") else None
    if table is None:
        return None  # a forgotten row is blanked: its proposition lives in the trash
    prop = proposition(conn, row[0], _row(conn, table, rid) or {})
    if prop is None or not prop[0] or not prop[2]:
        return None
    return table, prop


def same_fact(conn: sqlite3.Connection, ids: Iterable[str]) -> bool:
    """True when these records are copies of one proposition (`commands._resolve_target`:
    identical candidates are one target)."""
    keys = {_fact_key(conn, rid) for rid in ids}
    return len(keys) == 1 and None not in keys


def _copies(conn: sqlite3.Connection, recs: set[str]) -> set[str]:
    """Every record stating the same proposition as one of these: «забудь» means the fact,
    not one copy - same subject entity, relation family and normalized value
    (`proposition`, the suppression's own definition), any status, any source. Copies an
    earlier forget still holds in its trash come back as ids too, so this forget claims
    them. A different value is another fact: a successor never comes this way (A14)."""
    keys = {k for k in (_fact_key(conn, rid) for rid in sorted(recs)) if k is not None}
    out: set[str] = set()
    for table, prop in keys:
        subject, _fam, _val = prop
        if table == "facts":
            sql = "SELECT k.*, r.kind AS _kind FROM facts k JOIN records r ON r.id = k.id "
            sql += "WHERE k.subject_id = ?"
            args: tuple[Any, ...] = (subject,)
        elif table == "impressions":
            sql = "SELECT k.*, r.kind AS _kind FROM impressions k JOIN records r ON r.id = k.id "
            sql += "WHERE k.about_entity_id = ?"
            args = (subject,)
        else:
            sql = f"SELECT k.*, r.kind AS _kind FROM {table} k JOIN records r ON r.id = k.id "
            sql += "WHERE r.kind = ?"
            args = (subject,)
        cur = conn.execute(sql + " AND r.status NOT IN ('forgotten', 'purged')", args)
        cols = [d[0] for d in cur.description]
        for values in cur.fetchall():
            row = dict(zip(cols, values, strict=True))
            if proposition(conn, row.pop("_kind"), row) == prop:
                out.add(str(row["id"]))
        # copies already forgotten by an earlier forget, still in its trash (`stopped`)
        owners = [
            r[0]
            for r in conn.execute(
                "SELECT forget_id FROM suppressions WHERE hmac = ?", (suppression_hmac(*prop),)
            )
        ]
        for ref, payload in (
            conn.execute(
                f"SELECT row_ref, payload FROM trash WHERE store = 'memory' AND forget_id IN "
                f"({_qs(owners)}) AND row_ref LIKE 'record:%' AND payload IS NOT NULL",
                owners,
            ).fetchall()
            if owners
            else []
        ):
            t = {"row_ref": ref, "payload": json.loads(payload)}
            if (t["payload"] or {}).get("table") == table and _trash_proposition(conn, t) == prop:
                out.add(ref.partition(":")[2])
    return out


def _fact_closure(conn: sqlite3.Connection, recs: set[str]) -> set[str]:
    """The named records, their history (A14) and every copy of each, to a fixpoint: a
    copy's own predecessors are the same slot's history. Never a successor."""
    out = _history(conn, recs)
    for _ in range(16):
        more = _history(conn, _copies(conn, out) - out) - out
        if not more:
            break
        out |= more
    return out


def _resolve_targets(
    conn: sqlite3.Connection, targets: list[str], scope: ForgetScope
) -> tuple[set[str], set[str], set[str], set[str]]:
    """(records forgotten whole, raw forgotten whole, entity ids X for §9.8, ids in scope
    an earlier forget already holds in its trash: this forget claims them).

    A superseded or retracted record is a valid target (history is content); a record
    named (or about X) takes its `supersedes` predecessors along (§9.1, A14) and every
    copy of its proposition from any source, in any status; never a successor or another
    value. A chat's forget takes only what came from that chat, and a raw target only
    what was learned from it (a message is not the fact)."""
    alive = "status NOT IN ('forgotten', 'purged')"
    raws: set[str] = set()
    xs: set[str] = set()
    if scope is ForgetScope.CHAT:
        raws = _ids_in(conn, "SELECT id FROM raw_chunks WHERE chat_id IN ({qs})", targets)
        turns = _ids_in(conn, "SELECT turn_id FROM turn_sources WHERE chat_id IN ({qs})", targets)
        if turns:
            raws |= _ids_in(conn, "SELECT id FROM raw_chunks WHERE turn_id IN ({qs})", turns)
        # a mode change hands over the exact items policy.mode_scope picked (§5.2): the user's
        # own words in those turns stay, so item ids are taken as they are
        raws |= _ids_in(conn, "SELECT id FROM raw_chunks WHERE id IN ({qs})", targets)
        recs: set[str] = _ids_in(conn, "SELECT id FROM records WHERE id IN ({qs})", targets)
    elif scope is ForgetScope.ABOUT_ENTITY:
        xs = _ids_in(conn, "SELECT id FROM entities WHERE id IN ({qs})", targets)
        recs = set(xs)
        if xs:
            recs |= _ids_in(conn, "SELECT id FROM facts WHERE subject_id IN ({qs})", xs)
            recs |= _ids_in(conn, "SELECT id FROM facts WHERE value_entity_id IN ({qs})", xs)
            recs |= _ids_in(conn, "SELECT id FROM impressions WHERE about_entity_id IN ({qs})", xs)
            recs |= _ids_in(
                conn, "SELECT from_id FROM edges WHERE to_id IN ({qs}) AND etype = 'about'", xs
            )
            recs |= _ids_in(
                conn,
                "SELECT episode_id FROM episode_links WHERE target_id IN ({qs}) AND role = 'about'",
                xs,
            )
            recs = _ids_in(conn, "SELECT id FROM records WHERE id IN ({qs})", recs)
        recs = _fact_closure(conn, recs)  # a merged-away duplicate of X is X too
    else:
        named = _ids_in(conn, "SELECT id FROM records WHERE id IN ({qs})", targets)
        recs = _fact_closure(conn, named)  # its history and every copy of the fact
        raws = _ids_in(conn, "SELECT id FROM raw_chunks WHERE id IN ({qs})", targets)
    in_scope = set(raws)
    raws = _ids_in(conn, "SELECT id FROM raw_chunks WHERE id IN ({qs}) AND " + alive, raws)
    # what was learned from a forgotten message goes with it; a chat's forget also takes
    # what was learned from its messages an earlier forget already holds
    for raw_id in sorted(in_scope if scope is ForgetScope.CHAT else raws):
        recs |= _raw_users(conn, raw_id)
    held = "status = 'forgotten'"
    claimed = _ids_in(conn, "SELECT id FROM records WHERE id IN ({qs}) AND " + held, recs)
    claimed |= _ids_in(conn, "SELECT id FROM raw_chunks WHERE id IN ({qs}) AND " + held, in_scope)
    recs = _ids_in(conn, "SELECT id FROM records WHERE id IN ({qs}) AND " + alive, recs)
    return recs, raws, xs, claimed


def _cascade_derived_records(conn: sqlite3.Connection, recs: set[str]) -> set[str]:
    """Records built from forgotten records (`derived_from` edges), to a fixpoint."""
    out = set(recs)
    frontier = set(recs)
    for _ in range(32):
        if not frontier:
            break
        nxt = _ids_in(
            conn,
            "SELECT e.from_id FROM edges e JOIN records r ON r.id = e.from_id "
            "WHERE e.to_id IN ({qs}) AND e.etype = 'derived_from' "
            "AND r.status NOT IN ('forgotten', 'purged')",
            frontier,
        )
        frontier = nxt - out
        out |= frontier
    return out


def _x_terms(conn: sqlite3.Connection, xs: set[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """X's aliases (and display names) and stable ids."""
    if not xs:
        return [], []
    names = _aliases_of(conn, xs)
    stable: list[tuple[str, str]] = []
    for (raw,) in conn.execute(
        f"SELECT stable_ids FROM entities WHERE id IN ({_qs(xs)})", list(xs)
    ):
        try:
            for sid in json.loads(raw or "[]"):
                if sid.get("value"):
                    stable.append((str(sid.get("kind", "")), str(sid["value"])))
        except (ValueError, AttributeError):
            continue
    return sorted(set(names), key=len, reverse=True), stable


def _mentions(conn: sqlite3.Connection, xs: set[str], terms: list[str], skip: set[str]) -> set[str]:
    """§9.8: records about something else where X is speaker, author, participant or a name."""
    if not xs:
        return set()
    out = _ids_in(conn, "SELECT id FROM facts WHERE speaker_entity_id IN ({qs})", xs)
    out |= _ids_in(conn, "SELECT id FROM records WHERE author_entity_id IN ({qs})", xs)
    out |= _ids_in(
        conn,
        "SELECT episode_id FROM episode_links WHERE target_id IN ({qs}) AND role = 'participant'",
        xs,
    )
    # whole words: prefix matching is for turns, where over-redaction is accepted
    alts = _name_alts(match_norm(t).strip() for t in terms)
    pattern = re.compile(rf"(?<=\s)(?:{alts})(?=\s)") if alts else None
    if pattern is not None:
        for table, cols in _MASK_COLS.items():
            cur = conn.execute(
                f"SELECT k.id, {', '.join('k.' + c for c in cols)} FROM {table} k "
                "JOIN records r ON r.id = k.id WHERE r.status NOT IN ('forgotten', 'purged')"
            )
            for row in cur:
                if any(isinstance(v, str) and text_matches(v, pattern) for v in row[1:]):
                    out.add(row[0])
    out = _ids_in(
        conn,
        "SELECT id FROM records WHERE id IN ({qs}) AND status NOT IN ('forgotten', 'purged')",
        out,
    )
    return out - skip


def _plan(
    conn: sqlite3.Connection,
    targets: list[str],
    scope: ForgetScope,
    forget_id: str,
    *,
    extra_records: Iterable[str] = (),
    extra_raws: Iterable[str] = (),
    extra_masked: Iterable[str] = (),
) -> _Plan:
    """The stop-use plan. The `extra_*` ids come from a replayed journal line (§9.3 F1):
    records it forgot whole, raw it forgot or redacted (decided again here: redacted when
    other usable records still read it) and records it masked (re-masked against X)."""
    plan = _Plan(forget_id=forget_id, scope=scope, targets=list(targets))
    recs, raws, xs, claimed = _resolve_targets(conn, targets, scope)
    plan.claimed = sorted(claimed)
    recs |= _ids_in(
        conn, "SELECT id FROM records WHERE id IN ({qs}) AND status NOT IN ('forgotten', 'purged')",
        extra_records,
    )  # fmt: skip
    direct = set(recs)  # raw follows these; records built from them follow by cascade
    recs = _cascade_derived_records(conn, recs)
    terms, stable = _x_terms(conn, xs)
    plan.xs = sorted(xs)
    plan.terms = terms + [v for _k, v in stable]
    masked = _mentions(conn, xs, plan.terms, recs)
    if xs:
        masked |= (
            _ids_in(
                conn,
                "SELECT id FROM records WHERE id IN ({qs}) AND status NOT IN ('forgotten', 'purged')",
                extra_masked,
            )
            - recs
        )
    plan.records = sorted(recs)
    plan.masked = sorted(masked)
    plan.keys = match_keys(conn, plan.records)

    # §9.6 raw follows its records.
    redact: dict[str, list[str]] = {}
    quote_map: dict[str, list[str]] = {}
    minted_all: set[str] = set()
    for rid in sorted(direct):
        for raw_id in _lineage_raw(conn, rid):
            if raw_id in raws:
                continue
            others = _usable(conn, _raw_users(conn, raw_id) - recs)
            minted = _minted_only(conn, others, recs | {rid}, raws | {raw_id})
            names = _ids_in(conn, "SELECT id FROM records WHERE id IN ({qs}) AND kind = 'entity'",
                            others)  # fmt: skip
            if others - minted - names:  # a name read from a message never keeps it alive
                redact.setdefault(raw_id, []).append(rid)
            else:
                raws.add(raw_id)
                minted_all |= minted
    for raw_id in sorted(raws):  # a forgotten message takes what was minted only from it
        minted_all |= _minted_only(conn, _usable(conn, _raw_users(conn, raw_id) - recs), recs, raws)
    if minted_all - recs:
        recs |= minted_all
        plan.records = sorted(recs)
        plan.keys = match_keys(conn, plan.records)
    extra_raw = _ids_in(
        conn, "SELECT id FROM raw_chunks WHERE id IN ({qs}) AND status = 'active'", extra_raws
    )
    for raw_id in sorted(extra_raw):
        if raw_id in raws or raw_id in redact:
            continue
        users = _raw_users(conn, raw_id)
        mine = sorted(users & recs)
        if mine and _usable(conn, users - recs):
            redact[raw_id] = mine  # the same span again, never the whole message
        else:
            raws.add(raw_id)
    for raw_id in list(redact):
        if raw_id in raws:
            redact.pop(raw_id)
            continue
        text = conn.execute("SELECT text FROM raw_chunks WHERE id = ?", (raw_id,)).fetchone()[0]
        quotes = [
            (_row(conn, _KIND_TABLE[k], r) or {}).get("quote_orig")
            for r in redact[raw_id]
            for (k,) in conn.execute("SELECT kind FROM records WHERE id = ?", (r,))
        ]
        if not text or not all(_locate(text, q) for q in quotes):
            raws.add(raw_id)
            redact.pop(raw_id)
        else:
            quote_map[raw_id] = [q for q in quotes if q]
    raws = _ids_in(conn, "SELECT id FROM raw_chunks WHERE id IN ({qs}) AND status = 'active'", raws)
    plan.raw_forgotten = sorted(raws)
    plan.raw_redacted = sorted(r for r in redact if r not in raws)
    plan.redact_quotes = {r: quote_map[r] for r in plan.raw_redacted}
    if xs:
        masked_raw: set[str] = set()
        for rid in plan.masked:
            masked_raw |= _lineage_raw(conn, rid)
        plan.raw_masked = sorted(masked_raw - raws - set(plan.raw_redacted))

    # Derived items reading any of it (§9.1a, §7.3).
    gone = set(plan.records) | raws | set(plan.raw_redacted)
    plan.derived = sorted(
        _ids_in(
            conn,
            "SELECT holder_id FROM read_deps WHERE holder_kind = 'derived' AND item_id IN ({qs})",
            gone,
        )
    )
    if xs:
        plan.derived_masked = sorted(
            _ids_in(
                conn,
                "SELECT holder_id FROM read_deps WHERE holder_kind = 'derived' "
                "AND item_id IN ({qs})",
                set(plan.masked) | set(plan.raw_masked),
            )
            - set(plan.derived)
        )
    plan.raw_hmacs = sorted(
        _ids_in(
            conn,
            "SELECT content_hmac FROM raw_chunks WHERE id IN ({qs})",
            raws | set(plan.raw_redacted),
        )
    )
    labels = 0
    for (lab,) in conn.execute(
        f"SELECT labels FROM records WHERE id IN ({_qs(plan.records)})", plan.records
    ):
        labels |= int(lab)
    for (lab,) in conn.execute(
        f"SELECT labels FROM raw_chunks WHERE id IN ({_qs(plan.raw_forgotten)})",
        plan.raw_forgotten,
    ):
        labels |= int(lab)
    plan.labels = labels

    # Suppressions (§9.6, §9.8).
    sups: set[str] = set()
    alias_norms = (
        [
            r[0]
            for r in conn.execute(
                f"SELECT alias_norm FROM aliases WHERE entity_id IN ({_qs(xs)})", list(xs)
            )
        ]
        if xs
        else []
    )
    for rid in plan.records:
        (kind,) = conn.execute("SELECT kind FROM records WHERE id = ?", (rid,)).fetchone()
        prop = proposition(conn, kind, _row(conn, _KIND_TABLE[kind], rid) or {})
        if prop is None:
            continue
        subject, fam, val = prop
        sups.add(suppression_hmac(subject, fam, val))
        for a in alias_norms:
            sups.add(alias_hmac(a, fam if fam else kind, val))
    for kind, value in stable:
        sups.add(stable_id_hmac(kind, value))
    claimed_sups: set[str] = set()
    for t in _held_trash(conn, plan.claimed, forget_id):
        prop = _trash_proposition(conn, t)
        if prop is not None:
            subject, fam, val = prop
            claimed_sups.add(suppression_hmac(subject, fam, val))
            kind = t["row_ref"].partition(":")[2]
            for a in alias_norms:
                sups.add(alias_hmac(a, fam if fam else _record_kind(conn, kind), val))
    plan.claimed_sups = sorted(claimed_sups)
    plan.suppressions = sorted(sups | claimed_sups)

    # Served sets (§9.5, §9.1c): turns and threads that saw a forgotten or masked id.
    served_items = set(plan.records) | set(plan.derived)
    for thread_id, turn_id, created in (
        conn.execute(
            f"SELECT thread_id, turn_id, created_at FROM served WHERE item_id IN ({_qs(served_items)}) "
            "ORDER BY created_at",
            list(served_items),
        )
        if served_items
        else []
    ):
        plan.threads.append(thread_id)
        if turn_id:
            plan.turns.append(turn_id)
            plan.first_turn.setdefault(thread_id, turn_id)
        del created
    sess_items = set(plan.masked) | set(plan.derived_masked)
    if sess_items:
        plan.session_threads = sorted(
            _ids_in(conn, "SELECT thread_id FROM served WHERE item_id IN ({qs})", sess_items)
        )
    if scope is not ForgetScope.CHAT:
        lineage_raw = raws | set(plan.raw_redacted)
        for turn_id, chat_id in (
            conn.execute(
                f"SELECT turn_id, chat_id FROM raw_chunks WHERE id IN ({_qs(lineage_raw)}) "
                "AND turn_id IS NOT NULL ORDER BY ts",
                list(lineage_raw),
            )
            if lineage_raw
            else []
        ):
            plan.turns.append(turn_id)
            if chat_id:
                plan.first_turn.setdefault(chat_id, turn_id)
    plan.turns = sorted(set(plan.turns))
    plan.threads = sorted(set(plan.threads))
    return plan


# ---------------------------------------------------------------- applying stop-use


def _trash(
    conn: sqlite3.Connection, fid: str, store_name: str, ref: str, payload: Any, until: str
) -> None:
    conn.execute(
        "INSERT INTO trash (id, forget_id, store, row_ref, payload, until, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            _new_id("tr"),
            fid,
            store_name,
            ref,
            json.dumps(payload, ensure_ascii=False),
            until,
            _ts(),
        ),
    )


def _fts_row(conn: sqlite3.Connection, table: str, rowid: int | None) -> dict[str, Any] | None:
    if rowid is None:
        return None
    cur = conn.execute(f"SELECT * FROM {table} WHERE rowid = ?", (rowid,))
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], tuple(row), strict=True))


def _apply_stop(conn: sqlite3.Connection, plan: _Plan, version: int) -> list[str]:
    ts, until, fid = _ts(), plan.until, plan.forget_id
    touched: list[str] = []
    for rid in plan.records:
        head = _row(conn, "records", rid)
        assert head is not None
        table = _KIND_TABLE[head["kind"]]
        kind_row = _row(conn, table, rid)
        aliases = [
            dict(zip([d[0] for d in c.description], tuple(r), strict=True))
            for c in [conn.execute("SELECT * FROM aliases WHERE entity_id = ?", (rid,))]
            for r in c.fetchall()
        ]
        names_fts = [
            {"rowid": a["fts_rowid"], **(_fts_row(conn, "fts_names", a["fts_rowid"]) or {})}
            for a in aliases
            if a.get("fts_rowid") is not None
        ]
        links = [
            {"episode_id": r[0], "role": r[1], "target_id": r[2]}
            for r in conn.execute(
                "SELECT episode_id, role, target_id FROM episode_links WHERE episode_id = ?",
                (rid,),
            )
        ]
        _trash(
            conn,
            fid,
            "memory",
            f"record:{rid}",
            {
                "header": {
                    k: head[k] for k in ("status", "origin_quote", "retired_at", "author_entity_id")
                },
                "table": table,
                "row": kind_row,
                "aliases": aliases,
                "fts": _fts_row(conn, "fts_records", head["fts_rowid"]),
                "names_fts": names_fts,
                "links": links,
            },
            until,
        )
        blank = _BLANK[table]
        conn.execute(
            f"UPDATE {table} SET {', '.join(f'{c} = ?' for c in blank)} WHERE id = ?",
            [*blank.values(), rid],
        )
        for a in names_fts:
            conn.execute("DELETE FROM fts_names WHERE rowid = ?", (a["rowid"],))
        conn.execute("DELETE FROM aliases WHERE entity_id = ?", (rid,))
        if head["fts_rowid"] is not None:
            conn.execute("DELETE FROM fts_records WHERE rowid = ?", (head["fts_rowid"],))
        conn.execute(
            "UPDATE records SET status = 'forgotten', version = ?, origin_quote = NULL, "
            "retired_at = ?, updated_at = ? WHERE id = ?",
            (version, ts, ts, rid),
        )
        touched.append(rid)

    for rid in plan.masked:
        head = _row(conn, "records", rid)
        assert head is not None
        table = _KIND_TABLE[head["kind"]]
        kind_row = _row(conn, table, rid) or {}
        cols = _MASK_COLS.get(table, ())
        new = {c: _mask_text(kind_row.get(c), plan.terms) for c in cols}
        changed = {c: kind_row.get(c) for c in cols if new[c] != kind_row.get(c)}
        speaker = kind_row.get("speaker_entity_id") if table == "facts" else None
        xs_speaker = speaker is not None and speaker in set(_x_ids(plan))
        links = [
            {"episode_id": r[0], "role": r[1], "target_id": r[2]}
            for r in conn.execute(
                f"SELECT episode_id, role, target_id FROM episode_links WHERE episode_id = ? "
                f"AND role = 'participant' AND target_id IN ({_qs(_x_ids(plan))})",
                (rid, *_x_ids(plan)),
            )
        ]
        fts = _fts_row(conn, "fts_records", head["fts_rowid"])
        _trash(
            conn,
            fid,
            "memory",
            f"mask:{rid}",
            {
                "table": table,
                "cols": changed,
                "speaker": speaker if xs_speaker else None,
                "author_entity_id": head["author_entity_id"],
                "origin_quote": head["origin_quote"],
                "links": links,
                "fts": fts,
            },
            until,
        )
        if changed:
            conn.execute(
                f"UPDATE {table} SET {', '.join(f'{c} = ?' for c in changed)} WHERE id = ?",
                [new[c] for c in changed] + [rid],
            )
        if xs_speaker:
            conn.execute("UPDATE facts SET speaker_entity_id = NULL WHERE id = ?", (rid,))
        if links:
            conn.execute(
                f"DELETE FROM episode_links WHERE episode_id = ? AND role = 'participant' "
                f"AND target_id IN ({_qs(_x_ids(plan))})",
                (rid, *_x_ids(plan)),
            )
        author = head["author_entity_id"] if head["author_entity_id"] in _x_ids(plan) else None
        conn.execute(
            "UPDATE records SET version = ?, origin_quote = ?, author_entity_id = ?, "
            "updated_at = ? WHERE id = ?",
            (
                version,
                _mask_text(head["origin_quote"], plan.terms),
                None if author else head["author_entity_id"],
                ts,
                rid,
            ),
        )
        if fts is not None:
            stems = {_stem(t) for term in plan.terms for t in match_norm(term).split()}
            words = {t for term in plan.terms for t in match_norm(term).split()}
            drop = stems | words
            masked_fts = {
                k: " ".join(w for w in str(v).split() if w.casefold() not in drop)
                for k, v in fts.items()
                if isinstance(v, str)
            }
            masked_fts = {k: _mask_text(v, plan.terms) for k, v in masked_fts.items()}
            conn.execute("DELETE FROM fts_records WHERE rowid = ?", (head["fts_rowid"],))
            conn.execute(
                "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, ?, ?)",
                (
                    head["fts_rowid"],
                    masked_fts.get("text", ""),
                    masked_fts.get("keywords", ""),
                    masked_fts.get("aliases", ""),
                ),
            )
        touched.append(rid)

    for raw_id in plan.raw_forgotten:
        raw = _row(conn, "raw_chunks", raw_id)
        assert raw is not None
        _trash(
            conn,
            fid,
            "memory",
            f"raw:{raw_id}",
            # the file path, page URL or subject line names it too; kind and chat stay
            {
                "text": raw["text"],
                "source_ref": raw.get("source_ref"),
                "status": raw["status"],
                "mode": "forget",
            },
            until,
        )
        if raw["fts_rowid"] is not None:
            conn.execute("DELETE FROM fts_raw WHERE rowid = ?", (raw["fts_rowid"],))
        conn.execute(
            "UPDATE raw_chunks SET status = 'forgotten', text = NULL, source_ref = NULL, "
            "rev = rev + 1, updated_at = ? WHERE id = ?",
            (ts, raw_id),
        )
        touched.append(raw_id)

    for raw_id in plan.raw_redacted + plan.raw_masked:
        raw = _row(conn, "raw_chunks", raw_id)
        assert raw is not None
        text = raw["text"] or ""
        if raw_id in plan.raw_redacted:
            for quote in plan.redact_quotes.get(raw_id, []):
                span = _locate(text, quote)
                if span:
                    text = text[: span[0]] + FORGOTTEN + text[span[1] :]
        text = _mask_text(text, plan.terms) if plan.terms else text
        if text == raw["text"]:
            continue
        _trash(conn, fid, "memory", f"raw:{raw_id}", {"text": raw["text"], "mode": "redact"}, until)
        if raw["fts_rowid"] is not None:
            conn.execute("DELETE FROM fts_raw WHERE rowid = ?", (raw["fts_rowid"],))
            conn.execute(
                "INSERT INTO fts_raw (rowid, text) VALUES (?, ?)",
                (raw["fts_rowid"], _stem_for_fts(text)),
            )
        conn.execute(
            "UPDATE raw_chunks SET text = ?, rev = rev + 1, updated_at = ? WHERE id = ?",
            (text, ts, raw_id),
        )
        touched.append(raw_id)

    for did in plan.derived + plan.derived_masked:
        d = _row(conn, "derived", did)
        if d is None:
            continue
        mode = "forget" if did in plan.derived else "mask"
        new = FORGOTTEN if mode == "forget" else _mask_text(d["content"], plan.terms)
        _trash(
            conn,
            fid,
            "memory",
            f"derived:{did}",
            {"content": d["content"], "valid": d["valid"], "mode": mode, "rev": d.get("rev")},
            until,
        )
        conn.execute(
            "UPDATE derived SET content = ?, valid = 0, updated_at = ? WHERE id = ?",
            (new, ts, did),
        )
        touched.append(did)

    # The retraction overlay (§9.6) holds the corrected value in plain text: it leaves
    # with the record it corrected and with the message it came from.
    for raw_id, rec_id, value, created in conn.execute(
        f"SELECT raw_id, record_id, retracted_value_norm, created_at FROM raw_retractions "
        f"WHERE record_id IN ({_qs(plan.records)}) OR raw_id IN ({_qs(plan.raw_forgotten)})",
        [*plan.records, *plan.raw_forgotten],
    ).fetchall():
        _trash(
            conn, fid, "memory", f"retraction:{rec_id}",
            {"raw_id": raw_id, "record_id": rec_id, "value": value, "created_at": created},
            until,
        )  # fmt: skip
        conn.execute(
            "DELETE FROM raw_retractions WHERE raw_id = ? AND record_id = ? "
            "AND retracted_value_norm = ?",
            (raw_id, rec_id, value),
        )

    # Action summaries naming the forgotten (§9.2: action summaries).
    gone = set(plan.records) | set(plan.raw_forgotten)
    acts = _ids_in(
        conn,
        "SELECT holder_id FROM read_deps WHERE holder_kind = 'action' AND item_id IN ({qs})",
        gone,
    ) | _ids_in(conn, "SELECT id FROM actions WHERE target_entity_id IN ({qs})", gone)
    for aid in sorted(acts):
        a = _row(conn, "actions", aid) or _row_by(conn, "actions", "action_key", aid)
        if a is None or a.get("summary_en") is None:
            continue
        _trash(conn, fid, "memory", f"action:{a['id']}", {"summary_en": a["summary_en"]}, until)
        conn.execute("UPDATE actions SET summary_en = NULL WHERE id = ?", (a["id"],))

    # §7.6 (v0.4): a forget never leaves a thread's gate shut - its pending updates of
    # forgotten or masked ids go in this transaction; the session restarts instead (1c)
    ids = (
        set(plan.records) | set(plan.masked) | set(plan.derived) | set(plan.derived_masked)
        | set(plan.raw_forgotten) | set(plan.raw_redacted) | set(plan.raw_masked)
    )  # fmt: skip
    if ids:
        conn.execute(f"DELETE FROM pending_updates WHERE item_id IN ({_qs(ids)})", list(ids))
    for h in plan.suppressions:
        conn.execute(
            "INSERT OR IGNORE INTO suppressions (hmac, forget_id, created_at) VALUES (?, ?, ?)",
            (h, fid, ts),
        )
    # A wider later forget takes over what an earlier one holds in its trash:
    # «Вернуть» on the earlier one then restores only what no later forget covers.
    claims = _claim_trash(conn, plan)
    if plan.claimed_sups:
        conn.execute(
            f"UPDATE suppressions SET forget_id = ? WHERE hmac IN ({_qs(plan.claimed_sups)})",
            [fid, *plan.claimed_sups],
        )
    conn.execute(
        "INSERT INTO jobs (id, jkind, payload, state, attempts, not_before, idempotency_key, "
        "created_at, updated_at) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?, ?)",
        (
            _new_id("job"),
            PURGE_JOB,
            json.dumps(
                {
                    "forget_id": fid,
                    "stage": "stopped",
                    "scope": plan.scope.value,
                    "target_ids": plan.targets,
                    "masked_ids": plan.masked + plan.raw_masked,
                    "until": until,
                    "claimed": claims,
                    "record_ids": plan.records,
                    "raw_ids": plan.raw_forgotten + plan.raw_redacted,
                    "turn_ids": [],
                    "thread_ids": sorted(set(plan.threads) | set(plan.session_threads)),
                    "stopped_at": ts,
                }
            ),
            until,
            f"{PURGE_JOB}:{fid}",
            ts,
            ts,
        ),
    )
    conn.execute(
        "INSERT INTO audit (ts, event, record_id, actor) VALUES (?, 'forget_stopped', ?, 'memory')",
        (ts, fid),
    )
    return touched


def _x_ids(plan: _Plan) -> list[str]:
    return plan.xs


def _held_trash(
    conn: sqlite3.Connection, ids: Iterable[str], fid: str, *, kinds: tuple[str, ...] = ("record",)
) -> list[dict[str, Any]]:
    """Trash entries of other forgets still in the trash (`stopped`) for these ids."""
    refs = [f"{k}:{i}" for i in ids for k in kinds]
    if not refs:
        return []
    out = []
    for tid, owner, ref, payload in conn.execute(
        f"SELECT id, forget_id, row_ref, payload FROM trash WHERE store = 'memory' "
        f"AND forget_id <> ? AND payload IS NOT NULL AND row_ref IN ({_qs(refs)}) "
        "ORDER BY created_at, id",
        [fid, *refs],
    ).fetchall():
        job = _job(conn, owner)
        if job is not None and job["payload"].get("stage") == "stopped":
            out.append({"id": tid, "owner": owner, "row_ref": ref, "payload": json.loads(payload)})
    return out


def _record_kind(conn: sqlite3.Connection, rid: str) -> str:
    row = conn.execute("SELECT kind FROM records WHERE id = ?", (rid,)).fetchone()
    return str(row[0]) if row else ""


def _trash_proposition(conn: sqlite3.Connection, t: dict[str, Any]) -> tuple[str, str, str] | None:
    kind, _, rid = t["row_ref"].partition(":")
    body = t["payload"] or {}
    if kind != "record" or not body.get("row"):
        return None
    return proposition(conn, _record_kind(conn, rid), body["row"])


def _claim_trash(conn: sqlite3.Connection, plan: _Plan) -> dict[str, str]:
    """Move other forgets' trash entries for what this forget covers to it; returns
    {trash id: the forget it came from} (the job payload's `claimed`, for «Вернуть»)."""
    recs = plan.claimed + plan.records + plan.masked
    raws = plan.claimed + plan.raw_forgotten + plan.raw_redacted + plan.raw_masked
    held = _held_trash(
        conn, recs, plan.forget_id, kinds=("record", "mask", "retraction")
    ) + _held_trash(conn, raws, plan.forget_id, kinds=("raw",))
    claims = {t["id"]: t["owner"] for t in held}
    if claims:
        conn.execute(
            f"UPDATE trash SET forget_id = ? WHERE id IN ({_qs(claims)})",
            [plan.forget_id, *claims],
        )
    return claims


def _row_by(conn: sqlite3.Connection, table: str, col: str, value: str) -> dict[str, Any] | None:
    cur = conn.execute(f"SELECT * FROM {table} WHERE {col} = ?", (value,))
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], tuple(row), strict=True))


# ---------------------------------------------------------------- jobs (purge state)


def _job(conn: sqlite3.Connection, forget_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, payload, state, attempts, not_before FROM jobs WHERE idempotency_key = ?",
        (f"{PURGE_JOB}:{forget_id}",),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "payload": json.loads(row[1]),
        "state": row[2],
        "attempts": row[3],
        "not_before": row[4],
    }


def _set_job(
    conn: sqlite3.Connection,
    forget_id: str,
    *,
    state: str | None = None,
    not_before: str | None = None,
    attempts: int | None = None,
    **payload: Any,
) -> None:
    job = _job(conn, forget_id)
    if job is None:
        return
    body = {**job["payload"], **payload}
    conn.execute(
        "UPDATE jobs SET payload = ?, state = ?, not_before = ?, attempts = ?, updated_at = ? "
        "WHERE id = ?",
        (
            json.dumps(body),
            state or job["state"],
            not_before if not_before is not None else job["not_before"],
            attempts if attempts is not None else job["attempts"],
            _ts(),
            job["id"],
        ),
    )


def _locked(conn: sqlite3.Connection, fn: Callable[[], Any]) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        fn()
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def status(conn: sqlite3.Connection, forget_id: str) -> str:
    """`in_trash` | `purging` | `purged` | `undone` (api.PurgeStatus.state)."""
    job = _job(conn, forget_id)
    if job is None:
        raise ForgetError(f"unknown forget {forget_id}")
    stage = job["payload"].get("stage", "stopped")
    if stage == "stopped" and job["not_before"] <= job["payload"].get("stopped_at", ""):
        return "purging"  # «насовсем» / sensitive: due at once, the pool runs it (F11)
    return {"stopped": "in_trash"}.get(stage, stage)


def due(conn: sqlite3.Connection, *, now: datetime | None = None) -> list[str]:
    """Forgets whose purge is due: trash window over (or «насовсем», sensitive), or a retry."""
    rows = conn.execute(
        "SELECT payload FROM jobs WHERE jkind = ? AND state = 'queued' AND not_before <= ? "
        "ORDER BY not_before",
        (PURGE_JOB, _ts(now)),
    ).fetchall()
    return [json.loads(r[0])["forget_id"] for r in rows]


def run_due(conn: sqlite3.Connection, *, now: datetime | None = None) -> dict[str, bool]:
    """Walk due trash entries through purge (api.run_purges)."""
    return {fid: purge(conn, fid) for fid in due(conn, now=now)}


# ---------------------------------------------------------------- stop use (§9.1 stage 1)


def stop_use(
    conn: sqlite3.Connection,
    target_ids: list[str],
    *,
    scope: ForgetScope,
    forever: bool,
    conversation: ConversationStore | None = None,
) -> str:
    """§9.1 stage 1 (a-e): ledger, conversation rows, sessions, caches, journal. Returns
    the forget_id.

    `target_ids`: record or raw chunk ids (`item`), entity ids (`about_entity`), chat ids
    or the item ids `policy.mode_scope` picked (`chat`). `conversation` defaults to the wired store; `chat` scope never touches the
    island history (§5.2) but still restarts sessions."""
    return _stop_use(
        conn,
        target_ids,
        scope=scope,
        forever=forever,
        conversation=conversation if conversation is not None else _W.conversation,
    )


def _stop_use(
    conn: sqlite3.Connection,
    target_ids: list[str],
    *,
    scope: ForgetScope,
    forever: bool,
    conversation: ConversationStore | None,
    forget_id: str | None = None,
    journal: bool = True,
    extra_turns: Iterable[str] = (),
    until: str | None = None,
    extra_records: Iterable[str] = (),
    extra_raws: Iterable[str] = (),
    extra_masked: Iterable[str] = (),
    write_ahead: bool = True,
) -> str:
    """`until`: a replayed forget keeps its trash window (§9.3 F1) unless `forever`.
    `write_ahead`: False when a `stopping` line for this forget is already on disk."""
    fid = forget_id or _new_id("fg")
    targets = list(target_ids)
    extras = (list(extra_records), list(extra_raws), list(extra_masked))
    holder: dict[str, Any] = {}
    conv = conversation if scope is not ForgetScope.CHAT else None
    path = journal_path(conn) if journal else None
    if path is not None and write_ahead:
        # §9.1 A4 write-ahead: on disk before the stop-use commit, so a crash between them
        # leaves a line startup can finish from (never a forgotten row with no line)
        recs, raws, _xs, held = _resolve_targets(conn, targets, scope)
        if not (recs or raws or any(extras) or (held and scope is not ForgetScope.ITEM)):
            raise ForgetNotFound(f"nothing usable to forget in {targets!r}")
        append_journal(
            path,
            JournalLine(
                ts=_ts(),
                stage=ForgetStage.STOPPING,
                forget_id=fid,
                scope=scope,
                target_ids=targets,
                until=_ts() if forever else None,
                memory_version=store.memory_version(conn),
            ),
        )

    def apply(ctx: store.CommitCtx, _op: ProposalOp) -> list[str]:
        plan = _plan(
            ctx.conn, targets, scope, fid,
            extra_records=extras[0], extra_raws=extras[1], extra_masked=extras[2],
        )  # fmt: skip
        if not (
            plan.records
            or plan.raw_forgotten
            or plan.raw_redacted
            or plan.masked
            or (plan.claimed and scope is not ForgetScope.ITEM)
        ):
            raise ForgetNotFound(f"nothing usable to forget in {targets!r}")
        sensitive = bool(plan.labels & SENSITIVE_DEFAULT)
        if forever or sensitive:
            plan.until = _ts()
        else:
            plan.until = until or _ts(_now() + TRASH_WINDOW)
        plan.version = ctx.version
        holder["plan"] = plan
        touched = _apply_stop(ctx.conn, plan, ctx.version)
        # §9.5 affected turns (read-only on the conversation store), in the stop commit: the
        # purge job knows them before any is blanked, and a crash before the blanking ends
        # leaves `turns_pending`, which startup finishes
        holder["mask_only"] = mask_only = _affected_turns(plan, conv, extra_turns)
        restart = sorted(set(plan.threads) | set(plan.session_threads))
        _set_job(
            ctx.conn,
            fid,
            turn_ids=sorted(set(plan.turns) | mask_only),
            mask_turns=sorted(mask_only),
            mask_xs=plan.xs if mask_only else [],
            thread_ids=restart,
            turns_pending=bool(plan.turns or mask_only),
        )
        return touched

    proposal = Proposal(ops=[ProposalOp(op=OpKind.FORGET, group=f"forget:{fid}")])
    store.commit(conn, proposal, apply, event_kind=EventKind.FORGET)
    plan: _Plan = holder["plan"]
    mask_only = holder["mask_only"]

    _delete_vectors(conn, plan.records)
    if plan.until <= _ts():  # no «Вернуть» for this one: the hot log lets go of it now
        _scrub_hot_ops(conn, plan.records + plan.raw_forgotten)
    if path is not None:
        append_journal(
            path,
            JournalLine(
                ts=_ts(),
                stage=ForgetStage.STOPPED,
                forget_id=fid,
                record_ids=plan.records,
                raw_hmacs=plan.raw_hmacs,
                suppression_hmacs=plan.suppressions,
                turn_ids=plan.turns + sorted(mask_only),
                scope_done=list(_STOP_SCOPE),
                memory_version=plan.version,
                scope=plan.scope,
                target_ids=plan.targets,
                masked_ids=plan.masked + plan.raw_masked,
                until=plan.until,
            ),
        )

    if conv is not None and (plan.turns or mask_only):
        _blank_and_mask(conn, conv, fid, plan.turns, sorted(mask_only), plan.terms, plan.until)
        plan.turns = sorted(set(plan.turns) | mask_only)
        _locked(conn, lambda: _set_job(conn, fid, turns_pending=False))
    restart = sorted(set(plan.threads) | set(plan.session_threads))
    if conversation is not None and restart:
        conversation.restart_sessions(restart, exclude_turns=plan.turns)
    event = StopEvent(
        forget_id=fid,
        record_ids=plan.records + plan.masked,
        raw_ids=plan.raw_forgotten + plan.raw_redacted + plan.raw_masked,
        turn_ids=plan.turns,
        thread_ids=restart,
    )
    for hook in list(_W.stop_hooks):
        hook(event)
    return fid


def _affected_turns(
    plan: _Plan, conv: ConversationStore | None, extra_turns: Iterable[str]
) -> set[str]:
    """§9.5: the island turns this forget blanks (`plan.turns`, set here) and the ones it
    only masks (returned); `plan.threads` grows by the threads they live in. Read-only."""
    turns = set(plan.turns) | set(extra_turns)
    mask_only: set[str] = set()
    if conv is None:
        plan.turns = []
        return mask_only
    names = {match_norm(t).strip() for t in plan.terms}
    blank_keys = [k for k in plan.keys if match_norm(k).strip() not in names]
    threads = set(plan.threads)
    for t in sorted(turns):
        threads |= {str(r["thread_id"]) for r in conv.turn_rows(t) if r.get("thread_id")}
    for thread in sorted(threads):
        after = plan.first_turn.get(thread)
        turns |= set(conv.find_turns([thread], blank_keys, after_turn=after))
        if plan.terms:
            mask_only |= set(conv.find_turns([thread], plan.terms, after_turn=after))
    plan.threads = sorted(threads)
    plan.turns = sorted(turns)
    return mask_only - turns


def _blank_and_mask(
    conn: sqlite3.Connection,
    conv: ConversationStore,
    fid: str,
    blank: list[str],
    mask: list[str],
    terms: list[str],
    until: str,
) -> None:
    if blank:
        _blank_turns(conn, conv, fid, blank, until)
    if not mask:
        return
    masker = getattr(conv, "mask_turns", None)
    if masker is None or not terms:  # nothing to mask with: blank (more forgotten, never less)
        _blank_turns(conn, conv, fid, mask, until)
    else:
        _trash_turn_payload(conn, fid, masker(mask, terms, forget_id=fid), until)


def finish_turns(conn: sqlite3.Connection, conversation: ConversationStore | None = None) -> int:
    """`api.startup` step 2: a stop-use whose process died after its commit, before the
    island turns it names were all blanked (`turns_pending`), blanks the rest now. A
    turn with a trash row of this forget is done; one only masked is masked again with X's
    names, or blanked when they are gone. Returns the forgets finished."""
    conv = conversation if conversation is not None else _W.conversation
    if conv is None:
        return 0
    done = 0
    for (payload,) in conn.execute(
        "SELECT payload FROM jobs WHERE jkind = ? ORDER BY created_at", (PURGE_JOB,)
    ).fetchall():
        body = json.loads(payload)
        if not body.get("turns_pending") or body.get("stage") in ("undone", "purged"):
            continue
        fid = body["forget_id"]
        have = {
            r[0][5:]
            for r in conn.execute(
                "SELECT row_ref FROM trash WHERE forget_id = ? AND store = 'career' "
                "AND row_ref LIKE 'turn:%'",
                (fid,),
            )
        }
        masks = set(body.get("mask_turns") or [])
        left = [t for t in body.get("turn_ids") or [] if t not in have]
        terms, stable = _x_terms(conn, set(body.get("mask_xs") or []))
        _blank_and_mask(
            conn,
            conv,
            fid,
            [t for t in left if t not in masks],
            [t for t in left if t in masks],
            terms + [v for _k, v in stable],
            body.get("until") or _ts(),
        )
        _locked(conn, lambda fid=fid: _set_job(conn, fid, turns_pending=False))
        restart = list(body.get("thread_ids") or [])
        if restart:
            conv.restart_sessions(restart, exclude_turns=list(body.get("turn_ids") or []))
        done += 1
    return done


def _claims(conn: sqlite3.Connection, fid: str, turn_id: str) -> list[str]:
    """Other forgets still holding a turn blanked (not undone)."""
    return [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT forget_id FROM trash WHERE store = 'career' AND row_ref = ? "
            "AND forget_id <> ?",
            (f"turn:{turn_id}", fid),
        )
    ]


def _trash_turn_payload(
    conn: sqlite3.Connection, fid: str, payload: list[dict[str, Any]], until: str
) -> None:
    by_turn: dict[str, list[dict[str, Any]]] = {}
    for item in payload:
        by_turn.setdefault(str(item.get("turn_id", "")), []).append(item)

    def write() -> None:
        for turn_id, items in by_turn.items():
            _trash(conn, fid, "career", f"turn:{turn_id}" if turn_id else "turns", items, until)

    _locked(conn, write)


def _blank_turns(
    conn: sqlite3.Connection, conv: ConversationStore, fid: str, turns: list[str], until: str
) -> None:
    fresh = [t for t in turns if not _claims(conn, fid, t)]
    held = [t for t in turns if t not in fresh]
    payload = conv.blank_turns(fresh, forget_id=fid) if fresh else []
    _trash_turn_payload(conn, fid, payload, until)

    def claim() -> None:  # already blank under another forget; the payload moves on undo
        for t in held:
            conn.execute(
                "INSERT INTO trash (id, forget_id, store, row_ref, payload, until, created_at) "
                "VALUES (?, ?, 'career', ?, NULL, ?, ?)",
                (_new_id("tr"), fid, f"turn:{t}", until, _ts()),
            )

    if held:
        _locked(conn, claim)


def rematch_turn(
    conn: sqlite3.Connection,
    turn_id: str,
    *,
    conversation: ConversationStore | None = None,
    window_s: int = REMATCH_WINDOW_S,
) -> list[str]:
    """§9.1c: after a turn writes its rows, match them against forgets stopped in the last
    `window_s` seconds (the answer that streamed during the forget, the «забудь» turn
    itself) and blank the turn under each matching forget. Returns those forget ids."""
    conv = conversation if conversation is not None else _W.conversation
    if conv is None:
        return []
    text = " ".join(str(r.get("text") or "") for r in conv.turn_rows(turn_id))
    since = _ts(_now() - timedelta(seconds=window_s))
    hit: list[str] = []
    for (payload,) in conn.execute(
        "SELECT payload FROM jobs WHERE jkind = ? AND created_at >= ?", (PURGE_JOB, since)
    ).fetchall():
        body = json.loads(payload)
        fid = body["forget_id"]
        if body.get("stage") not in ("stopped", "purging") or body.get("scope") == "chat":
            continue
        if turn_id in body.get("turn_ids", []):
            continue
        if not text_matches(text, compile_keys(_payload_match_keys(conn, fid))):
            continue
        until = (_job(conn, fid) or {}).get("not_before") or _ts()
        _blank_turns(conn, conv, fid, [turn_id], until)
        turns = sorted(set(body.get("turn_ids", [])) | {turn_id})
        _locked(conn, lambda fid=fid, turns=turns: _set_job(conn, fid, turn_ids=turns))
        if (path := journal_path(conn)) is not None:
            append_journal(
                path,
                JournalLine(
                    ts=_ts(),
                    stage=ForgetStage.STOPPED,
                    forget_id=fid,
                    turn_ids=[turn_id],
                    scope_done=["conversation"],
                    memory_version=store.memory_version(conn),
                ),
            )
        hit.append(fid)
    return hit


def _trash_rows(
    conn: sqlite3.Connection, fid: str, store_name: str | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT id, store, row_ref, payload, until FROM trash WHERE forget_id = ?"
    params: list[Any] = [fid]
    if store_name:
        sql += " AND store = ?"
        params.append(store_name)
    return [
        {
            "id": r[0],
            "store": r[1],
            "row_ref": r[2],
            "payload": json.loads(r[3]) if r[3] is not None else None,
            "until": r[4],
        }
        for r in conn.execute(sql + " ORDER BY created_at, id", params)
    ]


def _payload_match_keys(conn: sqlite3.Connection, fid: str) -> list[str]:
    rows = []
    for t in _trash_rows(conn, fid, "memory"):
        ref, body = t["row_ref"], t["payload"]
        if body is None or not ref.startswith("record:"):
            continue
        row = dict(body.get("row") or {})
        row["_origin_quote"] = (body.get("header") or {}).get("origin_quote")
        own = [a["alias"] for a in body.get("aliases", [])]
        rows.append((body["table"], ref.split(":", 1)[1], row, own))
    return _keys_from_rows(conn, rows)


# ---------------------------------------------------------------- vectors.db and hotlog.db


def _side_db(conn: sqlite3.Connection, name: str, mode: str = "WAL") -> sqlite3.Connection | None:
    root = memory_dir(conn)
    if root is None or not (root / name).exists():
        return None
    return store.connect(root / name, journal_mode=mode)


def _tables_with(db: sqlite3.Connection, column: str) -> list[str]:
    out = []
    for (name,) in db.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        if column in {r[1] for r in db.execute(f"PRAGMA table_info({name})")}:
            out.append(name)
    return out


def _delete_vectors(conn: sqlite3.Connection, record_ids: list[str]) -> None:
    """§9.1a: vectors of forgotten records go; a crash before this is healed by the
    orphan sweep at the next open (vec) and by the purge's own sweep. Every table of
    vectors.db with a `record_id` column is covered, whatever vec names it."""
    vdb = _side_db(conn, "vectors.db")
    if vdb is None or not record_ids:
        if vdb is not None:
            vdb.close()
        return
    try:
        for table in _tables_with(vdb, "record_id"):
            vdb.execute(f"DELETE FROM {table} WHERE record_id IN ({_qs(record_ids)})", record_ids)
    finally:
        vdb.close()


def _sweep_vectors(conn: sqlite3.Connection) -> bool:
    """Orphan sweep (vectors of records that are not usable), then a verified checkpoint."""
    vdb = _side_db(conn, "vectors.db")
    if vdb is None:
        return True
    try:
        try:
            from .recall import vectors as _vectors

            _vectors.sweep_orphans(conn, vdb)
        except NotImplementedError:
            for table in _tables_with(vdb, "record_id"):
                ids = [r[0] for r in vdb.execute(f"SELECT DISTINCT record_id FROM {table}")]
                bad = sorted(set(ids) - _usable(conn, ids))
                for i in range(0, len(bad), 400):
                    chunk = bad[i : i + 400]
                    vdb.execute(f"DELETE FROM {table} WHERE record_id IN ({_qs(chunk)})", chunk)
        busy, log, done = vdb.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return busy == 0 and log == done
    finally:
        vdb.close()


def _scrub_hot_ops(conn: sqlite3.Connection, ids: list[str]) -> None:
    """§9.4 at stop-use (`backup.hotlog_scrub`): hot ops naming the forgotten ids lose their
    payload, so neither a restore nor a backup's hot-log snapshot carries it. Only for a
    forget with no «Вернуть» (forever or sensitive): a 24 h one keeps its ops until the
    purge, or a restore after «Вернуть» would lose what only the hot log still held."""
    root = memory_dir(conn)
    if root is None or not ids or not (root / "hotlog.db").exists():
        return
    hot = backup.open_hotlog(root / "hotlog.db")
    try:
        backup.hotlog_scrub(hot, ids)
    finally:
        hot.close()


def _scrub_hotlog(conn: sqlite3.Connection, ids: set[str], pattern: re.Pattern[str] | None) -> None:
    """§9.4: forgets delete their ops' payloads (ops naming the ids or carrying the text)."""
    hot = _side_db(conn, "hotlog.db", "DELETE")
    if hot is None:
        return
    try:
        for table in _tables_with(hot, "payload"):
            for rowid, payload in hot.execute(f"SELECT rowid, payload FROM {table}").fetchall():
                text = payload if isinstance(payload, str) else str(payload or "")
                if any(i in text for i in ids) or (pattern is not None and pattern.search(text)):
                    hot.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
    finally:
        hot.close()


# ---------------------------------------------------------------- undo (§9.1 stage 2)


def undo(conn: sqlite3.Connection, forget_id: str) -> None:
    """§9.1 stage 2: «Вернуть» from the trash."""
    _undo(conn, forget_id, _W.conversation, journal=True)


def _undo(
    conn: sqlite3.Connection,
    fid: str,
    conversation: ConversationStore | None,
    *,
    journal: bool,
    back_ids: set[str] | None = None,
) -> None:
    """`back_ids` (a replayed partial «Вернуть»): the record, mask, raw and retraction
    rows named here come back and the rest stays, instead of deciding it by chat bans now."""
    job = _job(conn, fid)
    if job is None:
        raise ForgetError(f"unknown forget {fid}")
    stage = job["payload"].get("stage")
    if stage == "undone":
        return
    if stage != "stopped":
        raise ForgetError(f"forget {fid} is {stage}; «Вернуть» works only from the trash")
    trash = _trash_rows(conn, fid, "memory")
    # What this forget took over from an earlier one goes back to it unless that one was
    # undone; what came from a chat banned since then stays forgotten here.
    claims = job["payload"].get("claimed") or {}
    back = {
        t["id"]: claims[t["id"]]
        for t in trash
        if t["id"] in claims and _stage_of(conn, claims[t["id"]]) not in (None, "undone")
    }
    if back_ids is None:
        kept = _banned_entries(conn, [t for t in trash if t["id"] not in back])
    else:
        kept = {
            t["id"]
            for t in trash
            if t["id"] not in back
            and t["row_ref"].partition(":")[0] in ("record", "mask", "raw", "retraction")
            and t["row_ref"].partition(":")[2] not in back_ids
        }
    restored: list[str] = []
    came_back: dict[str, list[str]] = {"records": [], "raws": [], "sups": []}

    def apply(ctx: store.CommitCtx, _op: ProposalOp) -> list[str]:
        c, ts, v = ctx.conn, _ts(), ctx.version
        stay: set[str] = set()
        for t in trash:
            if t["id"] in back or t["id"] in kept:
                prop = _trash_proposition(c, t)
                if prop is not None:
                    h = suppression_hmac(*prop)
                    owner = back.get(t["id"], fid)
                    c.execute("UPDATE suppressions SET forget_id = ? WHERE hmac = ?", (owner, h))
                    stay.add(h)
                continue
            kind, _, item = t["row_ref"].partition(":")
            body = t["payload"]
            if body is None:
                continue
            if kind == "record":
                _restore_record(c, item, body, v, ts)
            elif kind == "mask":
                _restore_mask(c, item, body, v, ts)
            elif kind == "raw":
                _restore_raw(c, item, body, ts)
            elif kind == "derived":
                cur = c.execute(
                    "SELECT rev, valid, dkind, dkey FROM derived WHERE id = ?", (item,)
                ).fetchone()
                if cur is None:
                    continue
                rev, valid, dkind, dkey = tuple(cur)
                if valid or (body.get("rev") is not None and rev != body["rev"]):
                    # Rebuilt since the forget. Its read set lacks what comes back, so
                    # the old text must not return over it under the same rev: it is marked
                    # for a rebuild (new rev) that reads the restored items.
                    from . import ledger

                    c.execute(
                        "UPDATE derived SET valid = 0, rev = rev + 1, updated_at = ? WHERE id = ?",
                        (ts, item),
                    )
                    ledger._queue_rebuild(ctx, item, dkind, dkey)
                else:  # stays invalid: its read set predates the new versions
                    c.execute(
                        "UPDATE derived SET content = ?, updated_at = ? WHERE id = ?",
                        (body["content"], ts, item),
                    )
            elif kind == "action":
                c.execute(
                    "UPDATE actions SET summary_en = ? WHERE id = ?", (body["summary_en"], item)
                )
            elif kind == "retraction":
                c.execute(
                    "INSERT OR IGNORE INTO raw_retractions (raw_id, record_id, "
                    "retracted_value_norm, created_at) VALUES (?, ?, ?, ?)",
                    (body["raw_id"], body["record_id"], body["value"], body["created_at"]),
                )
                continue
            restored.append(item)
            if kind in ("record", "mask"):
                came_back["records"].append(item)
            elif kind == "raw":
                came_back["raws"].append(item)
        for tid, owner in back.items():
            c.execute("UPDATE trash SET forget_id = ? WHERE id = ?", (owner, tid))
        came_back["sups"] = [
            r[0]
            for r in c.execute(
                f"SELECT hmac FROM suppressions WHERE forget_id = ? AND hmac NOT IN ({_qs(stay)})",
                [fid, *stay],
            )
        ]
        c.execute(
            f"DELETE FROM suppressions WHERE forget_id = ? AND hmac NOT IN ({_qs(stay)})",
            [fid, *stay],
        )
        c.execute(
            f"DELETE FROM trash WHERE forget_id = ? AND store = 'memory' AND id NOT IN "
            f"({_qs(kept)})",
            [fid, *kept],
        )
        if kept:  # still in the trash: the banned chat's part waits for its purge
            refs = [t["row_ref"].partition(":") for t in trash if t["id"] in kept]
            _set_job(
                c,
                fid,
                record_ids=sorted({i for k, _, i in refs if k == "record"}),
                raw_ids=sorted({i for k, _, i in refs if k == "raw"}),
            )
        else:
            _set_job(c, fid, state="done", stage="undone")
        c.execute(
            "INSERT INTO audit (ts, event, record_id, actor) "
            "VALUES (?, 'forget_undone', ?, 'memory')",
            (ts, fid),
        )
        return restored

    proposal = Proposal(ops=[ProposalOp(op=OpKind.FORGET, group=f"undo:{fid}")])
    result = store.commit(conn, proposal, apply, event_kind=EventKind.UNDO)

    # Conversation rows: restore, or hand the payload to a forget that still holds the turn.
    give_back: list[dict[str, Any]] = []

    def settle() -> None:
        for t in _trash_rows(conn, fid, "career"):
            if t["payload"] is None:
                continue
            heirs = _claims(conn, fid, t["row_ref"][5:]) if t["row_ref"].startswith("turn:") else []
            if heirs:
                conn.execute(
                    "UPDATE trash SET payload = ? WHERE forget_id = ? AND row_ref = ? "
                    "AND payload IS NULL",
                    (json.dumps(t["payload"], ensure_ascii=False), heirs[0], t["row_ref"]),
                )
            else:
                give_back.extend(t["payload"])
        conn.execute("DELETE FROM trash WHERE forget_id = ? AND store <> 'memory'", (fid,))

    _locked(conn, settle)
    if give_back and conversation is not None:
        conversation.restore_turns(give_back)
    for owner in sorted(set(back.values())):  # handed back to a forget already purging
        if _stage_of(conn, owner) in ("purging", "purged"):
            _purge_rows(conn, owner, drop_payloads=True)
    version = next(iter(result.versions.values()), store.memory_version(conn))
    if not journal or (path := journal_path(conn)) is None:
        return
    if not kept:  # the whole forget is undone
        append_journal(
            path,
            JournalLine(
                ts=_ts(),
                stage=ForgetStage.UNDONE,
                forget_id=fid,
                record_ids=[
                    t["row_ref"].partition(":")[2]
                    for t in trash
                    if t["row_ref"].startswith(("record:", "mask:"))
                ],
                memory_version=version,
            ),
        )
        return
    # A partial «Вернуть» names what came back, and its `until` (the rest still waits
    # in the trash) marks it partial: a restore replays the forget minus these, so what the
    # user took back is not forgotten again
    append_journal(
        path,
        JournalLine(
            ts=_ts(),
            stage=ForgetStage.UNDONE,
            forget_id=fid,
            record_ids=sorted(came_back["records"]),
            raw_hmacs=sorted(
                _ids_in(
                    conn,
                    "SELECT content_hmac FROM raw_chunks WHERE id IN ({qs}) "
                    "AND content_hmac IS NOT NULL",
                    came_back["raws"],
                )
            ),
            suppression_hmacs=sorted(came_back["sups"]),
            memory_version=version,
            until=str(job["payload"].get("until") or _ts()),
        ),
    )


def _stage_of(conn: sqlite3.Connection, fid: str) -> str | None:
    job = _job(conn, fid)
    return job["payload"].get("stage") if job is not None else None


def _banned_entries(conn: sqlite3.Connection, trash: list[dict[str, Any]]) -> set[str]:
    """Trash ids «Вернуть» must not restore: a message from a chat banned since the forget
    (§5.2 «ничего, никогда»), the records learned from it and their retraction rows."""
    from . import policy

    raws: set[str] = set()
    for t in trash:
        kind, _, rid = t["row_ref"].partition(":")
        if kind != "raw" or (t["payload"] or {}).get("mode", "forget") != "forget":
            continue
        row = conn.execute("SELECT chat_id FROM raw_chunks WHERE id = ?", (rid,)).fetchone()
        if row and row[0] and policy.chat_mode(conn, row[0]) is ChatMode.BANNED:
            raws.add(rid)
    if not raws:
        return set()
    out: set[str] = set()
    for t in trash:
        kind, _, rid = t["row_ref"].partition(":")
        if (kind == "raw" and rid in raws) or (
            kind in ("record", "retraction") and _lineage_raw(conn, rid) & raws
        ):
            out.add(t["id"])
    return out


def _restore_raw(c: sqlite3.Connection, raw_id: str, body: dict[str, Any], ts: str) -> None:
    raw = _row(c, "raw_chunks", raw_id) or {}
    status_ = "active" if body.get("mode") == "forget" else raw.get("status", "active")
    if raw.get("fts_rowid") is not None:
        c.execute("DELETE FROM fts_raw WHERE rowid = ?", (raw["fts_rowid"],))
        if body.get("text"):
            c.execute(
                "INSERT INTO fts_raw (rowid, text) VALUES (?, ?)",
                (raw["fts_rowid"], _stem_for_fts(body["text"])),
            )
    c.execute(
        "UPDATE raw_chunks SET text = ?, status = ?, rev = rev + 1, updated_at = ? WHERE id = ?",
        (body.get("text"), status_, ts, raw_id),
    )
    if body.get("source_ref") is not None:
        c.execute("UPDATE raw_chunks SET source_ref = ? WHERE id = ?", (body["source_ref"], raw_id))


def _restore_record(
    c: sqlite3.Connection, rid: str, body: dict[str, Any], version: int, ts: str
) -> None:
    table, row, head = body["table"], body.get("row") or {}, body["header"]
    cols = [k for k in row if k != "id"]
    if cols:
        c.execute(
            f"UPDATE {table} SET {', '.join(f'{k} = ?' for k in cols)} WHERE id = ?",
            [row[k] for k in cols] + [rid],
        )
    for a in body.get("aliases", []):
        keys = list(a)
        c.execute(
            f"INSERT OR REPLACE INTO aliases ({', '.join(keys)}) VALUES ({_qs(keys)})",
            [a[k] for k in keys],
        )
    for n in body.get("names_fts", []):
        if n.get("name") is not None:
            c.execute("INSERT INTO fts_names (rowid, name) VALUES (?, ?)", (n["rowid"], n["name"]))
    for link in body.get("links", []):
        c.execute(
            "INSERT OR IGNORE INTO episode_links (episode_id, role, target_id) VALUES (?, ?, ?)",
            (link["episode_id"], link["role"], link["target_id"]),
        )
    fts = body.get("fts")
    rowid = c.execute("SELECT fts_rowid FROM records WHERE id = ?", (rid,)).fetchone()[0]
    if fts and rowid is not None:
        c.execute(
            "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, ?, ?)",
            (rowid, fts.get("text"), fts.get("keywords"), fts.get("aliases")),
        )
    c.execute(
        "UPDATE records SET status = ?, origin_quote = ?, retired_at = ?, version = ?, "
        "updated_at = ? WHERE id = ?",
        (head["status"], head["origin_quote"], head["retired_at"], version, ts, rid),
    )


def _restore_mask(
    c: sqlite3.Connection, rid: str, body: dict[str, Any], version: int, ts: str
) -> None:
    table, cols = body["table"], body.get("cols") or {}
    if cols:
        c.execute(
            f"UPDATE {table} SET {', '.join(f'{k} = ?' for k in cols)} WHERE id = ?",
            [*cols.values(), rid],
        )
    if body.get("speaker"):
        c.execute("UPDATE facts SET speaker_entity_id = ? WHERE id = ?", (body["speaker"], rid))
    for link in body.get("links", []):
        c.execute(
            "INSERT OR IGNORE INTO episode_links (episode_id, role, target_id) VALUES (?, ?, ?)",
            (link["episode_id"], link["role"], link["target_id"]),
        )
    rowid = c.execute("SELECT fts_rowid FROM records WHERE id = ?", (rid,)).fetchone()[0]
    fts = body.get("fts")
    if fts and rowid is not None:
        c.execute("DELETE FROM fts_records WHERE rowid = ?", (rowid,))
        c.execute(
            "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, ?, ?)",
            (rowid, fts.get("text"), fts.get("keywords"), fts.get("aliases")),
        )
    c.execute(
        "UPDATE records SET author_entity_id = ?, origin_quote = ?, version = ?, updated_at = ? "
        "WHERE id = ?",
        (body.get("author_entity_id"), body.get("origin_quote"), version, ts, rid),
    )


# ---------------------------------------------------------------- purge (§9.1 stage 3)


def _scan_strings(conn: sqlite3.Connection, fid: str) -> list[str]:
    """The exact strings the purge must not find anywhere (content; RAM only)."""
    out: set[str] = set()

    def add(text: Any) -> None:
        if not isinstance(text, str) or text == FORGOTTEN:
            return
        norm = match_norm(text).strip()
        if 6 <= len(text) <= 300:
            out.add(text)
        if len(norm.split()) > 4:
            out.update(_shingles(text))
        elif len(norm) >= 6:
            out.add(norm)

    for t in _trash_rows(conn, fid, "memory"):
        body = t["payload"]
        if body is None:
            continue
        kind = t["row_ref"].partition(":")[0]
        if kind == "record":
            row = body.get("row") or {}
            for col in _BLANK.get(body["table"], {}):
                if col != "stable_ids":
                    add(row.get(col))
            add(body["header"].get("origin_quote"))
            for a in body.get("aliases", []):
                if len(a.get("alias", "")) >= 4:
                    out.add(a["alias"])
        elif kind in ("raw", "derived") and body.get("mode", "forget") == "forget":
            add(body.get("text") if kind == "raw" else body.get("content"))
            add(body.get("source_ref"))
        elif kind == "retraction":
            add(body.get("value"))
    return sorted(out, key=len, reverse=True)[:_MAX_SCAN_KEYS]


def _scan_pattern(keys: Iterable[str]) -> re.Pattern[str] | None:
    """Plain substring pattern over decoded bytes: original, casefolded and JSON-escaped."""
    variants: set[str] = set()
    for k in keys:
        for v in (k, k.casefold(), match_norm(k).strip()):
            if v:
                variants.add(v)
                esc = json.dumps(v, ensure_ascii=True)[1:-1]
                if esc != v:
                    variants.add(esc)
    if not variants:
        return None
    alt = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    return re.compile(alt, re.IGNORECASE)


def scan_bytes(data: bytes, pattern: re.Pattern[str] | None) -> bool:
    """§19.5: UTF-8, JSON `\\uXXXX` escapes and UTF-16LE (both alignments)."""
    if pattern is None or not data:
        return False
    for text in (
        data.decode("utf-8", "ignore"),
        data.decode("utf-16-le", "ignore"),
        data[1:].decode("utf-16-le", "ignore"),
    ):
        if pattern.search(text):
            return True
    return False


def scan_keys(keys: Iterable[str]) -> re.Pattern[str] | None:
    """The §19.5 token pattern for a set of strings (canary sweeps, purge steps)."""
    return _scan_pattern(keys)


def _memory_files(conn: sqlite3.Connection, *, main: bool) -> list[Path]:
    root = memory_dir(conn)
    if root is None:
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        is_main = p.parent == root and p.name in ("memory.db", "memory.db-wal", "memory.db-shm")
        if is_main == main:
            out.append(p)
    return out


_SHADOW = re.compile(r"^fts_\w+_(data|idx|docsize|config|content)$")


def _scan_db_rows(
    conn: sqlite3.Connection, pattern: re.Pattern[str], *, usable_only: bool
) -> set[str]:
    """Matched strings in memory.db rows. `usable_only`: legit content (the kind rows,
    aliases and quotes of records that keep their content, active raw, valid derived);
    otherwise every table but trash.

    A record keeps its content in every §4.2 status but `forgotten` and `purged`: a
    `superseded`, `retracted` or `needs_review` record outside this forget's scope is
    another fact (A14 never takes a successor; a different value is not a copy), not a
    leak. Copies of the forgotten proposition are in the scope (`_copies`), so they are
    `forgotten` here and their strings count as leaks until the purge clears them."""
    hits: set[str] = set()

    def look(values: Iterable[Any]) -> None:
        for v in values:
            if isinstance(v, str):
                hits.update(m.group(0).casefold() for m in pattern.finditer(v))

    if usable_only:
        kept = "status NOT IN ('forgotten', 'purged')"
        sqls = [
            f"SELECT k.* FROM {t} k JOIN records r ON r.id = k.id WHERE r.{kept}"
            for t in sorted(set(_KIND_TABLE.values()))
        ] + [
            f"SELECT a.alias FROM aliases a JOIN records r ON r.id = a.entity_id WHERE r.{kept}",
            f"SELECT origin_quote FROM records WHERE {kept}",
            "SELECT text FROM raw_chunks WHERE status = 'active'",
            "SELECT content FROM derived WHERE valid = 1",
        ]
    else:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "AND name NOT IN ('trash', 'schema_migrations')"
            ).fetchall()
            if not _SHADOW.match(r[0])
        ]
        sqls = [f"SELECT * FROM {n}" for n in names]
    for sql in sqls:
        for row in conn.execute(sql):
            look(tuple(row))
    return hits


def _other_trash_hits(conn: sqlite3.Connection, fid: str, pattern: re.Pattern[str]) -> set[str]:
    hits: set[str] = set()
    for (payload,) in conn.execute(
        "SELECT payload FROM trash WHERE forget_id <> ? AND payload IS NOT NULL", (fid,)
    ):
        hits.update(m.group(0).casefold() for m in pattern.finditer(payload))
    return hits


def purge(conn: sqlite3.Connection, forget_id: str) -> bool:
    """§9.1 stage 3: walk the §9.2 scope; True only when every check passed."""
    job = _job(conn, forget_id)
    if job is None:
        raise ForgetError(f"unknown forget {forget_id}")
    body = job["payload"]
    if body.get("stage") == "purged":
        return True
    if body.get("stage") == "undone":
        return False
    # claim with a lease (§18.1): a crash leaves `running` that startup requeues
    lease = store.lease_until()
    _locked(
        conn,
        lambda: _set_job(
            conn,
            forget_id,
            state="running",
            stage="purging",
            not_before=lease,
            lease_holder=store.lease_holder(),
        ),
    )

    keys = _scan_strings(conn, forget_id)
    pattern = _scan_pattern(keys)
    while pattern is not None:  # strings legitimately alive elsewhere are not leaks
        allowed = _scan_db_rows(conn, pattern, usable_only=True)
        allowed |= _other_trash_hits(conn, forget_id, pattern)
        left = [k for k in keys if not ({k.casefold(), match_norm(k).strip()} & allowed)]
        if len(left) == len(keys):
            break
        # again without them: a longer key matched first hides a shorter one inside it
        # («X lives in Berlin» alive in another message also keeps «Berlin»)
        keys, pattern = left, _scan_pattern(left)

    ok = True
    store.renew_lease(conn, job["id"])  # the scan is done; the purge steps start
    _purge_rows(conn, forget_id, drop_payloads=False)
    ok &= _sweep_vectors(conn)
    _scrub_hotlog(conn, set(body.get("record_ids", [])) | set(body.get("raw_ids", [])), pattern)
    checkpoint = getattr(_W.conversation, "checkpoint", None)
    if checkpoint is not None:
        ok &= bool(checkpoint())
    ctx = PurgeCtx(
        conn=conn,
        forget_id=forget_id,
        record_ids=list(body.get("record_ids", [])),
        raw_ids=list(body.get("raw_ids", [])),
        turn_ids=list(body.get("turn_ids", [])),
        thread_ids=list(body.get("thread_ids", [])),
        keys=keys,
    )
    for _name, step in sorted(_W.purge_steps.items()):
        ok &= bool(step(ctx))
    if pattern is not None:
        ok &= not _scan_db_rows(conn, pattern, usable_only=False)
        ok &= not any(scan_bytes(p.read_bytes(), pattern) for p in _memory_files(conn, main=False))
    if not ok:
        _retry(conn, forget_id)
        return False

    _purge_rows(conn, forget_id, drop_payloads=True)
    backup.optimize_fts(conn)  # secure-delete alone can leave a term in a shared segment page
    if not _checkpoint(conn):  # a reader (a streaming answer) holds a snapshot: it ends soon
        _retry(conn, forget_id, reader=True)
        return False
    if pattern is not None and any(
        scan_bytes(p.read_bytes(), pattern) for p in _memory_files(conn, main=True)
    ):
        _locked(conn, lambda: _set_job(conn, forget_id, state="needs_attention"))
        return False
    _locked(conn, lambda: _set_job(conn, forget_id, state="done", stage="purged"))
    if (path := journal_path(conn)) is not None:
        append_journal(
            path,
            JournalLine(
                ts=_ts(),
                stage=ForgetStage.PURGED,
                forget_id=forget_id,
                record_ids=list(body.get("record_ids", [])),
                turn_ids=list(body.get("turn_ids", [])),
                scope_done=[
                    "memory",
                    "fts",
                    "vectors",
                    "hotlog",
                    "checkpoint",
                    "scan",
                    *sorted(_W.purge_steps),
                ],
                memory_version=store.memory_version(conn),
            ),
        )
    return True


def _checkpoint(conn: sqlite3.Connection, wait_ms: int = 250) -> bool:
    """Verified `wal_checkpoint(TRUNCATE)`: busy = 0 and log = checkpointed. A short busy
    wait: the purge retries with backoff instead of holding the connection for 5 s."""
    (prev,) = conn.execute("PRAGMA busy_timeout").fetchone()
    conn.execute(f"PRAGMA busy_timeout = {int(wait_ms)}")
    try:
        busy, log, done = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.execute(f"PRAGMA busy_timeout = {int(prev)}")
    return busy == 0 and log == done


def _retry(conn: sqlite3.Connection, fid: str, *, reader: bool = False) -> None:
    """Back off and queue again. `retries` counts these apart from crashes: startup parks a
    job only after `JOB_ATTEMPTS_CAP` crashes, and a purge waiting on an old backup is no
    crash. `reader`: only the checkpoint was blocked by an open reader, which ends with its
    answer: the first `JOB_ATTEMPTS_CAP` such retries are due at once."""
    job = _job(conn, fid) or {"attempts": 0, "payload": {}}
    attempts = int(job["attempts"]) + 1
    retries = int(job["payload"].get("retries") or 0) + 1
    wait = BACKOFF_S[min(attempts - 1, len(BACKOFF_S) - 1)]
    if reader and retries <= store.JOB_ATTEMPTS_CAP:
        wait = 0
    not_before = _ts(_now() + timedelta(seconds=wait))
    _locked(
        conn,
        lambda: _set_job(
            conn, fid, state="queued", attempts=attempts, not_before=not_before, retries=retries
        ),
    )


def _purge_rows(conn: sqlite3.Connection, fid: str, *, drop_payloads: bool) -> None:
    """The memory.db part of the purge, idempotent. Without `drop_payloads` it clears
    everything but the trash (the scan still needs it); with it, the trash payloads go and
    headers become `purged` (one commit, event `forget`)."""
    trash = _trash_rows(conn, fid)
    recs = [t["row_ref"][7:] for t in trash if t["row_ref"].startswith("record:")]
    raws = [
        t["row_ref"][4:]
        for t in trash
        if t["row_ref"].startswith("raw:")
        and (t["payload"] or {}).get("mode", "forget") == "forget"
    ]
    dels = [
        t["row_ref"][8:]
        for t in trash
        if t["row_ref"].startswith("derived:") and (t["payload"] or {}).get("mode") == "forget"
    ]

    def apply(c: sqlite3.Connection, version: int | None) -> list[str]:
        ts = _ts()
        gone = recs + dels
        if gone:
            c.execute(f"DELETE FROM served WHERE item_id IN ({_qs(gone)})", gone)
            c.execute(f"DELETE FROM pending_updates WHERE item_id IN ({_qs(gone)})", gone)
        if recs:
            c.execute(f"DELETE FROM record_stats WHERE record_id IN ({_qs(recs)})", recs)
        if dels:
            c.execute(
                "DELETE FROM read_deps WHERE holder_kind = 'derived' "
                f"AND holder_id IN ({_qs(dels)})",
                dels,
            )
            c.execute(f"DELETE FROM derived WHERE id IN ({_qs(dels)})", dels)
        items = recs + raws
        for jid in sorted(
            _ids_in(
                c,
                "SELECT holder_id FROM read_deps WHERE holder_kind = 'job' AND item_id IN ({qs})",
                items,
            )
        ):
            c.execute(
                "UPDATE jobs SET payload = '{}', state = CASE WHEN state IN ('done', 'failed') "
                "THEN state ELSE 'failed' END, updated_at = ? WHERE id = ?",
                (ts, jid),
            )
        if raws:
            c.execute(f"UPDATE raw_chunks SET text = NULL WHERE id IN ({_qs(raws)})", raws)
        if version is not None:
            if recs:
                c.execute(
                    "UPDATE records SET status = 'purged', version = ?, origin_quote = NULL, "
                    f"updated_at = ? WHERE id IN ({_qs(recs)}) AND status = 'forgotten'",
                    [version, ts, *recs],
                )
            if raws:
                c.execute(
                    "UPDATE raw_chunks SET status = 'purged', updated_at = ? "
                    f"WHERE id IN ({_qs(raws)}) AND status = 'forgotten'",
                    [ts, *raws],
                )
            c.execute("UPDATE trash SET payload = NULL WHERE forget_id = ?", (fid,))
            c.execute(
                "INSERT INTO audit (ts, event, record_id, actor) "
                "VALUES (?, 'forget_purged', ?, 'memory')",
                (ts, fid),
            )
        return recs + raws + dels

    if drop_payloads:
        store.commit(
            conn,
            Proposal(ops=[ProposalOp(op=OpKind.FORGET, group=f"purge:{fid}")]),
            lambda ctx, _op: apply(ctx.conn, ctx.version),
            event_kind=EventKind.FORGET,
        )
    else:
        _locked(conn, lambda: apply(conn, None))
    _delete_vectors(conn, recs)


# ---------------------------------------------------------------- replay on restore (§9.4)


def purge_started(conn: sqlite3.Connection) -> list[str]:
    """Forgets whose purge has begun (`purging`) or finished (`purged`) on this store."""
    out = []
    for (payload,) in conn.execute("SELECT payload FROM jobs WHERE jkind = ?", (PURGE_JOB,)):
        body = json.loads(payload)
        if body.get("stage") in ("purging", "purged"):
            out.append(body["forget_id"])
    return sorted(out)


def replay_journal(
    conn: sqlite3.Connection,
    journal: Path | Iterable[JournalLine] | None,
    *,
    after_version: int = 0,
    conversation: ConversationStore | None = None,
    purge: Iterable[str] = (),
    forever: Iterable[str] = (),
) -> list[str]:
    """§9.4 step (2), the one journal replay: `backup.restore` and the nightly backup
    rewrite call it. Forgets replay through the full pipeline (cascade, FTS, derived,
    suppressions, conversation rows when a store is given), not row deletes, keyed by the
    original forget_id (no new id, no new journal line).

    `journal`: the journal file, or lines already merged (a restore merges the live journal
    with the backup set's copy). Lines are taken in the order they happened (ts, then
    stopped < undone < purged); per forget_id the last stage wins. A forget whose newest
    line is at or below `after_version` is skipped. `undone` is undone (only while the store
    holds it stopped); a partial `undone` (one with `until`) names what came back, and the
    forget replays without it; `purged` is purged locally (rows and trash payloads; the live
    purge verifies files); one already purged here is left alone, so replay is idempotent.
    Ids replay without the HMAC key: suppressions are copied as HMACs, raw is found by its
    stored `content_hmac`, and a raw HMAC with no row here gets a content-free tombstone
    (`tomb:<hmac>`, purged) so the sweep cannot re-ingest that text (§9.4 step 4).
    `purge`: forget ids also purged locally although their newest line is `stopped` - the
    nightly rewrite passes `purge_started(live)`, so a live purge waiting on old backups is
    not blocked by the trash payload the replay itself puts into them.
    `forever`: forget ids re-stopped with no trash window (due for purge now) although their
    newest line is `stopped` - a restore passes the ones the old live store was already
    purging or had due, so «насовсем» does not turn into a 24 h trash.
    FTS segments are optimized once at the end (secure-delete alone can leave a forgotten
    term in a shared segment page). Returns the forget ids that changed this store."""
    purge_now = set(purge)
    hurry = set(forever)
    if journal is None or isinstance(journal, (str, Path)):
        lines = read_journal(Path(journal) if journal is not None else None)
    else:
        lines = list(journal)
    groups = _groups(lines)
    applied: list[str] = []
    for fid, group in groups.items():
        final = group[-1].stage
        if final is not ForgetStage.STOPPING and max(x.memory_version for x in group) <= (
            after_version
        ):
            continue  # a `stopping`-only forget has no version of its own yet
        job = _job(conn, fid)
        here = job["payload"].get("stage") if job is not None else None
        if final is ForgetStage.UNDONE and group[-1].until is None:  # the whole forget
            if here == "stopped":
                _undo(conn, fid, None, journal=False)
                applied.append(fid)
            continue
        if here in ("purged", "undone"):
            continue
        back = _came_back(group)
        changed = _replay_ids(conn, fid, group, back)
        if back is not None:  # a partial «Вернуть»: the forget minus what came back
            changed = _replay_partial(conn, fid, group, here, back, fid in hurry) or changed
        elif job is None and any(x.scope is not None for x in group):
            changed = _replay_same(conn, fid, group, conversation, fid in hurry) or changed
        elif job is None:  # a v0.3 line: no scope, replayed as `item`
            rec_ids = sorted({r for x in group for r in x.record_ids})
            hmacs = sorted({h for x in group for h in x.raw_hmacs})
            targets = _ids_in(
                conn,
                "SELECT id FROM records WHERE id IN ({qs}) AND status NOT IN ('forgotten', 'purged')",
                rec_ids,
            ) | _ids_in(
                conn,
                "SELECT id FROM raw_chunks WHERE content_hmac IN ({qs}) AND status = 'active'",
                hmacs,
            )
            if targets:
                with contextlib.suppress(ForgetNotFound):
                    _stop_use(
                        conn,
                        sorted(targets),
                        scope=ForgetScope.ITEM,
                        forever=final is ForgetStage.PURGED or fid in hurry,
                        conversation=conversation,
                        forget_id=fid,
                        journal=False,
                        extra_turns=sorted({t for x in group for t in x.turn_ids}),
                    )
                    changed = True
        if (final is ForgetStage.PURGED or fid in purge_now) and _job(conn, fid) is not None:
            _purge_rows(conn, fid, drop_payloads=True)
            _locked(conn, lambda fid=fid: _set_job(conn, fid, state="done", stage="purged"))
            changed = True
        if changed:
            applied.append(fid)
    if applied:
        backup.optimize_fts(conn)
    return applied


def _came_back(group: list[JournalLine]) -> tuple[set[str], set[str], set[str]] | None:
    """What partial `undone` lines name as come back: (record and mask ids, raw HMACs,
    suppression HMACs); None when the forget has none."""
    lines = [x for x in group if x.stage is ForgetStage.UNDONE and x.until]
    if not lines:
        return None
    return (
        {r for x in lines for r in x.record_ids},
        {h for x in lines for h in x.raw_hmacs},
        {h for x in lines for h in x.suppression_hmacs},
    )


def _replay_partial(
    conn: sqlite3.Connection,
    fid: str,
    group: list[JournalLine],
    here: str | None,
    back: tuple[set[str], set[str], set[str]],
    hurry: bool,
) -> bool:
    """A forget the user partly took back. Held stopped here: what came back is undone by
    name. Not here: only the rest is stopped again, as an `item` forget of the records and
    raw its `stopped` lines name (no island turns: a «Вернуть» gave them all back)."""
    recs, hmacs, _sups = back
    raw_back = _ids_in(conn, "SELECT id FROM raw_chunks WHERE content_hmac IN ({qs})", hmacs)
    if here == "stopped":
        named = recs | raw_back
        if not any(
            t["row_ref"].partition(":")[2] in named for t in _trash_rows(conn, fid, "memory")
        ):
            return False
        _undo(conn, fid, None, journal=False, back_ids=named)
        return True
    if here is not None:
        return False
    stopped = [x for x in group if x.stage is ForgetStage.STOPPED]
    targets = _ids_in(
        conn,
        "SELECT id FROM records WHERE id IN ({qs}) AND status NOT IN ('forgotten', 'purged')",
        {r for x in stopped for r in x.record_ids} - recs,
    )
    raws = _ids_in(
        conn,
        "SELECT id FROM raw_chunks WHERE content_hmac IN ({qs}) AND status = 'active'",
        {h for x in stopped for h in x.raw_hmacs} - hmacs,
    )
    if not (targets or raws):
        return False
    final = group[-1].stage
    try:
        _stop_use(
            conn,
            sorted(targets),
            scope=ForgetScope.ITEM,
            forever=hurry or final is ForgetStage.PURGED,
            conversation=None,
            forget_id=fid,
            journal=False,
            until=next((x.until for x in reversed(group) if x.until), None),
            extra_raws=sorted(raws),
        )
    except ForgetNotFound:
        return False
    return True


def _groups(lines: Iterable[JournalLine]) -> dict[str, list[JournalLine]]:
    groups: dict[str, list[JournalLine]] = {}
    for line in journal_order(lines):
        groups.setdefault(line.forget_id, []).append(line)
    return groups


def _replay_same(
    conn: sqlite3.Connection,
    fid: str,
    group: list[JournalLine],
    conversation: ConversationStore | None,
    hurry: bool,
) -> bool:
    """§9.3 (v0.4): the same forget again, not a wider one. Re-planned from `scope` and
    `target_ids` (a `stopping`-only forget has nothing else); `record_ids` are forgotten
    whole, raw named by its HMAC is forgotten or span-redacted as the plan decides here
    (another usable record still reading it keeps it), `masked_ids` are re-masked
    against X and stay usable, and `until` is kept: the trash window does not restart."""
    scope = next(x.scope for x in reversed(group) if x.scope is not None)
    hmacs = sorted({h for x in group for h in x.raw_hmacs})
    raw_ids = _ids_in(
        conn, "SELECT id FROM raw_chunks WHERE content_hmac IN ({qs}) AND status = 'active'", hmacs
    )
    until = next((x.until for x in reversed(group) if x.until), None)
    final = group[-1].stage
    forever = (
        hurry or final is ForgetStage.PURGED or (final is ForgetStage.STOPPING and bool(until))
    )
    try:
        _stop_use(
            conn,
            sorted({t for x in group for t in x.target_ids}),
            scope=scope,
            forever=forever,
            conversation=conversation,
            forget_id=fid,
            journal=False,
            extra_turns=sorted({t for x in group for t in x.turn_ids}),
            until=until,
            extra_records=sorted({r for x in group for r in x.record_ids}),
            extra_raws=sorted(raw_ids),
            extra_masked=sorted({m for x in group for m in x.masked_ids}),
        )
    except ForgetNotFound:
        return False
    return True


def heal_journal(conn: sqlite3.Connection, *, conversation: ConversationStore | None = None) -> int:
    """§9.1 order (A4), `api.startup` step 2: "forgotten ⇒ journaled" at every crash point.
    A purge job with no `stopped` line gets the line, built from the job and the
    `suppressions` rows; a `stopping` line with no purge job is finished now (the user's word
    stands; no second user line). Idempotent; ids only. Returns the forgets healed."""
    path = journal_path(conn)
    if path is None:
        return 0
    groups = _groups(read_journal(path))
    healed = 0
    for (payload,) in conn.execute(
        "SELECT payload FROM jobs WHERE jkind = ? ORDER BY created_at", (PURGE_JOB,)
    ).fetchall():
        body = json.loads(payload)
        fid = body["forget_id"]
        if any(x.stage is not ForgetStage.STOPPING for x in groups.get(fid, [])):
            continue
        for line in _lines_from_job(conn, body):
            append_journal(path, line)
        healed += 1
    for fid, group in groups.items():
        if any(x.stage is not ForgetStage.STOPPING for x in group) or _job(conn, fid):
            continue
        line = group[-1]
        try:
            _stop_use(
                conn,
                list(line.target_ids),
                scope=line.scope or ForgetScope.ITEM,
                forever=bool(line.until),
                conversation=conversation if conversation is not None else _W.conversation,
                forget_id=fid,
                write_ahead=False,
            )
        except ForgetNotFound:
            continue  # nothing usable is left under those ids: nothing to finish
        healed += 1
    return healed


def _lines_from_job(conn: sqlite3.Connection, body: dict[str, Any]) -> list[JournalLine]:
    """The journal lines a purge job stands for (heal: the crash fell before its line)."""
    fid = body["forget_id"]
    recs = list(body.get("record_ids") or [])
    raw_ids = list(body.get("raw_ids") or [])
    raw_hmacs = sorted(
        _ids_in(conn, "SELECT content_hmac FROM raw_chunks WHERE id IN ({qs})", raw_ids)
    )
    sups = [
        r[0]
        for r in conn.execute(
            "SELECT hmac FROM suppressions WHERE forget_id = ? ORDER BY hmac", (fid,)
        )
    ]
    (version,) = conn.execute(
        f"SELECT coalesce(max(version), 0) FROM records WHERE id IN ({_qs(recs)})", recs
    ).fetchone()
    version = int(version) or store.memory_version(conn)
    stopped_at = body.get("stopped_at") or _ts()
    scope = body.get("scope")
    out = [
        JournalLine(
            ts=stopped_at,
            stage=ForgetStage.STOPPED,
            forget_id=fid,
            record_ids=recs,
            raw_hmacs=raw_hmacs,
            suppression_hmacs=sups,
            turn_ids=list(body.get("turn_ids") or []),
            scope_done=list(_STOP_SCOPE),
            memory_version=version,
            scope=ForgetScope(scope) if scope else None,
            target_ids=list(body.get("target_ids") or []),
            masked_ids=list(body.get("masked_ids") or []),
            until=body.get("until"),
        )
    ]
    stage = body.get("stage")
    if stage in ("undone", "purged"):
        out.append(
            JournalLine(
                ts=_ts(),
                stage=ForgetStage(stage),
                forget_id=fid,
                record_ids=recs if stage == "undone" else [],
                memory_version=store.memory_version(conn),
            )
        )
    return out


def _replay_ids(
    conn: sqlite3.Connection,
    fid: str,
    group: list[JournalLine],
    back: tuple[set[str], set[str], set[str]] | None = None,
) -> bool:
    """The id-only half of a replayed forget: its suppressions (copied as HMACs, dated by
    its first line) and tombstones for raw HMACs this store has no row for, minus what a
    partial «Вернуть» brought back (`back`). True when a row was written."""
    since = group[0].ts
    _recs, raw_back, sup_back = back or (set(), set(), set())
    lines = [x for x in group if x.stage is not ForgetStage.UNDONE]
    sups = sorted({h for x in lines for h in x.suppression_hmacs} - sup_back)
    hmacs = sorted({h for x in lines for h in x.raw_hmacs} - raw_back)
    have_sups = _ids_in(conn, "SELECT hmac FROM suppressions WHERE hmac IN ({qs})", sups)
    have_raw = _ids_in(
        conn, "SELECT content_hmac FROM raw_chunks WHERE content_hmac IN ({qs})", hmacs
    )
    new_sups = [h for h in sups if h not in have_sups]
    tombs = [h for h in hmacs if h not in have_raw]
    if not (new_sups or tombs):
        return False

    def write() -> None:
        conn.executemany(
            "INSERT OR IGNORE INTO suppressions (hmac, forget_id, created_at) VALUES (?, ?, ?)",
            [(h, fid, since) for h in new_sups],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO raw_chunks (id, status, source_kind, author_tier, ts, text,"
            " content_hmac, policy_version, created_at, updated_at)"
            " VALUES (?, 'purged', 'tombstone', 'system', ?, NULL, ?, 0, ?, ?)",
            [(f"tomb:{h}", since, h, since, since) for h in tombs],
        )

    _locked(conn, write)
    return True


# ---------------------------------------------------------------- no re-derivation (§9.6, §9.8)


def blocked_by(
    conn: sqlite3.Connection,
    *,
    source_ts: str | None,
    author_tier: AuthorTier | str,
    subject_id: str | None = None,
    relation: str | None = None,
    family: str | None = None,
    value_norm: str | None = None,
    alias_norms: Iterable[str] = (),
    stable_ids: Iterable[tuple[str, str]] = (),
) -> str | None:
    """The commit-side check for ledger's applier, workers and builders: the forget_id
    whose suppression an op would undo, or None. An op is admitted only as the user's new
    word: `user`/`user_confirmed` tier from a source newer than the forget (§9.6)."""
    fam = family if family is not None else _family(conn, relation)
    val = value_norm or ""
    cand: list[str] = []
    if subject_id is not None:
        cand.append(suppression_hmac(subject_id, fam, val))
    cand += [alias_hmac(a, fam, val) for a in alias_norms]
    cand += [stable_id_hmac(k, v) for k, v in stable_ids]
    if not cand:
        return None
    tier = author_tier.value if isinstance(author_tier, AuthorTier) else str(author_tier)
    for fid, created in conn.execute(
        f"SELECT forget_id, created_at FROM suppressions WHERE hmac IN ({_qs(cand)}) "
        "ORDER BY created_at DESC",
        cand,
    ).fetchall():
        if tier in _USER_TIERS and source_ts is not None and source_ts > created:
            continue
        return fid
    return None


# ---------------------------------------------------------------- read paths


def raw_is_forgotten(conn: sqlite3.Connection, raw_id: str) -> bool:
    """Read-only, for the basket's raw lane before it shows a raw (`conn` is the memory db's
    sqlite3 connection, as everywhere in this module).

    True when the raw is missing or not `active`; when its content HMAC is in a forget of the
    journal whose last line is not «Вернуть» (this covers a raw redacted in place, which keeps
    its original HMAC); or when its text still quotes a forgotten value of a forget that is
    stopped or purging: the value phrase, a forgotten entity's name or a 4-word shingle of a
    forgotten quote, read from that forget's trash. Subject aliases and stems are not used
    here (they would hide every raw that names the person), and the user's own raw newer than
    the forget is the user's new word (§9.6), never hidden by a quote match."""
    from .ingest import _journal_hmacs

    row = conn.execute(
        "SELECT status, content_hmac, text, author_tier, ts FROM raw_chunks WHERE id = ?",
        (raw_id,),
    ).fetchone()
    if row is None or row[0] != "active":
        return True
    if row[1] and row[1] in _journal_hmacs(conn):
        return True
    text, tier, ts = row[2] or "", row[3], row[4] or ""
    if not text.strip():
        return False
    for created, payload in conn.execute(
        "SELECT created_at, payload FROM jobs WHERE jkind = ?", (PURGE_JOB,)
    ).fetchall():
        try:
            body = json.loads(payload)
        except (TypeError, ValueError):
            continue
        fid = body.get("forget_id")
        if not fid or body.get("stage") not in ("stopped", "purging") or body.get("scope") == "chat":
            continue
        if tier in _USER_TIERS and ts and created and ts > created:
            continue  # the user's new word after the forget (§9.6)
        keys: set[str] = set()
        for t in _trash_rows(conn, fid, "memory"):
            item = t["payload"]
            if item is None or not str(t["row_ref"]).startswith("record:"):
                continue
            rec = item.get("row") or {}
            value = rec.get("value_text")
            if value and len(match_norm(value).strip()) >= 4:
                keys.add(value)
            name = rec.get("display_name")
            if item.get("table") == "entities" and name and name != FORGOTTEN:
                keys.add(name)
            keys.update(_shingles(rec.get("quote_orig")))
            keys.update(_shingles((item.get("header") or {}).get("origin_quote")))
        if text_matches(text, compile_keys(keys)):
            return True
    return False
