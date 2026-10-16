"""Arrival by code, no model: normalize, HMAC, journal match, tier, junk gate, foreign
marking and quarantine, labels, raw store.

Spec: §6.2, §13.2.

Order per row (§2, G1): the row is read as `model.IngestRow`; its `source_kind`, chat id and
author tier are derived from role and platform metadata (never from model text); then
`policy.admit_ex` runs before any write. Only `Admission.text` is stored (F2). A stored text
whose HMAC matches a forgotten raw (the deletion journal, or a forgotten / purged / redacted
raw chunk) is refused, unless it is shorter than 20 normalized characters (P2). One write
transaction stores the chunks and their stemmed `fts_raw` rows and enqueues one `extract`
job for the chunks that are neither junk nor quarantined. Content never reaches a log: the
only trace of a refusal is a content-free `audit` line.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from .model import (
    OWNER_ENTITY_ID,
    SOURCE_FAMILIES,
    AuthorTier,
    ForgetStage,
    IngestRow,
    Label,
    RawChunk,
    TurnCtx,
    TurnOrigin,
    tier_rank,
)

EXTRACT_JOB = "extract"
SHORT_EXEMPT = 20  # §6.2 step 3: normalized texts under this length skip the journal match

# ---------------------------------------------------------------- normalize and HMAC


def normalize(text: str) -> str:
    """NFKC, casefold, collapsed whitespace (§6.2 step 3)."""
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


def content_hmac(normalized: str) -> str:
    """HMAC-SHA256 with the Keychain key; texts < 20 chars are exempt from journal match."""
    from .forget import keyed_hmac

    return keyed_hmac("raw", normalized)


# ---------------------------------------------------------------- rows: kind, chat, tier

_LEGACY_KINDS = {"cube_turn", "island", "owner_message"}
_ROLE_FAMILY = {"owner": "owner", "cube": "cube", "tool": "tool", "other": "telegram"}
_DB_ROLE = {"user": "owner", "assistant": "cube", "tool": "tool", "owner": "owner", "cube": "cube"}


def row_of(row: dict[str, Any] | IngestRow) -> IngestRow:
    """The §6.2 row; a raw `koob_messages` row (`role` user/assistant, `created_at`, `id`)
    from the conversation adapter is mapped to the same shape."""
    if isinstance(row, IngestRow):
        return row
    d = dict(row)
    if "created_at" in d or d.get("role") in ("user", "assistant"):
        d = {
            "role": _DB_ROLE.get(str(d.get("role")), "tool"),
            "ts": d.get("ts") or d.get("created_at") or "",
            "text": d.get("text") or "",
            "turn_id": None if d.get("turn_id") is None else str(d["turn_id"]),
            "thread_id": None if d.get("thread_id") is None else str(d["thread_id"]),
            "row_id": d.get("row_id", d.get("id")),
            "source": d.get("source"),
            "tool": d.get("tool"),
        }
    ch = d.get("channel")
    if ch is not None and ch not in ("typed", "voice", "speaker"):
        # the eval harness's channels (island / app / tg_bot) are typed words (§6.2)
        d = {**d, "channel": "typed"}
    return IngestRow.model_validate(d)


def source_kind_of(row: IngestRow) -> str:
    """§6.2 (A10): `<family>` or `<family>:<detail>` from `SOURCE_FAMILIES`; `capture:` is
    `screen:`; legacy `cube_turn` maps by role; an unknown family falls back to the role."""
    src = (row.source or "").strip()
    head, sep, detail = src.partition(":")
    head = head.strip().casefold()
    if head == "capture":
        head = "screen"
    if row.role == "owner":
        return "owner"  # the user's words, any channel
    if row.role == "cube":
        return head if head in ("bot", "report") else "cube"
    if row.role == "tool":
        if head in SOURCE_FAMILIES and head not in ("owner", "cube", "tool") and head:
            return f"{head}:{detail.strip()}" if sep and detail.strip() else head
        name = (row.tool or detail or "").strip()
        return f"tool:{name}" if name else "tool"
    # role == "other": a person's message
    if head in SOURCE_FAMILIES and head not in ("owner", "cube", "tool", "bot", "report"):
        return f"{head}:{detail.strip()}" if sep and detail.strip() else head
    return _ROLE_FAMILY["other"]


def chat_of(row: IngestRow) -> str | None:
    """The user's words and the Cube's answer carry no chat (`island` is NULL); a tool row
    that read a chat, and a person's message, carry that chat's id."""
    if row.role in ("owner", "cube"):
        return None
    return row.chat_id or None


def source_ref_of(row: IngestRow, thread_id: str | None) -> str | None:
    if row.role in ("owner", "cube"):
        tid = row.thread_id or thread_id
        if tid:
            return f"thread:{tid}"
        return f"row:{row.row_id}" if row.row_id is not None else None
    if row.msg_id:
        return f"msg:{row.msg_id}"
    return f"row:{row.row_id}" if row.row_id is not None else None


def _lower(a: AuthorTier, b: AuthorTier | None) -> AuthorTier:
    """The lower of a computed tier and a metadata hint; a hint never raises a tier."""
    if b is None:
        return a
    return b if tier_rank(b) < tier_rank(a) else a


def tier_of(row: dict[str, Any] | IngestRow, turn: TurnCtx | None) -> AuthorTier:
    """§6.2 step 4: the lowest among the text's origins; never `user` for Cube text."""
    r = row_of(row)
    hint = r.author_tier
    if r.role == "owner":
        if r.fwd_from:  # a forwarded message is the forwarder's origin, not the user's words
            return _lower(AuthorTier.OTHER, hint)
        if r.channel == "speaker" or (turn is not None and turn.speaker):
            return _lower(AuthorTier.USER_ROOM, hint)
        if hint is AuthorTier.USER_CONFIRMED:
            return AuthorTier.USER_CONFIRMED
        return _lower(AuthorTier.USER, hint)
    if r.role == "cube":
        tainted = turn is not None and turn.is_tainted
        return AuthorTier.TOOL if tainted or hint is AuthorTier.TOOL else AuthorTier.SYSTEM
    if r.role == "tool":
        if hint is AuthorTier.OTHER and (r.msg_id or r.sender_id):
            return AuthorTier.OTHER  # a person's message read by a tool, with its id
        return AuthorTier.TOOL if hint is not AuthorTier.SYSTEM else AuthorTier.SYSTEM
    return _lower(AuthorTier.OTHER, hint)


def turn_ctx(
    rows: list[dict[str, Any]] | list[IngestRow],
    *,
    turn_id: str,
    thread_id: str,
    origin: str = "owner",
) -> TurnCtx:
    """The after-turn adapter's ids plus what the rows say: their words, foreign text, room
    speech. Tainting comes from roles and metadata, never from model text."""
    rs = [row_of(r) for r in rows]
    own = [r.text for r in rs if r.role == "owner" and not r.fwd_from and r.text]
    foreign = any(r.role in ("tool", "other") or r.fwd_from for r in rs)
    try:
        org = TurnOrigin(origin)
    except ValueError:
        org = TurnOrigin.OWNER
    return TurnCtx(
        turn_id=str(turn_id),
        thread_id=str(thread_id),
        origin=org,
        owner_words="\n".join(own),
        tainted=foreign,
        speaker=any(r.channel == "speaker" for r in rs),
        clock=next((r.ts for r in rs if r.ts), None),
    )


# ---------------------------------------------------------------- junk gate (step 5)

_ACKS = frozenset(
    [
        "ок",
        "окей",
        "ok",
        "okay",
        "k",
        "kk",
        "ага",
        "угу",
        "да",
        "нет",
        "не",
        "неа",
        "ну",
        "ладно",
        "хорошо",
        "хор",
        "понял",
        "поняла",
        "понятно",
        "ясно",
        "спс",
        "спасибо",
        "благодарю",
        "пасиб",
        "thx",
        "thanks",
        "thank",
        "ty",
        "yes",
        "no",
        "yep",
        "nope",
        "sure",
        "лол",
        "lol",
        "хаха",
        "haha",
        "ахах",
        "ахаха",
        "пон",
        "норм",
        "ок,спс",
        "good",
        "nice",
        "cool",
        "круто",
        "супер",
        "класс",
        "отлично",
        "done",
        "готово",
        "принял",
        "принято",
        "договорились",
    ]
)
_NOISE_SENDER = re.compile(r"(?i)(?:^|[\s@_])\w*bot$|^cron\b|^system$|^noreply")
_NOISE_TEXT = re.compile(
    r"(?i)^\s*[\[(]?(?:cron|heartbeat|healthcheck|ping|keepalive|scheduled job|job \S+ "
    r"(?:started|finished|completed|succeeded|failed))\b"
)
_CUBE_RULES = (
    "не отправлять без его слова",
    "этот чат под полным запретом",
    "наш с тобой разговор режима не получает",
    "режим чата меняю только по твоим словам",
    "перестал использовать",
    "удаляю полностью",
    "удалил полностью",
    "капчу вводишь ты",
    "enter — твой",
    "enter - твой",
    "пароли и ключи я не",
    "по твоему правилу",
    "по моим правилам",
)


def _letters(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def is_junk(text: str) -> bool:
    """§6.2 step 5: stickers, «ок»/«+»/emoji-only, cron noise. (Duplicates are not stored
    twice at all: `ingest_rows` keeps one copy per content HMAC.)"""
    n = normalize(text)
    if not n:
        return True
    if not _letters(n):  # «+», «)))», emoji, stickers' emoji text
        return True
    words = re.findall(r"[^\W_]+", n)
    if words and len(n) < 40 and all(w in _ACKS for w in words):
        return True
    return bool(_NOISE_TEXT.match(n))


def _junk_row(row: IngestRow, text: str) -> bool:
    if is_junk(text):
        return True
    if row.role == "other" and row.sender and _NOISE_SENDER.search(row.sender.strip()):
        return True  # another bot's message
    if row.role == "cube":
        n = normalize(text)
        return any(rule in n for rule in _CUBE_RULES)  # the Cube restating its rules
    return False


# ---------------------------------------------------------------- foreign marking (step 6)

_SENT_END = re.compile(r"[.!?…](?=\s|$)|\n")
_POISON = [
    # MemGhost-style memory orders, Russian and English
    re.compile(
        r"(?i)(?<![^\W_])(?:запомни(?:те)?|запиши(?:те)? в память|сохрани(?:те)? в памят\w*|"
        r"забудь(?:те)?|забыть|сотри(?:те)? из памяти|удали(?:те)? из памяти|"
        r"исправь(?:те)? в памяти|обнови(?:те)? в памяти)(?![^\W_])"
    ),
    re.compile(
        r"(?i)\b(?:remember\s+(?:that|this|:)|forget\s+(?:that|this|about|everything|all|his|her)|"
        r"note\s+that|(?:save|store|add|write)\s+(?:this|that|it)?\s*(?:to|in|into)\s+(?:your\s+)?"
        r"memory|update\s+(?:your|the)\s+memory|(?:memory|records?)\s+(?:update|correction)\s*:|"
        r"correction\s*:)"
    ),
    # addressing the assistant from foreign text
    re.compile(
        r"(?i)(?<![^\W_])(?:куб|koob|куб|cube|ассистент|assistant|нейросеть|ии|ai)\s*[,:!—–-]"
    ),
    # fake role and system tags
    re.compile(r"(?i)<\s*/?\s*(?:system|assistant|user|instructions?|his_[a-z_]+)\b[^>]*>"),
    re.compile(r"(?i)\[\s*(?:system|инструкция|instruction)s?\s*\]"),
]


def quarantine_spans(text: str) -> list[tuple[int, int]]:
    """§6.2 step 6: instruction-like spans (start of the order to the end of its sentence),
    merged and sorted."""
    from ..sanitize import INJECTION_PATTERNS

    body = text or ""
    starts: list[int] = []
    for pat in [p for _, p in INJECTION_PATTERNS] + _POISON:
        starts.extend(m.start() for m in pat.finditer(body))
    spans: list[tuple[int, int]] = []
    for s in sorted(set(starts)):
        m = _SENT_END.search(body, s + 1)
        e = m.end() if m else len(body)
        if spans and s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans


# ---------------------------------------------------------------- journal and retired values

_JOURNAL_CACHE: dict[str, tuple[float, int, frozenset[str]]] = {}


def _journal_hmacs(conn: sqlite3.Connection) -> frozenset[str]:
    """Raw HMACs of forgets whose last journal line is not «Вернуть» (§9.3)."""
    from . import forget

    path = forget.journal_path(conn)
    if path is None or not Path(path).exists():
        return frozenset()
    st = Path(path).stat()
    key = str(path)
    hit = _JOURNAL_CACHE.get(key)
    if hit is not None and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    last: dict[str, ForgetStage] = {}
    hm: dict[str, set[str]] = {}
    for line in forget.journal_order(forget.read_journal(Path(path))):
        last[line.forget_id] = line.stage
        hm.setdefault(line.forget_id, set()).update(line.raw_hmacs)
    out = frozenset(h for fid, hs in hm.items() if last[fid] is not ForgetStage.UNDONE for h in hs)
    _JOURNAL_CACHE[key] = (st.st_mtime, st.st_size, out)
    return out


def forgotten_match(conn: sqlite3.Connection, h: str, normalized: str) -> bool:
    """§6.2 step 3: the HMAC names forgotten content (journal, forgotten/purged/tombstone
    raw, or a raw whose stored text was redacted). Short texts never match (P2)."""
    if len(normalized) < SHORT_EXEMPT:
        return False
    if h in _journal_hmacs(conn):
        return True
    for status, text in conn.execute(
        "SELECT status, text FROM raw_chunks WHERE content_hmac = ?", (h,)
    ).fetchall():
        if status != "active":
            return True
        if text is not None and content_hmac(normalize(text)) != h:
            return True  # redacted in place («[забыто]», §9.6): the original was forgotten
    return False


_CONTACT = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<![\w@])@[A-Za-z]\w{3,}|\+?\d[\d\s().-]{6,}\d"
)


def _contact_hmacs(text: str) -> set[str]:
    out: set[str] = set()
    for m in _CONTACT.finditer(text or ""):
        tok = normalize(m.group(0))
        out.add(content_hmac(tok))
        digits = re.sub(r"\D", "", tok)
        if len(digits) >= 7 and not tok.startswith("@") and "@" not in tok:
            out.update({content_hmac(digits), content_hmac("+" + digits)})
    return out


def _retired(conn: sqlite3.Connection, hmacs: set[str]) -> bool:
    if not hmacs:
        return False
    qs = ",".join("?" * len(hmacs))
    q = f"SELECT 1 FROM retired_values WHERE hmac IN ({qs}) LIMIT 1"
    return conn.execute(q, sorted(hmacs)).fetchone() is not None


# ---------------------------------------------------------------- store


def _active_copy(conn: sqlite3.Connection, h: str) -> bool:
    q = "SELECT 1 FROM raw_chunks WHERE content_hmac = ? AND status = 'active' LIMIT 1"
    return conn.execute(q, (h,)).fetchone() is not None


def _audit(conn: sqlite3.Connection, event: str) -> None:
    from . import store as _store

    conn.execute(
        "INSERT INTO audit (ts, event, record_id, actor) VALUES (?, ?, NULL, 'ingest')",
        (_store.now(), event),
    )


def store_raw(conn: sqlite3.Connection, chunk: RawChunk) -> str | None:
    """Admitted chunk -> raw_chunks + fts_raw; None when the journal rejects it. The caller
    holds the write transaction (or none: then the insert autocommits as one statement pair)."""
    from . import store as _store
    from .forget import _stem_for_fts

    n = normalize(chunk.text or "")
    # re-checked inside the write transaction: a forget that committed after planning wins
    if forgotten_match(conn, chunk.content_hmac, n):
        _audit(conn, "ingest.reject.forgotten")
        return None
    if len(n) >= SHORT_EXEMPT and _active_copy(conn, chunk.content_hmac):
        return None
    ts = _store.now()
    conn.execute(
        "INSERT INTO raw_chunks (id, status, rev, source_kind, source_ref, chat_id, turn_id,"
        " author_tier, author_entity_id, ts, text, content_hmac, labels, labels_final,"
        " cloud_seen, policy_version, quarantined, junk, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk.id,
            chunk.status.value,
            chunk.rev,
            chunk.source_kind,
            chunk.source_ref,
            chunk.chat_id,
            chunk.turn_id,
            chunk.author_tier.value,
            chunk.author_entity_id,
            chunk.ts,
            chunk.text,
            chunk.content_hmac,
            int(chunk.labels),
            int(chunk.labels_final),
            int(chunk.cloud_seen),
            chunk.policy_version,
            int(chunk.quarantined),
            int(chunk.junk),
            ts,
            ts,
        ),
    )
    (rowid,) = conn.execute("SELECT fts_rowid FROM raw_chunks WHERE id = ?", (chunk.id,)).fetchone()
    conn.execute(
        "INSERT INTO fts_raw (rowid, text) VALUES (?, ?)", (rowid, _stem_for_fts(chunk.text or ""))
    )
    return chunk.id


def _enqueue(conn: sqlite3.Connection, raw_ids: list[str], turn_id: str | None) -> None:
    payload = {"raw_ids": raw_ids, "turn_id": turn_id}
    key = f"{EXTRACT_JOB}:{raw_ids[0]}"
    from .workers import scheduler

    try:
        scheduler.enqueue(conn, EXTRACT_JOB, payload, idempotency_key=key)
        return
    except NotImplementedError:  # queue helper may be unimplemented; the row shape is §3.2's
        pass
    from . import store as _store
    from .ledger import new_id

    ts = _store.now()
    conn.execute(
        "INSERT OR IGNORE INTO jobs (id, jkind, payload, state, attempts, idempotency_key,"
        " created_at, updated_at) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?)",
        (new_id(), EXTRACT_JOB, json.dumps(payload), key, ts, ts),
    )


def _redelivered(
    conn: sqlite3.Connection, h: str, kind: str, chat: str | None, turn: str | None, ref: str | None
) -> bool:
    """The same message again (a redelivered event, a re-run after-turn adapter)."""
    if ref is not None and ref.startswith(("msg:", "row:")):
        q = "SELECT 1 FROM raw_chunks WHERE source_ref = ? AND chat_id IS ? AND content_hmac = ?"
        return conn.execute(q, (ref, chat, h)).fetchone() is not None
    if turn is not None:
        q = (
            "SELECT 1 FROM raw_chunks WHERE turn_id = ? AND source_kind = ? AND chat_id IS ?"
            " AND content_hmac = ?"
        )
        return conn.execute(q, (turn, kind, chat, h)).fetchone() is not None
    return False


def ingest_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]], turn: TurnCtx | None) -> int:
    """Admit -> store -> enqueue for a batch of `koob_messages` rows."""
    from . import entities, policy
    from . import store as _store
    from .ledger import new_id

    if not rows:
        return 0
    pv = conn.execute("SELECT policy_version FROM meta WHERE id = 1").fetchone()
    policy_version = int(pv[0]) if pv else 0
    planned: list[tuple[RawChunk, bool]] = []  # (chunk, process)
    for raw in rows:
        r = row_of(raw)
        text = r.text or r.caption or ""
        if not text.strip():
            continue  # a sticker or media with no words: nothing to keep
        kind = source_kind_of(r)
        chat = chat_of(r)
        tier = tier_of(r, turn)
        # §5.1: admission before any write; its text is the only text arrival stores (F2)
        a = policy.admit_ex(
            conn,
            source_kind=kind,
            chat_id=chat,
            text=text,
            author_tier=tier,
            turn=turn if r.turn_id is None or turn is None or r.turn_id == turn.turn_id else None,
            sender_id=r.sender_id,
            reply_to=r.reply_to,
            ts=r.ts or None,
        )
        if not a.admit or not a.text:
            continue
        body = a.text
        n = normalize(body)
        h = content_hmac(n)
        turn_id = r.turn_id or (turn.turn_id if turn is not None else None)
        ref = source_ref_of(r, turn.thread_id if turn is not None else None)
        if _redelivered(conn, h, kind, chat, turn_id, ref) or any(
            c.content_hmac == h and c.source_ref == ref and c.turn_id == turn_id for c, _ in planned
        ):
            continue
        own = tier in policy.OWN_TIERS
        contacts = _contact_hmacs(body) if not own else set()
        if contacts and content_hmac(n) in contacts and _retired(conn, {h}):
            _audit(conn, "ingest.reject.retired")  # the bare old value again (§9.6)
            continue
        if len(n) >= SHORT_EXEMPT and forgotten_match(conn, h, n):
            _audit(conn, "ingest.reject.forgotten")
            continue
        if len(n) >= SHORT_EXEMPT and (
            _active_copy(conn, h) or any(c.content_hmac == h for c, _ in planned)
        ):
            continue  # content-hash dedupe: one copy, so a forget of it reaches every copy
        junk = _junk_row(r, body)
        quarantined = False
        if tier not in policy.OWN_TIERS and tier is not AuthorTier.SYSTEM:
            quarantined = bool(quarantine_spans(body)) or _retired(conn, contacts)
        labels = Label.NONE if junk else policy.label_topics(policy.words(body))
        author = OWNER_ENTITY_ID if own else None
        if r.role == "other" and not r.fwd_from and not junk and not quarantined:
            # §13.3: a person's own message names its author by the sender's stable id; a
            # forward is someone else's words. Extraction reads «я» as this person.
            author = entities.author_entity(
                conn, sender_id=r.sender_id, sender=r.sender, source=kind
            )
        chunk = RawChunk(
            id=new_id(),
            source_kind=kind,
            source_ref=ref,
            chat_id=chat,
            turn_id=turn_id,
            author_tier=tier,
            author_entity_id=author,
            ts=r.ts or _store.now(),
            text=body,
            content_hmac=h,
            labels=labels,
            labels_final=False,  # the in-process classifier (§5.3) finalizes; until then local only
            policy_version=policy_version,
            quarantined=quarantined,
            junk=junk,
        )
        planned.append((chunk, not junk and not quarantined))
    if not planned:
        return 0
    stored: list[str] = []
    with _store.write_tx(conn, priority="background"):
        process: list[str] = []
        for chunk, go in planned:
            rid = store_raw(conn, chunk)
            if rid is None:
                continue
            stored.append(rid)
            if go:
                process.append(rid)
        if process:
            _enqueue(conn, process, turn.turn_id if turn is not None else None)
    if process and not conn.in_transaction:
        # §5.3: the classifier pass right after arrival, off the turn; until it lands (or with
        # a cold classifier) the rows stay not final and local only. The extract batch
        # finalizes again before it routes (`scheduler.finalize_labels`).
        from . import labeler

        labeler.finalize_later(conn, raw_ids=process)
    return len(stored)


# ---------------------------------------------------------------- the two entry points


def _conversation() -> Any:
    from . import forget

    return getattr(forget._W, "conversation", None)


def ingest_turn(
    conn: sqlite3.Connection,
    *,
    turn_id: str,
    thread_id: str,
    origin: str,
    rows: list[dict[str, Any]] | None,
) -> int:
    """After-turn arrival: the turn's rows (the wired conversation adapter when `rows` is
    None), admitted on the turn's sources."""
    if rows is None:
        conv = _conversation()
        if conv is None:
            return 0  # unwired host: no conversation side
        rows = conv.turn_rows(str(turn_id))
    ctx = turn_ctx(rows, turn_id=turn_id, thread_id=thread_id, origin=origin)
    return ingest_rows(conn, [r for r in rows], ctx)


def sweep(conn: sqlite3.Connection, rows: list[dict[str, Any]] | None = None) -> int:
    """§6.2 step 1: rows with no turn, above `meta.sweep_watermark`; the watermark moves to
    the highest `row_id` seen. Rows that belong to a turn are left to the after-turn adapter."""
    from . import store as _store

    (mark,) = conn.execute("SELECT sweep_watermark FROM meta WHERE id = 1").fetchone()
    if rows is None:
        conv = _conversation()
        if conv is None:
            return 0
        total = 0
        while True:
            batch = conv.rows_without_turn(after_id=int(mark))
            if not batch:
                return total
            total += sweep(conn, batch)
            (new_mark,) = conn.execute("SELECT sweep_watermark FROM meta WHERE id = 1").fetchone()
            if new_mark <= mark:
                return total
            mark = new_mark
    todo: list[IngestRow] = []
    top = int(mark)
    for raw in rows:
        r = row_of(raw)
        if r.turn_id is not None:
            continue
        if r.row_id is not None:
            if r.row_id <= int(mark):
                continue
            top = max(top, int(r.row_id))
        if r.role == "cube" and not r.source:
            r = r.model_copy(update={"source": "report"})  # a no-turn Cube line (§6.2)
        todo.append(r)
    n = ingest_rows(conn, todo, None)  # type: ignore[arg-type]
    if top > int(mark):
        with _store.write_tx(conn, priority="background"):
            conn.execute(
                "UPDATE meta SET sweep_watermark = ?, updated_at = ? WHERE id = 1"
                " AND sweep_watermark < ?",
                (top, _store.now(), top),
            )
    return n
