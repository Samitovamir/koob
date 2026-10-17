"""In-turn writes before the next answer: remember, promises, corrections, retractions,
constraints; the after-turn safety net (freeze, urgent job, failure line) and the Stop-hook
claim check. «забудь» ("forget") itself is `commands.forget_now`; the safety net watches it.

Spec: §6.1 (the core), §4.1-§4.5, §5.5, §6.5, §7.1-§7.3, §7.6, §8.3-§8.4, §9.6, §10.4
(target grammar), §12 (promises), §13.1-§13.2 (tainted turns).

No model runs here: the brain supplies the structure (`Assertion`), code resolves names,
checks the vocabulary, normalizes values and commits synchronously through `store.commit`
with the ledger's applier (p95 ≤ 50 ms). Every record carries `origin_quote`.

Seams for other modules:
- `on_core_change(fn)`: core registers its core-section rebuild; a fix calls it with the
  touched record ids after the commit (the ledger already invalidated derived readers and
  queued `rebuild_derived` jobs, §7.3).
- `justify.cascade` runs inside the fix's transaction once it is built.
- `intentions.arm` arms a promise once it is built; until then the intention record is
  written by an ADD and stays `armed`.
- `frozen_entities` / `frozen_records` / `is_frozen`: recall drops these from the basket
  and update blocks, the hands gate denies actions on them. The urgent job
  (`jkind='safety_net'`) is run by extract; it lifts the freeze by finishing.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from math import ceil
from typing import Any

from .. import clock as _clockmod
from .. import ledger, policy, store
from ..model import (
    OWNER_ENTITY_ID,
    USABLE,
    Applicability,
    Assertion,
    AuthorTier,
    EventKind,
    Fact,
    FKind,
    Intention,
    Label,
    Loop,
    OpKind,
    Polarity,
    Preference,
    Proposal,
    ProposalOp,
    RecordKind,
    ServedVia,
    Status,
    Trigger,
    TriggerKind,
    TurnCtx,
    TurnOrigin,
)
from ..recall import lexical, safe_text

FREEZE_S = 60.0
SAFETY_JOB = "safety_net"
_HOT_OPS_MAX_CHARS = 500
log = logging.getLogger(__name__)


@dataclass
class Written:
    """What `api.WriteResult` carries (same field names)."""

    outcome: str
    record_ids: list[str] = field(default_factory=list)
    memory_version: int | None = None
    dependents_to_review: int = 0
    line: str | None = None


class _Refused(store.OpRejected):
    """Raised inside the fix's transaction: the target is gone or was undone meanwhile."""

    def __init__(self, line: str) -> None:
        super().__init__(line)
        self.line = line


# ---------------------------------------------------------------- seams

_CORE_HOOKS: list[Callable[[sqlite3.Connection, list[str]], None]] = []


def on_core_change(fn: Callable[[sqlite3.Connection, list[str]], None]) -> None:
    """core: called after a correction commits, with the touched record ids (§6.1)."""
    if fn not in _CORE_HOOKS:
        _CORE_HOOKS.append(fn)


def _core_changed(conn: sqlite3.Connection, ids: list[str]) -> None:
    for fn in list(_CORE_HOOKS):
        try:
            fn(conn, ids)
        except Exception:  # noqa: BLE001 - a view rebuild never fails his correction
            continue


# ---------------------------------------------------------------- the user's words


def _words(text: str) -> list[str]:
    return [w for w in policy.words(text or "") if len(w) >= 3]


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n >= 4 and n >= min(len(a), len(b)) - 2


def quote_is_his(quote: str, turn: TurnCtx) -> bool:
    q = policy.norm(quote or "")
    own = policy.norm(policy.turn_words(turn))
    return len(q) >= 2 and bool(own) and q in own


def _his_word(w: str, own: list[str], skels: set[str]) -> bool:
    return any(_same(w, o) for o in own) or policy.word_matches(w, skels)


def content_named(text: str, quote: str, turn: TurnCtx, *, strict: bool | None = None) -> bool:
    """The user's own words name the content: the quote is a substring of the user's words, the
    content words of the text are in those words, and every value in it (dates, numbers,
    contacts) is one the user said. In a tainted turn (`strict`) EVERY content word must be the
    user's: one word from the page makes it the page's claim; elsewhere most (60 %) will do."""
    if not quote_is_his(quote, turn):
        return False
    strict = turn.is_tainted if strict is None else strict
    own_text = policy.turn_words(turn)
    own = _words(own_text)
    skels = {policy.skeleton(o) for o in own}
    content = [w for w in _words(text) if w not in policy._STOP]
    if not content:
        return False
    hit = sum(1 for w in content if _his_word(w, own, skels))
    if hit < (len(content) if strict else max(1, ceil(0.6 * len(content)))):
        return False
    return ledger.value_tokens(text) <= ledger.value_tokens(own_text)


def _entity_named(conn: sqlite3.Connection, ref: str, quote: str, turn: TurnCtx) -> bool:
    """A value entity counts as the user's only when the user's words name it (a name or an alias)."""
    if ref.casefold() in ("self", "m:self"):
        return True
    if ref.startswith("m:"):
        names = policy._entity_names(conn, [ref[2:]])
        return bool(names) and policy.names_in(policy.words(policy.turn_words(turn)), names)
    return content_named(ref, quote, turn)


def scrub(text: str | None) -> str | None:
    """I-3: never-store spans out of anything the hot writer stores."""
    return policy.redact(text) if text else text


def _scrub_assertion(a: Assertion | None) -> Assertion | None:
    if a is None:
        return None
    fields = ("subject", "value", "value_entity", "text_en", "keywords_ru", "keywords_en")
    return a.model_copy(update={k: scrub(getattr(a, k)) for k in fields if getattr(a, k)})


def _needs_his_word(turn: TurnCtx) -> bool:
    return turn.is_tainted or turn.origin is not TurnOrigin.OWNER


def _tier(turn: TurnCtx, named: bool) -> AuthorTier:
    if not _needs_his_word(turn) or named:
        return AuthorTier.USER_ROOM if turn.speaker else AuthorTier.USER
    return AuthorTier.TOOL if turn.is_tainted else AuthorTier.SYSTEM


def _origin_quote(quote: str, turn: TurnCtx, fallback: str) -> str:
    """The stored quote: the user's matching words, else the turn's words, else the brain's quote,
    always through the never-store pass (the user's raw words may hold a password the brain
    never saw)."""
    if quote_is_his(quote, turn):
        out = quote.strip()
    elif not _needs_his_word(turn) and (turn.owner_words or "").strip():
        out = turn.owner_words.strip()[:_HOT_OPS_MAX_CHARS]
    else:
        out = (quote or "").strip() or fallback.strip()[:_HOT_OPS_MAX_CHARS]
    return (scrub(out) or "").strip() or "(no quote)"


def _lang(text: str) -> str:
    return "ru" if re.search(r"[а-яё]", text or "", re.IGNORECASE) else "en"


# ---------------------------------------------------------------- vocabulary and values


def _relation(conn: sqlite3.Connection, relation: str | None) -> tuple[str, str]:
    """(relation, value_type); unknown → `other` (§4.1)."""
    rel = (relation or "other").strip().casefold().replace(" ", "_")
    row = conn.execute("SELECT value_type FROM relations WHERE relation = ?", (rel,)).fetchone()
    if row is None:
        rel = "other"
        row = conn.execute("SELECT value_type FROM relations WHERE relation = 'other'").fetchone()
    return rel, (row[0] if row is not None else "text")


_DATE_PATTERNS = (ledger._ISO, ledger._DAY_MONTH, ledger._MONTH_DAY, ledger._DOTTED)


def normalize_value(value: str | None, value_type: str) -> str | None:
    """§6.1: `value_norm` by the relation's `value_type` (ledger conventions)."""
    if value is None or not value.strip():
        return None
    v = value.strip()
    toks = ledger.value_tokens(v)
    if value_type == "date":
        dates = sorted((t[5:] for t in toks if t.startswith("date:")), key=len, reverse=True)
        if dates:
            return dates[0]
    elif value_type == "email":
        m = ledger._EMAIL.search(v)
        return (m.group() if m else v).casefold()
    elif value_type == "phone":
        digits = re.sub(r"\D", "", v)
        return digits or v.casefold()
    elif value_type == "handle":
        return v.casefold().lstrip("@")
    elif value_type in ("number", "money"):
        nums = [t[4:] for t in toks if t.startswith("num:")]
        if nums:
            return nums[0]
    return " ".join(v.casefold().split())


def value_from(text: str, value_type: str) -> str:
    """The value span of a correction without an `assertion` (dates, contacts, numbers);
    other types take the whole correction."""
    low = text.casefold()
    pats: tuple[re.Pattern[str], ...] = ()
    if value_type == "date":
        pats = _DATE_PATTERNS
    elif value_type == "email":
        pats = (ledger._EMAIL,)
    elif value_type == "phone":
        pats = (ledger._PHONE,)
    elif value_type == "handle":
        pats = (ledger._HANDLE,)
    elif value_type in ("number", "money"):
        pats = (ledger._NUMBER,)
    for p in pats:
        m = p.search(low)
        if m:
            return text[m.start() : m.end()].strip()
    return text.strip()


# ---------------------------------------------------------------- names


@dataclass
class _Subject:
    entity_id: str | None = None
    new_name: str | None = None  # an unknown name: a new entity in the same commit
    ask: str | None = None


_SELF_WORDS = frozenset({"self", "я", "меня", "мне", "мной", "me", "myself", "i", "owner"})


def _resolve_name(conn: sqlite3.Connection, name: str | None, *, create: bool) -> _Subject:
    """`self`, `m:<id>`, else aliases (§16.1); ambiguous → ask; unknown → new (subject only)."""
    n = (name or "").strip()
    if not n or n.casefold() in _SELF_WORDS:
        return _Subject(OWNER_ENTITY_ID)
    if n.startswith("m:"):
        row = conn.execute(
            "SELECT status FROM records WHERE id = ? AND kind = 'entity'", (n[2:],)
        ).fetchone()
        if row is not None and Status(row[0]) in USABLE:
            return _Subject(n[2:])
        return _Subject(ask=f"Не нашёл, кто это — {n}.")
    from .. import entities

    hits = entities.resolve_rows(conn, n, "any")
    ids = list(dict.fromkeys(h.entity_id for h in hits if h.confidence >= 0.6))
    if len(ids) == 1:
        return _Subject(ids[0])
    if len(ids) > 1:
        opts = ", ".join(f"{h.display_name} [m:{h.entity_id}]" for h in hits[:4])
        return _Subject(ask=f"Про кого именно «{n}»: {opts}?")
    return _Subject(new_name=n) if create else _Subject()


# ---------------------------------------------------------------- remember


def _now() -> str:
    return store.now()


def _serve_write(
    ctx: store.CommitCtx, turn: TurnCtx, record_id: str, kind: str, version: int | None = None
) -> None:
    """§7.6: the write's result reached this thread; also the turn's commit mark (§6.1)."""
    ctx.conn.execute(
        "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id, via,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            turn.thread_id,
            turn.session_id,
            record_id,
            kind,
            ctx.version if version is None else version,
            turn.turn_id,
            ServedVia.WRITE.value,
            ctx.ts,
        ),
    )


def _hotlog(conn: sqlite3.Connection, proposal: Proposal, version: int | None, ids: list[str]):
    """§9.4: append the committed proposal to hotlog.db beside memory.db (replayed on restore)."""
    db = store.db_file(conn)
    if db is None:
        return
    from .. import backup

    h = backup.open_hotlog(db.parent / "hotlog.db")
    try:
        backup.hotlog_append(
            h, "proposal", proposal.model_dump(mode="json"), memory_version=version, item_ids=ids
        )
    finally:
        h.close()


def code_labels(text: str) -> Label:
    """`policy.label` without the classifier: lexicons, money, precise address, weak
    hits -> `unknown_sensitive`. The hot path stays within its 50 ms; the classifier's labels
    arrive right after the commit (`_classify_later`), and the record stays not final."""
    ws = policy.words(text)
    out = Label.NONE
    weak = False
    for lbl, lex in policy._STRONG.items():
        if policy._hits(ws, lex):
            out |= lbl
    for lbl, lex in policy._WEAK.items():
        if not out & lbl and policy._hits(ws, lex):
            weak = True
    if policy._MONEY_AMOUNT.search(text or ""):
        out |= Label.MONEY
    if policy._PRECISE_ADDR.search(text or ""):
        out |= Label.LOCATION_PRECISE
    if weak and not out & policy.sensitive_mask():
        out |= Label.UNKNOWN_SENSITIVE
    return out


def _labels(*texts: str) -> Label:
    out = Label.NONE
    for t in texts:
        if t:
            out |= code_labels(t)
    return out


# The classifier pass runs right after the commit on one background thread
# (`labeler.finalize_later`; an embedder call is 15-20 ms warm, two of them broke the 50 ms
# budget). It goes through the ledger (an UPDATE at the version committed; an edit under it
# makes it re-score), labels only grow, and `labels_final = 1` lands only when the classifier
# was warm for the whole pass; a cold or failing one leaves the record not final (fail-closed;
# `api.run_jobs` sweeps it later).


def _classify_later(conn: sqlite3.Connection, rid: str, *texts: str) -> None:
    """The classifier half of `policy.label` for one committed record, off the turn; then
    its labels are final (§5.3)."""
    from .. import labeler

    labeler.finalize_later(conn, record_id=rid, texts=[t for t in texts if t])


def drain_labels(timeout: float = 5.0) -> None:
    """Wait for the background classifier passes (tests, shutdown)."""
    from .. import labeler

    labeler.drain(timeout)


def _fact_record(
    conn: sqlite3.Connection,
    *,
    rid: str,
    subject_id: str,
    text: str,
    quote: str,
    origin: str,
    tier: AuthorTier,
    labels: Label,
    a: Assertion | None,
    kind: RecordKind,
    applicability: Applicability | None,
    ts: str,
) -> Fact:
    rel, vtype = _relation(
        conn, (a.relation if a else None) or ("prefers" if kind is RecordKind.PREFERENCE else None)
    )
    value = a.value if a else None
    common: dict[str, Any] = {
        "id": rid,
        "author_tier": tier,
        "recorded_at": ts,
        "labels": labels,
        "labels_final": False,
        "origin_quote": origin,
        "subject_id": subject_id,
        "relation": rel,
        "value_text": value,
        "value_norm": normalize_value(value, vtype),
        "text_en": ((a.text_en if a else None) or text).strip(),
        "keywords_ru": a.keywords_ru if a else None,
        "keywords_en": a.keywords_en if a else None,
        "quote_orig": quote,
        "lang_orig": _lang(quote),
        "fkind": (a.fkind if a and a.fkind else FKind.STATE),
        "polarity": (a.polarity if a and a.polarity else Polarity.AFFIRMED),
        "valid_from": a.valid_from if a else None,
        "valid_to": a.valid_to if a else None,
        "stability": a.stability if a else None,
        "applicability": applicability,
    }
    if kind is RecordKind.PREFERENCE:
        common["applicability"] = applicability or Applicability(unstated=True)
        return Preference(**common)
    return Fact(**common)


def remember(
    conn: sqlite3.Connection,
    text: str,
    turn: TurnCtx,
    *,
    because_quote: str,
    kind: RecordKind | None = None,
    applicability: Applicability | None = None,
    assertion: Assertion | None = None,
) -> Written:
    """§6.1 «запомни» / a promise / a constraint: one structured record, committed now."""
    text = (text or "").strip()
    if not text and not (assertion and (assertion.text_en or assertion.value)):
        return Written("rejected", line="Нечего запоминать.")
    named = content_named(text or (assertion.text_en or ""), because_quote, turn)
    if turn.is_tainted:  # the value and the value entity must be the user's words too
        if named and assertion is not None and assertion.value:
            named = content_named(assertion.value, because_quote, turn)
        if named and assertion is not None and assertion.value_entity:
            named = _entity_named(conn, assertion.value_entity, because_quote, turn)
    elif not named and assertion is not None and assertion.value:
        named = content_named(assertion.value, because_quote, turn)
    assertion = _scrub_assertion(assertion)
    tier = _tier(turn, named)
    adm = policy.admit_ex(
        conn, source_kind="island", chat_id=None, text=text, author_tier=tier, turn=turn
    )
    if not adm.admit:
        return Written("rejected", line="Это я не сохраняю.")
    text = (adm.text or text).strip()
    kind = RecordKind(kind) if kind is not None else RecordKind.FACT
    a = assertion
    if kind is RecordKind.INTENTION and a is not None and a.due and a.trigger is None:
        kind = RecordKind.LOOP
    origin = _origin_quote(because_quote, turn, text)
    quote = origin
    labels = _labels(text, quote)
    rid = ledger.new_id()
    ts = turn.clock or _now()
    ops: list[ProposalOp] = []
    group = OWNER_ENTITY_ID
    if kind in (RecordKind.FACT, RecordKind.PREFERENCE):
        subj = _resolve_name(conn, a.subject if a else None, create=True)
        if subj.ask:
            return Written("ask_first", line=subj.ask)
        subject_id = subj.entity_id
        if subj.new_name:
            subject_id = ledger.new_id()
            ent = {
                "kind": "entity",
                "id": subject_id,
                "author_tier": tier.value,
                "recorded_at": ts,
                "etype": "person" if subj.new_name[:1].isupper() else "thing",
                "display_name": subj.new_name,
                "origin_quote": origin,
            }
            ops.append(
                ProposalOp(
                    op=OpKind.ADD,
                    group=subject_id,
                    payload={"record": ent, "aliases": [{"alias": subj.new_name}]},
                )
            )
        group = subject_id or OWNER_ENTITY_ID
        rec = _fact_record(
            conn,
            rid=rid,
            subject_id=group,
            text=text,
            quote=quote,
            origin=origin,
            tier=tier,
            labels=labels,
            a=a,
            kind=kind,
            applicability=applicability,
            ts=ts,
        )
        if a is not None and a.value_entity:
            ve = _resolve_name(conn, a.value_entity, create=False)
            if ve.ask:
                return Written("ask_first", line=ve.ask)
            if ve.entity_id:
                rec = rec.model_copy(update={"value_entity_id": ve.entity_id})
        record = rec.model_dump(mode="json", exclude_none=True)
    elif kind is RecordKind.LOOP:
        owner = (
            "cube" if a is not None and (a.subject or "").casefold() in ("cube", "куб") else "user"
        )
        record = Loop(
            id=rid,
            author_tier=tier,
            recorded_at=ts,
            labels=labels,
            origin_quote=origin,
            owner=owner,
            text_en=((a.text_en if a else None) or text),
            due=a.due if a else None,
        ).model_dump(mode="json", exclude_none=True)
    elif kind is RecordKind.INTENTION:
        trig = a.trigger if a is not None and a.trigger is not None else None
        if trig is None:
            trig = Trigger(
                kind=TriggerKind.SOFT_OPPORTUNITY, goal=(a.text_en if a else None) or text
            )
        intention = Intention(
            id=rid,
            author_tier=tier,
            recorded_at=ts,
            labels=labels,
            origin_quote=origin,
            trigger=trig,
            action={"text": (a.text_en if a else None) or text},
        )
        armed = _arm(conn, intention, turn)
        if armed is not None:
            return armed
        record = intention.model_dump(mode="json", exclude_none=True)
    else:
        return Written("rejected", line="Такое через «запомни» не пишется.")
    ops.append(ProposalOp(op=OpKind.ADD, group=group, payload={"record": record}))
    proposal = Proposal(ops=ops, author_tier=tier)
    box: dict[str, Any] = {}

    def apply(ctx: store.CommitCtx, op: ProposalOp) -> list[str]:
        touched = list(ledger.apply_op(ctx, op))
        if op is ops[-1]:
            _serve_write(ctx, turn, rid, kind.value)
            box["v"] = ctx.version
        return touched

    try:
        res = store.commit(conn, proposal, apply, event_kind=EventKind.WRITE, priority="hot")
    except store.OpRejected as exc:
        return Written("rejected", line=_reject_line(exc))
    if not res.ok:
        return Written("rejected", line="Не смог сохранить — попробуй ещё раз.")
    version = box.get("v")
    _hotlog(conn, proposal, version, [rid])
    _classify_later(conn, rid, text, quote)
    line = (
        "Запомнил."
        if tier in (AuthorTier.USER, AuthorTier.USER_ROOM)
        else ("Записал как сведения со страницы, не как твои слова.")
    )
    return Written("committed", record_ids=[rid], memory_version=version, line=line)


def _arm(conn: sqlite3.Connection, intention: Intention, turn: TurnCtx) -> Written | None:
    """intent's `arm` when built (it writes the record); None → the ADD path writes it.
    The armed intention is stamped as this turn's write (§6.1: the safety net and the Stop
    hook see the promise landed, as they see every other hot write)."""
    from .. import intentions

    try:
        iid = intentions.arm(conn, intention)
    except NotImplementedError:
        return None
    except (store.OpRejected, ValueError) as exc:
        return Written("rejected", line=str(exc) or "Не могу это поставить.")
    version = store.memory_version(conn)
    with store.write_tx(conn):
        conn.execute(
            "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id, via,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn.thread_id, turn.session_id, iid, RecordKind.INTENTION.value, version,
                turn.turn_id, ServedVia.WRITE.value, store.now(),
            ),
        )  # fmt: skip
    return Written("committed", record_ids=[iid], memory_version=version, line="Запомнил.")


def _reject_line(exc: Exception) -> str:
    if isinstance(exc, _Refused):
        return exc.line
    reason = getattr(exc, "reason", None) or str(exc)
    if "suppress" in reason or "retract" in reason:
        return "Это ты просил забыть или назвал неправдой — не записываю."
    if "tier" in reason:
        return "Это пришло не из твоих слов — так не записываю."
    return "Не смог сохранить — попробуй ещё раз."


# ---------------------------------------------------------------- fix


_FIX_LEAD = re.compile(
    r"^\s*(?:нет[,.!]?\s*|исправь[:,]?\s*|поправь[:,]?\s*|это\s+неправда[:,.!]?\s*|неправда[:,.!]?\s*"
    r"|не\s+так[:,.!]?\s*|на\s+самом\s+деле[:,]?\s*|actually[:,]?\s*|no[,.!]\s*|wrong[:,.!]?\s*)+",
    re.IGNORECASE,
)


def _strip_fix_lead(text: str) -> str:
    return _FIX_LEAD.sub("", text or "").strip()


# «не X, а Y» / «Y, а не X» / "not X but Y" / "Y, not X": the old value the user's correction
# names and the new one. A clause keeps a time, a date or a decimal whole (14:20, 12.06, 10/14).
_V, _END = ledger._V, ledger._END
_CONTRAST = (
    re.compile(rf"(?<!\w)не\s+(?P<old>{_V}+?)\s*[,—–-]?\s+а\s+(?P<new>{_V}+)", re.IGNORECASE),
    re.compile(rf"(?P<new>{_V}+?)\s*,\s*а\s+не\s+(?P<old>{_V}+?){_END}", re.IGNORECASE),
    re.compile(rf"(?<!\w)not\s+(?P<old>{_V}+?)\s*,?\s+but\s+(?P<new>{_V}+)", re.IGNORECASE),
    # "15:40, not 14:20", "by Thursday, not Friday", "10/14 not 10/13": a value, then «not» + old
    re.compile(
        rf"(?P<new>[^\s,;!?]+)(?P<sep>\s*[,—–]\s*|\s+)not\s+(?P<old>{_V}+?){_END}", re.IGNORECASE
    ),
)
# "moved to Y", "pushed to Y", "it's Y now", «перенесли на Y», «теперь Y»: a correction that
# names only the new value (the old one is the record it corrects)
_SHIFT_WORD = (
    r"(?:moved|pushed|postponed|rescheduled|changed|bumped|shifted|delayed|перенес\w*"
    r"|перенёс\w*|сдвинул\w*|сдвинут\w*|передвинул\w*|поменял\w*|отложил\w*)"
)
_SHIFT = (
    re.compile(
        rf"(?<!\w){_SHIFT_WORD}(?:\s+(?:back|up|forward|again))?\s+(?:to|till|until|на)\s+"
        rf"(?P<new>{_V}+?){_END}",
        re.IGNORECASE,
    ),
    re.compile(rf"(?<!\w)(?:it'?s|it\s+is)\s+(?P<new>{_V}+?)\s+now(?!\w)", re.IGNORECASE),
    re.compile(rf"(?<!\w)now\s+(?:it'?s|it\s+is)\s+(?P<new>{_V}+?){_END}", re.IGNORECASE),
    re.compile(rf"(?<!\w)теперь\s+(?P<new>{_V}+?){_END}", re.IGNORECASE),
)
# "the demo got pushed - it's wed 10/14": after a shift word, «it's Y» names the new value
_SHIFTED_IS = re.compile(
    rf"(?<!\w){_SHIFT_WORD}(?!\w).*?(?<!\w)(?:it'?s|it\s+is|it\s+will\s+be|будет)\s+"
    rf"(?P<new>{_V}+?){_END}",
    re.IGNORECASE,
)
# the user's own «я спутала», «я перепутал», «я ошиблась» after the correction: not the value
_OWN_SLIP = re.compile(
    r"\s*[,;—–-]\s*(?:я\s+)?(?:\S+\s+)?(?:спутал\w*|перепутал\w*|ошибл\w*|ошибся"
    r"|не\s+так\s+понял\w*|my\s+mistake|i\s+was\s+wrong)\s*[.!]*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Contrast:
    old: str
    new: str
    start: int
    end: int


def contrast(text: str | None) -> Contrast | None:
    """The «не X, а Y» part of a correction: X is the value it retires, Y the new one. Also
    «Y, а не X», "not X but Y" and "Y, not X" (a negated verb - "it is not ready",
    "Dad not coming" - is no contrast). None when the words hold no such contrast."""
    for pat in _CONTRAST:
        for m in pat.finditer(text or ""):
            old, new = m.group("old").strip(" -—–"), m.group("new").strip(" -—–")
            if not old or not new:
                continue
            if "sep" in pat.groupindex and (
                new.casefold() in ledger._AUX
                or (not re.search(r"[,—–]", m.group("sep")) and not re.search(r"\d", new))
            ):
                continue
            return Contrast(old, new, m.start(), m.end())
    return None


def shifted(text: str | None) -> str | None:
    """The new value of a correction that names only it: "moved to Friday", "it's 10/14 now",
    "got pushed - it's wed 10/14 at 3", «перенесли на пятницу», «теперь в 15:00». None else."""
    for pat in (*_SHIFT, _SHIFTED_IS):
        m = pat.search(text or "")
        if m and m.group("new").strip(" -—–"):
            return m.group("new").strip(" -—–")
    return None


def positive_text(text: str | None) -> str:
    """The user's words without the value a contrast retires: «X is not in Tver but in Samara» →
    «X is in Samara». A topic match never takes the new value through the old one's words."""
    out = text or ""
    for _ in range(4):
        c = contrast(out)
        if c is None:
            break
        out = out[: c.start] + c.new + out[c.end :]
    return out


def corrected_text(correction: str) -> str:
    """What the correcting record states: the contrast collapsed to its new side, the fix lead
    («нет», «исправь») and the user's own «я спутала» dropped. The old value never stays in it."""
    out = _strip_fix_lead(positive_text(correction))
    out = _OWN_SLIP.sub("", out).strip(" ,;—–-")
    return out or correction.strip()


def _turn_raw(conn: sqlite3.Connection, turn: TurnCtx, quote: str) -> str | None:
    """Lineage of a correction: the user's own raw message of this turn that holds the quote."""
    if not turn.turn_id or not (quote or "").strip():
        return None
    want = " ".join(quote.casefold().split())
    for rid, text in conn.execute(
        "SELECT id, text FROM raw_chunks WHERE turn_id = ? AND status = 'active'"
        " AND quarantined = 0 AND author_tier IN (?, ?) ORDER BY ts",
        (turn.turn_id, AuthorTier.USER.value, AuthorTier.USER_CONFIRMED.value),
    ):
        if want and want in " ".join((text or "").casefold().split()):
            return str(rid)
    return None


def _mask_written_raws(conn: sqlite3.Connection, record_id: str, turn: TurnCtx, ts: str) -> None:
    """A record the hot path wrote («запомни…») carries no `source_raw_id` (the user's message is
    ingested after the write), so once a fix retires it the raw lane had no lineage to mask
    the user's old message by and quoted the retired value as current. The fix links the record
    to the user's own raws of the turns that wrote it, those holding its quote, through
    `raw_retractions` (the raw lane's lineage table). The fixing turn's own raw is the user's
    new word, never masked here; a raw already linked to the record keeps its link."""
    rec = conn.execute(
        "SELECT r.origin_quote, f.quote_orig, f.value_norm, f.source_raw_id FROM records r"
        " LEFT JOIN facts f ON f.id = r.id WHERE r.id = ?",
        (record_id,),
    ).fetchone()
    if rec is None:
        return
    quotes = [
        " ".join(q.casefold().split())
        for q in (rec["origin_quote"], rec["quote_orig"])
        if q and q.strip() and q.strip() != "(no quote)"
    ]
    if not quotes:
        return
    marks = ", ".join("?" * len(_MARK_KINDS))
    rows = conn.execute(
        "SELECT DISTINCT c.id, c.text FROM served s JOIN raw_chunks c ON c.turn_id = s.turn_id"
        f" WHERE s.item_id = ? AND s.via = 'write' AND s.item_kind NOT IN ({marks})"
        " AND c.status = 'active' AND c.author_tier IN (?, ?) AND c.turn_id IS NOT ?",
        (
            record_id,
            *_MARK_KINDS,
            AuthorTier.USER.value,
            AuthorTier.USER_CONFIRMED.value,
            turn.turn_id,
        ),
    ).fetchall()
    for raw_id, text in rows:
        if raw_id == rec["source_raw_id"]:
            continue
        said = " ".join((text or "").casefold().split())
        if not any(q in said for q in quotes):
            continue
        conn.execute(
            "INSERT OR IGNORE INTO raw_retractions (raw_id, record_id, retracted_value_norm,"
            " created_at) VALUES (?, ?, ?, ?)",
            (raw_id, record_id, rec["value_norm"] or "", ts),
        )


FIX_MARK = "fix:"  # `raw_retractions.record_id` prefix of a correction no record carried yet


def mark_corrected_raws(conn: sqlite3.Connection, turn: TurnCtx, words: str) -> int:
    """A correction leaves its mark on the user's own earlier messages holding the value it
    retires, whether or not a record holds that value yet (the message may still wait for a
    worker): a `raw_retractions` row keyed `fix:<turn id>` per message and retired value. The
    raw lanes cut it (`basket._raw_past`); a later extraction reads the correction itself
    (`ledger.retired_sides` orders it), so `ledger.check_retracted` skips these rows. The
    retired values are the user's contrast («не X, а Y», "Y, not X") and another name put in
    the same place slot; a message is marked only when it shares a topic stem with the
    correction besides the value itself. With no contrast the retired value is read by slot
    (`slot_sides`), and an amount takes the totals computed from it (`derived_amounts`); with
    one, a new value of the contrasted type stated besides it retires the old value of that
    slot as well (`restated_sides`)."""
    from ..recall import basket

    words = (words or "").strip()
    if not words or not (his_correction(words) or retired_values(words) or _slots(words)):
        return 0
    when = turn.clock or _now()
    own = tuple(t.value for t in ledger.USER_TIERS)
    rows = conn.execute(
        "SELECT id, rev, ts, chat_id, text, author_tier FROM raw_chunks WHERE status = 'active'"
        " AND quarantined = 0 AND junk = 0 AND author_tier IN (?, ?) AND turn_id IS NOT ?"
        " AND text IS NOT NULL AND ts < ? ORDER BY rowid DESC LIMIT ?",
        (*own, turn.turn_id, when, basket.RECENT_RAWS),
    ).fetchall()
    if not rows:
        return 0
    key = f"{FIX_MARK}{turn.turn_id}"
    said = {"id": key, "rev": 0, "ts": when, "text": words, "author_tier": own[0]}
    marks: dict[tuple[str, str], None] = {}
    for raw_id, pairs in basket._later_contrasts(conn, rows, [said]).items():
        for side, _cid in pairs:
            marks[(raw_id, side)] = None
    for side in retired_values(words):
        for r in rows:
            if ledger.holds_value(side, r["text"] or ""):
                marks.setdefault((str(r["id"]), side), None)
    # only a message on the topic of the user's correction: one shared topic stem besides the value
    topic = basket._topic(words)
    texts = {str(r["id"]): r["text"] or "" for r in rows}
    marks = {
        (raw_id, side): None
        for raw_id, side in marks
        if (basket._topic(texts.get(raw_id, "")) & topic) - basket._topic(side)
    }
    if not marks:  # no contrast: the old value is the one of the same slot said earlier
        for raw_id, side in _slot_marks(rows, words, topic):
            marks[(raw_id, side)] = None
    else:  # a contrast that also states a new total or balance retires the old one by slot
        for raw_id in dict.fromkeys(r for r, _side in marks):
            done = [side for r, side in marks if r == raw_id]
            for side in restated_sides(texts.get(raw_id, ""), words, done):
                marks.setdefault((raw_id, side), None)
    for raw_id, side in list(marks):  # a total or balance computed from a retired amount
        for d in derived_amounts(texts.get(raw_id, ""), side):
            marks.setdefault((raw_id, d), None)
    for raw_id, side in marks:
        conn.execute(
            "INSERT OR IGNORE INTO raw_retractions (raw_id, record_id, retracted_value_norm,"
            " created_at) VALUES (?, ?, ?, ?)",
            (raw_id, key, side, when),
        )
    if marks and conn.in_transaction:
        conn.commit()
    return len(marks)


def _status(conn: sqlite3.Connection, rid: str) -> Status | None:
    row = conn.execute("SELECT status FROM records WHERE id = ?", (rid,)).fetchone()
    return Status(row[0]) if row is not None else None


def head(conn: sqlite3.Connection, record_id: str) -> str | None:
    """§6.1: follow `supersedes` edges forward (new → old) to the slot's usable head."""
    seen = {record_id}
    cur = record_id
    for _ in range(64):
        st = _status(conn, cur)
        if st in USABLE:
            return cur
        row = conn.execute(
            "SELECT e.from_id FROM edges e JOIN records r ON r.id = e.from_id"
            " WHERE e.to_id = ? AND e.etype = 'supersedes' ORDER BY r.version DESC LIMIT 1",
            (cur,),
        ).fetchone()
        if row is None or row[0] in seen:
            return None
        cur = row[0]
        seen.add(cur)
    return None


_GONE_LINE = {
    Status.FORGOTTEN: "Это уже забыто.",
    Status.PURGED: "Это уже забыто.",
    Status.RETRACTED: "Это уже отменено.",
}


@dataclass
class _Target:
    id: str | None = None
    ask: str | None = None
    missing: str | None = None
    also: tuple[str, ...] = ()  # more of the user's records holding the retired value


def _slot_target(conn: sqlite3.Connection, a: Assertion) -> _Target | None:
    if not a.relation:
        return None
    subj = _resolve_name(conn, a.subject, create=False)
    if subj.ask:
        return _Target(ask=subj.ask)
    if subj.entity_id is None:
        return None
    rel, _vt = _relation(conn, a.relation)
    if rel == "other":
        return None
    row = conn.execute(
        "SELECT f.id FROM facts f JOIN records r ON r.id = f.id WHERE f.subject_id = ?"
        " AND f.relation = ? AND r.status IN ('active', 'archived', 'superseded')"
        " ORDER BY (r.status IN ('active', 'archived')) DESC, r.version DESC LIMIT 1",
        (subj.entity_id, rel),
    ).fetchone()
    return _Target(id=row[0]) if row is not None else None


def seed_fact(conn: sqlite3.Connection, rid: str) -> str | None:
    """An episode's one seed fact: the episode only carries the value its seed
    holds, so a correction fixes the fact and the episode goes with it (`ledger.carriers`).
    None for any other record, or an episode of several seeds."""
    rows = conn.execute(
        "SELECT e.to_id FROM edges e JOIN records r ON r.id = e.to_id"
        " JOIN records ep ON ep.id = e.from_id AND ep.kind = 'episode'"
        " WHERE e.from_id = ? AND e.etype = 'derived_from' AND r.kind IN ('fact', 'preference')"
        " AND r.status IN ('active', 'archived', 'superseded')"
        " AND e.to_id NOT IN (SELECT target_id FROM episode_links"
        "  WHERE episode_id = e.from_id AND role = 'cause')",
        (rid,),
    ).fetchall()
    return rows[0][0] if len(rows) == 1 else None


# ---------------------------------------------------- a correction that names only the new value
# "Moved to the 21st, same time": no contrast names the old value, so it is read by slot - the
# value of the same type (day, time, amount, place) in the user's earlier message on the same
# topic that differs from the new one. A day without a month replaces the day of a full date; a
# value the correction repeats is never retired.

_SLOT_TIME = re.compile(
    r"(?<![\w:.])(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?\s*(?P<ap>am|pm|a\.m\.|p\.m\.)(?!\w)"
    r"|(?<![\w:.])(?P<h2>\d{1,2}):(?P<m2>\d{2})(?![\d:])"
    r"|(?<!\w)(?P<h3>\d{1,2})\s*час\w*"
    r"|(?<!\w)(?P<word>noon|midnight|полдень|полночь)(?!\w)",
    re.IGNORECASE,
)
_SLOT_DATES = (ledger._ISO, ledger._DAY_MONTH, ledger._MONTH_DAY, ledger._DOTTED)
_UNIT = (
    r"(?:k|тыс\w*|млн|млрд|руб\w*|₽|долл\w*|евро|бакс\w*|thousand|million|billion|dollars?"
    r"|bucks|euros?|pounds?|usd|eur|rub|%)"
)
_SLOT_AMOUNT = re.compile(
    rf"(?P<pre>[$€£₽]\s?)?(?<![\w.,])(?P<n>\d+(?:[.,]\d+)?)(?![\w.,]*\d)(?:\s*(?P<unit>{_UNIT})(?!\w))?",
    re.IGNORECASE,
)
_SLOT_PLACE = re.compile(
    r"(?<!\w)(?P<prep>in|at|to|from|with|в|во|на|из|с|со|у)\s+"
    r"(?P<name>[A-ZА-ЯЁ][\w-]+(?:\s+[A-ZА-ЯЁ][\w-]+){0,2})"
)


@dataclass(frozen=True)
class _Slot:
    kind: str  # day | time | amount | place
    surface: str  # as it stands in the text (case folded for the dated kinds)
    key: tuple[Any, ...]
    at: int


def _blank(text: str, s: int, e: int) -> str:
    return text[:s] + " " * (e - s) + text[e:]


def _slots(text: str) -> list[_Slot]:
    """The typed values in their words, each with a comparable key."""
    out: list[_Slot] = []
    for m in _SLOT_PLACE.finditer(text or ""):
        out.append(
            _Slot("place", m["name"], (m["prep"].casefold(), m["name"].casefold()), m.start())
        )
    low = (text or "").casefold()
    for m in list(_SLOT_TIME.finditer(low)):
        if m["word"]:
            key: tuple[Any, ...] = ({"noon": 12, "полдень": 12}.get(m["word"], 0), 0)
        else:
            h = int(m["h"] or m["h2"] or m["h3"])
            if (m["ap"] or "").startswith("p") and h < 12:
                h += 12
            key = (h, int(m["m"] or m["m2"] or 0))
        out.append(_Slot("time", m.group().strip(), key, m.start()))
        low = _blank(low, *m.span())
    for pat in _SLOT_DATES:
        for m in list(pat.finditer(low)):
            toks = ledger.value_tokens(m.group())
            md = sorted(t for t in toks if t.startswith("date:") and len(t) == 10)
            if not md:
                continue
            mo, dd = (int(x) for x in md[0][5:].split("-"))
            out.append(_Slot("day", m.group().strip(), (mo, dd), m.start()))
            low = _blank(low, *m.span())
    for m in list(ledger._BARE_DAY.finditer(low)):
        out.append(_Slot("day", m.group(), (None, int(m[1])), m.start()))
        low = _blank(low, *m.span())
    for pat, days in ledger._WEEKDAY_WORDS:
        for m in pat.finditer(low):
            out.append(_Slot("day", m.group(), ("weekday", days), m.start()))
    for m in _SLOT_AMOUNT.finditer(low):
        n = float(m["n"].replace(",", "."))
        unit = (m["unit"] or m["pre"] or "").strip().casefold()
        out.append(_Slot("amount", m.group().strip(), (n, unit), m.start()))
    return sorted(out, key=lambda s: s.at)


def _same_slot(old: _Slot, new: _Slot) -> bool:
    if old.kind == "day":
        a, b = old.key, new.key
        if a[0] == "weekday" or b[0] == "weekday":
            return a == b
        if a[0] is None or b[0] is None:  # a day without a month: the day part only
            return a[1] == b[1]
        return a == b
    if old.kind == "amount":
        return old.key[0] == new.key[0]
    if old.kind == "place":
        return old.key[1] == new.key[1]
    return old.key == new.key


def slot_sides(old_text: str, words: str) -> list[str]:
    """The values of `old_text` the correction `words` replaces by slot (see above)."""
    new, old = _slots(words), _slots(old_text)
    said = " ".join(words.casefold().split())
    out: list[str] = []
    for kind in ("day", "time", "amount", "place"):
        nv = [s for s in new if s.kind == kind]
        if not nv:
            continue
        ov = [s for s in old if s.kind == kind]
        if kind == "place":  # a name in the same place slot: the same preposition
            ov = [o for o in ov if any(o.key[0] == n.key[0] for n in nv)]
        if any(_same_slot(o, n) for o in ov for n in nv) and len(nv) == len(
            [o for o in ov if any(_same_slot(o, n) for n in nv)]
        ):
            continue  # every value named is the one said before: a repeat, not a change
        gone = [
            o
            for o in ov
            if not any(_same_slot(o, n) for n in nv) and o.surface.casefold() not in said
        ]
        if kind == "amount" and len(gone) > 1:  # the one in the same unit, else the first
            units = {n.key[1] for n in nv if n.key[1]}
            same = [o for o in gone if o.key[1] in units]
            gone = same[:1] if len(same) == 1 else gone[:1]
        out += [o.surface for o in gone]
    return list(dict.fromkeys(out))


def derived_amounts(text: str, side: str) -> list[str]:
    """Amounts in `text` computed from the retired amount `side`: a later number that is its
    sum or difference with another number said before it («lent 500, got 200 back, so 300
    left»). The old total or balance goes with the old amount."""
    retired = [s for s in _slots(side) if s.kind == "amount"]
    if not retired:
        return []
    amounts = [s for s in _slots(text) if s.kind == "amount"]
    out: list[str] = []
    for a in amounts:
        if not any(a.key[0] == r.key[0] for r in retired):
            continue
        for d in amounts:
            if d.at <= a.at or d.key[0] == a.key[0]:
                continue
            for x in amounts:
                if x is d or x is a or x.at >= d.at:
                    continue
                if d.key[0] in (a.key[0] + x.key[0], a.key[0] - x.key[0], x.key[0] - a.key[0]):
                    out.append(d.surface)
    return list(dict.fromkeys(out))


_CLAUSE_BREAK = re.compile(
    r"[,.;:!?()]|(?<!\w)(?:and|so|but|then|which|и|а|но|так что|поэтому|значит)(?!\w)",
    re.IGNORECASE,
)


def _clause_topic(text: str, at: int, surface: str) -> set[str]:
    """Topic stems of the clause of `text` around position `at`, the value itself aside."""
    from ..recall import basket

    start, end = 0, len(text)
    for m in _CLAUSE_BREAK.finditer(text):
        if m.end() <= at:
            start = m.end()
        elif m.start() > at:
            end = m.start()
            break
    return basket._topic(text[start:end]) - basket._topic(surface)


def restated_sides(old_text: str, words: str, retired: Sequence[str]) -> list[str]:
    """Values of `old_text` a correction carrying a contrast restates by slot: besides the new
    side of the contrast it states another new value of the contrasted type («450, not 400, so
    the fund is at 1550» against «paid 400, so the fund is down to 1600»), and that value
    replaces the old one of its slot even when nothing in their words computes it. The new side
    is the new value nearest the retired one; each other new value takes the old value of its
    type whose clause shares a topic stem with its own clause (or, for an amount, one of the
    same order), the most shared and then the latest first. A value the correction repeats or
    the contrast retires stays."""
    low_w, low_o = (words or "").casefold(), (old_text or "").casefold()
    said = " ".join(low_w.split())
    new, old = _slots(words), _slots(old_text)
    out: list[str] = []
    for side in retired:
        gone = [g for g in _slots(side) if g.kind != "place"]
        pos = low_w.find(side.casefold())
        for kind in dict.fromkeys(g.kind for g in gone):
            mine = [g for g in gone if g.kind == kind]
            ov = [o for o in old if o.kind == kind]
            nv = [
                n
                for n in new
                if n.kind == kind
                and not any(_same_slot(g, n) for g in mine)
                and not any(_same_slot(o, n) for o in ov)
            ]
            if pos >= 0 and nv:  # the contrast's own new side
                nv.remove(min(nv, key=lambda n: abs(n.at - pos)))
            free = [
                o
                for o in ov
                if not any(_same_slot(o, n) for n in new)
                and not any(_same_slot(o, g) for g in mine)
                and o.surface.casefold() not in said
                and o.surface not in out
            ]
            for n in nv:
                near = _clause_topic(low_w, n.at, n.surface)

                def fit(o: _Slot, n: _Slot = n, near: set[str] = near, k: str = kind) -> int:
                    shared = len(near & _clause_topic(low_o, o.at, o.surface))
                    if shared:
                        return shared
                    a, b = o.key[0], n.key[0]
                    return int(k == "amount" and a > 0 and b > 0 and 1 / 3 <= a / b <= 3)

                fits = [o for o in reversed(free) if fit(o)]
                if fits:
                    best = max(fits, key=fit)
                    out.append(best.surface)
                    free.remove(best)
    return list(dict.fromkeys(out))


# words of the change itself, not of its topic: they never pick the message it corrects
_CHANGE_WORDS = (
    "moved", "move", "pushed", "push", "rescheduled", "changed", "change", "make", "made",
    "same", "time", "still", "now", "again", "actually", "update", "instead", "перенесли",
    "перенеси", "перенёс", "сдвинули", "поменяли", "тоже", "время", "теперь", "сделай",
)  # fmt: skip
_CHANGE_STEMS = frozenset(lexical.stem(w) for w in _CHANGE_WORDS)


def _slot_marks(rows: Sequence[Any], words: str, topic: set[str]) -> list[tuple[str, str]]:
    """(raw id, retired value) by slot, in ONE earlier message of the user's: the one sharing most
    topic stems with the correction (the change words and the values aside), the newest on a
    tie, among those holding a different value of a type the correction names."""
    from ..recall import basket

    topic = topic - _CHANGE_STEMS
    best: tuple[int, str, list[str]] | None = None
    for r in rows:  # newest first
        text = r["text"] or ""
        shared = (basket._topic(text) & topic) - _CHANGE_STEMS
        if not shared or (best is not None and len(shared) <= best[0]):
            continue
        sides = slot_sides(text, words)
        shared -= basket._topic(" ".join(sides))
        if sides and shared and (best is None or len(shared) > best[0]):
            best = (len(shared), str(r["id"]), sides)
    return [(best[1], side) for side in best[2]] if best else []


_HELD_ATTRS = ("value_text", "value_norm", "quote_orig", "text_en", "origin_quote")


def retired_values(words: str | None) -> list[str]:
    """The old values the user's correction retires: the ledger's reading of «не X, а Y» /
    "Y, not X" (the same one that orders a late extraction), else this module's contrast."""
    sides = ledger.retired_sides(words)
    if not sides and (c := contrast(words)) is not None:
        sides = [c.old]
    return sides


def _holders(conn: sqlite3.Connection, words: str, rids: Sequence[str]) -> list[str]:
    """Slot heads among `rids` whose own value or words hold a value the user's correction
    retires. Only the user's own records count: a tie never resolves onto someone else's claim."""
    sides = retired_values(words)
    if not sides:
        return []
    out: list[str] = []
    for rid in dict.fromkeys(seed_fact(conn, r) or r for r in rids):
        hd = head(conn, rid) or rid
        rec = ledger.get_record(conn, hd)
        if rec is None or rec.kind is RecordKind.ENTITY or hd in out:
            continue
        if AuthorTier(getattr(rec, "author_tier", AuthorTier.OTHER)) not in ledger.USER_TIERS:
            continue
        text = " ".join(str(v) for a in _HELD_ATTRS if (v := getattr(rec, a, None)))
        if any(ledger.holds_value(s, text) for s in sides):
            out.append(hd)
    return out


def _value_holders(
    conn: sqlite3.Connection, words: str, held: Sequence[str], *, turn_words: str
) -> list[str]:
    """The holders whose own VALUE (not only their quote) is the value the user's «не X, а Y»
    retires - each is a copy of the claim being corrected (one per message that stated it), so
    the fix retires all of them instead of asking which. A record that only mentions X in its
    words (a reason, a decision «из-за X») is a dependant: the cascade reviews it, never this.
    Copies on several subjects count only when their words name each subject (or it is the
    user); else [] and the caller asks as before."""
    sides = retired_values(words)
    if len(held) < 2 or not sides:
        return []
    out: list[tuple[str, str]] = []
    for rid in held:
        rec = ledger.get_record(conn, rid)
        if rec is None or rec.kind not in (RecordKind.FACT, RecordKind.PREFERENCE):
            continue
        value = " ".join(
            str(v) for a in ("value_text", "value_norm") if (v := getattr(rec, a, None))
        )
        if value and any(ledger.holds_value(s, value) for s in sides):
            out.append((rid, str(getattr(rec, "subject_id", "") or "")))
    if len(out) < 2:
        return [rid for rid, _ in out] if len(out) == 1 else []
    subjects = {subj for _, subj in out}
    if len(subjects) > 1:
        ws = policy.words(turn_words)
        named = {
            subj
            for subj in subjects
            if subj == "self" or policy.names_in(ws, policy._entity_names(conn, [subj]))
        }
        if named != subjects:
            return []
    return [rid for rid, _ in out]


def resolve_target(
    conn: sqlite3.Connection,
    target: str,
    because_quote: str,
    assertion: Assertion | None,
    *,
    thread_id: str | None = None,
) -> _Target:
    """§10.4 grammar for `fix`: handles first; then the assertion's slot; then the user's words,
    ranked lexically over usable and superseded records (the eval harness passes the user's
    whole message as both `target` and `because_quote`)."""
    from .. import commands

    target = (target or "").strip()
    handles = list(dict.fromkeys(commands.handles_in(target)))
    if handles:
        if len(handles) > 1:
            return _Target(
                ask="Что именно исправить: " + ", ".join(f"m:{h}" for h in handles) + "?"
            )
        if _status(conn, handles[0]) is None:
            return _Target(missing="Такой записи нет — нечего исправлять.")
        return _Target(id=handles[0])
    if assertion is not None:
        slot = _slot_target(conn, assertion)
        if slot is not None:
            return slot
    phrase = _strip_fix_lead(target or because_quote)
    if not phrase:
        return _Target(missing="Не понял, что исправить.")
    found = commands._find(conn, phrase, include_history=True, min_cover=0.0)
    scored = [(s, rid) for s, rid in found.scored if found.recs[rid].kind is not RecordKind.ENTITY]
    if not scored:  # «нет, в среду»: the words name no record - the user's subject and value kind do
        kin = list(
            dict.fromkeys(
                head(conn, r) or r
                for r in commands.find_by_value_kind(conn, phrase, thread_id=thread_id)
            )
        )
        if len(kin) == 1:
            return _Target(id=kin[0])
        if 1 < len(kin) <= 4:
            return _Target(ask=_which(conn, kin))
        return _Target(missing="Не нашёл в памяти, что исправить.")
    best = scored[0][0]
    held = _holders(conn, phrase, [rid for s, rid in scored if s >= best - 1.0])
    if len(held) == 1:  # the user's «не X, а Y» names the record that holds X
        return _Target(id=held[0])
    every = _value_holders(conn, phrase, held, turn_words=target or because_quote)
    if every:  # several of the user's records hold X as their value: the fix retires each one
        return _Target(id=every[0], also=tuple(every[1:]))
    top = list(dict.fromkeys(seed_fact(conn, rid) or rid for s, rid in scored if s == best))
    heads = list(dict.fromkeys(head(conn, rid) or rid for rid in top))
    if len(heads) > 1:
        return _Target(ask=_which(conn, heads))
    return _Target(id=top[0])


def _which(conn: sqlite3.Connection, heads: Sequence[str]) -> str:
    opts = []
    for rid in heads[:4]:
        rec = ledger.get_record(conn, rid)
        shown = getattr(rec, "quote_orig", None) or getattr(rec, "text_en", "") or rid
        opts.append(f"«{shown[:80]}» [m:{rid}]")
    return "Что именно исправить: " + " или ".join(opts) + "?"


def _new_from(
    conn: sqlite3.Connection,
    base: Any,
    *,
    rid: str,
    correction: str,
    quote: str,
    origin: str,
    tier: AuthorTier,
    a: Assertion | None,
    ts: str,
    source_raw: str | None = None,
) -> dict[str, Any]:
    """The correcting record: the base's slot (subject, relation, fkind, applicability) with
    the new value (§6.1). Non-fact kinds copy the base and take the corrected text. Without
    an `assertion` the value and text are the new side of the user's «не X, а Y», and
    the user's own raw message of the turn is the fact's lineage."""
    data = base.model_dump(mode="json", exclude_none=True)
    for k in (
        "version",
        "status",
        "review",
        "retired_at",
        "created_at",
        "updated_at",
        "importance",
        "confidence",
        "labels_final",
        "author_entity_id",
        "pinned_category",
    ):
        data.pop(k, None)
    data.update(id=rid, author_tier=tier.value, recorded_at=ts, origin_quote=origin)
    data["labels"] = int(base.labels | _labels(correction))
    said = corrected_text(correction)
    if isinstance(base, Fact):
        _rel, vtype = _relation(conn, base.relation)
        turn_over = contrast(correction)
        part = turn_over.new if turn_over is not None else _strip_fix_lead(correction)
        value = (a.value if a and a.value else None) or value_from(part, vtype)
        data.update(
            value_text=value,
            value_norm=normalize_value(value, vtype),
            text_en=((a.text_en if a else None) or said).strip(),
            quote_orig=quote,
            lang_orig=_lang(quote),
        )
        for k in (
            "value_entity_id",
            "speaker_entity_id",
            "source_raw_id",
            "keywords_ru",
            "keywords_en",
            "valid_from",
            "valid_to",
        ):
            data.pop(k, None)
        if a is not None:
            for k in ("keywords_ru", "keywords_en", "valid_from", "valid_to"):
                if getattr(a, k):
                    data[k] = getattr(a, k)
            if a.polarity:
                data["polarity"] = a.polarity.value
            if a.stability:
                data["stability"] = a.stability.value
            if a.value_entity:
                ve = _resolve_name(conn, a.value_entity, create=False)
                if ve.entity_id:
                    data["value_entity_id"] = ve.entity_id
        if source_raw:
            data["source_raw_id"] = source_raw
    elif "text_en" in data:
        data["text_en"] = ((a.text_en if a else None) or said).strip()
    return data


# ------------------------------------------------- speaker correction (e2e-d01/d02)

_MIXUP = re.compile(
    r"(?<!\w)(?:перепутал\w*|спутал\w*|ошибл\w*|ошибся|mixed\s+(?:them\s+)?up|my\s+mistake)",
    re.IGNORECASE,
)
_NAME_RUN = re.compile(r"(?<![\w-])[A-ZА-ЯЁ][\w-]+(?:\s+[A-ZА-ЯЁ][\w-]+){0,2}")


@dataclass
class _Swap:
    old_id: str
    old_name: str
    new_name: str
    claims: list[str] = field(default_factory=list)


def _swap_words(text: str) -> set[str]:
    return ledger._value_words(text)  # noqa: SLF001


def _claims_of(conn: sqlite3.Connection, eid: str, words: set[str]) -> list[str]:
    """The user's mistaken claim: usable records that name the wrong person (subject, speaker,
    value, impression or episode link) and share a content word with the user's correction."""
    usable = "r.status IN ('active', 'archived')"
    rows = conn.execute(
        "SELECT f.id, coalesce(f.text_en, '') || ' ' || coalesce(f.quote_orig, '') || ' '"
        " || coalesce(f.value_text, '') FROM facts f JOIN records r ON r.id = f.id"
        f" WHERE {usable} AND ? IN (f.subject_id, f.speaker_entity_id, f.value_entity_id)"
        " UNION ALL SELECT i.id, coalesce(i.text_en, '') || ' ' || coalesce(i.quote_orig, '')"
        f" FROM impressions i JOIN records r ON r.id = i.id WHERE {usable} AND i.about_entity_id = ?"
        " UNION ALL SELECT ep.id, ep.summary_en FROM episodes ep JOIN records r ON r.id = ep.id"
        f" WHERE {usable} AND ep.id IN (SELECT episode_id FROM episode_links WHERE target_id = ?)",
        (eid, eid, eid),
    ).fetchall()
    return [rid for rid, text in rows if _swap_words(text) & words]


def _speaker_swap(conn: sqlite3.Connection, text: str) -> _Swap | None:
    """«я перепутала: X сказала Y, а не Z» ("I mixed up: X said Y, not Z") - the claim is right,
    the person is wrong. The wrong person is a known entity named on the contrast's old side;
    the right one is the name before it (or on its new side) that shares a first name with it,
    or the nearest full name. None when the user's words do not swap one person for another."""
    from .. import entities

    c = contrast(text)
    if c is None or not _MIXUP.search(text):
        return None
    old_runs = _NAME_RUN.findall(c.old)
    if not old_runs:
        return None
    old_name = old_runs[0]
    hits = [h for h in entities.resolve_rows(conn, old_name, "any") if h.confidence >= 0.6]
    old_ids = list(dict.fromkeys(h.entity_id for h in hits))
    if len(old_ids) != 1 or old_ids[0] == OWNER_ENTITY_ID:
        return None
    first = old_name.split()[0].casefold()
    runs = [
        r for r in _NAME_RUN.findall(c.new) + _NAME_RUN.findall(text[: c.start]) if r != old_name
    ]
    pick = [r for r in runs if len(r.split()) > 1 and r.split()[0].casefold() == first]
    pick = pick or [r for r in runs[::-1] if len(r.split()) > 1]
    if not pick:
        return None
    new_name = pick[0]
    names = {w.casefold() for w in [*old_name.split(), *new_name.split()]}
    words = _swap_words(" ".join(w for w in _words(text) if w.casefold() not in names))
    claims = _claims_of(conn, old_ids[0], words)
    if not claims:
        return None
    return _Swap(old_ids[0], old_name, new_name, claims)


def _rename_in(text: Any, old_name: str, new_name: str) -> Any:
    """The wrong name's forms (Cyrillic stems and transliterations) → the right name."""
    if not isinstance(text, str) or not text:
        return text
    from .. import entities

    olds, news = old_name.split(), new_name.split()
    pairs = [(old_name, new_name)] + (
        list(zip(olds, news, strict=True)) if len(olds) == len(news) else []
    )
    out = text
    for o, n in pairs:
        latin_o, latin_n = entities.transliterations(o), entities.transliterations(n)
        for src, dst in [(o, n), *((lo, latin_n[0]) for lo in latin_o if latin_n)]:
            stem = re.escape(src[:-1] if len(src) > 4 else src)
            out = re.sub(rf"(?<!\w){stem}\w*", dst, out, flags=re.IGNORECASE)
    return out


_FRESH_DROP = ("version", "status", "review", "retired_at", "created_at", "updated_at",
               "importance", "confidence", "labels_final", "author_entity_id", "pinned_category")  # fmt: skip


def _reattribute(
    conn: sqlite3.Connection, swap: _Swap, turn: TurnCtx, *, because_quote: str
) -> Written:
    """The user's speaker correction: each fact of the claim is superseded by the same fact told
    by the right person (its episodes go with it, `ledger.carriers`), impressions and episodes
    of the claim stop being current, and a person minted only from the claim is retracted."""
    subj = _resolve_name(conn, swap.new_name, create=True)
    if subj.ask:
        return Written("ask_first", line=subj.ask)
    ts = turn.clock or _now()
    tier = _tier(turn, True)
    origin = _origin_quote(because_quote, turn, because_quote)
    new_id = subj.entity_id or ledger.new_id()
    ops: list[ProposalOp] = []
    if subj.entity_id is None:
        ent = {"kind": "entity", "id": new_id, "author_tier": tier.value, "recorded_at": ts,
               "etype": "person", "display_name": swap.new_name, "origin_quote": origin}  # fmt: skip
        ops.append(ProposalOp(op=OpKind.ADD, group=new_id,
                              payload={"record": ent, "aliases": [{"alias": swap.new_name}]}))  # fmt: skip
    moved: list[str] = []
    for rid in swap.claims:
        base = ledger.get_record(conn, rid)
        if base is None or base.kind not in (RecordKind.FACT, RecordKind.PREFERENCE):
            continue
        data = base.model_dump(mode="json", exclude_none=True)
        for k in _FRESH_DROP:
            data.pop(k, None)
        nid = ledger.new_id()
        data.update(id=nid, author_tier=tier.value, recorded_at=ts, origin_quote=origin)
        for k in ("subject_id", "speaker_entity_id", "value_entity_id"):
            if data.get(k) == swap.old_id:
                data[k] = new_id
        for k in ("text_en", "quote_orig", "value_text", "keywords_ru", "keywords_en"):
            data[k] = _rename_in(data.get(k), swap.old_name, swap.new_name)
        ops.append(ProposalOp(op=OpKind.SUPERSEDE, target=rid, group=new_id,
                              payload={"record": data, "fix": True}))  # fmt: skip
        moved.append(nid)
    rest = [r for r in swap.claims if ledger.get_record(conn, r) is not None
            and ledger.get_record(conn, r).kind in (RecordKind.IMPRESSION, RecordKind.EPISODE)]  # fmt: skip
    box: dict[str, Any] = {}

    def apply(ctx: store.CommitCtx, op: ProposalOp) -> list[str]:
        touched = list(ledger.apply_op(ctx, op))
        if op is ops[-1]:
            for rid in rest:
                if _status(ctx.conn, rid) in USABLE:
                    ledger.set_status(ctx, rid, Status.SUPERSEDED)
                    touched.append(rid)
            if _minted_by(ctx.conn, swap.old_id, swap.claims):
                ledger.set_status(ctx, swap.old_id, Status.RETRACTED)
                touched.append(swap.old_id)
            for nid in moved:
                _serve_write(ctx, turn, nid, "fact")
            box["v"] = ctx.version
        return touched

    if not ops:
        return Written("rejected", line="Не нашёл, что переписать на другого человека.")
    proposal = Proposal(ops=ops, author_tier=tier)
    try:
        res = store.commit(conn, proposal, apply, event_kind=EventKind.FIX, priority="hot")
    except store.OpRejected as exc:
        return Written("rejected", line=_reject_line(exc))
    if not res.ok:
        return Written("rejected", line="Не смог исправить — попробуй ещё раз.")
    touched = [*moved, *swap.claims, swap.old_id]
    _hotlog(conn, proposal, box.get("v"), touched)
    _core_changed(conn, touched)
    return Written("committed", record_ids=moved, memory_version=box.get("v"),
                   line=f"Исправил: это от {swap.new_name}, не от {swap.old_name}.")  # fmt: skip


def _minted_by(conn: sqlite3.Connection, eid: str, claims: Sequence[str]) -> bool:
    """The wrong person exists only because of the mistaken claim: no stable ids, no usable
    record left that names it, and every message it came from is one the claim came from."""
    row = conn.execute("SELECT stable_ids FROM entities WHERE id = ?", (eid,)).fetchone()
    if row is None or (row[0] or "[]") != "[]":
        return False
    left = conn.execute(
        "SELECT 1 FROM facts f JOIN records r ON r.id = f.id WHERE r.status IN ('active',"
        " 'archived') AND ? IN (f.subject_id, f.speaker_entity_id, f.value_entity_id)"
        " UNION ALL SELECT 1 FROM impressions i JOIN records r ON r.id = i.id"
        " WHERE r.status IN ('active', 'archived') AND i.about_entity_id = ? LIMIT 1",
        (eid, eid),
    ).fetchone()
    if left is not None:
        return False
    srcs = ledger._source_raws(conn, eid)  # noqa: SLF001
    claimed = set().union(*(ledger._source_raws(conn, c) for c in claims)) if claims else set()  # noqa: SLF001
    return bool(srcs) and srcs <= claimed


def _dependents(conn: sqlite3.Connection, rid: str) -> int:
    return _dependents_of(conn, [rid])


def _merged_copies(conn: sqlite3.Connection, rid: str) -> list[str]:
    """The duplicates already merged into `rid` (a `supersedes` edge onto the same subject,
    relation, value and polarity): what rested on them rests on the user's one claim, so a
    correction of it reaches them too. An earlier value `rid` replaced is not a copy."""
    return [
        r[0]
        for r in conn.execute(
            "SELECT e.to_id FROM edges e JOIN facts a ON a.id = e.from_id"
            " JOIN facts b ON b.id = e.to_id WHERE e.from_id = ? AND e.etype = 'supersedes'"
            " AND a.subject_id = b.subject_id AND a.relation = b.relation"
            " AND a.value_norm IS b.value_norm AND a.polarity = b.polarity",
            (rid,),
        )
    ]


def _dependents_of(conn: sqlite3.Connection, rids: Sequence[str]) -> int:
    """How many records rest on any of these (one count when one rests on several)."""
    ids = list(dict.fromkeys(rids))
    if not ids:
        return 0
    marks = ", ".join("?" for _ in ids)
    row = conn.execute(
        "SELECT count(DISTINCT target_record_id) FROM justifications"  # noqa: S608
        f" WHERE member_record_id IN ({marks}) AND target_record_id NOT IN ({marks})",
        ids * 2,
    ).fetchone()
    return int(row[0]) if row else 0


def fix(
    conn: sqlite3.Connection,
    target: str,
    correction: str,
    turn: TurnCtx,
    *,
    because_quote: str,
    assertion: Assertion | None = None,
    retract: bool = False,
) -> Written:
    """§6.1 «исправь» / «это неправда»: retarget to the slot head inside the transaction,
    SUPERSEDE (fix=true) or RETRACT; ledger invalidation and pending updates; dependents
    via cascade when built; core hook; hot log.

    Tainted turn or a turn not from the user (the user-only rule 4, as `commands.forget_now`): no
    correction word in the user's own words → `ask_first`, before the target is even resolved.

    A cancel sent as a correction («отмени ужин в пятницу», "call off Friday's dinner") with
    no new value cancels the plan it names (`commands.forget_now`, its own tainted rule);
    not found there, it is corrected as usual."""
    cancelled = _cancel(conn, (correction or "").strip() or because_quote, turn, because_quote)
    if cancelled is not None:
        return cancelled
    if _needs_his_word(turn) and not his_correction(policy.turn_words(turn)):
        return Written(
            "ask_first",
            line="Исправлять ты не просил — это из чужого текста. Исправить — скажешь сам?",
        )
    if not _needs_his_word(turn):
        swap = _speaker_swap(conn, (correction or "").strip() or because_quote)
        if swap is not None:  # «я перепутала, это сказала Лера Пак»: the person, not the value
            return _reattribute(conn, swap, turn, because_quote=because_quote)
    t = resolve_target(conn, target, because_quote, assertion, thread_id=turn.thread_id)
    if t.missing:
        return Written("rejected", line=t.missing)
    if t.ask:
        return Written("ask_first", line=t.ask)
    assert t.id is not None
    if t.also:  # every copy of the retired value, each by its own handle
        said, deps = correction, _dependents_of(conn, [t.id, *t.also])
        first = fix(
            conn, f"m:{t.id}", said, turn,
            because_quote=because_quote, assertion=assertion, retract=retract,
        )  # fmt: skip
        if first.outcome != "committed":
            return first
        ids, version = list(first.record_ids), first.memory_version
        for aid in t.also:
            w = fix(conn, f"m:{aid}", said, turn, because_quote=because_quote, retract=retract)
            if w.outcome == "committed":
                ids += [i for i in w.record_ids if i not in ids]
                version = w.memory_version or version
        return replace(first, record_ids=ids, dependents_to_review=deps, memory_version=version)
    t.id = seed_fact(conn, t.id) or t.id  # an episode's value lives in its seed fact
    st = _status(conn, t.id)
    if st in _GONE_LINE:
        return Written("rejected", line=_GONE_LINE[st])
    rec = ledger.get_record(conn, t.id)
    if rec is None or rec.kind is RecordKind.ENTITY:
        return Written("rejected", line="Это не запись, которую можно исправить.")
    named = True
    if _needs_his_word(turn):
        from .. import commands

        hd = head(conn, t.id)
        recs = [r for r in (rec, ledger.get_record(conn, hd) if hd else None) if r is not None]
        named = any(commands._named_by_him(conn, [r], turn, because_quote) for r in recs)
        if not named:
            return Written(
                "ask_first", line="Это пришло не из твоих слов. Что именно исправить — скажешь сам?"
            )
    tier = _tier(turn, named)
    correction = scrub((correction or "").strip() or because_quote.strip()) or ""
    assertion = _scrub_assertion(assertion)
    origin = _origin_quote(because_quote, turn, correction)
    quote = origin
    source_raw = _turn_raw(conn, turn, quote) if quote_is_his(quote, turn) else None
    rid = ledger.new_id()
    ts = turn.clock or _now()
    group = getattr(rec, "subject_id", None) or t.id
    placeholder = ProposalOp(op=OpKind.SUPERSEDE, target=t.id, group=group)
    box: dict[str, Any] = {}

    def apply(ctx: store.CommitCtx, _op: ProposalOp) -> list[str]:
        now_st = _status(ctx.conn, t.id)
        if now_st in _GONE_LINE:
            raise _Refused(_GONE_LINE[now_st])
        hd = head(ctx.conn, t.id)
        if retract:
            if hd is None:
                raise _Refused("Это уже отменено.")
            op = ProposalOp(op=OpKind.RETRACT, target=hd, group=group)
        else:
            base = ledger.get_record(ctx.conn, hd or t.id)
            data = _new_from(
                ctx.conn, base, rid=rid, correction=correction, quote=quote, origin=origin,
                tier=tier, a=assertion, ts=ts, source_raw=source_raw,
            )  # fmt: skip
            if hd is None:  # the chain ends in no usable record: a new fact of that slot
                op = ProposalOp(
                    op=OpKind.ADD, group=group, payload={"record": data, "rule_match": False}
                )
            else:
                op = ProposalOp(
                    op=OpKind.SUPERSEDE, target=hd, group=group,
                    payload={"record": data, "fix": True},
                )  # fmt: skip
        copies = (
            [*ledger.carriers(ctx.conn, hd, exclude=(rid,)), *_merged_copies(ctx.conn, hd)]
            if hd and not retract
            else []
        )
        touched = list(ledger.apply_op(ctx, op))
        version = ctx.version
        if not retract and hd is not None:
            _close_old(ctx, hd, ts)
        if not retract:
            version = _after_retired(ctx, rid)
            _serve_write(ctx, turn, rid, str(data.get("kind", "fact")), version)
        elif hd is not None:
            _serve_write(ctx, turn, hd, rec.kind.value)
        old = hd or t.id
        # the copies the ledger retired with it (`ledger.carriers`) carry dependants too
        gone = [old, *(c for c in copies if _status(ctx.conn, c) is Status.SUPERSEDED)]
        box.update(op=op, v=version, old=old, touched=touched, deps=_dependents_of(ctx.conn, gone))
        _cascade(ctx, gone)
        _mask_written_raws(ctx.conn, old, turn, ts)
        return touched

    try:
        res = store.commit(
            conn, Proposal(ops=[placeholder], author_tier=tier), apply,
            event_kind=EventKind.FIX, priority="hot",
        )  # fmt: skip
    except store.OpRejected as exc:
        return Written("rejected", line=_reject_line(exc))
    if not res.ok or "op" not in box:
        return Written("rejected", line="Не смог исправить — попробуй ещё раз.")
    final = Proposal(ops=[box["op"]], author_tier=tier)
    touched = list(dict.fromkeys(box["touched"]))
    _hotlog(conn, final, box["v"], touched)
    _core_changed(conn, touched)
    if retract:
        ids, line = [box["old"]], "Понял, это неправда — больше не использую."
    else:
        new = ledger.get_record(conn, rid)
        shown = getattr(new, "value_text", None) or correction
        ids, line = [rid], f"Исправил: {shown}."
        _classify_later(conn, rid, correction)
    return Written(
        "committed",
        record_ids=ids,
        memory_version=box["v"],
        dependents_to_review=box["deps"],
        line=line,
    )


def _cancel(
    conn: sqlite3.Connection, said: str, turn: TurnCtx, because_quote: str
) -> Written | None:
    """A cancel order with no new value (no «не X, а Y», no «перенеси на Y»): the plan it
    names is cancelled the way «отмени» is (`commands.forget_now`, plan targets). None when
    the words are not a plain cancel or name no plan: the correction path takes them.

    A cancel said without the verb («ужин в пятницу отменяется», «встречу уже не надо»,
    "never mind Friday's dinner") counts on the user's own untainted turn when it brings no
    new value (one that does is a move or a fix, handled by the correction path)."""
    from .. import commands

    plain = safe_text.order_kind(said) == "cancel"
    spoken = (
        not plain
        and commands.said_cancel(said)
        and not commands._his_word_needed(turn)  # noqa: SLF001
        and not commands.brings_new_value(said)
    )
    if not (plain or spoken) or contrast(said) or shifted(said):
        return None
    res = commands.forget_now(conn, said, turn=turn, because_quote=because_quote)
    if res.outcome == "not_found":
        return None
    if res.outcome == "ask_first":
        return Written("ask_first", line=res.line)
    return Written(
        "committed",
        record_ids=list(res.record_ids),
        memory_version=store.memory_version(conn),
        line=res.line,
    )


def _close_old(ctx: store.CommitCtx, old_id: str, ts: str) -> None:
    """A committed fix retires the value it corrects: besides `superseded` and the
    `supersedes` edge, the old fact stops being valid at the fix (dated history, never a
    current value); an end it already had before the fix stays."""
    st = _status(ctx.conn, old_id)
    if st is not Status.SUPERSEDED:
        return  # a model match parked for review, not retired
    day = (ts or _now())[:10]
    ctx.conn.execute(
        "UPDATE facts SET valid_to = ? WHERE id = ? AND (valid_to IS NULL OR valid_to > ?)",
        (day, old_id, day),
    )


def _after_retired(ctx: store.CommitCtx, rid: str) -> int:
    """The correcting record takes the next version after the one its commit retired the old
    head at, in the same transaction: the slot's newest value is the one with the highest
    version even when two fixes race (§6.1, §8.1 «the latest wins»; versions only grow)."""
    v2 = ctx.conn.execute(
        "UPDATE meta SET memory_version = memory_version + 1, updated_at = ? WHERE id = 1"
        " RETURNING memory_version",
        (ctx.ts,),
    ).fetchone()[0]
    store.reserve_version(ctx.conn, v2)
    ctx.conn.execute("UPDATE records SET version = ? WHERE id = ?", (v2, rid))
    return int(v2)


def _cascade(ctx: store.CommitCtx, changed: list[str]) -> None:
    """§8.4 outcomes for dependents: `justify.cascade` once it lands."""
    from .. import justify

    try:
        justify.cascade(ctx, changed)
    except NotImplementedError:
        return


# ---------------------------------------------------------------- safety net (§6.1)

_CORRECTION = re.compile(
    r"(?:^|[\s,.!?])(?:нет[,!.]?\s+(?:\S+\s+){2,}\S+|это\s+не\s+(?:так|тот|та|то|правда)"
    r"|неправда|не\s+тот\s|не\s+та\s|я\s+же\s+(?:говорил|сказал|писал)\w*|исправь|поправь"
    r"|ты\s+перепутал\w*|ошиб\w+|на\s+самом\s+деле|actually\b|that'?s\s+(?:wrong|not\s+(?:true|right))"
    r"|not\s+true|i\s+(?:already\s+)?(?:said|told\s+you))",
    re.IGNORECASE,
)
# In a tainted turn a fix is the user's order only when the user's own words correct something
# («исправь», «это неправда», «нет, …», «не 12, а 13»); a question about a paste orders nothing.
_FIX_WORDS = re.compile(
    r"(?<!\w)(?:исправ\w*|поправ\w*|неправд\w*|не\s+правда|неверн\w*|не\s+верн\w*"
    r"|неправильн\w*|не\s+правильн\w*|не\s+так|ошиб\w+|перепута\w*|на\s+самом\s+деле"
    r"|вообще-?\s?то|измени\w*|поменя\w*|обнови\w*|замени\w*|перепиши|уточни|ложь"
    r"|я\s+же\s+(?:говорил|сказал|писал)\w*|нет[,.!—–-]|не\s+[\w.-]+\s*,\s*а\s"
    r"|fix\w*|correct\w*|wrong|incorrect|mistake\w*|not\s+(?:true|right)|actually|update\w*"
    r"|change\w*|i\s+(?:already\s+)?(?:said|told\s+you)|no[,.!])",
    re.IGNORECASE,
)


# Retraction at write: a statement whose own words take back an earlier claim
# («беру назад», «забудь, что я говорила про…», "takes back", "retracts"), in the user's
# words or in the extractor's English line for them. Such a record is a retraction, not a
# new claim.
_RETRACTS = re.compile(
    r"(?<!\w)(?:беру\s+(?:свои\s+)?(?:слова\s+)?(?:назад|обратно)|забираю\s+(?:свои\s+)?слова"
    r"|(?:я\s+)?был[аи]?\s+неправ\w*|зря\s+я\s+(?:так\s+)?(?:говорил|сказал|писал|наговорил)\w*"
    r"|(?:забудь|не\s+помни|выкинь\s+из\s+головы),?\s+(?:что|то,\s+что)\s+я\s+(?:говорил|сказал|писал|думал)\w*"
    r"|это\s+было\s+(?:нечестно|несправедливо)|отзываю\s+(?:свои\s+)?слова"
    r"|takes?\s+(?:it\s+|that\s+|(?:his|her|their|my)\s+words\s+)?back|took\s+(?:it\s+)?back|taken\s+back"
    r"|retract(?:s|ed|ing)?\b|withdr[ae]ws?\s+(?:what|(?:his|her|their|my)\s+(?:words|remark|claim))"
    r"|(?:was|were)\s+wrong\s+about|no\s+longer\s+(?:thinks|believes|stands\s+by)"
    r"|(?:asks?|asked)\s+(?:the\s+cube\s+|me\s+)?to\s+forget\s+(?:what|that|his|her|the))",
    re.IGNORECASE,
)


def retracts(*texts: str | None) -> bool:
    """True when one of these texts takes back an earlier claim (retraction at write)."""
    return any(t and _RETRACTS.search(t) for t in texts)


def his_correction(words: str) -> bool:
    """The user's own words carry a correction (the fix analogue of `commands._FORGET_VERB`): a
    fix word, a correction phrase, or a «не X, а Y» / "Y, not X" contrast of any length."""
    return bool(words) and bool(
        _FIX_WORDS.search(words) or _CORRECTION.search(words) or contrast(words) is not None
    )


_FORGET = re.compile(
    r"(?:^|[\s,.!?])(?:забудь|забыть|не\s+запоминай|удали\s+из\s+памяти|сотри\s+из\s+памяти"
    r"|forget\b|don'?t\s+remember)",
    re.IGNORECASE,
)
_REMEMBER = re.compile(
    r"(?:^|[\s,.!?])(?:запомни|имей\s+в\s+виду|remember\s+(?:that|this)|note\s+that)",
    re.IGNORECASE,
)
_CONSTRAINT = re.compile(
    r"(?:^|[\s,.!?])(?:(?:никогда\s+)?не\s+(?:пиши|звони|присылай|напоминай|буди|беспокой)\w*"
    r"|don'?t\s+(?:text|message|call|email|remind|wake))",
    re.IGNORECASE,
)
_PROMISE = re.compile(
    r"(?:^|[\s,.!?])(?:напомню|сделаю|проверю|пришлю|отправлю|напишу\s+(?:тебе|ему|ей)|вернусь\s+к"
    r"|дам\s+знать|сообщу|i'?ll\s+(?:remind|check|send|get\s+back|let\s+you\s+know|follow\s+up)"
    r"|i\s+will\s+(?:remind|check|send|follow\s+up))",
    re.IGNORECASE,
)
# what the net lands as the Cube's promise (wider than the freeze cue above: a promise it
# can land never needs the freeze; one it cannot keeps the old behaviour)
_PROMISE_LAND = re.compile(
    rf"{_PROMISE.pattern}|(?:^|[\s,.!?:])(?:спрошу|пну|растормошу|уточню|подготовлю|соберу"
    r"|позвоню|разбужу|подскажу|i'?ll\s+(?:ask|ping|nudge|text|prepare|make\s+sure)"
    r"|i\s+will\s+(?:ask|ping|nudge))(?!\w)",
    re.IGNORECASE,
)
_CLAIM = re.compile(
    r"(?:^|[\s,.!?«\"])(?:запомнил[аи]?|запомню|исправил[аи]?|поправил[аи]?|забыл[аи]?(?!\s+(?:про|о)\s)"
    r"|удалил[аи]?\s+из\s+памяти|перестал\w*\s+использовать|отменил[аи]?|noted|remembered"
    r"|i'?ll\s+remember|updated\s+(?:my\s+)?memory|forgot(?:ten)?\s+(?:it|that))(?!\w)",
    re.IGNORECASE,
)


# «не забудь», "don't forget": the opposite of a forget order (a reminder), never its cue;
# so is a forget the user only fears: "nudge me if I forget", "in case I forget", «если вдруг забыть»
_NOT_FORGET = re.compile(
    r"(?<!\w)(?:не|never|not|don'?t|do\s+not)\s+(?:за|по)?(?:забудь|забыть|забывай|forget)\w*"
    r"|(?<!\w)(?:if|in\s+case|when|before|so|lest)\s+(?:i|you|we)\s+(?:ever\s+|might\s+|do\s+)?"
    r"forget\w*|(?<!\w)(?:если|вдруг|чтобы)\s+(?:я\s+|ты\s+)?(?:вдруг\s+)?(?:забыть|забуду)",
    re.IGNORECASE,
)


# where a forget cue's clause starts (`_cue_text`): the plain forms and every colloquial one
_FORGET_AT = re.compile(f"{_FORGET.pattern}|{safe_text.ORDER.pattern}", re.IGNORECASE)


def _forget_cue(own: str) -> bool:
    """The user's words order a forget or a cancel in any form (`safe_text.order_kind`: «выкинь из
    головы», "scratch that", «отмени встречу»), or the plain forms `_FORGET` knows."""
    return safe_text.order_kind(own) is not None or bool(_FORGET.search(_NOT_FORGET.sub(" ", own)))


def detect(turn: TurnCtx, reply_text: str) -> list[str]:
    """§6.1 safety net detector (code): cues in the user's words (correction, forget, remember,
    constraint) and promises in the Cube's reply."""
    cues: list[str] = []
    own = policy.turn_words(turn) if turn.origin is TurnOrigin.OWNER else ""
    for name, pat in (
        ("correction", _CORRECTION),
        ("forget", None),
        ("remember", _REMEMBER),
        ("constraint", _CONSTRAINT),
    ):
        if own and (_forget_cue(own) if pat is None else pat.search(own)):
            cues.append(name)
    if reply_text and _PROMISE.search(reply_text):
        cues.append("promise")
    return cues


_MARK_KINDS = ("forget", "undo", "mode")  # commits that are not record writes
# What backs each cue / claim (§6.1 "no matching commit"). Forget, undo and mode commits
# are stamped with their turn by the commands (note_commit); nothing else counts.
_CUE_NEEDS: dict[str, set[str]] = {
    "correction": {"fix", "forget"},
    "forget": {"forget", "mode"},
    "remember": {"write", "fix"},
    "constraint": {"write", "mode"},
    "promise": {"write"},
}


def note_commit(
    conn: sqlite3.Connection, turn: TurnCtx, kind: str, item_ids: Sequence[str] = ()
) -> None:
    """Stamp a forget / undo / mode commit with its turn: the safety net and the Stop hook match
    marks of this turn and thread only. Callers: `commands.forget_now`, `commands.undo` and
    `api.set_chat_mode`."""
    if kind not in _MARK_KINDS:
        raise ValueError(f"note_commit: {kind}")
    ts = store.now()
    mv = conn.execute("SELECT memory_version FROM meta WHERE id = 1").fetchone()
    rows = [
        (turn.thread_id, turn.session_id, i, kind, int(mv[0]) if mv else 0, turn.turn_id, ts)
        for i in (list(item_ids) or [turn.turn_id])
    ]
    with store.write_tx(conn):
        conn.executemany(
            "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id, via,"
            f" created_at) VALUES (?, ?, ?, ?, ?, ?, '{ServedVia.WRITE.value}', ?)",
            rows,
        )


def turn_commits(conn: sqlite3.Connection, turn: TurnCtx) -> set[str]:
    """The kinds of commit this turn made in this thread: `write`, `fix`, `forget`, `undo`,
    `mode` (served marks of this turn and thread)."""
    kinds: set[str] = set()
    for item_kind, created in conn.execute(
        "SELECT item_kind, created_at FROM served"
        " WHERE turn_id = ? AND thread_id = ? AND via = 'write'",
        (turn.turn_id, turn.thread_id),
    ):
        if item_kind in _MARK_KINDS:
            kinds.add(item_kind)
            continue
        ev = conn.execute("SELECT kind FROM events WHERE ts = ? LIMIT 1", (created,)).fetchone()
        kinds.add(ev[0] if ev is not None and ev[0] in ("write", "fix") else "write")
    return kinds


def committed_in_turn(conn: sqlite3.Connection, turn: TurnCtx) -> bool:
    """A hot commit of this turn and thread (a write, fix or stamped forget / undo / mode)."""
    return bool(turn_commits(conn, turn))


def unbacked_cues(conn: sqlite3.Connection, turn: TurnCtx, cues: Sequence[str]) -> list[str]:
    """Each cue needs its own kind of commit in this turn; the rest are left over."""
    if not cues:
        return []
    have = turn_commits(conn, turn)
    return [c for c in cues if not (_CUE_NEEDS.get(c, set()) & have)]


def _claim_needs(claim: str) -> set[str]:
    c = claim.casefold()
    if c.startswith(("исправ", "поправ", "отмен")):
        return {"fix", "forget", "undo"}
    if c.startswith(("забыл", "удалил", "перестал", "forgot")):
        return {"forget", "mode"}
    return {"write", "fix"}


def claims_unbacked_change(conn: sqlite3.Connection, turn: TurnCtx, reply_text: str) -> bool:
    """§6.1 step 4: a user-turn reply claims a memory change with no matching commit of
    this turn («запомнил» needs a write, «исправил» a fix, «забыл» a forget)."""
    if turn.origin is not TurnOrigin.OWNER or not reply_text or not _CLAIM.search(reply_text):
        return False
    have = turn_commits(conn, turn)
    claims = [m.group(0).lstrip(' ,.!?«"') for m in _CLAIM.finditer(reply_text)]
    return any(not (_claim_needs(c) & have) for c in claims)


def _cue_text(words: str, cues: Sequence[str]) -> str:
    """The clauses of `words` that belong to `cues` (only their entities get frozen)."""
    pats = {
        "correction": _CORRECTION,
        "forget": _FORGET_AT,
        "remember": _REMEMBER,
        "constraint": _CONSTRAINT,
    }
    if any(c not in pats for c in cues):
        return words
    marks = sorted((m.start(), name) for name, p in pats.items() for m in p.finditer(words))
    if not marks:
        return words
    parts = []
    for k, (start, name) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(words)
        if name in cues:
            parts.append(words[(0 if k == 0 else start) : end])
    return " ".join(parts)


def _cue_entities(conn: sqlite3.Connection, words: str) -> list[str]:
    from .. import commands

    try:
        named, _rest = commands._named(conn, words)
    except sqlite3.Error:
        return []
    return list(dict.fromkeys(e.entity_id for e in named))


def freeze(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str],
    *,
    seconds: float = FREEZE_S,
    turn: TurnCtx | None = None,
    cues: Sequence[str] = (),
) -> str:
    """Freeze cue entities (≤ `seconds`) and queue the urgent job that lifts it (§6.1 1-2).
    One job per turn; content-free payload (ids, cues, labels), the worker reads the turn."""
    now = _clockmod.now()
    until = (now + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ts = store.now()
    key = f"{SAFETY_JOB}:{turn.turn_id}" if turn is not None else f"{SAFETY_JOB}:{ledger.new_id()}"
    payload = {
        "entities": sorted(set(entity_ids)),
        "frozen_until": until,
        "cues": list(cues),
        "turn_id": turn.turn_id if turn else None,
        "thread_id": turn.thread_id if turn else None,
        "message_id": turn.message_id if turn else None,
        "labels": int(_labels(turn.owner_words)) if turn else 0,
        "urgent": True,
    }
    jid = ledger.new_id()
    with store.write_tx(conn):
        conn.execute(
            "INSERT OR IGNORE INTO jobs (id, jkind, payload, state, attempts, not_before,"
            " idempotency_key, created_at, updated_at) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?, ?)",
            (jid, SAFETY_JOB, json.dumps(payload), ts, key, ts, ts),
        )
    row = conn.execute("SELECT id FROM jobs WHERE idempotency_key = ?", (key,)).fetchone()
    return row[0]


def _freeze_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    now = _clockmod.now()
    out = []
    for (raw,) in conn.execute(
        "SELECT payload FROM jobs WHERE jkind = ? AND state IN ('queued', 'running')",
        (SAFETY_JOB,),
    ):
        p = json.loads(raw)
        if policy._ts(p.get("frozen_until")) > now:
            out.append(p)
    return out


def frozen_entities(conn: sqlite3.Connection) -> set[str]:
    return {e for p in _freeze_rows(conn) for e in p.get("entities", [])}


def is_frozen(conn: sqlite3.Connection, entity_id: str) -> bool:
    return entity_id in frozen_entities(conn)


def frozen_records(conn: sqlite3.Connection) -> set[str]:
    """Records about frozen entities: recall drops them, hands are denied on them (§6.1)."""
    ents = sorted(frozen_entities(conn))
    if not ents:
        return set()
    qs = ", ".join("?" for _ in ents)
    out = set(ents)
    for sql in (
        f"SELECT id FROM facts WHERE subject_id IN ({qs}) OR value_entity_id IN ({qs})",  # noqa: S608
        f"SELECT id FROM impressions WHERE about_entity_id IN ({qs})",  # noqa: S608
    ):
        args = ents * sql.count("IN (")
        out |= {r[0] for r in conn.execute(sql, args)}
    return out


# What the net lands itself, in the turn (probes and the next answer come before any
# worker). «запомни» as a whole word with its content after it; a promise is the reply's
# sentence that holds it, kept only when it says about what (not «напомню завтра»).
_REMEMBER_AT = re.compile(
    r"(?<!\w)(?:запомни|имей\s+в\s+виду|remember\s+(?:that|this)|note\s+that)(?!\w)"
    r"(?:\s*[:,—–-])?\s*",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_FILLER = re.compile(
    r"(?:хорошо|ладно|окей|ок|ага|договорились|конечно|обязательно|тогда|тебе|тебя|потом|позже"
    r"|скоро|завтра|сегодня|послезавтра|вечером|утром|днём|днем|ночью|через|минут\w*|час\w*"
    r"|понедельник\w*|вторник\w*|сред\w*|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*"
    r"|sure|okay|ok|will|do|you|your|it|that|this|then|later|soon|tomorrow|tonight|today"
    r"|morning|evening|night|afternoon|minutes?|hours?|monday|tuesday|wednesday|thursday"
    r"|friday|saturday|sunday|got|and|the|about|again)",
    re.IGNORECASE,
)


def _promise_clause(reply_text: str) -> str | None:
    """The reply's sentence holding the Cube's promise, when it says what it is about."""
    for m in _SENTENCE.finditer(reply_text or ""):
        sent = m.group(0).strip()
        hit = _PROMISE_LAND.search(sent)
        if hit is None:
            continue
        rest = (sent[: hit.start()] + " " + sent[hit.end() :]).casefold()
        if any(len(w) > 2 and not _FILLER.fullmatch(w) for w in re.findall(r"[^\W\d_]+", rest)):
            return sent.split(":", 1)[1].strip() if ":" in sent[:30] else sent
    return None


_WEEKDAYS = (
    ("понедельник", "monday"), ("вторник", "tuesday"), ("сред[аеуы](?!\\w)", "wednesday"),
    ("четверг", "thursday"), ("пятниц", "friday"), ("суббот", "saturday"),
    ("воскресень", "sunday"),
)  # fmt: skip
_NEXT_DAY = re.compile(r"(?<!\w)(?:завтра|tomorrow)(?!\w)", re.IGNORECASE)
_THIS_DAY = re.compile(r"(?<!\w)(?:сегодня|today|tonight)(?!\w)", re.IGNORECASE)


def _due_of(text: str, clock: str) -> str | None:
    """The day a promise names (YYYY-MM-DD), from the turn's day: an explicit date, a
    weekday (its next one), «завтра» / "tomorrow", «сегодня» / "tonight". None: no day."""
    try:
        today = datetime.fromisoformat(clock.replace("Z", "+00:00")).date()
    except ValueError:
        return None
    low = text.casefold()
    for pat, md in (
        (ledger._ISO, lambda m: (int(m[1]), int(m[2]), int(m[3]))),
        (ledger._DAY_MONTH, lambda m: (int(m[3] or 0), ledger._MONTHS[m[2]], int(m[1]))),
        (ledger._MONTH_DAY, lambda m: (int(m[3] or 0), ledger._MONTHS[m[1]], int(m[2]))),
    ):
        m = pat.search(low)
        if m is None:
            continue
        try:
            y, mo, d = md(m)
            day = date(y or today.year, mo, d)
        except (KeyError, ValueError):
            continue
        if not m.groups()[-1] and day < today:
            day = date(today.year + 1, mo, d)
        return day.isoformat()
    for n, stems in enumerate(_WEEKDAYS):
        if any(re.search(rf"(?<!\w){s}\w*", low) for s in stems):
            return (today + timedelta(days=(n - today.weekday()) % 7 or 7)).isoformat()
    if _NEXT_DAY.search(low):
        return (today + timedelta(days=1)).isoformat()
    if _THIS_DAY.search(low):
        return today.isoformat()
    return None


def land_in_turn(
    conn: sqlite3.Connection, turn: TurnCtx, reply_text: str, cues: Sequence[str]
) -> bool:
    """Land the cues code can land, in the turn, as the brain's call would have: the user's
    correction with a new value (one target), the user's «запомни …», the Cube's promise. The
    user's own untainted turns only (foreign text is never the user's); a forget cue lands
    nothing here (the forget path owns it). True when something committed."""
    if turn.is_tainted or turn.origin not in (TurnOrigin.OWNER, TurnOrigin.TASK):
        return False
    own = policy.turn_words(turn).strip()
    if not own or "forget" in cues:
        return False
    done = False
    try:
        if "correction" in cues and (contrast(own) is not None or shifted(own)):
            t = resolve_target(conn, own, own, None, thread_id=turn.thread_id)
            if t.id is not None and not t.ask and not t.missing:
                done |= fix(conn, own, own, turn, because_quote=own).outcome == "committed"
        m = _REMEMBER_AT.search(own) if "remember" in cues else None
        if m is not None and len(own[m.end() :].split()) >= 2:
            text = own[m.end() :].strip()
            done |= remember(conn, text, turn, because_quote=own).outcome == "committed"
        clause = _promise_clause(reply_text) if "promise" in cues else None
        due = _due_of(clause, turn.clock or _now()) if clause else None
        if clause and due:  # the Cube's open loop with its day: the core shows it
            a = Assertion(subject="cube", text_en=clause, due=due)
            w = remember(conn, clause, turn, because_quote=own, kind=RecordKind.LOOP, assertion=a)
            done |= w.outcome == "committed"
    except (sqlite3.Error, store.OpRejected, ValueError) as exc:
        log.warning("memory hot: in-turn landing failed: %s", type(exc).__name__)
    return done


def after_turn_check(
    conn: sqlite3.Connection, turn: TurnCtx, reply_text: str
) -> tuple[bool, list[str], str | None]:
    """§6.1 safety net: cues with no matching commit first land on the hot path when code
    can land them (`land_in_turn`); the rest → freeze + urgent job."""
    cues = unbacked_cues(conn, turn, detect(turn, reply_text))
    wide = [] if "promise" in cues or not _PROMISE_LAND.search(reply_text or "") else ["promise"]
    wide = unbacked_cues(conn, turn, wide)
    if (cues or wide) and land_in_turn(conn, turn, reply_text, [*cues, *wide]):
        cues = unbacked_cues(conn, turn, cues)
    if not cues:
        return False, [], None
    words = _cue_text(policy.turn_words(turn), cues)
    ents = _cue_entities(conn, words) if turn.origin is TurnOrigin.OWNER else []
    jid = freeze(conn, ents, turn=turn, cues=cues)
    return True, ents, jid


def failure_line(conn: sqlite3.Connection, thread_id: str) -> str | None:
    """§6.1 step 3: once per failed urgent job of this thread, «Не смог применить … — уточни.»"""
    rows = conn.execute(
        "SELECT id, payload FROM jobs WHERE jkind = ? AND state IN ('failed', 'needs_attention')"
        " ORDER BY created_at",
        (SAFETY_JOB,),
    ).fetchall()
    for jid, raw in rows:
        p = json.loads(raw)
        if p.get("thread_id") != thread_id or p.get("told"):
            continue
        p["told"] = True
        with store.write_tx(conn):
            conn.execute(
                "UPDATE jobs SET payload = ?, updated_at = ? WHERE id = ?",
                (json.dumps(p), store.now(), jid),
            )
        about = "это"
        if p.get("entities"):
            row = conn.execute(
                "SELECT display_name FROM entities WHERE id = ?", (p["entities"][0],)
            ).fetchone()
            if row is not None:
                about = row[0]
        return f"Не смог применить исправление про {about} — уточни."
    return None
