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
from dataclasses import dataclass, field
from datetime import timedelta
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
        armed = _arm(conn, intention)
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


def _arm(conn: sqlite3.Connection, intention: Intention) -> Written | None:
    """`intentions.arm` when built (it writes the record); None → the ADD path writes it."""
    from .. import intentions

    try:
        iid = intentions.arm(conn, intention)
    except NotImplementedError:
        return None
    except (store.OpRejected, ValueError) as exc:
        return Written("rejected", line=str(exc) or "Не могу это поставить.")
    return Written(
        "committed", record_ids=[iid], memory_version=store.memory_version(conn), line="Запомнил."
    )


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


# «не X, а Y» / «Y, а не X» / "not X but Y": the old value the user's correction names
# and the new one
_CONTRAST = (
    re.compile(
        r"(?<!\w)не\s+(?P<old>[^,;:.!?]+?)\s*[,—–-]?\s+а\s+(?P<new>[^,;:.!?]+)", re.IGNORECASE
    ),
    re.compile(r"(?P<new>[^,;:.!?]+?)\s*,\s*а\s+не\s+(?P<old>[^,;:.!?]+)", re.IGNORECASE),
    re.compile(
        r"(?<!\w)not\s+(?P<old>[^,;:.!?]+?)\s*,?\s+but\s+(?P<new>[^,;:.!?]+)", re.IGNORECASE
    ),
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
    """The «не X, а Y» ("not X but Y") part of a correction: X is the value it retires, Y the
    new one. Also «Y, а не X» and "not X but Y". None when the user's words hold no such
    contrast."""
    for pat in _CONTRAST:
        m = pat.search(text or "")
        if m and m.group("old").strip() and m.group("new").strip():
            return Contrast(m.group("old").strip(), m.group("new").strip(), m.start(), m.end())
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


def resolve_target(
    conn: sqlite3.Connection, target: str, because_quote: str, assertion: Assertion | None
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
    if not scored:
        return _Target(missing="Не нашёл в памяти, что исправить.")
    best = scored[0][0]
    top = list(dict.fromkeys(seed_fact(conn, rid) or rid for s, rid in scored if s == best))
    heads = list(dict.fromkeys(head(conn, rid) or rid for rid in top))
    if len(heads) > 1:
        opts = []
        for rid in heads[:4]:
            rec = ledger.get_record(conn, rid)
            shown = getattr(rec, "quote_orig", None) or getattr(rec, "text_en", "") or rid
            opts.append(f"«{shown[:80]}» [m:{rid}]")
        return _Target(ask="Что именно исправить: " + " или ".join(opts) + "?")
    return _Target(id=top[0])


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
    row = conn.execute(
        "SELECT count(DISTINCT target_record_id) FROM justifications WHERE member_record_id = ?",
        (rid,),
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
    """§6.1 «исправь» / «это неправда» ("fix" / "that's not true"): retarget to the slot head
    inside the transaction, SUPERSEDE (fix=true) or RETRACT; ledger invalidation and pending
    updates; dependents via cascade when built; core hook; hot log.

    Tainted or non-owner turn (same rule as `commands.forget_now`): no correction word in the
    user's own words → `ask_first`, before the target is even resolved."""
    if _needs_his_word(turn) and not his_correction(policy.turn_words(turn)):
        return Written(
            "ask_first",
            line="Исправлять ты не просил — это из чужого текста. Исправить — скажешь сам?",
        )
    if not _needs_his_word(turn):
        swap = _speaker_swap(conn, (correction or "").strip() or because_quote)
        if swap is not None:  # «я перепутала, это сказала Лера Пак»: the person, not the value
            return _reattribute(conn, swap, turn, because_quote=because_quote)
    t = resolve_target(conn, target, because_quote, assertion)
    if t.missing:
        return Written("rejected", line=t.missing)
    if t.ask:
        return Written("ask_first", line=t.ask)
    assert t.id is not None
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
        box.update(op=op, v=version, old=old, touched=touched, deps=_dependents(ctx.conn, old))
        _cascade(ctx, [old])
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
    """The user's own words carry a correction (the fix analogue of `commands._FORGET_VERB`)."""
    return bool(words) and bool(_FIX_WORDS.search(words) or _CORRECTION.search(words))


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
_CLAIM = re.compile(
    r"(?:^|[\s,.!?«\"])(?:запомнил[аи]?|запомню|исправил[аи]?|поправил[аи]?|забыл[аи]?(?!\s+(?:про|о)\s)"
    r"|удалил[аи]?\s+из\s+памяти|перестал\w*\s+использовать|отменил[аи]?|noted|remembered"
    r"|i'?ll\s+remember|updated\s+(?:my\s+)?memory|forgot(?:ten)?\s+(?:it|that))(?!\w)",
    re.IGNORECASE,
)


def detect(turn: TurnCtx, reply_text: str) -> list[str]:
    """§6.1 safety net detector (code): cues in the user's words (correction, forget, remember,
    constraint) and promises in the Cube's reply."""
    cues: list[str] = []
    own = policy.turn_words(turn) if turn.origin is TurnOrigin.OWNER else ""
    for name, pat in (
        ("correction", _CORRECTION),
        ("forget", _FORGET),
        ("remember", _REMEMBER),
        ("constraint", _CONSTRAINT),
    ):
        if own and pat.search(own):
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
        "forget": _FORGET,
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


def after_turn_check(
    conn: sqlite3.Connection, turn: TurnCtx, reply_text: str
) -> tuple[bool, list[str], str | None]:
    """§6.1 safety net: cues with no matching commit → freeze + urgent job."""
    cues = unbacked_cues(conn, turn, detect(turn, reply_text))
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
