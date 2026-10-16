"""Core view (~6k tokens): About you, People, Working now, Lessons, What changed recently;
cache-stable order; each section with its read set; built through use-now with a neutral turn.

Spec: §10.1, §7.6.

How it holds together:
- `compute` is a pure read of the ledger: the same ledger on the same UTC day gives the same
  bytes (no clock text, no heat or use counters in the text; display order by stable keys).
  The day is the only clock input: the 7-day windows move once a day.
- Every line passes the policy fence (`policy.decide_many` with a neutral turn: the user's own
  sensitive items stay, other people's sensitive items never enter), frozen entities and
  their records are left out (`hot.frozen_records`), and other people's precise places
  stay out of the core (the core rides on send turns too, unmarked).
- `persist` stores each section as a `derived` row `core:<section>` with its read set,
  through `store.commit` (the ledger checks usability, labels and retracted values); an
  unchanged section is not rewritten. The ledger invalidates a section when an item it read
  changes and queues `rebuild_derived`; `_run_rebuild` and the hot `on_core_change` hook
  rebuild it.
- `diff_lines` compares what a session was served (`served` rows) with the current core and
  renders §7.6 update lines; `api.core_diff` sends them when they fit in 1k tokens.
- The always-on core shows a thing only when it applies to any turn today.
  People: only the recently active (`WINDOW_DAYS`); a person named in the question comes
  through the basket. About you: a preference scoped to someone else or to an audience stays
  out, a day- or date-bound one shows on its day (and the day before), today's time-bound
  override hides the standing rule it overrides, a future plan beyond the window and an
  expired one-off stay out. Topic-sensitive labels (health, identity documents,
  intimate, legal) never enter any section; they still answer a direct
  question through the basket. The budgets are unchanged, and freed room is not refilled.
- Since every sender is an entity, «recently active» people means the user's
  interactions (`_his_touch`): the user's records about, pointing at or quoting them, the
  user's messages naming them, the user's messages in their one-to-one chat; their messages
  arriving do not count. An undated one-off is dated by when the user said it; a habit with
  no period field takes its day from its own words or the message it rests on.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .. import clock as _clock
from .. import entities, ledger, policy, store
from ..model import (
    OWNER_ENTITY_ID,
    AuthorTier,
    Label,
    OpKind,
    Proposal,
    ProposalOp,
    ReadRef,
    ReadSrc,
    ServedVia,
    TurnCtx,
    TurnOrigin,
)
from ..writers import hot
from .safe_text import clean

SECTIONS: tuple[str, ...] = ("about_you", "people", "working_now", "lessons", "what_changed")
TITLES: dict[str, str] = {
    "about_you": "About you",
    "people": "People",
    "working_now": "Working now",
    "lessons": "Lessons",
    "what_changed": "What changed recently",
}
BUDGET_TOKENS = 6000
SECTION_BUDGET: dict[str, int] = {
    "about_you": 1600,
    "people": 1700,
    "working_now": 1200,
    "lessons": 600,
    "what_changed": 800,
}
DIFF_TOKENS = 1000  # §10.1: a resumed session gets the changed lines up to 1k, else a restart
WINDOW_DAYS = 7
LESSONS_TOP = 10
PEOPLE_CAP = 40
FACTS_PER_PERSON = 3
DKIND = "core_section"
REBUILD_JOB = "rebuild_derived"
OPEN, CLOSE = "<memory-core>", "</memory-core>"
UPDATE_OPEN, UPDATE_CLOSE = "<memory-update>", "</memory-update>"

_TEXT_MAX = 300
_PERSON_FACT_MAX = 120
_VALUE_MAX = 80  # the source-language value next to an English line
_PERSON_VALUE_MAX = 50
_RECENCY_HALF_LIFE_DAYS = 30.0
_GONE = frozenset({"forgotten", "purged"})
_OWN = frozenset({"user", "user_confirmed"})
_PRONOUNS = frozenset(
    [
        "я",
        "меня",
        "мне",
        "мной",
        "мой",
        "моя",
        "моё",
        "мое",
        "мои",
        "себя",
        "себе",
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "self",
        "owner",
    ]
) | frozenset(a.casefold() for a in ledger.OWNER_ALIASES)
_APPL_KEYS = ("task", "audience", "situation", "period", "channel")
# Labels whose topic must not ride on an unrelated turn. The rest of the §5.5 sensitive set
# stays with the policy fence: `minor` and the fail-closed `unknown_sensitive` mark who or how
# sure, not a topic; `money` and `third_party_private` are set by the classifier on rent, a
# carpenter's price, a thesis on banking apps and a colleague's release (why_bench), so hiding
# them cut 7 of the user's everyday facts.
_TOPIC_SENSITIVE = Label.HEALTH | Label.IDENTITY_DOCS | Label.INTIMATE | Label.LEGAL
_SELF_SCOPE = frozenset({"self", "user", "owner", "general", OWNER_ENTITY_ID.casefold()})
_EVERYONE = frozenset({"all", "everyone", "anyone", "general", "все", "всем", "всех", "любой"})
_WEEKDAY_RES: tuple[tuple[re.Pattern[str], frozenset[int]], ...] = tuple(
    (re.compile(p), frozenset(days))
    for p, days in (
        (r"\bпонедельник\w*|\bmondays?\b", {0}),
        (r"\bвторник\w*|\btuesdays?\b", {1}),
        (r"\bсред(?:а|у|ы|ам|ой)\b|\bwednesdays?\b", {2}),
        (r"\bчетверг\w*|\bthursdays?\b", {3}),
        (r"\bпятниц\w*|\bfridays?\b", {4}),
        (r"\bсуббот\w*|\bsaturdays?\b", {5}),
        (r"\bвоскресен\w*|\bsundays?\b", {6}),
        (r"\bвыходн\w*|\bweekends?\b", {5, 6}),
        (r"\bбудн\w*|\bweekdays?\b", {0, 1, 2, 3, 4}),
    )
)
_ISO_DAY = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_WORD = re.compile(r"\w+")


def tokens(text: str) -> int:
    """Conservative estimate: ~4 characters a token in Latin script, ~2 in Cyrillic and other
    scripts (no tokenizer dependency; over-counts, so the budget holds)."""
    if not text:
        return 0
    ascii_n = sum(1 for ch in text if ord(ch) < 128)
    return math.ceil(ascii_n / 4 + (len(text) - ascii_n) / 2)


# ---------------------------------------------------------------- shapes


@dataclass(frozen=True)
class Line:
    """One core line: `- <text> [m:<key>]`. `refs` are the usable items the text was built
    from: they go to the prompt's read set and the pre-send check. `deps` are retired items
    the text names (a superseded old value, a retracted claim): they only invalidate the
    section (a forget of one rebuilds it) and never reach the pre-send check, which would
    find them unusable and drop the whole core (RC-2)."""

    key: str
    text: str
    refs: tuple[ReadRef, ...]
    order: tuple[Any, ...] = ()
    prio: int = 0  # position in selection priority (0 = kept first)
    deps: tuple[ReadRef, ...] = ()

    @property
    def rendered(self) -> str:
        return f"- {self.text} [m:{self.key}]"


@dataclass
class Section:
    name: str
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        if not self.lines:
            return ""
        return "\n".join([f"## {TITLES[self.name]}", *(ln.rendered for ln in self.lines)])

    @property
    def read_set(self) -> list[ReadRef]:
        """What a prompt holding this section reads: the lines' usable refs."""
        seen: dict[str, ReadRef] = {}
        for ln in self.lines:
            for r in ln.refs:
                seen.setdefault(r.id, r)
        return [seen[k] for k in sorted(seen)]

    @property
    def read_deps(self) -> list[ReadRef]:
        """What the stored section depends on: refs and retired deps (its derived read set)."""
        seen: dict[str, ReadRef] = {}
        for ln in self.lines:
            for r in (*ln.refs, *ln.deps):
                seen.setdefault(r.id, r)
        return [seen[k] for k in sorted(seen)]


def render(sections: Iterable[Section]) -> str:
    body = [s.text for s in sections if s.lines]
    if not body:
        return ""
    return "\n".join([OPEN, "\n\n".join(body), CLOSE])


@dataclass
class _Ctx:
    conn: sqlite3.Connection
    today: date
    since: str  # ISO date: the 7-day window start
    until: str  # ISO date: due within 7 days
    frozen: set[str]
    names: dict[str, str] = field(default_factory=dict)
    turn: TurnCtx = field(default_factory=lambda: neutral_turn())
    sensitive: Label = _TOPIC_SENSITIVE  # never in the always-on part

    @property
    def today_s(self) -> str:
        return self.today.isoformat()

    @property
    def tomorrow_s(self) -> str:
        return (self.today + timedelta(days=1)).isoformat()

    @property
    def near_weekdays(self) -> frozenset[int]:
        """Today and tomorrow: a day-bound rule shows on its day and the evening before."""
        return frozenset({self.today.weekday(), (self.today.weekday() + 1) % 7})


def neutral_turn() -> TurnCtx:
    """§10.1: the core is built through use-now with a turn that says nothing."""
    return TurnCtx(turn_id="core-view", thread_id="core-view", origin=TurnOrigin.OWNER)


def _ctx(conn: sqlite3.Connection, today: date | None = None) -> _Ctx:
    day = today or _clock.now().date()
    try:
        frozen = hot.frozen_records(conn) | hot.frozen_entities(conn)
    except sqlite3.Error:
        frozen = set()
    return _Ctx(
        conn=conn,
        today=day,
        since=(day - timedelta(days=WINDOW_DAYS)).isoformat(),
        until=(day + timedelta(days=WINDOW_DAYS)).isoformat(),
        frozen=frozen,
        sensitive=policy.sensitive_mask() & _TOPIC_SENSITIVE,
    )


# ---------------------------------------------------------------- helpers


def _short(text: str | None, n: int = _TEXT_MAX, tier: Any = None) -> str:
    """Fence-safe (RC-1: quarantined spans cut, `<memory-…>` tags and `[m:` defused), then
    clipped. `tier` is the text's author tier: the user's own words are never quarantined."""
    t = clean(text, handles=True, tier=tier)
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _source_value(r: sqlite3.Row, shown: str, n: int = _VALUE_MAX) -> str:
    """The value in the language the user said it in, for a line built on the English
    `text_en`: `value_text`, or the verbatim quote when the record has none (money keeps
    the currency the user named). Empty when the English line already holds it."""
    keys = r.keys()
    val = (r["value_text"] if "value_text" in keys else None) or ""
    if not val.strip() and "quote_orig" in keys:
        val = r["quote_orig"] or ""
    v = _short(val, n, r["author_tier"] if "author_tier" in keys else None)
    if not v or v.strip("«»\"' ").casefold() in shown.casefold():
        return ""
    return v


def _with_value(text: str, value: str) -> str:
    return f"{text} — «{value}»" if value else text


def _subject_of(ctx: _Ctx, subject_id: str | None, relation: str | None) -> str:
    """What changed, without the value: «(you · email)», «(<person> · works at)», «(<person>)»."""
    who = _name(ctx, subject_id) or "someone"
    rel = (relation or "").replace("_", " ").strip()
    return f"({who} · {rel})" if rel and rel != "other" else f"({who})"


def is_current(valid_to: str | None, day: str) -> bool:
    """A fact is current on `day` (ISO date) unless its `valid_to` has passed (§4.1); the
    core view and the basket share this notion."""
    return not (valid_to and str(valid_to)[:10] < day)


def _ref(row: sqlite3.Row, id_col: str = "id", v_col: str = "version") -> ReadRef:
    return ReadRef(id=row[id_col], version=int(row[v_col]))


def _name(ctx: _Ctx, eid: str | None) -> str | None:
    if not eid:
        return None
    if eid == OWNER_ENTITY_ID:
        return "you"
    if eid not in ctx.names:
        r = ctx.conn.execute(
            "SELECT e.display_name FROM entities e JOIN records r ON r.id = e.id"
            " WHERE e.id = ? AND r.status = 'active'",
            (eid,),
        ).fetchone()
        ctx.names[eid] = clean(r[0], handles=True) if r and r[0] else ""
    return ctx.names[eid] or None


def _entity_ref(ctx: _Ctx, eid: str | None) -> ReadRef | None:
    if not eid or eid in ("user", "cube"):
        return None
    r = ctx.conn.execute(
        "SELECT r.id, r.version FROM records r JOIN entities e ON e.id = r.id"
        " WHERE r.id = ? AND r.status = 'active'",
        (eid,),
    ).fetchone()
    return _ref(r) if r else None


def _fence(ctx: _Ctx, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """Use-now with the neutral turn, frozen items out, others' precise places out, and
    topic-sensitive items out even when they are the user's own."""
    rows = [
        r for r in rows if r["id"] not in ctx.frozen and not Label(int(r["labels"])) & ctx.sensitive
    ]
    if not rows:
        return []
    verdicts = policy.decide_many(ctx.conn, [r["id"] for r in rows], ctx.turn)
    out = []
    for r in rows:
        if not verdicts[r["id"]].allowed:
            continue
        if Label(int(r["labels"])) & Label.LOCATION_PRECISE and r["author_tier"] not in _OWN:
            continue
        out.append(r)
    return out


def _retired_ok(ctx: _Ctx, rid: str) -> bool:
    if rid in ctx.frozen:
        return False
    r = ctx.conn.execute("SELECT labels, author_tier FROM records WHERE id = ?", (rid,)).fetchone()
    if r is None:
        return False
    labels = Label(int(r["labels"]))
    if labels & ctx.sensitive:
        return False
    if labels & Label.LOCATION_PRECISE and r["author_tier"] not in _OWN:
        return False
    return policy.decide_retired(ctx.conn, rid, ctx.turn).allowed


# ---------------------------------------------------------------- applicability


def _appl_data(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _norm(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _for_someone_else(data: dict[str, Any]) -> bool:
    """A rule for another person or an audience: it applies when they are in play, which the
    always-on core cannot see (a style for professors is not a style for a family member)."""
    fw = data.get("for_whom")
    if isinstance(fw, str) and fw.strip():
        key = _norm(fw.strip())
        if key not in _SELF_SCOPE and key not in _PRONOUNS:
            return True
    aud = data.get("audience")
    return isinstance(aud, str) and bool(aud.strip()) and _norm(aud.strip()) not in _EVERYONE


def _period_weekdays(period: Any) -> frozenset[int] | None:
    if not isinstance(period, str) or not period.strip():
        return None
    text = _norm(period)
    days: set[int] = set()
    for rx, ds in _WEEKDAY_RES:
        if rx.search(text):
            days |= ds
    return frozenset(days) if days else None


def _period_span(period: Any) -> tuple[str, str] | None:
    if not isinstance(period, str):
        return None
    found = sorted(_ISO_DAY.findall(period))
    return (found[0], found[-1]) if found else None


def _habit_days(r: sqlite3.Row) -> frozenset[int] | None:
    """A routine with no period field still names its day in its own words («каждую
    субботу», «every Saturday»), or else in the message it rests on (two habits of one
    sentence share its day)."""
    for text in (r["quote_orig"], r["value_text"], r["text_en"], r["said_text"]):
        days = _period_weekdays(text)
        if days is not None:
            return days
    return None


def _applies_now(ctx: _Ctx, r: sqlite3.Row, *, scoped: bool = True) -> bool:
    """Whether a fact or preference applies to any turn today. `scoped` checks the
    audience and for-whom scope; the time checks hold for every row."""
    data = _appl_data(r["applicability"])
    if scoped and _for_someone_else(data):
        return False
    days = _period_weekdays(data.get("period"))
    if days is None and r["stability"] == "habit":
        days = _habit_days(r)
    if days is not None and not days & ctx.near_weekdays:
        return False
    span = _period_span(data.get("period"))
    if span is not None and not (span[0] <= ctx.tomorrow_s and ctx.today_s <= span[1]):
        return False
    start = (r["valid_from"] or "")[:10]
    if r["fkind"] == "plan" and start and start > ctx.until:
        return False  # a plan beyond the window: the basket brings it when asked
    if r["stability"] == "one_off":
        # an undated one-off is dated by when the user said it («the meeting was
        # cancelled», «meet on Saturday»): months later it is not about any turn today, and
        # the basket still brings it when the turn asks
        end = (r["valid_to"] or r["valid_from"] or r["said_at"] or r["recorded_at"] or "")[:10]
        if end and end < ctx.since:
            return False  # an expired one-off
    return True


def _stems(text: str | None) -> set[str]:
    """Word stems of a value, script-agnostic: «кинза» and «кинзы» share «кинз»."""
    out: set[str] = set()
    for w in _WORD.findall(_norm(text or "")):
        if len(w) >= 4 and not w.isdigit():
            out.add(w[: max(4, len(w) - 2)])
    return out


def _is_override(ctx: _Ctx, r: sqlite3.Row) -> bool:
    if r["stability"] != "one_off":
        return False
    if r["valid_to"] and str(r["valid_to"])[:10] <= ctx.until:
        return True
    return _period_span(_appl_data(r["applicability"]).get("period")) is not None


def _drop_overridden(ctx: _Ctx, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """Today's override beats the standing rule: while a time-bound one-off about the same
    thing is current, the standing row it overrides stays out (it is back once the override
    ends). `rows` are one subject's current, applicable rows."""
    over = {r["id"]: _stems(r["value_text"]) for r in rows if _is_override(ctx, r)}
    stems = set().union(*over.values()) if over else set()
    if not stems:
        return rows
    return [
        r
        for r in rows
        if r["id"] in over
        or r["stability"] == "one_off"
        or r["valid_to"]
        or not _stems(r["value_text"]) & stems
    ]


def _appl(ctx: _Ctx, raw: str | None) -> list[str]:
    data = _appl_data(raw)
    out = []
    fw = data.get("for_whom")
    if isinstance(fw, str) and fw.strip() and fw not in ("self", "user"):
        out.append(f"for {_name(ctx, fw) or _short(fw, 40)}")
    for k in _APPL_KEYS:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            out.append(f"{k}: {_short(v, 60)}")
    return out


def _fact_text(
    ctx: _Ctx, r: sqlite3.Row, *, limit: int = _TEXT_MAX, value_max: int = _VALUE_MAX
) -> str:
    text = _short(r["text_en"], limit, r["author_tier"])
    text = _with_value(text, _source_value(r, text, value_max))
    if r["fkind"] == "plan":
        text = f"plan: {text}"
    marks: list[str] = []
    if r["valid_from"] and r["fkind"] == "plan":
        marks.append(f"from {r['valid_from'][:10]}")
    if r["valid_to"]:
        marks.append(f"until {r['valid_to'][:10]}")
    if r["stability"] == "one_off":
        marks.append("one-off")
    elif r["stability"] == "habit":
        marks.append("habit")
    marks += _appl(ctx, r["applicability"])
    tier = r["author_tier"]
    if tier not in _OWN:
        if tier == "system":
            marks.append("inferred")
        elif tier == "tool":
            marks.append("per a source")
        else:
            who = _name(ctx, r["speaker_entity_id"] or r["author_entity_id"])
            marks.append(f"per {who}" if who and who != "you" else "per someone")
    if r["review"]:
        marks.append("re-check")
    return f"{text} ({'; '.join(marks)})" if marks else text


def _fit(lines: list[Line], budget: int) -> list[Line]:
    """Keep lines in priority order while the section fits; then display by stable keys."""
    kept: list[Line] = []
    used = tokens("## " + "x" * 20)
    for i, ln in enumerate(lines):
        cost = tokens(ln.rendered) + 1
        if used + cost > budget:
            continue
        used += cost
        kept.append(Line(ln.key, ln.text, ln.refs, ln.order, i, ln.deps))
    return sorted(kept, key=lambda ln: ln.order)


def _clean(ctx: _Ctx, lines: list[Line]) -> list[Line]:
    """§9.6 at line grain: a line may not hold a retracted value of what it read (the ledger
    refuses such derived text); a history clause («previously …») stays allowed."""
    out = []
    for ln in lines:
        ids = [r.id for r in (*ln.refs, *ln.deps) if r.src is ReadSrc.RECORD]
        try:
            ledger.check_retracted(ctx.conn, [ln.rendered], raws=[], records=ids)
        except store.OpRejected:
            continue
        out.append(ln)
    return out


# ---------------------------------------------------------------- sections

_FACT_COLS = (
    "r.id, r.kind, r.version, r.labels, r.author_tier, r.author_entity_id, r.review,"
    " r.importance, r.pinned_category, r.recorded_at, f.relation, f.text_en, f.fkind,"
    " f.polarity, f.valid_from, f.valid_to, f.stability, f.applicability,"
    " f.speaker_entity_id, f.value_entity_id, f.value_text, f.quote_orig,"
    " (SELECT rc.ts FROM raw_chunks rc WHERE rc.id = f.source_raw_id) AS said_at,"
    " (SELECT substr(rc.text, 1, 600) FROM raw_chunks rc WHERE rc.id = f.source_raw_id"
    "   AND rc.status = 'active') AS said_text"
)


def _current(ctx: _Ctx, r: sqlite3.Row) -> bool:
    return is_current(r["valid_to"], ctx.today_s)


def _about_you(ctx: _Ctx) -> list[Line]:
    conn = ctx.conn
    lines: list[Line] = []
    me = conn.execute(
        "SELECT r.id, r.version, r.status, e.display_name FROM records r"
        " JOIN entities e ON e.id = r.id WHERE r.id = ?",
        (OWNER_ENTITY_ID,),
    ).fetchone()
    if me is not None and me["status"] == "active" and me["id"] not in ctx.frozen:
        shown = me["display_name"].casefold()
        seen: dict[str, str] = {}
        for a in conn.execute(
            "SELECT alias, alias_norm FROM aliases WHERE entity_id = ? AND kind != 'translit'"
            " ORDER BY alias_norm, id",
            (OWNER_ENTITY_ID,),
        ):
            norm = a["alias_norm"]
            if norm in _PRONOUNS or norm == shown or len(norm) < 2:
                continue
            seen.setdefault(norm, a["alias"])
        aka = list(seen.values())[:6]
        if aka or me["display_name"] != "Owner":
            text = f"You are {_short(me['display_name'], 80)}"
            if aka:
                text += f" (also {', '.join(_short(a, 40) for a in aka)})"
            lines.append(Line(OWNER_ENTITY_ID, text, (_ref(me),), order=("", "", "")))
    rows = conn.execute(
        f"SELECT {_FACT_COLS} FROM facts f JOIN records r ON r.id = f.id"  # noqa: S608
        " WHERE f.subject_id = ? AND r.status = 'active' AND r.kind IN ('fact', 'preference')",
        (OWNER_ENTITY_ID,),
    ).fetchall()
    rows = [r for r in _fence(ctx, rows) if _current(ctx, r) and _applies_now(ctx, r)]
    rows = _drop_overridden(ctx, rows)
    rows.sort(key=lambda r: (r["recorded_at"], r["id"]), reverse=True)
    rows.sort(
        key=lambda r: (
            r["pinned_category"] is None,
            r["author_tier"] not in _OWN,
            -(r["importance"] if r["importance"] is not None else 0.5),
        )
    )
    for r in rows:
        refs = [_ref(r)]
        er = _entity_ref(ctx, r["value_entity_id"])
        if er is not None:
            refs.append(er)
        lines.append(
            Line(
                r["id"],
                _fact_text(ctx, r),
                tuple(refs),
                order=("~", r["relation"], r["recorded_at"], r["id"]),
            )
        )
    return lines


def _days(ctx: _Ctx, ts: str | None) -> int:
    if not ts:
        return 3650
    try:
        d = date.fromisoformat(ts[:10])
    except ValueError:
        return 3650
    return max(0, (ctx.today - d).days)


_HIS_RAWS_MAX = 500  # the user's own messages of the window scanned for names, newest first


def _person_forms(ctx: _Ctx) -> dict[str, set[str]]:
    """Name forms of usable people → their ids: each alias word (any script; a translit alias
    row carries the other one) and, «~»-marked, its Cyrillic case roots (Гордей ~ Гордею). A
    word index, not a per-message name search: the core is built on the turn path."""
    out: dict[str, set[str]] = {}
    for a in ctx.conn.execute(
        "SELECT a.entity_id, a.alias_norm FROM aliases a JOIN entities e ON e.id = a.entity_id"
        " JOIN records r ON r.id = e.id"
        " WHERE e.etype = 'person' AND e.id != ? AND r.status = 'active'",
        (OWNER_ENTITY_ID,),
    ):
        for tok in _WORD.findall(_norm(a["alias_norm"] or "")):
            if len(tok) < 3 or tok.isdigit() or tok in _PRONOUNS:
                continue
            out.setdefault(tok, set()).add(a["entity_id"])
            for root in entities.name_roots(tok):
                if len(root) >= 3:
                    out.setdefault("~" + root, set()).add(a["entity_id"])
    return out


def _his_touch(ctx: _Ctx) -> dict[str, str]:
    """When the user last touched each person inside the window: a record in their words
    about them, pointing at them or quoting them (dated by the message it rests on), a
    message of the user's that names them, or a message of the user's in a chat where they
    are the only other voice. Their own messages arriving are not the user's activity (every
    sender is an entity), and a retracted or superseded record touches no one."""
    conn = ctx.conn
    out: dict[str, str] = {}

    def bump(eid: Any, at: Any) -> None:
        if eid and at and eid != OWNER_ENTITY_ID and str(at) > out.get(str(eid), ""):
            out[str(eid)] = str(at)

    own = tuple(sorted(_OWN))
    said = (
        "coalesce((SELECT rc.ts FROM raw_chunks rc WHERE rc.id = f.source_raw_id), r.recorded_at)"
    )
    for col in ("f.subject_id", "f.value_entity_id", "f.speaker_entity_id"):
        for row in conn.execute(
            f"SELECT {col}, {said} FROM facts f JOIN records r ON r.id = f.id"  # noqa: S608
            " WHERE r.status = 'active' AND r.author_tier IN (?, ?) AND r.recorded_at >= ?",
            (*own, ctx.since),
        ):
            bump(row[0], row[1])
    for row in conn.execute(
        "SELECT i.about_entity_id, coalesce(i.observed_at, r.recorded_at) FROM impressions i"
        " JOIN records r ON r.id = i.id"
        " WHERE r.status = 'active' AND r.author_tier IN (?, ?) AND r.recorded_at >= ?",
        (*own, ctx.since),
    ):
        bump(row[0], row[1])
    for row in conn.execute(
        "SELECT r.id, r.recorded_at FROM entities e JOIN records r ON r.id = e.id"
        " WHERE r.status = 'active' AND r.author_tier IN (?, ?) AND r.recorded_at >= ?",
        (*own, ctx.since),
    ):
        bump(row[0], row[1])
    mine = conn.execute(
        "SELECT chat_id, ts, text FROM raw_chunks WHERE author_tier IN (?, ?)"
        " AND status = 'active' AND junk = 0 AND quarantined = 0 AND ts >= ?"
        " ORDER BY ts DESC, id LIMIT ?",
        (*own, ctx.since, _HIS_RAWS_MAX),
    ).fetchall()
    if not mine:
        return out
    names = _person_forms(ctx)
    roots: dict[str, set[str]] = {}
    for row in mine:
        found: list[set[str]] = []
        for tok in set(_WORD.findall(_norm(row["text"] or "")[:4000])):
            hit = names.get(tok)
            if hit is None and len(tok) >= 3:
                if tok not in roots:
                    roots[tok] = {r for r in entities.name_roots(tok) if len(r) >= 3}
                hit = set().union(*(names.get("~" + r, set()) for r in roots[tok]))
            if hit:
                found.append(hit)
        # a shared first name goes to the people the same message names by a word of their
        # own («Женя Воронин» is not every Zhenya); alone, it touches all who share it
        sure = set().union(*(h for h in found if len(h) == 1))
        for hit in found:
            if len(hit) == 1 or not hit & sure:
                for eid in hit:
                    bump(eid, row["ts"])
    last_in: dict[str, str] = {}
    for row in mine:
        if row["chat_id"] and row["ts"] > last_in.get(row["chat_id"], ""):
            last_in[row["chat_id"]] = row["ts"]
    for chat, at in sorted(last_in.items()):
        peers = conn.execute(
            "SELECT DISTINCT author_entity_id FROM raw_chunks WHERE chat_id = ?"
            " AND author_tier NOT IN (?, ?) AND author_entity_id IS NOT NULL LIMIT 2",
            (chat, *own),
        ).fetchall()
        if len(peers) == 1:  # a one-to-one chat: writing there is writing to them
            bump(peers[0][0], at)
    return out


def _people(ctx: _Ctx) -> list[Line]:
    conn = ctx.conn
    rows = conn.execute(
        "SELECT r.id, r.version, r.labels, r.author_tier, r.review, r.importance, r.recorded_at,"
        " e.display_name, e.stable_ids"
        " FROM entities e JOIN records r ON r.id = e.id"
        " WHERE e.etype = 'person' AND e.id != ? AND r.status = 'active'",
        (OWNER_ENTITY_ID,),
    ).fetchall()
    touch = _his_touch(ctx)

    def score(r: sqlite3.Row) -> float:
        imp = r["importance"] if r["importance"] is not None else 0.5
        return imp * 0.5 ** (_days(ctx, touch[r["id"]]) / _RECENCY_HALF_LIFE_DAYS)

    # only the recently active; a person named in the question comes via the basket.
    # «recent» is the user's interaction with them, not their messages arriving.
    rows = [r for r in rows if r["id"] in touch and _days(ctx, touch[r["id"]]) <= WINDOW_DAYS]
    rows.sort(key=lambda r: r["id"])
    rows.sort(key=score, reverse=True)
    people = _fence(ctx, rows[: PEOPLE_CAP * 2])[:PEOPLE_CAP]
    lines: list[Line] = []
    for p in people:
        eid = p["id"]
        refs = [_ref(p)]
        shown = p["display_name"].casefold()
        aka: dict[str, str] = {}
        for a in conn.execute(
            "SELECT alias, alias_norm FROM aliases WHERE entity_id = ?"
            " AND kind NOT IN ('handle', 'translit')"
            " ORDER BY alias_norm, id",
            (eid,),
        ):
            if a["alias_norm"] != shown and a["alias_norm"] not in _PRONOUNS:
                aka.setdefault(a["alias_norm"], a["alias"])
        facts = conn.execute(
            f"SELECT {_FACT_COLS} FROM facts f JOIN records r ON r.id = f.id"  # noqa: S608
            " WHERE f.subject_id = ? AND r.status = 'active' AND r.kind IN ('fact', 'preference')"
            " ORDER BY coalesce(r.importance, 0.5) DESC, r.recorded_at DESC, r.id LIMIT 12",
            (eid,),
        ).fetchall()
        facts = [
            f for f in _fence(ctx, facts) if _current(ctx, f) and _applies_now(ctx, f, scoped=False)
        ]
        facts = _drop_overridden(ctx, facts)[:FACTS_PER_PERSON]
        facts.sort(key=lambda f: (f["recorded_at"], f["id"]))
        text = _short(p["display_name"], 80)
        if aka:
            text += f" (aka {', '.join(_short(a, 30) for a in list(aka.values())[:3])})"
        if facts:
            text += ": " + "; ".join(
                _fact_text(ctx, f, limit=_PERSON_FACT_MAX, value_max=_PERSON_VALUE_MAX)
                for f in facts
            )
            refs += [_ref(f) for f in facts]
        try:
            stable = json.loads(p["stable_ids"] or "[]")
        except ValueError:
            stable = []
        kinds = sorted(
            {str(s.get("kind")) for s in stable if isinstance(s, dict) and s.get("kind")}
        )
        if kinds:  # §10.1: hints only, never the raw ids of other people's accounts
            text += f" [reach: {', '.join(kinds)}]"
        if p["review"]:
            text += " (re-check)"
        lines.append(Line(eid, text, tuple(refs), order=(p["recorded_at"], eid)))
    return lines


def _working_now(ctx: _Ctx) -> list[Line]:
    conn = ctx.conn
    lines: list[Line] = []
    ws = conn.execute(
        "SELECT r.id, r.version, r.labels, r.author_tier, r.review, r.recorded_at, w.matter_id,"
        " w.state FROM work_states w JOIN records r ON r.id = w.id WHERE r.status = 'active'"
    ).fetchall()
    ws = sorted(_fence(ctx, ws), key=lambda r: (r["recorded_at"], r["id"]), reverse=True)
    for r in ws:
        try:
            st = json.loads(r["state"] or "{}")
        except ValueError:
            st = {}
        if not isinstance(st, dict):
            st = {}
        refs = [_ref(r)]
        er = _entity_ref(ctx, r["matter_id"])
        if er is not None:
            refs.append(er)
        matter = _name(ctx, r["matter_id"]) or _short(r["matter_id"], 60)
        tier = r["author_tier"]
        goal = st.get("goal") or st.get("direction") or "in progress"
        text = f"{matter}: {_short(goal, 160, tier)}"
        if st.get("next_step"):
            text += f"; next: {_short(st['next_step'], 120, tier)}"
        qs = [q for q in st.get("open_questions") or [] if isinstance(q, str)][:2]
        if qs:
            text += f"; open: {'; '.join(_short(q, 80, tier) for q in qs)}"
        if r["review"]:
            text += " (re-check)"
        lines.append(Line(r["id"], text, tuple(refs), order=(0, r["recorded_at"], r["id"])))
    loops = conn.execute(
        "SELECT r.id, r.version, r.labels, r.author_tier, r.review, l.owner, l.text_en, l.due"
        " FROM loops l JOIN records r ON r.id = l.id WHERE r.status = 'active'"
        " AND l.state IN ('open', 'overdue') AND l.due IS NOT NULL AND substr(l.due, 1, 10) <= ?",
        (ctx.until,),
    ).fetchall()
    for r in sorted(_fence(ctx, loops), key=lambda r: (r["due"], r["id"])):
        due = r["due"][:10]
        owner = r["owner"]
        refs = [_ref(r)]
        if owner in ("user", OWNER_ENTITY_ID):
            who = "you"
        elif owner == "cube":
            who = "Cube"
        else:
            who = _name(ctx, owner) or "someone"
            er = _entity_ref(ctx, owner)
            if er is not None:
                refs.append(er)
        late = " (overdue)" if due < ctx.today_s else ""
        text = f"{who}: {_short(r['text_en'], 200, r['author_tier'])}, due {due}{late}"
        lines.append(Line(r["id"], text, tuple(refs), order=(1, due, r["id"])))
    ints = conn.execute(
        "SELECT r.id, r.version, r.labels, r.author_tier, r.review, r.origin_quote, i.trigger,"
        " i.action FROM intentions i JOIN records r ON r.id = i.id WHERE r.status = 'active'"
        " AND i.state = 'armed' AND json_extract(i.trigger, '$.kind') = 'date_reached'"
        " AND substr(json_extract(i.trigger, '$.ts'), 1, 10) <= ?",
        (ctx.until,),
    ).fetchall()
    for r in _fence(ctx, ints):
        try:
            trig = json.loads(r["trigger"] or "{}")
            action = json.loads(r["action"] or "null")
        except ValueError:
            trig, action = {}, None
        what = None
        if isinstance(action, dict):
            what = action.get("text") or action.get("summary")
        what = what or r["origin_quote"] or "reminder"
        day = str(trig.get("ts") or "")[:10]
        text = f"reminder {day}: {_short(str(what), 200, r['author_tier'])}"
        lines.append(Line(r["id"], text, (_ref(r),), order=(2, day, r["id"])))
    # priority: the due soonest first, then the newest work states
    return sorted(lines, key=lambda ln: (ln.order[0] == 0, ln.order[1] if ln.order[0] else ""))


def _lessons(ctx: _Ctx) -> list[Line]:
    rows = ctx.conn.execute(
        "SELECT r.id, r.version, r.labels, r.author_tier, r.review, r.recorded_at, l.rule_en,"
        " l.applicability, l.helpful, l.harmful FROM lessons l JOIN records r ON r.id = l.id"
        " WHERE r.status = 'active'"
        " ORDER BY (l.helpful - l.harmful) DESC, r.recorded_at, r.id LIMIT ?",
        (LESSONS_TOP * 3,),
    ).fetchall()
    lines = []
    for n, r in enumerate(_fence(ctx, rows)[:LESSONS_TOP]):
        marks = _appl(ctx, r["applicability"])
        if r["review"]:
            marks.append("re-check")
        text = _short(r["rule_en"], _TEXT_MAX, r["author_tier"])
        if marks:
            text += f" ({'; '.join(marks)})"
        lines.append(Line(r["id"], text, (_ref(r),), order=(n,)))
    return lines


def _what_changed(ctx: _Ctx) -> list[Line]:
    conn = ctx.conn
    lines: list[Line] = []
    # A line says THAT a value changed or was withdrawn, never the retired value itself
    # (no «previously …»): a quoted old value reads as current to the brain and leaks what a
    # correction retired. The retired record stays a dep, so its forget rebuilds.
    sup = conn.execute(
        "SELECT n.id, n.version, n.labels, n.author_tier, n.review, o.id AS old_id,"
        " o.version AS old_version, o.retired_at, nf.text_en AS new_text, nf.value_text,"
        " nf.quote_orig, nf.subject_id, nf.relation"
        " FROM edges e JOIN records o ON o.id = e.to_id JOIN records n ON n.id = e.from_id"
        " JOIN facts nf ON nf.id = n.id JOIN facts pf ON pf.id = o.id"
        " WHERE e.etype = 'supersedes' AND o.status = 'superseded' AND n.status = 'active'"
        " AND o.retired_at >= ?",
        (ctx.since,),
    ).fetchall()
    for r in _fence(ctx, sup):
        day = r["retired_at"][:10]
        refs = (_ref(r),)
        deps = (_ref(r, "old_id", "old_version"),)  # retired: invalidates, never sent
        new = _short(r["new_text"], 160, r["author_tier"])
        new = _with_value(f'"{new}"', _source_value(r, new))
        text = f"{day}: changed {_subject_of(ctx, r['subject_id'], r['relation'])}; now {new}"
        lines.append(Line(r["id"], text, refs, order=(day, 0, r["id"]), deps=deps))
    ret = conn.execute(
        "SELECT r.id, r.version, r.retired_at, f.subject_id, f.relation FROM records r"
        " JOIN facts f ON f.id = r.id WHERE r.status = 'retracted' AND r.retired_at >= ?",
        (ctx.since,),
    ).fetchall()
    for r in ret:
        if not _retired_ok(ctx, r["id"]):
            continue
        day = r["retired_at"][:10]
        lines.append(
            Line(
                r["id"],
                f"{day}: withdrawn as not true {_subject_of(ctx, r['subject_id'], r['relation'])}",
                (),
                order=(day, 1, r["id"]),
                deps=(_ref(r),),  # retracted: invalidates the section, never sent
            )
        )
    dec = conn.execute(
        "SELECT r.id, r.version, r.labels, r.author_tier, r.review,"
        " coalesce(d.decided_at, r.recorded_at) AS at, d.chosen_text FROM decisions d"
        " JOIN records r ON r.id = d.id WHERE r.status = 'active'"
        " AND coalesce(d.decided_at, r.recorded_at) >= ?",
        (ctx.since,),
    ).fetchall()
    for r in _fence(ctx, dec):
        day = r["at"][:10]
        text = f"{day}: decided: {_short(r['chosen_text'], 200, r['author_tier'])}"
        lines.append(Line(r["id"], text, (_ref(r),), order=(day, 2, r["id"])))
    return sorted(lines, key=lambda ln: ln.order, reverse=True)  # newest kept first


_BUILDERS: dict[str, Callable[[_Ctx], list[Line]]] = {
    "about_you": _about_you,
    "people": _people,
    "working_now": _working_now,
    "lessons": _lessons,
    "what_changed": _what_changed,
}


def compute(conn: sqlite3.Connection, *, today: date | None = None) -> list[Section]:
    """A pure read: the five sections in cache-stable order, fenced and within budget."""
    ctx = _ctx(conn, today)
    sections = [
        Section(name, _fit(_clean(ctx, _BUILDERS[name](ctx)), SECTION_BUDGET[name]))
        for name in SECTIONS
    ]
    # the total guard: drop the lowest-priority line of the latest section until it fits
    while tokens(render(sections)) > BUDGET_TOKENS:
        last = next((s for s in reversed(sections) if s.lines), None)
        if last is None:
            break
        worst = max(last.lines, key=lambda ln: ln.prio)
        last.lines.remove(worst)
    return sections


# ---------------------------------------------------------------- persistence (derived rows)


def derived_id(section: str) -> str:
    return f"core:{section}"


def _same_refs(stored: str | None, refs: list[ReadRef]) -> bool:
    got = sorted((r.id, r.version) for r in store.load_read_set(stored))
    return got == sorted((r.id, r.version) for r in refs)


def persist(
    conn: sqlite3.Connection, sections: list[Section], *, priority: store.Priority = "background"
) -> list[str]:
    """Store changed sections as `derived` rows with their read sets; an unchanged one is left
    as it is (no commit, no version bump). Returns the rewritten derived ids. A group the
    ledger refuses or finds stale is skipped: the next build or rebuild job retries it."""
    written: list[str] = []
    for s in sections:
        if not s.lines:
            _drop_empty(conn, derived_id(s.name), priority)
            continue
        did, refs = derived_id(s.name), s.read_deps
        row = conn.execute(
            "SELECT content, read_set, valid FROM derived WHERE id = ?", (did,)
        ).fetchone()
        same = row is not None and row["valid"] == 1 and row["content"] == s.text
        if same and _same_refs(row["read_set"], refs):
            continue
        op = ProposalOp(
            op=OpKind.ADD,
            group=did,
            read_set=refs,
            payload={"derived": {"id": did, "dkind": DKIND, "dkey": s.name, "content": s.text}},
        )
        try:
            res = store.commit(
                conn,
                Proposal(ops=[op], author_tier=AuthorTier.SYSTEM),
                ledger.apply_op,
                priority=priority,
            )
        except store.OpRejected:
            continue
        if did in res.committed:
            written.append(did)
    return written


def _drop_empty(conn: sqlite3.Connection, did: str, priority: store.Priority) -> None:
    """A section with nothing left keeps no row: the old text (and its read deps) would only
    linger. Derived rows are rebuildable, so this is a plain delete, not a ledger op."""
    if conn.execute("SELECT 1 FROM derived WHERE id = ?", (did,)).fetchone() is None:
        return
    with store.write_tx(conn, priority=priority):
        conn.execute(
            "DELETE FROM read_deps WHERE holder_kind = 'derived' AND holder_id = ?", (did,)
        )
        conn.execute("DELETE FROM derived WHERE id = ?", (did,))


def build(conn: sqlite3.Connection) -> dict[str, str]:
    """Rebuild every section from the ledger and store the changed ones; section → text."""
    sections = compute(conn)
    persist(conn, sections)
    return {s.name: s.text for s in sections}


def view(
    conn: sqlite3.Connection, *, priority: store.Priority = "hot"
) -> tuple[str, list[Section]]:
    """The core as a process's first question gets it: fresh from the ledger, persisted.
    Hot priority: on the turn path a background writer would yield to other hot writers
    for up to 2 s per group (§3.1); an unchanged core commits nothing."""
    sections = compute(conn)
    persist(conn, sections, priority=priority)
    return render(sections), sections


def _on_core_change(conn: sqlite3.Connection, ids: list[str]) -> None:
    """hot seam: a correction rebuilds the core before the next turn (G9)."""
    persist(conn, compute(conn), priority="hot")


_REBUILDERS: dict[str, Callable[[sqlite3.Connection, dict[str, Any]], None]] = {}
_PREV_RUNNER: list[Callable[..., Any]] = []


def register_rebuild(dkind: str, fn: Callable[[sqlite3.Connection, dict[str, Any]], None]) -> None:
    """One `rebuild_derived` job kind serves three dkinds; the store keeps one runner per
    kind, so the other dkinds (summary, assoc_link) register their rebuild here."""
    _REBUILDERS[dkind] = fn


def _run_rebuild(conn: sqlite3.Connection, job: dict[str, Any]) -> None:
    payload = job.get("payload") or {}
    dkind = payload.get("dkind") if isinstance(payload, dict) else None
    if dkind == DKIND:
        persist(conn, compute(conn))
        return
    fn = _REBUILDERS.get(str(dkind))
    if fn is not None:
        fn(conn, payload)
        return
    if _PREV_RUNNER:
        _PREV_RUNNER[0](conn, job)
        return
    # not ours and no runner yet: back to the queue, not lost
    with store.write_tx(conn, priority="background"):
        conn.execute(
            "UPDATE jobs SET state = 'queued', not_before = ?, updated_at = ?"
            " WHERE id = ? AND state = 'running'",
            (store.lease_until(60.0), store.now(), job.get("id")),
        )


def install() -> None:
    """Register the hot hook and the rebuild runner (idempotent; runs at import)."""
    hot.on_core_change(_on_core_change)
    prev = store.job_runner(REBUILD_JOB)
    if prev is not None and prev[0] is not _run_rebuild and not _PREV_RUNNER:
        _PREV_RUNNER.append(prev[0])
    store.register_job_runner(REBUILD_JOB, _run_rebuild)


install()


# ---------------------------------------------------------------- served set and diff (§7.6)


def serve(
    conn: sqlite3.Connection,
    refs: Iterable[ReadRef],
    *,
    thread_id: str,
    session_id: str | None,
    via: ServedVia = ServedVia.CORE,
) -> int:
    """A `served` row per item that reached the session; an identical row is not repeated."""
    refs = list({r.id: r for r in refs if r.src is ReadSrc.RECORD}.values())
    if not refs:
        return 0
    kinds: dict[str, str] = {}
    for i in range(0, len(refs), 400):
        chunk = [r.id for r in refs[i : i + 400]]
        marks = ", ".join("?" for _ in chunk)
        for row in conn.execute(f"SELECT id, kind FROM records WHERE id IN ({marks})", chunk):  # noqa: S608
            kinds[row[0]] = row[1]
    ts = store.now()
    n = 0
    with store.write_tx(conn):
        for r in refs:
            if r.id not in kinds:
                continue
            cur = conn.execute(
                "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, via,"
                " created_at) SELECT ?, ?, ?, ?, ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM served"
                " WHERE thread_id = ? AND item_id = ? AND version = ? AND via = ?"
                " AND session_id IS ?)",
                (
                    thread_id,
                    session_id,
                    r.id,
                    kinds[r.id],
                    r.version,
                    via.value,
                    ts,
                    thread_id,
                    r.id,
                    r.version,
                    via.value,
                    session_id,
                ),
            )
            n += cur.rowcount
    return n


def served_core(
    conn: sqlite3.Connection, *, thread_id: str, session_id: str | None
) -> dict[str, int]:
    """What this session holds of the core: item → the newest version delivered to the thread
    (core rows of this session, plus update rows of the thread)."""
    rows = conn.execute(
        "SELECT item_id, max(version) FROM served WHERE thread_id = ?"
        " AND via IN ('core', 'update') AND item_id IN (SELECT item_id FROM served"
        " WHERE thread_id = ? AND via = 'core' AND session_id IS ?) GROUP BY item_id",
        (thread_id, thread_id, session_id),
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


def _item_text(conn: sqlite3.Connection, rid: str) -> str | None:
    r = conn.execute(
        "SELECT coalesce(f.text_en, l.text_en || ' (' || l.state || ')', le.rule_en,"
        " d.chosen_text, e.display_name, r.origin_quote) AS text, r.author_tier,"
        " f.value_text, f.quote_orig FROM records r"
        " LEFT JOIN facts f ON f.id = r.id LEFT JOIN loops l ON l.id = r.id"
        " LEFT JOIN lessons le ON le.id = r.id LEFT JOIN decisions d ON d.id = r.id"
        " LEFT JOIN entities e ON e.id = r.id WHERE r.id = ?",
        (rid,),
    ).fetchone()
    if not r or not r["text"]:
        return None
    text = _short(r["text"], 160, r["author_tier"])
    return _with_value(text, _source_value(r, text)) if r["value_text"] else text


def diff_lines(
    conn: sqlite3.Connection,
    served: dict[str, int],
    sections: list[Section] | None = None,
    *,
    today: date | None = None,
) -> tuple[list[str], list[ReadRef]]:
    """§7.6 lines for what changed since the session's core, and the refs it now holds.
    Forgotten and purged items never get a line; frozen ones are left out (§6.1)."""
    ctx = _ctx(conn, today)
    sections = compute(conn, today=today) if sections is None else sections
    out: list[str] = []
    held: list[ReadRef] = []
    current: set[str] = set()
    handled: set[str] = set()
    for s in sections:
        for ln in s.lines:
            every = (*ln.refs, *ln.deps)  # held = served rows only, never a pre-send set
            current.update(r.id for r in every)
            held.extend(every)
            if all(r.id in served and served[r.id] >= r.version for r in every):
                continue
            if any(r.id in served for r in every):
                out.append(f'[m:{ln.key}] outdated: now "{ln.text}"')
                continue
            olds = [
                o[0]
                for o in conn.execute(
                    "SELECT to_id FROM edges WHERE from_id = ? AND etype = 'supersedes'"
                    " ORDER BY to_id",
                    (ln.key,),
                )
                if o[0] in served
            ]
            if olds:
                old = olds[0]
                handled.update(olds)
                at = conn.execute("SELECT retired_at FROM records WHERE id = ?", (old,)).fetchone()
                since = f" (since {at[0][:10]})" if at and at[0] else ""
                # the handle names the old line; its retired value is never quoted
                out.append(f'[m:{old}] outdated: now "{ln.text}" [m:{ln.key}]{since}')
            else:
                out.append(f"[m:{ln.key}] new: {ln.text}")
    for rid in sorted(served):
        if rid in current or rid in handled or rid in ctx.frozen:
            continue
        r = conn.execute("SELECT status, version, retired_at FROM records WHERE id = ?", (rid,))
        row = r.fetchone()
        if row is None or row["status"] in _GONE or row["version"] <= served[rid]:
            continue
        held.append(ReadRef(id=rid, version=int(row["version"])))
        retired = row["status"] in ("retracted", "superseded")
        text = _item_text(conn, rid) if not retired and _retired_ok(ctx, rid) else None
        since = f" (since {row['retired_at'][:10]})" if row["retired_at"] else ""
        if row["status"] == "retracted":  # the handle names it; the value is never quoted
            out.append(f"[m:{rid}] retracted: not true, do not rely on it")
        elif row["status"] == "superseded":
            nxt = conn.execute(
                "SELECT e.from_id FROM edges e JOIN records n ON n.id = e.from_id"
                " WHERE e.to_id = ? AND e.etype = 'supersedes' AND n.status = 'active'"
                " ORDER BY e.from_id",
                (rid,),
            ).fetchone()
            new_ok = nxt is not None and _fence(
                ctx,
                conn.execute(
                    "SELECT id, labels, author_tier FROM records WHERE id = ?", (nxt[0],)
                ).fetchall(),
            )
            new_text = _item_text(conn, nxt[0]) if new_ok else None
            if new_text:
                out.append(f'[m:{rid}] outdated: now "{new_text}"{since}')
            else:
                out.append(f"[m:{rid}] outdated: no longer current{since}")
        elif row["status"] == "needs_review":
            out.append(f"[m:{rid}] on hold: needs re-check, do not rely on it")
        elif text:
            out.append(f'[m:{rid}] changed: now "{text}"')
        else:
            out.append(f"[m:{rid}] changed")
    return out, held


def update_block(lines: list[str]) -> str:
    return "\n".join([UPDATE_OPEN, *lines, UPDATE_CLOSE])


def diff(
    conn: sqlite3.Connection, served_version: int, *, served: dict[str, int] | None = None
) -> str | None:
    """Changed lines since a session's core, or None when over 1k tokens (restart). With no
    served map, every current item newer than `served_version` counts as changed. An empty
    string: nothing changed."""
    sections = compute(conn)
    if served is None:
        served = {
            r.id: (r.version if r.version <= served_version else served_version)
            for s in sections
            for ln in s.lines
            for r in (*ln.refs, *ln.deps)
        }
    lines, _held = diff_lines(conn, served, sections)
    if not lines:
        return ""
    block = update_block(lines)
    return block if tokens(block) <= DIFF_TOKENS else None
