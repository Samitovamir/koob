"""Scenario format (evals/memory/README.md) as strict pydantic models.

Every model forbids unknown fields: a typo in a scenario must fail the load, not be ignored.
Spec: the memory spec section 19.2; the format file is authoritative where it extends the spec.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GROUPS: dict[str, set[str]] = {
    "knows": {"as_of", "cross_lingual", "namesakes"},
    "understands": {
        "scope",
        "impressions",
        "c0.no_longer_vs_not_today",
        "c0.own_vs_for_other",
        "c0.plan_vs_done",
        "c0.habit_vs_one_off",
        "c0.observation_vs_preference",
    },
    "keeps_true": {"g2", "g3", "g5"},
    "work": {"decision", "resume", "intention"},
    "c1": {"positive", "negative"},
    "c2": {"transfer", "over_generalisation", "exception", "retirement"},
    "c3": {"one_match", "two_close", "none"},
    "poisoning": {
        "forwarded_remember",
        "fake_correction",
        "page_forget",
        "page_rule",
        "laundering",
        "alias_hijack",
        "room_speech",
    },
    "policy_leaks": {
        "banned",
        "no_dig",
        "direct_question",
        "summary_label",
        "cloud_route",
        "never_store",
        "send_turn",
    },
    "hands": {
        "changed_account",
        "fix_then_send",
        "two_namesakes",
        "intention_restart",
        "stale_token",
    },
    "safety_net": {"lands", "detector", "freeze", "failure_line", "claim_check"},
    "inspect": {"what", "where_from", "fix", "forget"},
    "e2e": {"work", "life", "study"},
    "scout": {"practice", "demo"},
}

BEHAVIOURS = {
    "answer",
    "ask_one",
    "offer",
    "silence",
    "none_then_ask",
    "refuse_line",
    "action",
    "no_action",
}
CHANNELS = {
    "answer",
    "cli_prompt",
    "core",
    "basket",
    "update_block",
    "store_tools",
    "summaries",
    "episodes",
    "assoc",
    "criteria",
    "intentions",
    "outbox",
    "actions",
    "island",
    "tg_bot",
    "cloud_input",
    "worker_input",
    "memory_files",
    "logs",
    "all",
}
DENY_REASONS = {
    "ask_which",
    "retired_value",
    "stale_token",
    "frozen",
    "duplicate",
    "unconfirmed_alias",
    "claim_without_commit",  # the Stop hook (tool: stop) blocked a memory-change claim (section 6.1)
}
TAGS = {"correction", "forget", "undo", "forever", "mode", "remember", "promise_ask"}
CHAT_MODES = {"normal", "no_dig", "banned"}
WORKER_JOBS = {"extract", "summary", "episode", "assoc", "criteria", "consolidate", "night"}
CRASH_POINTS = {
    "after_tg_delivered_before_post_tool",
    "between_ledger_commit_and_vector_delete",
    "mid_cascade",
    "during_purge",
    "mid_worker_commit",
}
CANARY_ALLOWED_RE = re.compile(
    r"^(none|trash_until_purge|history|turn_conversation:[A-Za-z0-9_\-]+)$"
)
REL_RE = re.compile(r"^\+(\d+)([mhdw])$")
TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\.(lat|cyr|dig)\s*\}\}")
TYPED_KEYS = {"date", "money", "cur", "time"}


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


# A value to look for: a string, a typed matcher, or a list of alternatives (any one counts).
Alt = str | dict[str, Any] | list[Any]


def _check_alt(v: Any, where: str) -> Any:
    if isinstance(v, bool):
        raise ValueError(f"{where}: a bare yes/no/on/off was parsed as a boolean; quote it")
    if isinstance(v, int | float):
        # A bare 6000 could mean the text "6000" or an amount in any spelling: make it explicit.
        raise ValueError(
            f"{where}: bare number {v!r}; quote it as text or use a typed matcher "
            "({money: ..., cur: ...}, {date: ...}, {time: ...})"
        )
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        bad = set(v) - TYPED_KEYS
        if bad or not (set(v) & {"date", "money", "time"}):
            raise ValueError(f"{where}: bad typed matcher {sorted(v)}")
        return v
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, list):
                raise ValueError(f"{where}: alternatives nest only one level")
            out.append(_check_alt(x, where))
        return out
    raise ValueError(
        f"{where}: value must be a string, a typed matcher or a list, got {type(v).__name__}"
    )


def _alts(v: list[Any], where: str) -> list[Any]:
    return [_check_alt(x, where) for x in v]


# ------------------------------------------------------------------ pieces


class ClockSpec(_S):
    start: datetime
    tz: str


class Setup(_S):
    workers: Literal["scheduled", "manual"] = "scheduled"
    chat_policy: dict[str, str] = Field(default_factory=dict)
    routing: str | dict[str, list[str]] = "default"
    known: list[Any] = Field(default_factory=list)
    hands_dryrun: bool = False

    @field_validator("chat_policy")
    @classmethod
    def _modes(cls, v: dict[str, str]) -> dict[str, str]:
        for chat, mode in v.items():
            if mode not in CHAT_MODES:
                raise ValueError(f"chat_policy {chat}: unknown mode {mode!r}")
        return v


class Assumes(_S):
    trash_delay_h: int | None = None
    no_dig_backfill: str | None = None
    forget_blanks_island: bool | None = None
    room_confirm: bool | None = None
    cloud_for_own_sensitive: bool | None = None


class Canary(_S):
    id: str
    lat: str
    cyr: str
    dig: str
    allowed: list[str] = Field(default_factory=lambda: ["none"])

    @field_validator("allowed")
    @classmethod
    def _allowed(cls, v: list[str]) -> list[str]:
        for a in v:
            if not CANARY_ALLOWED_RE.match(a):
                raise ValueError(f"unknown canary allowance {a!r}")
        return v


class Contact(_S):
    en: str
    ru: str | None = None
    aliases: list[str] = Field(default_factory=list)
    gender: str | None = None
    role: str | None = None
    minor: bool = False
    etype: Literal["person", "org"] | None = None  # None = person; README "Fields defined here"
    ids: dict[str, Any] = Field(default_factory=dict)


class Read(_S):
    tool: Literal["mail_read", "web_fetch", "file_read", "tg_read", "screen_look"]
    source: str
    chat: str | None = None
    text: str


# ------------------------------------------------------------------ events


class _Ev(_S):
    id: str | None = None
    at: str | None = (
        None  # absolute ISO local time, or relative "+10m"; YAML datetimes are normalised
    )
    during: str | None = None

    @field_validator("at", mode="before")
    @classmethod
    def _at(cls, v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, str) and not REL_RE.match(v):
            try:
                datetime.fromisoformat(v)
            except ValueError as exc:
                raise ValueError(
                    f"bad time {v!r}: use 2026-03-02T10:15 or +10m/+3h/+2d/+1w"
                ) from exc
        return v


class UserMsg(_Ev):
    ev: Literal["user_msg", "correction", "forget"]
    text: str
    channel: Literal["island", "voice", "speaker", "app", "tg_bot"] = "island"
    thread: str = "main"
    turn: Literal["owner", "task"] = "owner"
    reads: list[Read] = Field(default_factory=list)
    reply: str | None = None
    tag: str | None = None
    actual_speaker: str | None = None

    @field_validator("tag")
    @classmethod
    def _tag(cls, v: str | None) -> str | None:
        if v is not None and v not in TAGS:
            raise ValueError(f"unknown tag {v!r}")
        return v


class OtherMsg(_Ev):
    ev: Literal["other_msg"]
    from_: str = Field(alias="from")
    chat: str
    msg_id: int | str
    text: str | None = None
    reply_to: str | None = None
    fwd_from: str | None = None
    media: Literal["photo", "sticker", "voice"] | None = None
    caption: str | None = None
    sender_tg_id: int | str | None = None
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Redeliver(_Ev):
    ev: Literal["redeliver"]
    of: str


class ModeChange(_Ev):
    ev: Literal["mode_change"]
    chat: str
    mode: str
    text: str | None = None

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in CHAT_MODES:
            raise ValueError(f"unknown mode {v!r}")
        return v


class Worker(_Ev):
    ev: Literal["worker"]
    job: str
    snapshot_at: str | None = None
    commit_after: str | None = None

    @field_validator("job")
    @classmethod
    def _job(cls, v: str) -> str:
        if v not in WORKER_JOBS:
            raise ValueError(f"unknown worker job {v!r}")
        return v


class Restart(_Ev):
    ev: Literal["restart"]
    kind: Literal["session", "process", "app"]


class Crash(_Ev):
    ev: Literal["crash"]
    point: str

    @field_validator("point")
    @classmethod
    def _p(cls, v: str) -> str:
        if v not in CRASH_POINTS:
            raise ValueError(f"unknown crash point {v!r}")
        return v


class Backup(_Ev):
    ev: Literal["backup", "restore"]
    name: str


class Corrupt(_Ev):
    ev: Literal["corrupt"]
    store: str


class Advance(_Ev):
    ev: Literal["advance"]
    by: str

    @field_validator("by")
    @classmethod
    def _by(cls, v: str) -> str:
        if not REL_RE.match(v):
            raise ValueError(f"bad duration {v!r}")
        return v


class NoiseSpan(_S):
    from_: date = Field(alias="from")
    to: date
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Protect(_S):
    values: list[str] = Field(default_factory=list)
    slots: list[dict[str, str]] = Field(default_factory=list)


NOISE_KINDS = {
    "smalltalk",
    "sticker",
    "emoji_only",
    "bot_spam",
    "cron",
    "link_share",
    "logistics",
    "media_caption",
    "work_chatter",
    "namesake",
    "retelling",
    "contradiction_other",
    "distractor_plan",
    "distractor_wish",
    "cancelled_meeting",
    "forwarded_mail",
    "tool_output",
}


class Noise(_Ev):
    ev: Literal["noise"]
    seed: int
    count: int = Field(ge=0)
    span: NoiseSpan
    chats: dict[str, float] = Field(default_factory=dict)
    kinds: dict[str, float] = Field(default_factory=dict)
    lang: dict[str, float] = Field(default_factory=lambda: {"ru": 0.8, "en": 0.1, "mixed": 0.1})
    cast: list[str] = Field(default_factory=list)
    protect: Protect = Field(default_factory=Protect)
    kind: str | None = None  # appears in ev_after copies of the format file; unused

    @field_validator("kinds")
    @classmethod
    def _kinds(cls, v: dict[str, float]) -> dict[str, float]:
        bad = set(v) - NOISE_KINDS
        if bad:
            raise ValueError(f"unknown noise kinds {sorted(bad)}")
        return v


Event = Annotated[
    UserMsg
    | OtherMsg
    | Redeliver
    | ModeChange
    | Worker
    | Restart
    | Crash
    | Backup
    | Corrupt
    | Advance
    | Noise,
    Field(discriminator="ev"),
]


# ------------------------------------------------------------------ probes


class AskSpec(_S):
    must_name: list[Any] = Field(default_factory=list)
    features: list[Any] = Field(default_factory=list)
    max_questions: int = 1


class OfferSpec(_S):
    grounds: list[Any] = Field(default_factory=list)
    max: int = 1
    check_phrase: bool = False


class Forbidden(_S):
    values: list[Any]
    channels: list[str]
    allow: Literal["history"] | None = None

    @field_validator("channels")
    @classmethod
    def _ch(cls, v: list[str]) -> list[str]:
        bad = set(v) - CHANNELS
        if bad:
            raise ValueError(f"unknown channels {sorted(bad)}")
        return v


class ActionCheck(_S):
    tool: str
    target: str = "any"
    count: int | None = None
    text_must_include: list[Any] = Field(default_factory=list)
    text_must_not_include: list[Any] = Field(default_factory=list)


class Denied(_S):
    tool: str
    target: str | None = None
    reason_any: list[str] = Field(default_factory=list)

    @field_validator("reason_any")
    @classmethod
    def _r(cls, v: list[str]) -> list[str]:
        bad = set(v) - DENY_REASONS
        if bad:
            raise ValueError(f"unknown denial reasons {sorted(bad)}")
        return v


class Lines(_S):
    of: str | None = None
    must: list[Any] = Field(default_factory=list)
    must_not: list[Any] = Field(default_factory=list)


class NotifyMust(_S):
    contains: list[Any] = Field(default_factory=list)
    count: int | None = None
    count_min: int | None = None
    count_max: int | None = None


class Notifications(_S):
    window: list[Any]
    must: list[NotifyMust] = Field(default_factory=list)
    must_not_contain: list[Any] = Field(default_factory=list)


class Scan(_S):
    canaries: list[str]
    stores: Literal["all"] = "all"


class Journal(_S):
    forget_entries: int


class RecordCheck(_S):
    where: dict[str, Any]
    expect: dict[str, Any] = Field(default_factory=dict)
    count: int | str | None = None  # N, ">=N", "<=N"


WHERE_KEYS = {
    "kind", "subject", "about", "family", "relation", "value", "speaker", "status", "polarity",
    "stability", "fkind", "author_tier", "source_chat", "target", "dkind",
}  # fmt: skip
EXPECT_KEYS = {
    "polarity", "stability", "status", "author_tier", "review", "labels", "labels_any", "applicability",
    "provenance", "domain", "fkind", "speaker", "subject_or_speaker", "valid_to", "valid_from",
    "quote_orig", "read_set_includes", "evidence_count_min", "domains", "exceptions_count_min",
    "state", "value", "dkind", "confirmed", "ask", "target",
}  # fmt: skip


class Expect(_S):
    behaviour: str | None = None
    behaviour_any: list[str] = Field(default_factory=list)
    must_include: list[Any] = Field(default_factory=list)
    must_not_include: list[Any] = Field(default_factory=list)
    forbidden: list[Forbidden] = Field(default_factory=list)
    records: list[RecordCheck] = Field(default_factory=list)
    ask: AskSpec | None = None
    offer: OfferSpec | None = None
    actions: list[ActionCheck] = Field(default_factory=list)
    denied: list[Denied] = Field(default_factory=list)
    lines: Lines | None = None
    judge: str | None = None
    scan: Scan | None = None
    notifications: Notifications | None = None
    journal: Journal | None = None
    chat_policy: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Expect:
        for b in [self.behaviour, *self.behaviour_any]:
            if b is not None and b not in BEHAVIOURS:
                raise ValueError(f"unknown behaviour {b!r}")
        self.must_include = _alts(self.must_include, "must_include")
        self.must_not_include = _alts(self.must_not_include, "must_not_include")
        for f in self.forbidden:
            f.values = _alts(f.values, "forbidden.values")
        for rc in self.records:
            bad = set(rc.where) - WHERE_KEYS
            if bad:
                raise ValueError(f"records.where: unknown keys {sorted(bad)}")
            for k in rc.expect:
                if k not in EXPECT_KEYS and k.split(".")[0] not in {"state"}:
                    raise ValueError(f"records.expect: unknown key {k!r}")
            if isinstance(rc.count, str) and not re.fullmatch(r"(>=|<=)?\d+", rc.count):
                raise ValueError(f"records.count: bad {rc.count!r}")
        return self


class Budget(_S):
    basket_ms_p95: float = 150
    first_word_s: float = 3
    basket_tokens_max: int = 1500
    offers_max: int = 1
    model_calls_max: int | None = None


class Followup(_S):
    say: str
    expect: Expect


class Probe(_S):
    id: str
    at: datetime | None = None
    after: str | None = None
    thread: str = "main"
    turn: Literal["owner", "task"] = "owner"
    level: Literal["memory", "e2e", "both"] = "both"
    gate: bool = False
    # Expected route of the answer (README: route): `view` (core + basket + search), `tool`
    # (the memory tools too), or one tool. None = the group's default (routes.GROUP_ROUTE).
    route: Literal["view", "tool", "memory_what", "memory_where_from", "memory_resolve"] | None = (
        None
    )
    ask: list[str] = Field(default_factory=list)
    # `correction`: each wording is itself the user's correction and the scenario expects it to land;
    # the memory-level replay (bench, compare) reads the view after that turn is processed,
    # for every system alike. A grader hint, like user_msg `tag`: the system never sees it.
    tag: Literal["correction"] | None = None
    expect: Expect | None = None
    expect_any: list[Expect] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    followup: list[Followup] = Field(default_factory=list)
    ev_after: list[Annotated[Noise | Restart, Field(discriminator="ev")]] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _check(self) -> Probe:
        if self.at is None and self.after is None:
            raise ValueError(f"probe {self.id}: needs `at` or `after`")
        if self.expect is None and not self.expect_any:
            raise ValueError(f"probe {self.id}: needs `expect` or `expect_any`")
        return self


# ------------------------------------------------------------------ scenario


class Scenario(_S):
    id: str
    group: str
    subgroup: str
    set: Literal["dev", "hidden"]
    title: str
    covers: list[str] = Field(default_factory=list)
    persona: str
    assumes: Assumes = Field(default_factory=Assumes)
    clock: ClockSpec
    date_order: Literal["dmy", "mdy"] | None = None  # numeric dates; default: persona tz/langs
    setup: Setup = Field(default_factory=Setup)
    contacts: dict[str, Contact] = Field(default_factory=dict)
    canaries: list[Canary] = Field(default_factory=list)
    timeline: list[Event]
    probes: list[Probe]
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Scenario:
        if self.group not in GROUPS:
            raise ValueError(f"unknown group {self.group!r}")
        if self.subgroup not in GROUPS[self.group]:
            raise ValueError(f"group {self.group}: unknown subgroup {self.subgroup!r}")
        if not self.probes:
            raise ValueError("a scenario needs at least one probe")
        ev_ids = [e.id for e in self.timeline if e.id]
        probe_ids = [p.id for p in self.probes]
        if len(set(ev_ids)) != len(ev_ids):
            raise ValueError("duplicate event ids")
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError("duplicate probe ids")
        known = set(ev_ids)
        for e in self.timeline:
            for attr in ("during", "of", "snapshot_at", "commit_after"):
                ref = getattr(e, attr, None)
                if ref and ref not in known:
                    raise ValueError(f"event {e.id}: {attr} -> unknown event {ref!r}")
        for p in self.probes:
            if p.after and p.after not in known | set(probe_ids):
                raise ValueError(f"probe {p.id}: after -> unknown id {p.after!r}")
            if p.at and p.after and p.after not in probe_ids:
                # `at` with `after: <probe>` sets the clock of a chained ask; with an event it is ambiguous
                raise ValueError(f"probe {p.id}: `at` and `after: <event>` together; keep one")
        canary_ids = {c.id for c in self.canaries}
        for ref in _templates(self.model_dump(by_alias=True, mode="json")):
            if ref not in canary_ids:
                raise ValueError(f"template refers to undeclared canary {ref!r}")
        for p in self.probes:
            for ex in [p.expect, *p.expect_any, *(f.expect for f in p.followup)]:
                if ex and ex.scan:
                    missing = set(ex.scan.canaries) - canary_ids
                    if missing:
                        raise ValueError(
                            f"probe {p.id}: scan names undeclared canaries {sorted(missing)}"
                        )
        for c in self.canaries:
            for a in c.allowed:
                if a.startswith("turn_conversation:") and a.split(":", 1)[1] not in known:
                    raise ValueError(f"canary {c.id}: allowance names unknown event {a!r}")
        return self


def _templates(obj: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, str):
        out |= {m.group(1) for m in TEMPLATE_RE.finditer(obj)}
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == "canaries":
                continue  # declarations hold the raw tokens
            out |= _templates(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _templates(v)
    return out
