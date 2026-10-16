"""Typed records of the memory ledger: enums, the labels bitmask and one pydantic model per
table of the memory spec §3.2, plus the shared proposal and turn shapes.

Spec: §1 (terms), §3.2 (schema), §4 (records and G4 dimensions), §5.3 (labels), §6.3
(proposals), §8 (tiers, provenance), §12 (intentions), §16 (actions).

Unknown stays None: writers MUST NOT invent values (§4.1).
"""

from __future__ import annotations

from enum import IntFlag, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------- enums


class RecordKind(StrEnum):
    ENTITY = "entity"
    FACT = "fact"
    PREFERENCE = "preference"  # lives in `facts` with relation = 'prefers'
    IMPRESSION = "impression"
    DECISION = "decision"
    WORK_STATE = "work_state"
    INTENTION = "intention"
    LOOP = "loop"
    LESSON = "lesson"
    EPISODE = "episode"
    CRITERION = "criterion"


class Status(StrEnum):
    """§4.2 lifecycle."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    NEEDS_REVIEW = "needs_review"
    FORGOTTEN = "forgotten"
    PURGED = "purged"


USABLE: frozenset[Status] = frozenset({Status.ACTIVE, Status.ARCHIVED})


class AuthorTier(StrEnum):
    """§13.1."""

    USER = "user"
    USER_CONFIRMED = "user_confirmed"
    USER_ROOM = "user_room"
    OTHER = "other"
    TOOL = "tool"
    SYSTEM = "system"


def tier_rank(tier: AuthorTier, *, about_self: bool = False) -> int:
    """§8.2 order: user = user_confirmed > user_room > other about themselves > other >
    tool > system. Higher wins; equal may supersede only for the same speaker."""
    if tier in (AuthorTier.USER, AuthorTier.USER_CONFIRMED):
        return 50
    if tier is AuthorTier.USER_ROOM:
        return 40
    if tier is AuthorTier.OTHER:
        return 30 if about_self else 20
    if tier is AuthorTier.TOOL:
        return 10
    return 0


class Label(IntFlag):
    """§5.3 sensitivity bits. Bits 9+ are reserved."""

    NONE = 0
    HEALTH = 1 << 0
    MONEY = 1 << 1
    IDENTITY_DOCS = 1 << 2
    INTIMATE = 1 << 3
    LOCATION_PRECISE = 1 << 4
    THIRD_PARTY_PRIVATE = 1 << 5
    LEGAL = 1 << 6
    MINOR = 1 << 7
    UNKNOWN_SENSITIVE = 1 << 8  # fail-closed


LABEL_NAMES: tuple[str, ...] = (
    "health",
    "money",
    "identity_docs",
    "intimate",
    "location_precise",
    "third_party_private",
    "legal",
    "minor",
    "unknown_sensitive",
)
# §5.5 default: every label but location_precise is sensitive.
SENSITIVE_DEFAULT: Label = Label(sum(1 << i for i in range(9))) & ~Label.LOCATION_PRECISE


def label_from_names(names: list[str] | tuple[str, ...]) -> Label:
    out = Label.NONE
    for name in names:
        out |= Label[name.upper()]
    return out


class FKind(StrEnum):
    STATE = "state"
    EVENT = "event"
    PLAN = "plan"
    CONDITION = "condition"
    OPINION = "opinion"


class Polarity(StrEnum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"


class Stability(StrEnum):
    LASTING = "lasting"
    ONE_OFF = "one_off"
    HABIT = "habit"


class Cardinality(StrEnum):
    SINGLE = "single"
    MULTI = "multi"


class AliasKind(StrEnum):
    NAME = "name"
    DIMINUTIVE = "diminutive"
    HANDLE = "handle"
    NICKNAME = "nickname"
    TRANSLIT = "translit"


class Provenance(StrEnum):
    """§8.1 member links."""

    STATED = "stated"
    RULE = "rule"
    INFERRED = "inferred"


class EdgeType(StrEnum):
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"
    MENTIONS = "mentions"
    ABOUT = "about"


class DerivedKind(StrEnum):
    CORE_SECTION = "core_section"
    SUMMARY = "summary"
    ASSOC_LINK = "assoc_link"


class RawStatus(StrEnum):
    ACTIVE = "active"
    FORGOTTEN = "forgotten"
    PURGED = "purged"


class ChatMode(StrEnum):
    """§5.2."""

    NORMAL = "normal"
    NO_DIG = "no_dig"
    BANNED = "banned"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    PARKED = "parked"
    NEEDS_ATTENTION = "needs_attention"


class ActionState(StrEnum):
    """§16.2."""

    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"
    UNKNOWN = "unknown"
    UNDONE = "undone"


class IntentionState(StrEnum):
    """§12 lifecycle."""

    ARMED = "armed"
    FIRED = "fired"
    DONE = "done"
    CANCELLED = "cancelled"
    NEEDS_ATTENTION = "needs_attention"


class TriggerKind(StrEnum):
    MESSAGE_FROM = "message_from"
    STATUS_CHANGE = "status_change"
    DATE_REACHED = "date_reached"
    SOFT_OPPORTUNITY = "soft_opportunity"


class LoopState(StrEnum):
    OPEN = "open"
    DONE = "done"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    UNCLEAR = "unclear"


class CriterionStatus(StrEnum):
    HYPOTHESIS = "hypothesis"
    CONFIRMED = "confirmed"
    RETIRED = "retired"


class Domain(StrEnum):
    """§15.2 closed list."""

    PURCHASE = "purchase"
    TRAVEL = "travel"
    STUDY = "study"
    WORK = "work"
    CAREER = "career"
    HOME = "home"
    HEALTH = "health"
    SOCIAL = "social"
    MONEY = "money"
    TOOLS = "tools"


class EpisodeRole(StrEnum):
    PARTICIPANT = "participant"
    PLACE = "place"
    CAUSE = "cause"
    OUTCOME = "outcome"
    ABOUT = "about"


class EventKind(StrEnum):
    """`events.kind` (§7.7). `write` is every commit that is not one of the named kinds."""

    WRITE = "write"
    FIX = "fix"
    FORGET = "forget"
    MODE = "mode"
    RESTORE = "restore"
    UNDO = "undo"


class ServedVia(StrEnum):
    CORE = "core"
    BASKET = "basket"
    TOOL = "tool"
    WRITE = "write"
    UPDATE = "update"
    SESSION = "session"  # the after-turn adapter's session-id row (§9.2)


class ForgetScope(StrEnum):
    ITEM = "item"
    ABOUT_ENTITY = "about_entity"
    CHAT = "chat"


class ForgetStage(StrEnum):
    STOPPING = "stopping"  # v0.4 (§9.3): write-ahead line before the stop-use commit
    STOPPED = "stopped"
    UNDONE = "undone"
    PURGED = "purged"


class TurnOrigin(StrEnum):
    OWNER = "owner"
    TASK = "task"
    DUTY = "duty"
    EVENT = "event"


class Route(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class OpKind(StrEnum):
    """Delta ops only (§6.4)."""

    ADD = "add"
    UPDATE = "update"
    SUPERSEDE = "supersede"
    RETRACT = "retract"
    MERGE = "merge"
    SPLIT = "split"
    FLAG = "flag"
    LINK = "link"
    FORGET = "forget"


class ReadSrc(StrEnum):
    RECORD = "record"
    RAW = "raw"
    DERIVED = "derived"


# ---------------------------------------------------------------- v0.4 conventions

OWNER_ENTITY_ID = "self"
"""§4.5: the owner's own entity has this reserved id; facts about him use it as `subject_id`."""

ISLAND_CHAT = "island"
"""§6.2: reserved chat id of the Cube's own conversation; stored as NULL, never has a mode."""

SOURCE_FAMILIES: frozenset[str] = frozenset(
    {
        "owner",
        "cube",
        "tool",
        "telegram",
        "bot",
        "report",
        "screen",
        "file",
        "mail",
        "web",
        "import",
    }
)
"""§6.2 closed `source_kind` families; a kind is `<family>` or `<family>:<detail>`."""


def message_event_key(source: str, chat_id: str, message_id: str) -> str:
    """§12 event key of a message: `<source>:<chat_id>:<message_id>` (`telegram:tg:777:m-42`)."""
    return f"{source}:{chat_id}:{message_id}"


def record_event_key(record_id: str, version: int) -> str:
    """§12 event key of a slot change: `record:<record_id>:<version>`."""
    return f"record:{record_id}:{version}"


def date_event_key(day: str) -> str:
    """§12 event key of a clock tick: `date:<YYYY-MM-DD>` (the tick's UTC date)."""
    return f"date:{day}"


# ---------------------------------------------------------------- shared shapes


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class ReadRef(_M):
    """One (id, version) pair of a read set (§7.2); raw uses its `rev`, derived ignores it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    version: int
    src: ReadSrc = ReadSrc.RECORD


class Applicability(_M):
    """Where a fact applies (§4.1). Preferences MUST carry a key (§4.3)."""

    for_whom: str | None = None
    task: str | None = None
    audience: str | None = None
    situation: str | None = None
    period: str | None = None
    channel: str | None = None
    general: bool | None = None
    unstated: bool | None = None

    def has_key(self) -> bool:
        return any(v is not None for v in self.model_dump().values())


class StableId(_M):
    """§13.3: from user / user_confirmed tier or platform metadata only."""

    kind: Literal["tg_user_id", "phone_hash", "email", "path", "url"]
    value: str
    tier: AuthorTier | Literal["platform"]
    source: str | None = None


# ---------------------------------------------------------------- records


class RecordHeader(_M):
    """`records` row (§3.2). Version-bearing: status, labels, author_tier and the
    kind row's content columns (§7.1); not importance, confidence, review or stats."""

    id: str
    kind: RecordKind
    version: int = 0
    status: Status = Status.ACTIVE
    review: bool = False
    labels: Label = Label.NONE
    labels_final: bool = False
    author_tier: AuthorTier
    author_entity_id: str | None = None
    importance: float | None = None
    confidence: float | None = None
    recorded_at: str
    retired_at: str | None = None
    scope_key: str | None = None
    pinned_category: str | None = None
    origin_quote: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def usable(self) -> bool:
        return self.status in USABLE


class RecordStats(_M):
    record_id: str
    uses: int = 0
    last_used_at: str | None = None
    heat: float = 0.0


class Relation(_M):
    """Closed vocabulary row (§4.1)."""

    relation: str
    family: str
    cardinality: Cardinality
    value_type: str


class Entity(RecordHeader):
    kind: Literal[RecordKind.ENTITY] = RecordKind.ENTITY
    etype: str  # person, place, org, matter, thing, account, ...
    display_name: str
    stable_ids: list[StableId] = Field(default_factory=list)


class Alias(_M):
    id: str
    entity_id: str
    alias: str
    alias_norm: str
    script: str
    kind: AliasKind
    tier: AuthorTier
    confirmed: bool = False
    created_at: str | None = None
    updated_at: str | None = None


class Fact(RecordHeader):
    """G4 dimensions (§4.1)."""

    kind: Literal[RecordKind.FACT, RecordKind.PREFERENCE] = RecordKind.FACT
    subject_id: str
    relation: str
    value_text: str | None = None
    value_norm: str | None = None
    value_entity_id: str | None = None
    text_en: str
    keywords_ru: str | None = None
    keywords_en: str | None = None
    quote_orig: str | None = None
    lang_orig: str | None = None
    fkind: FKind
    polarity: Polarity = Polarity.AFFIRMED
    valid_from: str | None = None
    valid_to: str | None = None
    stability: Stability | None = None
    applicability: Applicability | None = None
    speaker_entity_id: str | None = None
    source_raw_id: str | None = None


class Preference(Fact):
    """A fact with relation 'prefers' and a required applicability key (§4.3)."""

    kind: Literal[RecordKind.PREFERENCE] = RecordKind.PREFERENCE
    relation: str = "prefers"
    applicability: Applicability

    @field_validator("applicability")
    @classmethod
    def _keyed(cls, v: Applicability) -> Applicability:
        if not v.has_key():
            raise ValueError('preference needs a scope, {"general": true} or {"unstated": true}')
        return v


class Impression(RecordHeader):
    """Soft, dated, quoted, decaying; never promoted to a fact (§4.3)."""

    kind: Literal[RecordKind.IMPRESSION] = RecordKind.IMPRESSION
    about_entity_id: str
    text_en: str
    quote_orig: str | None = None
    observed_at: str
    decay_half_life_days: float = 90.0


class Alternative(_M):
    text: str
    reasons: list[str] = Field(default_factory=list)


class Decision(RecordHeader):
    kind: Literal[RecordKind.DECISION] = RecordKind.DECISION
    matter_id: str | None = None
    domain: Domain | None = None
    chosen_text: str
    alternatives: list[Alternative] = Field(default_factory=list)
    revision_conditions: list[dict[str, Any]] = Field(default_factory=list)
    decided_at: str | None = None


class Justification(_M):
    """One member link of a justification group (§8.1): AND within `group_no`, OR across."""

    id: str
    target_record_id: str
    group_no: int
    member_record_id: str  # a record id or a raw chunk id (a raw root)
    provenance: Provenance
    created_at: str | None = None


class WorkStateBody(_M):
    """§14: JSON, never prose."""

    goal: str | None = None
    direction: str | None = None
    rejected: list[Alternative] = Field(default_factory=list)
    dead_ends: list[Alternative] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    next_step: str | None = None
    premises: list[ReadRef] = Field(default_factory=list)


class WorkState(RecordHeader):
    kind: Literal[RecordKind.WORK_STATE] = RecordKind.WORK_STATE
    matter_id: str
    state: WorkStateBody
    last_resumed_at: str | None = None


class Trigger(_M):
    """§12 triggers; `semantic` is the only clause a model judges."""

    kind: TriggerKind
    entity_id: str | None = None
    after_ts: str | None = None
    subject_id: str | None = None
    relation: str | None = None
    to: str | None = None
    ts: str | None = None
    goal: str | None = None
    semantic: str | None = None


class Intention(RecordHeader):
    kind: Literal[RecordKind.INTENTION] = RecordKind.INTENTION
    trigger: Trigger
    completion: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    state: IntentionState = IntentionState.ARMED
    fired_at: str | None = None
    outcome: str | None = None


class IntentionEvent(_M):
    intention_id: str
    event_key: str  # (source, chat, message) | (record id, version) | date
    created_at: str | None = None


class Loop(RecordHeader):
    kind: Literal[RecordKind.LOOP] = RecordKind.LOOP
    owner: str  # 'user', 'cube' or an entity id
    text_en: str
    due: str | None = None
    state: LoopState = LoopState.OPEN


class Lesson(RecordHeader):
    """Created only from user or user_confirmed tier (§4.3)."""

    kind: Literal[RecordKind.LESSON] = RecordKind.LESSON
    rule_en: str
    applicability: Applicability | None = None
    helpful: int = 0
    harmful: int = 0


class Criterion(RecordHeader):
    """§15.2: created `inferred` as a hypothesis."""

    kind: Literal[RecordKind.CRITERION] = RecordKind.CRITERION
    text_en: str
    applicability: Applicability | None = None
    evidence: list[str] = Field(default_factory=list)
    exceptions: list[dict[str, Any]] = Field(default_factory=list)
    # column `criteria.status`; named apart from the header's lifecycle `status`
    criterion_status: CriterionStatus = CriterionStatus.HYPOTHESIS


class Episode(RecordHeader):
    kind: Literal[RecordKind.EPISODE] = RecordKind.EPISODE
    event_type: str | None = None
    span_start: str | None = None
    span_end: str | None = None
    place: str | None = None
    outcome: str | None = None
    summary_en: str


class EpisodeLink(_M):
    episode_id: str
    role: EpisodeRole
    target_id: str


AnyRecord = (
    Entity
    | Fact
    | Preference
    | Impression
    | Decision
    | WorkState
    | Intention
    | Loop
    | Lesson
    | Criterion
    | Episode
)


# ---------------------------------------------------------------- raw, lineage, derived


class RawChunk(_M):
    """One normalized message (§6.2). Evidence, never instruction."""

    id: str
    status: RawStatus = RawStatus.ACTIVE
    rev: int = 1
    source_kind: str
    source_ref: str | None = None
    chat_id: str | None = None
    turn_id: str | None = None
    author_tier: AuthorTier
    author_entity_id: str | None = None
    ts: str
    text: str | None
    content_hmac: str
    labels: Label = Label.NONE
    labels_final: bool = False
    cloud_seen: bool = False
    policy_version: int
    quarantined: bool = False
    junk: bool = False


class TurnSource(_M):
    turn_id: str
    source_kind: str
    chat_id: str  # a chat id, or a path for files
    requested: bool = False


class Edge(_M):
    from_id: str
    to_id: str
    etype: EdgeType


class Derived(_M):
    id: str
    dkind: DerivedKind
    dkey: str | None = None
    content: str
    read_set: list[ReadRef]
    labels: Label = Label.NONE
    built_at: str
    valid: bool = True


class Served(_M):
    thread_id: str
    session_id: str | None = None
    item_id: str
    item_kind: str
    version: int
    turn_id: str | None = None
    via: ServedVia


class PendingUpdate(_M):
    thread_id: str
    item_id: str
    old_version: int
    new_version: int


class Suppression(_M):
    hmac: str
    forget_id: str
    created_at: str | None = None


class RawRetraction(_M):
    raw_id: str
    record_id: str
    retracted_value_norm: str


class RetiredValue(_M):
    hmac: str
    record_id: str


class TrashItem(_M):
    id: str
    forget_id: str
    store: str
    row_ref: str
    payload: str | None
    until: str


class Job(_M):
    id: str
    jkind: str
    payload: dict[str, Any]
    state: JobState = JobState.QUEUED
    attempts: int = 0
    not_before: str | None = None
    idempotency_key: str


class Action(_M):
    """Outbox-first action (§16.2)."""

    id: str
    action_key: str
    origin_key: str
    seq: int
    tool: str
    target_entity_id: str | None = None
    stable_target: str
    summary_en: str | None = None
    state: ActionState = ActionState.PENDING
    read_set: list[ReadRef] = Field(default_factory=list)
    turn_id: str | None = None
    intention_id: str | None = None
    reserved_at: str
    closed_at: str | None = None


class ChatPolicy(_M):
    chat_id: str
    mode: ChatMode = ChatMode.NORMAL
    policy_version: int


class AdmitWindow(_M):
    """§5.6: keyed by the send's action_key or by an intention."""

    id: str
    chat_id: str
    sender_id: str | None = None
    after_ts: str
    until_ts: str | None = None
    reason: str
    action_key: str | None = None
    intention_id: str | None = None
    cube_message_id: str | None = None  # v0.4 §5.6: the send's message id (migration 0002)

    @model_validator(mode="after")
    def _keyed(self) -> AdmitWindow:
        if self.action_key is None and self.intention_id is None:
            raise ValueError("an admit window needs an action_key or an intention_id")
        return self


class Event(_M):
    id: int
    ts: str
    kind: EventKind
    memory_version: int
    item_ids: list[str] = Field(default_factory=list)


class AuditLine(_M):
    ts: str
    event: str
    record_id: str | None = None
    actor: str


class Meta(_M):
    memory_version: int
    policy_version: int
    schema_version: int
    embedder_id: str | None = None
    sweep_watermark: int = 0


class JournalLine(_M):
    """`journal/forget.jsonl` (§9.3). No content, ever."""

    ts: str
    stage: ForgetStage
    forget_id: str
    record_ids: list[str] = Field(default_factory=list)
    raw_hmacs: list[str] = Field(default_factory=list)
    suppression_hmacs: list[str] = Field(default_factory=list)
    turn_ids: list[str] = Field(default_factory=list)
    scope_done: list[str] = Field(default_factory=list)
    memory_version: int
    # v0.4 (§9.3): what replay needs to redo the same forget, not a wider one. None = a v0.3
    # line, replayed as `item`.
    scope: ForgetScope | None = None
    target_ids: list[str] = Field(default_factory=list)  # what the user named: records, entities, a chat
    masked_ids: list[str] = Field(default_factory=list)  # records and raw masked, not forgotten
    until: str | None = None  # end of the trash window; a restore keeps it


# ---------------------------------------------------------------- proposals (§6.3)


class ProposalOp(_M):
    op: OpKind
    target: str | None = None  # the record the op changes; None for a new record
    group: str  # entity or matter id: ops of one group commit together
    read_set: list[ReadRef] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class Proposal(_M):
    """`propose(snapshot) -> Proposal`; `store.commit` validates and applies it (§6.3)."""

    ops: list[ProposalOp]
    model_id: str | None = None  # None for hot-path and code writes
    prompt_id: str | None = None
    author_tier: AuthorTier = AuthorTier.SYSTEM


# ---------------------------------------------------------------- the turn as memory sees it


class TurnCtx(_M):
    """What the adapters hand memory about the current turn. Ids and flags only, plus the
    user's own words (§1 terms; guard.TurnState.owner_words / task_words)."""

    turn_id: str
    thread_id: str
    session_id: str | None = None
    origin: TurnOrigin = TurnOrigin.OWNER
    owner_words: str = ""
    task_words: str | None = None
    tainted: bool = False
    foreign_in_chat: bool = False
    screen_seen: bool = False
    speaker: bool = False
    message_id: str | None = None  # the user's message id, the origin_key of the user's actions (§16.2)
    clock: str | None = None  # the turn's UTC time; None = now
    # v0.4 §5.5: set by the hooks adapter from the tool about to run (a send or forward to
    # another person); None = detect from the user's words (send-verb lexicon). OR-ed, never
    # cleared.
    sends: bool | None = None

    @property
    def is_tainted(self) -> bool:
        """§1: foreign text, a screen look or room speech."""
        return self.tainted or self.foreign_in_chat or self.screen_seen or self.speaker


# ---------------------------------------------------------------- v0.4: op payloads (§6.5)
#
# `ProposalOp.payload` stays a dict (the ledger reads it); these models are its normative
# shape per `OpKind`. Build a payload with `payload_dict(model)` (drops None, so a FLAG with
# only `status` does not also send `review: null`). `validate_payload(op)` checks one op.


class JustificationIn(_M):
    """One justification member (§8.1). `target` is read only by LINK; ADD uses the new record."""

    member: str  # a record id or a raw chunk id
    provenance: Provenance
    group_no: int = 0
    target: str | None = None


class AliasIn(_M):
    alias: str
    kind: AliasKind | None = None  # None: `name` on ADD, `nickname` on UPDATE / LINK
    entity_id: str | None = None  # LINK only; else op.target


class DerivedIn(_M):
    """A derived item (§3.2 `derived`): summary, core section or assoc link."""

    id: str | None = None
    dkind: DerivedKind
    dkey: str | None = None
    content: str
    read_set: list[ReadRef] = Field(default_factory=list)  # empty: the op's read set
    labels: int = 0
    built_at: str | None = None


class AddPayload(_M):
    """ADD: exactly one of `record` (an `AnyRecord` dict; `id` and `recorded_at` may be left
    out) or `derived`. `platform` is set by code only (a worker's proposal is refused)."""

    record: dict[str, Any] | None = None
    derived: DerivedIn | None = None
    aliases: list[AliasIn] = Field(default_factory=list)  # entity records only
    derived_from: list[str] = Field(default_factory=list)  # raw or record ids (lineage)
    mentions: list[str] = Field(default_factory=list)  # entity ids named in the text
    justifications: list[JustificationIn] = Field(default_factory=list)
    rule_match: bool = True  # §8.3 rule supersession of older facts in the same slot
    model_matches: list[str] = Field(default_factory=list)  # §8.3: records to flag `review`
    platform: bool = False

    @model_validator(mode="after")
    def _one(self) -> AddPayload:
        if (self.record is None) == (self.derived is None):
            raise ValueError("ADD needs exactly one of record / derived")
        return self


class UpdatePayload(_M):
    """UPDATE `target`: header or kind columns in `fields`; entity aliases and stable ids."""

    fields: dict[str, Any] = Field(default_factory=dict)
    add_aliases: list[AliasIn] = Field(default_factory=list)
    remove_aliases: list[str] = Field(default_factory=list)  # the user's word only
    add_stable_ids: list[StableId] = Field(default_factory=list)
    platform: bool = False


class SupersedePayload(_M):
    """SUPERSEDE `target` (the old record) by `by` (an existing record) or by a new `record`
    (with ADD's optional keys). `fix`: the user's correction, writes the retraction overlay
    (§9.6)."""

    record: dict[str, Any] | None = None
    by: str | None = None
    fix: bool = False
    aliases: list[AliasIn] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    justifications: list[JustificationIn] = Field(default_factory=list)
    model_matches: list[str] = Field(default_factory=list)
    platform: bool = False

    @model_validator(mode="after")
    def _one(self) -> SupersedePayload:
        if (self.record is None) == (self.by is None):
            raise ValueError("SUPERSEDE needs exactly one of record / by")
        return self


class RetractPayload(_M):
    raw_ids: list[str] | None = None  # raw to overlay; None: the target's own lineage


class FlagPayload(_M):
    review: bool | None = None
    status: Literal["needs_review", "archived", "active"] | None = None

    @model_validator(mode="after")
    def _some(self) -> FlagPayload:
        if self.review is None and self.status is None:
            raise ValueError("FLAG needs review or status")
        return self


class LinkPayload(_M):
    """LINK: exactly one of a justification member, an alias binding or a bare edge."""

    justification: JustificationIn | None = None
    alias: AliasIn | None = None
    platform: bool = False
    from_id: str | None = None
    to_id: str | None = None
    etype: EdgeType | None = None

    @model_validator(mode="after")
    def _one(self) -> LinkPayload:
        edge = (self.from_id, self.to_id, self.etype)
        whole = None not in edge
        partial = any(x is not None for x in edge) and not whole
        forms = [self.justification is not None, self.alias is not None, whole]
        if sum(forms) != 1 or partial:
            raise ValueError("LINK needs exactly one of justification / alias / edge")
        return self


class MergePayload(_M):
    drop: str  # merged into op.target, which stays


class SplitPayload(_M):
    record_ids: list[str] = Field(default_factory=list)  # records that move to the new entity
    display_name: str | None = None
    new_id: str | None = None


class ForgetPayload(_M):
    """FORGET is the ledger part of stop-use only (§9.1 1a); the user's tier; no payload."""


OP_PAYLOADS: dict[OpKind, type[_M]] = {
    OpKind.ADD: AddPayload,
    OpKind.UPDATE: UpdatePayload,
    OpKind.SUPERSEDE: SupersedePayload,
    OpKind.RETRACT: RetractPayload,
    OpKind.FLAG: FlagPayload,
    OpKind.LINK: LinkPayload,
    OpKind.MERGE: MergePayload,
    OpKind.SPLIT: SplitPayload,
    OpKind.FORGET: ForgetPayload,
}


def payload_dict(payload: BaseModel) -> dict[str, Any]:
    """The dict `ProposalOp.payload` carries: JSON mode, None dropped."""
    return payload.model_dump(mode="json", exclude_none=True)


def validate_payload(op: ProposalOp) -> BaseModel:
    """§6.5: the op's payload against its kind's model; raises pydantic's ValueError."""
    return OP_PAYLOADS[OpKind(op.op)].model_validate(op.payload)


# ---------------------------------------------------------------- v0.4: extraction output (§6.3)
#
# What an `Extractor` returns for a snapshot (and what a gold extraction contains); code
# turns it into ops (`propose`). Refs are local ("e1", "f2") or existing handles ("m:<id>");
# "self" is the user (§4.5). Unknown stays None (§4.1).


class XEntity(_M):
    ref: str
    etype: str  # person, place, org, matter, thing, account, pet
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    # @4: the common nouns the messages call it by («кот», "the car"): a pet or thing the user names
    kind_words: list[str] = Field(default_factory=list)


class XReason(_M):
    member: str  # a fact ref, an existing record handle or a raw id from the snapshot
    provenance: Provenance  # `stated` only when the user said the reason (§6.3)
    group_no: int = 0


class XFact(_M):
    ref: str
    kind: Literal["fact", "preference"] = "fact"
    subject: str
    relation: str  # the closed vocabulary; unknown -> `other` (§4.1)
    value_text: str | None = None
    value_entity: str | None = None
    text_en: str
    keywords_ru: str | None = None
    keywords_en: str | None = None
    quote_orig: str  # required (§6.3): verbatim from the source raw
    lang_orig: str | None = None
    fkind: FKind
    polarity: Polarity = Polarity.AFFIRMED
    valid_from: str | None = None
    valid_to: str | None = None
    stability: Stability | None = None
    applicability: Applicability | None = None  # required for preferences (§4.3)
    speaker: str | None = None  # None: the source raw's author
    source_raw_id: str
    reasons: list[XReason] = Field(default_factory=list)
    replaces: str | None = None  # a handle the model thinks this restates: `review` only (§8.3)


class XImpression(_M):
    ref: str
    about: str
    text_en: str
    quote_orig: str
    source_raw_id: str
    observed_at: str | None = None  # None: the raw's ts


class XAlternative(_M):
    text: str  # the option not taken, in the source language
    reasons: list[str] = Field(default_factory=list)  # why not, as the message says it


class XDecision(_M):
    """@3: a choice made, with its stated reasons and what it replaces (§8, §14)."""

    ref: str
    chosen: str  # what was chosen, in the language and words of the quote
    text_en: str  # one short English sentence of the decision
    quote_orig: str  # verbatim from the source raw
    source_raw_id: str
    domain: Domain | None = None  # §15.2 closed list; None when the message does not say
    matter: str | None = None  # the project / course / matter it belongs to (an entity ref)
    rejected: list[XAlternative] = Field(default_factory=list)
    reasons: list[XReason] = Field(default_factory=list)  # members: fact refs, handles, raw ids
    replaces: str | None = None  # an earlier decision or fact this one revises: review only
    decided_at: str | None = None
    speaker: str | None = None  # None: the source raw's author


class XTrigger(_M):
    """When an intention comes due: a date, a person's message, a value reached or a
    condition in words (an offer only, never an action)."""

    kind: Literal["date", "message", "status", "condition"]
    ts: str | None = None  # date: when
    person: str | None = None  # message: whose message (an entity ref)
    subject: str | None = None  # status: the slot's subject (an entity ref)
    relation: str | None = None
    to: str | None = None
    condition: str | None = None  # condition: the situation, in English


class XIntention(_M):
    """@3: something to do or to be reminded of later, with its trigger (§12)."""

    ref: str
    text_en: str
    value_text: str | None = None  # what, in the language and words of the quote
    quote_orig: str
    source_raw_id: str
    trigger: XTrigger
    concerns: str | None = None  # the person or matter it is about (an entity ref)


class XCriterion(_M):
    """@3: how the user judges a kind of choice («беру только с …»), §15.2."""

    ref: str
    text_en: str
    value_text: str | None = None  # the rule in the language and words of the quote
    quote_orig: str
    source_raw_id: str
    applicability: Applicability | None = None
    provenance: Provenance = Provenance.STATED  # inferred: drawn from decisions, not said
    evidence: list[str] = Field(default_factory=list)  # decision / fact refs, handles, raw ids


class XRetraction(_M):
    """@4: the author takes back an earlier claim of theirs («беру свои слова назад про …»,
    "scratch what I said about …"). Never a fact: code retires the claim when it is the
    author's own (§8.3)."""

    ref: str
    targets: list[str] = Field(default_factory=list)  # known fact handles or earlier fact refs
    quote_orig: str  # verbatim from the source raw
    source_raw_id: str


class Extraction(_M):
    entities: list[XEntity] = Field(default_factory=list)
    facts: list[XFact] = Field(default_factory=list)
    impressions: list[XImpression] = Field(default_factory=list)
    decisions: list[XDecision] = Field(default_factory=list)  # @3
    intentions: list[XIntention] = Field(default_factory=list)  # @3
    criteria: list[XCriterion] = Field(default_factory=list)  # @3
    retractions: list[XRetraction] = Field(default_factory=list)  # @4


# ---------------------------------------------------------------- v0.4: structured hot writes (§6.1)


class Assertion(_M):
    """The structured part of «запомни» / «исправь» (§6.1): the brain fills what the user's words
    say; code resolves names, checks the vocabulary and normalizes values. No model call on
    the hot path. Unknown stays None (§4.1)."""

    subject: str | None = None  # "self", "m:<id>", a name or alias; None: "self"
    relation: str | None = None  # unknown -> `other`
    value: str | None = None  # as the user said it; normalized by value_type
    value_entity: str | None = None  # "m:<id>" or a name, when the value is a person or place
    text_en: str | None = None  # English canonical text; None: the tool's text / correction
    keywords_ru: str | None = None
    keywords_en: str | None = None
    fkind: FKind | None = None  # None: `state`
    polarity: Polarity | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    stability: Stability | None = None
    trigger: Trigger | None = None  # kind=intention (§12)
    due: str | None = None  # kind=loop


# ---------------------------------------------------------------- v0.4: ingest rows (§6.2)


class IngestRow(_M):
    """One message as arrival reads it (§6.2): a `koob_messages` row through the
    conversation adapter, a sweep row, or an eval timeline row (the harness's shape)."""

    role: Literal["owner", "cube", "tool", "other"]
    ts: str
    text: str = ""
    chat_id: str | None = None  # `tg:<peer id>`, a file path, or None / "island" (Cube turn)
    source: str | None = None  # source_kind (§6.2 vocabulary); None: from role and tool
    tool: str | None = None  # role=tool: the tool name
    msg_id: str | None = None  # the platform message id
    reply_to: str | None = None
    fwd_from: str | None = None
    sender: str | None = None  # display name from platform metadata
    sender_id: str | None = None  # platform id, `tg:<id>`
    sender_tg_id: str | None = None  # accepted: a bare Telegram user id -> sender_id
    channel: Literal["typed", "voice", "speaker"] | None = None  # role=owner
    author_tier: AuthorTier | None = None  # from tool or platform metadata, never model text
    media: str | None = None
    caption: str | None = None
    turn_id: str | None = None
    thread_id: str | None = None
    row_id: int | None = None  # koob_messages.id (the sweep watermark)
    noise: str | None = None  # eval generator tag; ignored

    @field_validator("msg_id", "reply_to", "sender_tg_id", "fwd_from", mode="before")
    @classmethod
    def _str_id(cls, v: Any) -> Any:
        return str(v) if isinstance(v, int) else v

    @model_validator(mode="after")
    def _canon(self) -> IngestRow:
        if self.sender_id is None and self.sender_tg_id:
            self.sender_id = f"tg:{self.sender_tg_id}"
        if self.chat_id == ISLAND_CHAT:
            self.chat_id = None
        return self
