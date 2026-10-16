"""Inspection commands: what / where-from / forget / undo, in chat and in `cb memory`.

Spec: §17 (with §9.1 forget stages, §10.4 store tools, §16.1 resolve, §5.5 use-now, §6.1
tainted turns). Also covers the `cb memory` Typer group and, behind the flag, the MCP tools
in `assistant/tools.py`.

The ledger stores English; every line here is for the user and is in plain Russian
(«перестал использовать», «удалил полностью»). Handles `[m:<id>]` stay in the lines so the
brain can name an item back (§10.2 step 6, §10.4).

Targets in their words resolve through the ledger, never by a guess: names go through
`entities.resolve_rows` (stable ids, aliases, inflections, diminutives), other words through
a lexical match over usable records. Several plausible targets give one distinguishing
question (`ask_first`). Until a stemmed index exists, the lexical match is a prefix match
over the case-folded FTS rows plus a Russian-tolerant word comparison in Python.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import entities, forget, ledger, policy, store
from .model import (
    USABLE,
    AnyRecord,
    AuthorTier,
    FKind,
    ForgetScope,
    Label,
    ReadRef,
    ReadSrc,
    RecordKind,
    Status,
    TurnCtx,
    TurnOrigin,
)
from .writers import hot

if TYPE_CHECKING:
    from .api import ForgetResult, Item, Lineage, PurgeStatus, WhatAnswer

CLI_THREAD = "cli"  # `cb memory` turns: no session to restart, no served rows
TRASH_HOURS = int(forget.TRASH_WINDOW.total_seconds() // 3600)
_MAX_ASK = 4  # options in one distinguishing question
_QUOTE_MAX = 90

_USER_TIERS = frozenset({AuthorTier.USER, AuthorTier.USER_CONFIRMED})
_KIND_TITLE: dict[RecordKind, str] = {
    RecordKind.FACT: "Факты и планы",
    RecordKind.PREFERENCE: "Предпочтения",
    RecordKind.IMPRESSION: "Впечатления",
    RecordKind.DECISION: "Решения",
    RecordKind.INTENTION: "Намерения",
    RecordKind.LOOP: "Дела и обещания",
    RecordKind.LESSON: "Уроки",
    RecordKind.EPISODE: "События",
    RecordKind.CRITERION: "Критерии",
    RecordKind.WORK_STATE: "Работа",
    RecordKind.ENTITY: "Люди и вещи",
}
_KIND_ORDER = list(_KIND_TITLE)
_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
_SOURCES: dict[str, str] = {
    "owner_message": "разговор с Кубом",
    "cube_turn": "разговор с Кубом",
    "turn": "разговор с Кубом",
    "island": "разговор с Кубом",
    "chat": "разговор с Кубом",
    "voice": "голосом",
    "room": "голосом в комнате",
    "telegram": "Telegram",
    "tg": "Telegram",
    "tgbot": "Telegram-бот",
    "mail": "письмо",
    "email": "письмо",
    "file": "файл",
    "doc": "файл",
    "web": "страница в интернете",
    "page": "страница в интернете",
    "screen": "экран",
    "tool": "инструмент",
    "obsidian": "старые заметки",
    "import": "старые заметки",
}

# Words of the command itself, never the target («забудь всё про Сашу», «что ты помнишь о …»).
_LEAD = frozenset(
    [
        "забудь",
        "забыть",
        "забудешь",
        "забывай",
        "пожалуйста",
        "плиз",
        "удали",
        "удалить",
        "сотри",
        "стереть",
        "убери",
        "перестань",
        "память",
        "памяти",
        "помнить",
        "помни",
        "помнишь",
        "вспомни",
        "знаешь",
        "знаешь",
        "расскажи",
        "покажи",
        "скажи",
        "что",
        "ты",
        "мне",
        "про",
        "о",
        "об",
        "обо",
        "насчет",
        "насчёт",
        "касательно",
        "всё",
        "все",
        "всю",
        "весь",
        "это",
        "этот",
        "эту",
        "эти",
        "то",
        "тот",
        "ту",
        "насовсем",
        "навсегда",
        "совсем",
        "полностью",
        "нафиг",
        "откуда",
        "где-то",
        "forget",
        "about",
        "everything",
        "please",
        "the",
        "this",
        "that",
        "what",
        "do",
        "you",
        "remember",
        "know",
    ]
)
_ALL_ABOUT = re.compile(
    r"(?:^|\s)(?:вс[её]|everything)\s+(?:про|о|об|обо|насч[её]т|about)\s", re.IGNORECASE
)
# «всё, что я рассказывала про X», «всё, связанное с X», "everything I told you about X",
# "all about X" order the whole topic as «всё про X» does
_TOLD_ABOUT = re.compile(
    r"(?:^|(?<=[\s,.!?]))(?:вс[её]|everything)\s*,?\s*(?:(?:что|чего|i|we)\s+(?:я\s+|мы\s+)?"
    r"(?:тебе\s+|вам\s+|you\s+)?(?:рассказ\w*|говор\w*|сказа\w*|писа\w*|упомина\w*"
    r"|упомян\w*|сообща\w*|told|said|wrote|mentioned)(?:\s+(?:тебе|вам|you))?\s+"
    r"|связанн\w*\s+(?=с\s|со\s)|related\s+(?=to\s))",
    re.IGNORECASE,
)
# «забудь, что я говорила про X» (no «всё») is what the user told about X, too
_SAID_ABOUT = re.compile(
    r"^[\s,.!?:;—–-]*(?:то\s*,?\s*)?(?:что|what)\s+(?:я\s+|i\s+)?(?:тебе\s+|you\s+)?"
    r"(?:рассказ\w*|говор\w*|сказа\w*|писа\w*|упомина\w*|упомян\w*|told|said|wrote|mentioned)"
    r"(?:\s+(?:тебе|вам|you))?\s+",
    re.IGNORECASE,
)
# the user's reason after the target is not the target («…, это было нечестно», «…, не
# хочу, чтобы это где-то висело», "…, it was unfair")
_REASON_TAIL = re.compile(
    r"\s*[,;—–-]+\s*(?:это\s+было|это\s+нечестно|не\s+хочу|не\s+надо|потому\s+что|так\s+как"
    r"|чтобы|т\.\s?к\.|because|it\s+was|i\s+don'?t\s+want)(?!\w).*$",
    re.IGNORECASE | re.DOTALL,
)
_BACK_REF = re.compile(
    r"(?<!\w)(?:его|её|ее|ему|ей|их|им|него|неё|нее|нему|ней|ним|him|her|his|their|them)(?!\w)",
    re.IGNORECASE,
)
_ALL_WORD = re.compile(r"(?:^|\s)(?:all|вс[её])\s+(?:about|про|о|об|обо)\s", re.IGNORECASE)
# filler before the verb («Всё-таки забудь…», «Ладно, забудь…»): not the target
_FILLER = frozenset(
    [
        "всё-таки",
        "все-таки",
        "всетаки",
        "хотя",
        "ладно",
        "ну",
        "так",
        "короче",
        "слушай",
        "давай",
        "пожалуйста",
        "нет",
        "да",
        "ок",
        "окей",
        "а",
        "и",
        "тогда",
        "теперь",
        "всё",
        "все",
        "ещё",
        "еще",
        "лучше",
        "тоже",
        "ok",
        "okay",
        "well",
        "so",
        "then",
        "please",
        "also",
    ]
)
# words that describe the person, not a topic («всё про моего брата Диму»)
_DESCRIBE = frozenset(
    [
        "брат",
        "брата",
        "брату",
        "братом",
        "сестра",
        "сестру",
        "сестры",
        "сестрой",
        "мама",
        "маму",
        "мамы",
        "папа",
        "папу",
        "папы",
        "сын",
        "сына",
        "дочь",
        "дочку",
        "дочери",
        "муж",
        "мужа",
        "жена",
        "жену",
        "друг",
        "друга",
        "подруга",
        "подругу",
        "подруги",
        "коллега",
        "коллегу",
        "коллеги",
        "начальник",
        "начальника",
        "начальницу",
        "парень",
        "парня",
        "девушка",
        "девушку",
        "brother",
        "sister",
        "mom",
        "mother",
        "dad",
        "father",
        "son",
        "daughter",
        "husband",
        "wife",
        "friend",
        "colleague",
        "boss",
    ]
)
# a question word that points at the value's slot («до какого числа», «сколько»), not a topic
_SLOT_CUES = (
    (
        re.compile(
            r"(?<!\w)(?:(?:до|с|со|к|на|по)\s+)?(?:какого|какое|каким|каком)\s+числ\w*"
            r"|(?<!\w)(?:до|с)\s+когда(?!\w)|(?<!\w)как(?:ая|ой|ую)\s+дат\w*"
            r"|(?<!\w)(?:what|which)\s+(?:date|day)(?!\w)|(?<!\w)until\s+when(?!\w)",
            re.IGNORECASE,
        ),
        "date",
    ),
    (re.compile(r"(?<!\w)сколько(?!\w)|(?<!\w)how\s+(?:much|many)(?!\w)", re.IGNORECASE), "number"),
)
# In a tainted turn a forget is the user's order only when the user's own words say forget,
# and `forever` only when they say «насовсем»; a mention of the person next to foreign text
# («что тут Дима пишет?») orders nothing.
_FORGET_VERB = re.compile(
    r"(?<!\w)(?:забуд\w*|забыва\w*|забыть|позабудь|удали\w*|удалять|сотри|стереть|сотрите"
    r"|вычеркни\w*|выкинь|выбрось|убери\w*|не\s+запоминай|не\s+помни|forget|delete|erase"
    r"|remove|wipe)(?!\w)",
    re.IGNORECASE,
)
_FOREVER_WORD = re.compile(
    r"(?<!\w)(?:насовсем|навсегда|полностью|совсем|окончательно|начисто|бесследно|подчистую"
    r"|forever|permanently|completely|for\s+good|entirely)(?!\w)",
    re.IGNORECASE,
)
# Grammar words that carry no topic (kept out of the lexical match).
_STOP = frozenset(
    [
        "и",
        "в",
        "во",
        "на",
        "с",
        "со",
        "к",
        "ко",
        "у",
        "из",
        "за",
        "от",
        "до",
        "по",
        "для",
        "при",
        "над",
        "под",
        "без",
        "или",
        "но",
        "а",
        "же",
        "ли",
        "бы",
        "не",
        "ни",
        "он",
        "она",
        "оно",
        "они",
        "его",
        "её",
        "ее",
        "их",
        "ему",
        "ей",
        "им",
        "ним",
        "ней",
        "нему",
        "наш",
        "наша",
        "наше",
        "наши",
        "нашу",
        "нашего",
        "нашей",
        "наших",
        "нашим",
        "ваш",
        "ваши",
        "твою",
        "твоего",
        "твоей",
        "свою",
        "своего",
        "своей",
        "моих",
        "твоих",
        "своих",
        "our",
        "мой",
        "моя",
        "моё",
        "мое",
        "мои",
        "мою",
        "моего",
        "моей",
        "твой",
        "твоя",
        "твои",
        "свой",
        "своя",
        "свои",
        "как",
        "так",
        "там",
        "тут",
        "где",
        "когда",
        "кто",
        "чем",
        "чего",
        "кого",
        "кому",
        "был",
        "была",
        "были",
        "было",
        "есть",
        "быть",
        "будет",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "it",
        "its",
        "his",
        "her",
        "their",
        "my",
        "your",
        "from",
    ]
)
_WORD = re.compile(r"[0-9a-zа-яё][0-9a-zа-яё'\-]*", re.IGNORECASE)
_HANDLE = re.compile(r"m:([^\s,;»«\]\)\(\[]+)")
_SELF = frozenset({"меня", "мне", "я", "мной", "себя", "себе", "me", "myself", "i"})


# ---------------------------------------------------------------- small words


def ru_date(ts: str | None) -> str | None:
    """«12 октября 2026» from an ISO date or timestamp."""
    if not ts or len(ts) < 10:
        return None
    try:
        y, m, d = int(ts[0:4]), int(ts[5:7]), int(ts[8:10])
    except ValueError:
        return None
    if not 1 <= m <= 12:
        return None
    return f"{d} {_MONTHS[m - 1]} {y}"


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def genitive(name: str) -> str:
    """«Дима» → «Димы», «Саша» → «Саши», «Иван» → «Ивана»; anything else unchanged."""
    if not name or " " in name.strip() or not re.fullmatch(r"[А-Яа-яЁё\-]+", name):
        return name
    low = name[-1].lower()
    if low == "а":
        return name[:-1] + ("и" if name[-2:-1].lower() in "гкхжшщч" else "ы")
    if low == "я":
        return name[:-1] + "и"
    if low in "йь":
        return name[:-1] + "я"
    if low in "бвгджзклмнпрстфхцчшщ":
        return name + "а"
    return name


def _clip(text: str, n: int = _QUOTE_MAX) -> str:
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return (cut or text[:n]).rstrip(" ,;:") + "…"


def _q(text: str) -> str:
    return f"«{_clip(text)}»"


def handle(item_id: str) -> str:
    return f"m:{item_id}"


def handles_in(text: str) -> list[str]:
    """Record ids named by `m:<id>` handles in a target."""
    return [m.group(1).rstrip(".:") for m in _HANDLE.finditer(text or "")]


def _norm(text: str) -> str:
    return (text or "").casefold().replace("ё", "е")


def _words(text: str) -> list[str]:
    return [w.strip("-'") for w in _WORD.findall(_norm(text)) if w.strip("-'")]


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in _LEAD and w not in _STOP and len(w) >= 3]


def _same_word(a: str, b: str) -> bool:
    """Tolerant of Russian endings: «переезд» ~ «переезжает», «работа» ~ «работы»."""
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    if n >= 4 and n >= min(len(a), len(b)) - 2:
        return True
    if n < 6:
        return False
    # the same rule on Snowball stems: «выгоренность» ~ «выгоревшая», while
    # «переговоры» and «перегородка» stay apart
    from .recall.lexical import _stem_token

    sa, sb = _stem_token(a), _stem_token(b)
    return (
        sa[: len(sb)] == sb or sb[: len(sa)] == sa or _shared(sa, sb) >= min(len(sa), len(sb)) - 2
    )


def _shared(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _strip_lead(text: str) -> str:
    """The target without the command words around it («забудь всё про Сашу» → «Сашу»)."""
    words = [w for w in re.split(r"[\s,.!?:;]+", (text or "").strip()) if w]
    while words and _norm(words[0]) in _LEAD:
        words.pop(0)
    while words and _norm(words[-1]) in _LEAD:
        words.pop()
    return " ".join(words).strip(" «»\"'")


# ---------------------------------------------------------------- records as the user sees them


def _english(rec: AnyRecord) -> str:
    for attr in ("text_en", "chosen_text", "rule_en", "summary_en", "display_name"):
        v = getattr(rec, attr, None)
        if isinstance(v, str) and v:
            return v
    state = getattr(rec, "state", None)
    if state is not None:
        for attr in ("goal", "next_step", "direction"):
            v = getattr(state, attr, None)
            if v:
                return str(v)
    trigger = getattr(rec, "trigger", None)
    for v in (
        getattr(trigger, "semantic", None),
        getattr(trigger, "goal", None),
        getattr(rec, "action", None),
        getattr(rec, "completion", None),
    ):
        if v:
            return str(v)
    return rec.id


def _shown(rec: AnyRecord) -> str:
    """What the user reads: the user's own (or the speaker's) words when kept, else the text."""
    for attr in ("quote_orig", "origin_quote"):
        v = getattr(rec, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    return _english(rec)


def _speaker(rec: AnyRecord) -> str | None:
    return getattr(rec, "speaker_entity_id", None) or rec.author_entity_id


def _names(conn: sqlite3.Connection, ids: Iterable[str | None]) -> dict[str, str]:
    want = sorted({i for i in ids if i})
    if not want:
        return {}
    marks = ", ".join("?" for _ in want)
    return {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT id, display_name FROM entities WHERE id IN ({marks})",  # noqa: S608
            want,
        )
    }


def _who(rec: AnyRecord, names: dict[str, str], *, claim: bool = False) -> str:
    """Who said it, in words: «ты сам», «по словам Димы», «вывод Куба»."""
    tier = AuthorTier(rec.author_tier)
    if tier is AuthorTier.USER:
        return "по твоим словам" if claim else "ты сам"
    if tier is AuthorTier.USER_CONFIRMED:
        return "по твоим словам" if claim else "ты подтвердил"
    if tier is AuthorTier.USER_ROOM:
        return "ты вслух, в комнате"
    if tier is AuthorTier.OTHER:
        name = names.get(_speaker(rec) or "")
        return f"по словам {genitive(name)}" if name else "по чужим словам"
    if tier is AuthorTier.TOOL:
        return "из инструмента или страницы"
    return "вывод Куба"


def _date(rec: AnyRecord) -> str | None:
    return rec.recorded_at[:10] if rec.recorded_at else None


def _item(rec: AnyRecord, names: dict[str, str], use: policy.Use | None = None) -> Item:
    from .api import Item

    text = _english(rec)
    if use is not None and use.hold:
        text = f"{text} ({policy.HOLD_LINE})"
    tier = AuthorTier(rec.author_tier)
    said_by = None
    if tier not in _USER_TIERS and tier is not AuthorTier.USER_ROOM:
        said_by = names.get(_speaker(rec) or "")
    return Item(
        id=rec.id,
        kind=rec.kind,
        version=rec.version,
        text=text,
        status=rec.status,
        author_tier=tier,
        said_by=said_by,
        date=_date(rec),
        labels=Label(rec.labels),
        impression=rec.kind is RecordKind.IMPRESSION,
        review=bool(rec.review),
    )


def _records(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, AnyRecord]:
    out: dict[str, AnyRecord] = {}
    for rid in dict.fromkeys(ids):
        rec = ledger.get_record(conn, rid)
        if rec is not None:
            out[rid] = rec
    return out


def _usable(rec: AnyRecord, *, include_archived: bool = True) -> bool:
    if rec.status is Status.ACTIVE:
        return True
    return include_archived and rec.status in USABLE


# ---------------------------------------------------------------- served set (§10.4, §7.6)


def _serve(conn: sqlite3.Connection, turn: TurnCtx, recs: Sequence[AnyRecord]) -> None:
    """Every item the brain gets writes a `served` row (§10.4): a later forget restarts the
    sessions that saw it (§9.1c). `cb memory` turns have no session."""
    if not recs or turn.thread_id == CLI_THREAD:
        return
    now = store.now()
    with store.write_tx(conn):
        conn.executemany(
            "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, turn_id, via,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, 'tool', ?)",
            [
                (turn.thread_id, turn.session_id, r.id, r.kind.value, r.version, turn.turn_id, now)
                for r in recs
            ],
        )


# ---------------------------------------------------------------- finding (resolve + lexical)


def _about(conn: sqlite3.Connection, entity_ids: Sequence[str]) -> list[str]:
    """Record ids about these entities (subject, value, impression, `about` edge, episode)."""
    if not entity_ids:
        return []
    marks = ", ".join("?" for _ in entity_ids)
    ids = list(entity_ids)
    sql = [
        f"SELECT id FROM facts WHERE subject_id IN ({marks}) OR value_entity_id IN ({marks})",
        f"SELECT id FROM impressions WHERE about_entity_id IN ({marks})",
        f"SELECT from_id FROM edges WHERE etype = 'about' AND to_id IN ({marks})",
        f"SELECT episode_id FROM episode_links WHERE target_id IN ({marks})",
        f"SELECT id FROM decisions WHERE matter_id IN ({marks})",
        f"SELECT id FROM work_states WHERE matter_id IN ({marks})",
        f"SELECT id FROM loops WHERE owner IN ({marks})",
    ]
    out: dict[str, None] = {}
    for q in sql:
        params = ids * q.count("IN (")
        for r in conn.execute(q, params):  # noqa: S608
            if r[0] not in entity_ids:
                out[r[0]] = None
    return list(out)


def _fts_ids(conn: sqlite3.Connection, terms: Sequence[str], limit: int = 300) -> list[str]:
    """Candidate record ids for content words (to be replaced by a stemmed BM25 index)."""
    if not terms:
        return []
    from .recall import lexical

    try:
        hits = lexical.search_records(conn, " ".join(terms), label_mask=-1, limit=limit)
        return [h[0] for h in hits]
    except NotImplementedError:
        pass
    parts = []
    for t in terms:
        stem = t[: max(4, len(t) - 2)] if len(t) > 4 else t
        parts.append('"' + stem.replace('"', '""') + '"*')
    try:
        rows = conn.execute(
            "SELECT r.id FROM fts_records f JOIN records r ON r.fts_rowid = f.rowid"
            " WHERE fts_records MATCH ? LIMIT ?",
            (" OR ".join(parts), limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def _record_words(rec: AnyRecord) -> set[str]:
    texts = [_english(rec)]
    for attr in ("quote_orig", "origin_quote", "keywords_ru", "keywords_en", "value_text"):
        v = getattr(rec, attr, None)
        if isinstance(v, str) and v:
            texts.append(v)
    return {w for t in texts for w in _words(t)}


def _covered(terms: Sequence[str], words: set[str]) -> int:
    return sum(1 for t in terms if any(_same_word(t, w) for w in words))


@dataclass
class _Found:
    """What a phrase names: entities (by name) and/or records (by its other words)."""

    query: str
    entities: list[entities.Resolved] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    scored: list[tuple[float, str]] = field(default_factory=list)  # (score, record id)
    recs: dict[str, AnyRecord] = field(default_factory=dict)


def _founding_names(conn: sqlite3.Connection, entity_id: str) -> set[str]:
    """§13.3: the names a contact known only from other people's words was recorded
    under: its display name (and the transliterations stored with it), when those alias rows
    are as old as the entity itself. A later unconfirmed alias («Саша — это Глеб» in a
    forwarded message), or a display name a later op renamed, is only a candidate."""
    row = conn.execute(
        "SELECT e.display_name, r.created_at FROM entities e JOIN records r ON r.id = e.id"
        " WHERE e.id = ?",
        (entity_id,),
    ).fetchone()
    if row is None:
        return set()
    own = {entities.normalize_alias(n) for n in (row[0], *entities.transliterations(row[0]))}
    return {
        r[0]
        for r in conn.execute(
            "SELECT alias_norm FROM aliases WHERE entity_id = ? AND created_at <= ?",
            (entity_id, row[1]),
        )
        if r[0] in own
    }


def _trusted(conn: sqlite3.Connection, hits: list[entities.Resolved]) -> list[entities.Resolved]:
    """§13.3: an alias→entity link from foreign text is a candidate, so a hit
    that rests only on an unconfirmed alias is dropped and neither «забудь всё про Сашу» nor
    «что ты помнишь про Сашу» follows a planted «Саша — это Глеб». The one exception is a
    contact known only from other people's words (no confirmed name at all): the name it was
    recorded under still finds it, so «забудь всё про Глеба» works; a later alias does not."""
    shaky = list(dict.fromkeys(h.entity_id for h in hits if h.unconfirmed))
    if not shaky:
        return hits
    marks = ", ".join("?" for _ in shaky)
    known = {
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT entity_id FROM aliases WHERE confirmed = 1 AND entity_id IN ({marks})",  # noqa: S608
            shaky,
        )
    }
    founding = {eid: _founding_names(conn, eid) for eid in shaky if eid not in known}
    return [
        h
        for h in hits
        if not h.unconfirmed or (h.entity_id not in known and h.matched & founding[h.entity_id])
    ]


def _named(conn: sqlite3.Connection, query: str) -> tuple[list[entities.Resolved], list[str]]:
    """Entities the user's phrase names, and the words left once the names are taken out."""
    whole = _trusted(conn, entities.resolve_rows(conn, query, "any")) if query else []
    if whole:
        return whole, []
    found: dict[str, entities.Resolved] = {}
    rest: list[str] = []
    for w in _words(query):
        if w in _LEAD or w in _STOP:
            continue
        hits = _trusted(conn, entities.resolve_rows(conn, w, "any")) if len(w) >= 2 else []
        if hits:
            for h in hits:
                found.setdefault(h.entity_id, h)
        elif len(w) >= 3:
            rest.append(w)
    return list(found.values()), rest


def _find(
    conn: sqlite3.Connection,
    phrase: str,
    *,
    include_archived: bool = True,
    min_cover: float = 2 / 3,
    include_history: bool = False,
) -> _Found:
    """Records the user's phrase names. A record must cover at least `min_cover` of the user's
    content words (R-C3: «забудь мой номер карты в банке» does not take «мой банк Нордбанк» on
    one word of three); reads (`what`, `search`) pass 0 and rank partial matches instead.
    `include_history`: superseded / retracted records count too (a forget's target, A14)."""
    query = _strip_lead(phrase)
    named, rest = _named(conn, query)
    found = _Found(query=query, entities=named, terms=rest)
    if not rest:
        return found
    about = set(_about(conn, [e.entity_id for e in named]))
    # FTS rows keep «ё»: ask with the user's spelling as well as the folded one
    spelled = [w for w in _WORD.findall(query.casefold()) if _norm(w) in rest and "ё" in w]
    ids = list(dict.fromkeys([*_fts_ids(conn, [*rest, *spelled]), *about]))
    recs = _records(conn, ids)
    for rid, rec in recs.items():
        # «забудь» may aim at history (§9.1 A14): superseded / retracted records count,
        # a shade below a usable one with the same words
        old = include_history and rec.status in (Status.SUPERSEDED, Status.RETRACTED)
        if rec.kind is RecordKind.ENTITY or not (
            old or _usable(rec, include_archived=include_archived)
        ):
            continue
        hit = _covered(rest, _record_words(rec))
        if not hit or hit < min_cover * len(rest) - 1e-9:
            continue
        if named and rid not in about:
            continue  # «переезд Саши»: Sasha's, not anybody's
        found.scored.append((hit + (0.5 if rid in about else 0.0) - (0.25 if old else 0.0), rid))
        found.recs[rid] = rec
    found.scored.sort(key=lambda s: (-s[0], s[1]))
    return found


def _self_ids(conn: sqlite3.Connection) -> list[str]:
    """Records about the user («что ты помнишь про меня»): subjects 'user'/'self'/'owner'."""
    return [
        r[0]
        for r in conn.execute(
            "SELECT id FROM facts WHERE subject_id IN ('user', 'self', 'owner')"
            " UNION SELECT id FROM loops WHERE owner = 'user'"
        )
    ]


# ---------------------------------------------------------------- what (§17)


@dataclass
class _Group:
    title: str
    entity_id: str | None
    items: list[AnyRecord] = field(default_factory=list)
    competing: list[AnyRecord] = field(default_factory=list)


@dataclass
class _What:
    about: str
    groups: list[_Group] = field(default_factory=list)
    hidden: int = 0
    uses: dict[str, policy.Use] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)


def _single(conn: sqlite3.Connection, relations: Iterable[str]) -> set[str]:
    want = sorted(set(relations))
    if not want:
        return set()
    marks = ", ".join("?" for _ in want)
    return {
        r[0]
        for r in conn.execute(
            f"SELECT relation FROM relations WHERE cardinality = 'single' AND relation IN ({marks})",  # noqa: S608
            want,
        )
    }


def _split_competing(conn: sqlite3.Connection, recs: list[AnyRecord]) -> tuple[list, list]:
    """Usable claims in one single-valued slot with different values: attributed, not merged."""
    slots: dict[tuple[str, str], list[AnyRecord]] = {}
    for r in recs:
        if r.kind in (RecordKind.FACT, RecordKind.PREFERENCE):
            slots.setdefault((r.subject_id, r.relation), []).append(r)  # type: ignore[union-attr]
    single = _single(conn, [k[1] for k in slots])
    competing: set[str] = set()
    for (_s, rel), group in slots.items():
        values = {(g.value_norm or g.value_text or g.text_en).casefold() for g in group}  # type: ignore[union-attr]
        if rel in single and len(group) > 1 and len(values) > 1:
            competing |= {g.id for g in group}
    plain = [r for r in recs if r.id not in competing]
    rival = [r for r in recs if r.id in competing]
    return plain, rival


def _order(recs: list[AnyRecord]) -> list[AnyRecord]:
    return sorted(recs, key=lambda r: (_KIND_ORDER.index(r.kind), r.recorded_at or "", r.id))


def _what(conn: sqlite3.Connection, about: str, turn: TurnCtx) -> _What:
    out = _What(about=_strip_lead(about) or about.strip())
    groups: list[tuple[str, str | None, list[str]]] = []
    explicit = handles_in(about)
    if explicit:
        groups.append((out.about, None, explicit))
    elif set(_words(out.about)) & _SELF and len(_content_words(out.about)) == 0:
        groups.append(("тебя", None, _self_ids(conn)))
    else:
        found = _find(conn, about, min_cover=0.0)
        if found.terms:
            groups.append((found.query, None, [rid for _s, rid in found.scored]))
        else:
            for e in found.entities:
                groups.append((e.display_name, e.entity_id, _about(conn, [e.entity_id])))
    recs = _records(conn, [rid for _t, _e, ids in groups for rid in ids])
    usable = {rid: r for rid, r in recs.items() if _usable(r) and r.kind is not RecordKind.ENTITY}
    out.uses = policy.decide_many(conn, list(usable), turn) if usable else {}
    names_wanted: list[str | None] = []
    for title, eid, ids in groups:
        allowed = [usable[i] for i in ids if i in usable and out.uses[i].allowed]
        out.hidden += sum(1 for i in ids if i in usable and not out.uses[i].allowed)
        plain, rival = _split_competing(conn, allowed)
        out.groups.append(_Group(title, eid, _order(plain), _order(rival)))
        names_wanted += [_speaker(r) for r in allowed]
    out.names = _names(conn, names_wanted)
    return out


def what_answer(conn: sqlite3.Connection, about: str, turn: TurnCtx) -> WhatAnswer:
    """`api.what`: usable records by kind, impressions marked, competing claims attributed,
    sensitive items per §5.5. Writes `served` rows for what it returns."""
    from .api import WhatAnswer

    w = what_view(conn, about, turn)
    items = [r for g in w.groups for r in g.items]
    rival = [r for g in w.groups for r in g.competing]
    return WhatAnswer(
        about=w.about,
        items=[_item(r, w.names, w.uses.get(r.id)) for r in items],
        competing_claims=[_item(r, w.names, w.uses.get(r.id)) for r in rival],
    )


def _line(rec: AnyRecord, names: dict[str, str], use: policy.Use | None) -> str:
    text = _q(_shown(rec))
    if rec.kind is RecordKind.IMPRESSION:
        text = f"впечатление: {text}"
    elif getattr(rec, "fkind", None) is FKind.PLAN:
        text = f"план: {text}"
    when = ru_date(rec.recorded_at)
    tail = ", ".join(x for x in (_who(rec, names), when) if x)
    marks = []
    if rec.status is Status.ARCHIVED:
        marks.append("в архиве")
    if rec.review:
        marks.append("надо перепроверить")
    if use is not None and use.hold:
        marks.append("не отправлять без твоего слова")
    mark = f" ({'; '.join(marks)})" if marks else ""
    return f"— {text} — {tail}{mark} [{handle(rec.id)}]"


def _claim_line(rec: AnyRecord, names: dict[str, str]) -> str:
    when = ru_date(rec.recorded_at)
    tail = f", {when}" if when else ""
    return f"— {_who(rec, names, claim=True)}: {_q(_shown(rec))}{tail} [{handle(rec.id)}]"


def render_what(w: _What) -> list[str]:
    lines: list[str] = []
    for g in w.groups:
        if not g.items and not g.competing:
            continue
        title = "тебя" if g.title == "тебя" else f"«{g.title}»"
        tag = f" [{handle(g.entity_id)}]" if g.entity_id and len(w.groups) > 1 else ""
        lines.append(f"Про {title}{tag} помню:")
        kind = None
        for r in g.items:
            if r.kind is not kind:
                kind = r.kind
                lines.append(f"{_KIND_TITLE.get(kind, kind.value)}:")
            lines.append(_line(r, w.names, w.uses.get(r.id)))
        if g.competing:
            lines.append("Расходится — разные люди говорят разное:")
            lines += [_claim_line(r, w.names) for r in g.competing]
    if w.hidden:
        n = w.hidden
        lines.append(
            f"Ещё {n} {_plural(n, 'запись', 'записи', 'записей')} на личные темы — "
            "скажу, если спросишь о них прямо."
        )
    if not lines:
        return [f"Про «{w.about}» ничего не помню."]
    return lines


def what_view(conn: sqlite3.Connection, about: str, turn: TurnCtx) -> _What:
    """One pass for a caller that wants the items and the lines; serves what it returns."""
    w = _what(conn, about, turn)
    _serve(conn, turn, [r for g in w.groups for r in (*g.items, *g.competing)])
    return w


def what_ids(w: _What) -> list[str]:
    return [r.id for g in w.groups for r in (*g.items, *g.competing)]


def what(conn: sqlite3.Connection, about: str, turn: TurnCtx) -> list[str]:
    """«что ты помнишь про X» (§17) in plain Russian lines with handles."""
    return render_what(what_view(conn, about, turn))


# ---------------------------------------------------------------- get / search (§10.4)


def get_item(conn: sqlite3.Connection, item_id: str, turn: TurnCtx) -> Item | None:
    """`api.get`: one usable record the turn may see, or None. Writes a `served` row."""
    rid = (handles_in(item_id) or [item_id.strip()])[0]
    rec = ledger.get_record(conn, rid)
    if rec is None or not _usable(rec):
        return None
    use = policy.decide(conn, rid, turn)
    if not use.allowed:
        return None
    _serve(conn, turn, [rec])
    return _item(rec, _names(conn, [_speaker(rec)]), use)


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    turn: TurnCtx,
    kinds: Sequence[RecordKind] | None = None,
    as_of: str | None = None,
    include_archived: bool = False,
    limit: int = 20,
) -> list[Item]:
    """`api.search`, lexical only (no fusion or vectors yet): names through resolve,
    other words through FTS; use-now before rank; send turns mark held items."""
    found = _find(conn, query, include_archived=include_archived, min_cover=0.0)
    if found.terms:
        ids = [rid for _s, rid in found.scored]
    else:
        ids = _about(conn, [e.entity_id for e in found.entities])
    recs = _records(conn, ids)
    keep = [
        recs[i]
        for i in ids
        if i in recs
        and recs[i].kind is not RecordKind.ENTITY
        and _usable(recs[i], include_archived=include_archived)
        and (not kinds or recs[i].kind in kinds)
        and (not as_of or (recs[i].recorded_at or "") <= as_of)
    ]
    uses = policy.decide_many(conn, [r.id for r in keep], turn) if keep else {}
    keep = [r for r in keep if uses[r.id].allowed][:limit]
    _serve(conn, turn, keep)
    names = _names(conn, [_speaker(r) for r in keep])
    return [_item(r, names, uses[r.id]) for r in keep]


# ---------------------------------------------------------------- where from (§17)

_STATUS_LINE = {
    Status.SUPERSEDED: "Сейчас не используется: заменено новым.",
    Status.RETRACTED: "Сейчас не используется: ты сказал, что это неправда.",
    Status.NEEDS_REVIEW: "Сейчас не используется: опора под этим пропала, жду проверки.",
}


def _source_words(kind: str | None, chat_id: str | None) -> str:
    k = (kind or "").casefold()
    if k in _SOURCES:
        return _SOURCES[k]
    if (chat_id or "").startswith("tg:"):
        return "Telegram"
    return k or "неизвестно откуда"


def _raw_rows(conn: sqlite3.Connection, ids: Iterable[str]) -> list[sqlite3.Row]:
    want = list(dict.fromkeys(i for i in ids if i))
    if not want:
        return []
    marks = ", ".join("?" for _ in want)
    return conn.execute(
        "SELECT id, rev, status, source_kind, source_ref, chat_id, turn_id, author_tier,"
        f" author_entity_id, ts FROM raw_chunks WHERE id IN ({marks}) ORDER BY ts",  # noqa: S608
        want,
    ).fetchall()


def _raw_line(row: sqlite3.Row, names: dict[str, str]) -> str:
    src = _source_words(row["source_kind"], row["chat_id"])
    chat = row["chat_id"]
    where = src
    if chat and chat != "cube" and src != "разговор с Кубом":
        where = f"{src}, чат {chat}"
    tier = AuthorTier(row["author_tier"])
    if tier in _USER_TIERS:
        who = "ты сам"
    elif tier is AuthorTier.OTHER:
        who = names.get(row["author_entity_id"] or "") or "другой человек"
    elif tier is AuthorTier.USER_ROOM:
        who = "ты вслух, в комнате"
    else:
        who = "инструмент или страница" if tier is AuthorTier.TOOL else "Куб"
    when = ru_date(row["ts"])
    hhmm = row["ts"][11:16] if row["ts"] and len(row["ts"]) >= 16 else ""
    stamp = f"{when}, {hhmm}" if when and hhmm else (when or "")
    gone = " (сообщение уже забыто)" if row["status"] != "active" else ""
    return f"Откуда: {where} · написал: {who}{', ' + stamp if stamp else ''}{gone}"


def lineage(conn: sqlite3.Connection, item_id: str, turn: TurnCtx) -> Lineage:
    """`api.where_from`: lineage in plain words - quote, chat, date, who said it."""
    from .api import Lineage

    rid = (handles_in(item_id) or [item_id.strip()])[0]
    rec = ledger.get_record(conn, rid)
    if rec is None:
        return Lineage(item_id=rid, lines=["Такой записи нет."])
    if rec.status in (Status.FORGOTTEN, Status.PURGED):
        return Lineage(item_id=rid, lines=["Это забыто — откуда оно, больше не показываю."])
    # a retired record keeps the use-now fence
    use = (policy.decide if rec.status in USABLE else policy.decide_retired)(conn, rid, turn)
    if not use.allowed:
        return Lineage(
            item_id=rid,
            lines=["Это запись на личную тему — скажу, откуда она, если спросишь о ней прямо."],
        )

    raws = [getattr(rec, "source_raw_id", None)]
    parents: list[str] = []
    for r in conn.execute(
        "SELECT to_id FROM edges WHERE from_id = ? AND etype = 'derived_from'", (rid,)
    ):
        parents.append(r[0])
    raw_rows = _raw_rows(conn, [*raws, *parents])
    raw_ids = {r["id"] for r in raw_rows}
    support = [
        r[0]
        for r in conn.execute(
            "SELECT member_record_id FROM justifications WHERE target_record_id = ?"
            " UNION SELECT from_id FROM edges WHERE to_id = ? AND etype = 'supports'",
            (rid, rid),
        )
    ]
    later = [
        r[0]
        for r in conn.execute(
            "SELECT from_id FROM edges WHERE to_id = ? AND etype = 'supersedes'", (rid,)
        )
    ]
    others = _records(conn, [p for p in parents if p not in raw_ids] + support + later)
    shown_others = {
        i: o
        for i, o in others.items()
        if _usable(o) and o.kind is not RecordKind.ENTITY and policy.use_now(conn, i, turn)
    }
    names = _names(
        conn,
        [_speaker(rec), *(r["author_entity_id"] for r in raw_rows)]
        + [_speaker(o) for o in shown_others.values()],
    )

    lines = [f"{_q(_shown(rec))} [{handle(rid)}]"]
    when = ru_date(rec.recorded_at)
    lines.append(f"Кто сказал: {_who(rec, names)}" + (f" · записано {when}" if when else ""))
    quote = getattr(rec, "quote_orig", None) or rec.origin_quote
    if quote and quote.strip() != _shown(rec).strip():
        lines.append(f"Цитата: {_q(quote)}")
    elif quote:
        lines.append(f"Цитата: {_q(quote)} — дословно")
    lines += [_raw_line(r, names) for r in raw_rows]
    for i in parents:
        if i in shown_others:
            lines.append(f"Выведено из: {_q(_shown(shown_others[i]))} [{handle(i)}]")
    for i in support:
        if i in shown_others:
            lines.append(f"Опирается на: {_q(_shown(shown_others[i]))} [{handle(i)}]")
    if rec.status in _STATUS_LINE:
        line = _STATUS_LINE[rec.status]
        new = next((i for i in later if i in shown_others), None)
        if new is not None:
            line = f"Сейчас не используется: заменено на {_q(_shown(shown_others[new]))} [{handle(new)}]."
        lines.append(line)
    elif rec.status is Status.ARCHIVED:
        lines.append("Сейчас в архиве: прошлое, не текущее.")
    if rec.review:
        lines.append("Надо перепроверить: что-то из опор изменилось.")
    if not raw_rows and not parents:
        lines.append("Сообщения-источника нет: запись сделана напрямую.")
    sources = [ReadRef(id=r["id"], version=int(r["rev"]), src=ReadSrc.RAW) for r in raw_rows]
    sources += [ReadRef(id=o.id, version=o.version) for o in shown_others.values()]
    return Lineage(item_id=rid, lines=lines, sources=sources)


def where_from(conn: sqlite3.Connection, item_id: str, turn: TurnCtx) -> list[str]:
    """«откуда это» (§17) in plain Russian lines."""
    return lineage(conn, item_id, turn).lines


# ---------------------------------------------------------------- forget (§9, §17)


def _job(conn: sqlite3.Connection, forget_id: str) -> dict[str, Any] | None:
    import json

    row = conn.execute(
        "SELECT payload, state, attempts, not_before FROM jobs WHERE idempotency_key = ?",
        (f"{forget.PURGE_JOB}:{forget_id}",),
    ).fetchone()
    if row is None:
        return None
    return {
        "payload": json.loads(row[0]),
        "state": row[1],
        "attempts": row[2],
        "not_before": row[3],
    }


def _result(**kw: Any) -> ForgetResult:
    from .api import ForgetResult

    return ForgetResult(**kw)


def _his_word_needed(turn: TurnCtx) -> bool:
    """§6.1: in a tainted turn (and in any turn that is not the user's message) the user's own
    words must name the target."""
    return turn.is_tainted or turn.origin is not TurnOrigin.OWNER


def _target_names(conn: sqlite3.Connection, rec: AnyRecord) -> list[str]:
    """What names a record in their words: its subject's aliases, its value, the entity's own."""
    ents = [
        x
        for x in (
            rec.id if rec.kind is RecordKind.ENTITY else None,
            getattr(rec, "subject_id", None),
            getattr(rec, "value_entity_id", None),
            getattr(rec, "about_entity_id", None),
        )
        if x
    ]
    names: list[str] = []
    if ents:
        marks = ", ".join("?" for _ in ents)
        names += [
            r[0]
            for r in conn.execute(
                f"SELECT alias FROM aliases WHERE entity_id IN ({marks})",  # noqa: S608
                ents,
            )
        ]
    if getattr(rec, "value_text", None):
        names.append(rec.value_text)  # type: ignore[union-attr]
    return names


def _named_by_him(
    conn: sqlite3.Connection, recs: Sequence[AnyRecord], turn: TurnCtx, quote: str
) -> bool:
    for rec in recs:
        labels = Label(rec.labels)
        names = _target_names(conn, rec)
        by_name = any(policy.because_quote_ok(quote, turn, labels=labels, target=n) for n in names)
        by_topic = bool(labels) and policy.because_quote_ok(quote, turn, labels=labels, target=None)
        if not (by_name or by_topic):
            return False
    return True


def _hint(conn: sqlite3.Connection, entity_id: str, turn: TurnCtx) -> str:
    """One feature that tells a namesake apart: a usable, allowed record about it."""
    ids = _about(conn, [entity_id])
    recs = [
        r for r in _records(conn, ids).values() if _usable(r) and r.kind is not RecordKind.ENTITY
    ]
    uses = policy.decide_many(conn, [r.id for r in recs], turn) if recs else {}
    sens = policy.sensitive_mask()
    for r in sorted(recs, key=lambda r: (r.recorded_at or "", r.id)):
        if uses[r.id].allowed and not (Label(r.labels) & sens):
            return _q(_shown(r).replace("?", ""))
    row = conn.execute(
        "SELECT etype, stable_ids FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()
    return f"({row[0]})" if row else ""


def _ask_entities(
    conn: sqlite3.Connection, cands: Sequence[entities.Resolved], turn: TurnCtx
) -> str:
    opts = [
        f"«{c.display_name}» — {_hint(conn, c.entity_id, turn)} [{handle(c.entity_id)}]"
        for c in cands[:_MAX_ASK]
    ]
    name = cands[0].display_name
    return f"Про какого «{name}» забыть всё: " + _or(opts) + "?"


def _ask_records(conn: sqlite3.Connection, recs: Sequence[AnyRecord], turn: TurnCtx) -> str:
    uses = policy.decide_many(conn, [r.id for r in recs], turn)
    opts = []
    for r in recs[:_MAX_ASK]:
        text = _q(_shown(r).replace("?", "")) if uses[r.id].allowed else "запись на личную тему"
        opts.append(f"{text} [{handle(r.id)}]")
    return "Что именно забыть: " + _or(opts) + "?"


def _or(opts: Sequence[str]) -> str:
    if len(opts) == 1:
        return opts[0]
    return ", ".join(opts[:-1]) + " или " + opts[-1]


@dataclass
class _Target:
    ids: list[str] = field(default_factory=list)
    scope: ForgetScope = ForgetScope.ITEM
    recs: list[AnyRecord] = field(default_factory=list)
    title: str = ""
    ask: str | None = None
    missing: str | None = None
    one_fact: bool = False  # several copies of one proposition (FORGET DUPLICATES)


def _forget_query(target: str) -> tuple[str, bool]:
    """(the words that name what to forget, whether they order the whole topic). A sentence
    or filler before the verb is not the target («Всё, обсудили. Забудь насовсем.», «Всё-таки
    забудь про накопления»); «всё, что я рассказывала про X» is «всё про X»."""
    text = _REASON_TAIL.sub("", (target or "").strip())
    m = _FORGET_VERB.search(text)
    if m and m.start() > 0:
        before = text[: m.start()]
        words = _words(before)
        back = _BACK_REF.search(text[m.end() :])  # «Дима переехал. Забудь его адрес»
        if all(w in _FILLER for w in words) or (re.search(r"[.!?…]\s*$", before) and not back):
            text = text[m.end() :]
        elif not _words(text[m.end() :]):  # «Про давление забудь»: the target before the verb
            text = before
    elif m:
        said = _SAID_ABOUT.match(text[m.end() :])  # «Забудь, что я говорила про X»
        if said:
            text = "всё про " + text[m.end() + said.end() :]
    told = _TOLD_ABOUT.search(text)
    whole = bool(told) or bool(_ALL_ABOUT.search(f" {text} ")) or bool(_ALL_WORD.search(text))
    if told:
        text = text[: told.start()] + " " + text[told.end() :]
    return text, whole


def _slot_cue(query: str) -> tuple[str, str | None]:
    """The user's question word for the value's slot («до какого числа» → date) taken out of the
    topic words; the slot ranks a record holding such a value first."""
    for pat, slot in _SLOT_CUES:
        if pat.search(query):
            return " ".join(pat.sub(" ", query).split()), slot
    return query, None


def _slot_ok(rec: AnyRecord, slot: str | None) -> bool:
    if slot is None:
        return False
    if slot == "date" and (getattr(rec, "valid_to", None) or getattr(rec, "valid_from", None)):
        return True
    toks = ledger.value_tokens(
        " ".join(str(getattr(rec, a, None) or "") for a in ("value_text", "value_norm", "text_en"))
    )
    want = "date:" if slot == "date" else "num:"
    return any(t.startswith(want) for t in toks)


def _hits(terms: Iterable[str], words: Iterable[str]) -> set[str]:
    ws = list(words)
    return {t for t in terms if any(_same_word(t, w) for w in ws)}


def _prefix_expr(terms: Iterable[str]) -> str:
    """An FTS expression of 4-letter prefixes: a superset of what `_same_word` accepts
    («накопления» ~ «накопим»); Python checks every candidate after."""
    parts: dict[str, None] = {}
    for t in terms:
        w = _norm(t)
        if len(w) >= 4:
            parts['"' + w[:4].replace('"', '""') + '"*'] = None
        elif w:
            parts['"' + w.replace('"', '""') + '"'] = None
    return " OR ".join(parts)


def _prefix_records(conn: sqlite3.Connection, terms: Sequence[str], limit: int = 500) -> list[str]:
    expr = _prefix_expr(terms)
    if not expr:
        return []
    try:
        rows = conn.execute(
            "SELECT r.id FROM fts_records f JOIN records r ON r.fts_rowid = f.rowid"
            " WHERE fts_records MATCH ? LIMIT ?",
            (expr, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


_RAW_COLS = "c.id, c.text, c.chat_id, c.turn_id, c.ts, c.source_kind"
_RAW_LIVE = "c.status = 'active' AND c.quarantined = 0 AND c.junk = 0"
_REPLY_KINDS = ("cube", "bot", "report")


def _prefix_raws(conn: sqlite3.Connection, terms: Sequence[str], limit: int = 500) -> list[Any]:
    expr = _prefix_expr(terms)
    if not expr:
        return []
    try:
        return conn.execute(
            f"SELECT {_RAW_COLS} FROM fts_raw f JOIN raw_chunks c ON c.fts_rowid = f.rowid"  # noqa: S608
            f" WHERE fts_raw MATCH ? AND {_RAW_LIVE} LIMIT ?",
            (expr, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


_CODE_TOKEN = re.compile(r"\w[\w-]*\w")
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _codes(text: str | None) -> set[str]:
    """Distinctive tokens (a deposit «ЮУАЦФМШГ-4860», a contract «4860-4216», a handle with
    digits): the same code in another message is the same matter. Plain numbers and days
    are not codes."""
    out: set[str] = set()
    for m in _CODE_TOKEN.finditer(text or ""):
        tok = m.group().casefold()
        if len(tok) < 5 or tok.isdigit() or _ISO_DAY.match(tok):
            continue
        if any(c.isdigit() for c in tok) and (any(c.isalpha() for c in tok) or "-" in tok):
            out.add(tok)
    return out


def _code_candidates(conn: sqlite3.Connection, code: str) -> Iterable[Any]:
    """Live raws that may hold `code`: an `fts_raw` phrase of the code's own stemmed tokens
    (the index form, `lexical.stem`), so a lookup costs an index probe, not a scan of every
    message (a full `instr` scan was ~50 ms per code at 10^5 raws). A broken FTS query
    falls back to the scan on the code's digits."""
    from .recall import lexical

    phrase = lexical.stem(code)
    if phrase:
        try:
            return conn.execute(
                f"SELECT {_RAW_COLS} FROM fts_raw f JOIN raw_chunks c ON c.fts_rowid = f.rowid"  # noqa: S608
                f" WHERE fts_raw MATCH ? AND {_RAW_LIVE}",
                (f'"{phrase}"',),
            ).fetchall()
        except sqlite3.OperationalError:
            pass
    digits = max(re.findall(r"\d+", code), key=len)
    return conn.execute(
        f"SELECT {_RAW_COLS} FROM raw_chunks c WHERE {_RAW_LIVE}"  # noqa: S608
        " AND instr(c.text, ?) > 0",
        (digits,),
    ).fetchall()


def _raws_with_codes(
    conn: sqlite3.Connection, codes: set[str], cache: dict[str, list[Any]] | None = None
) -> list[Any]:
    out: dict[str, Any] = {}
    cache = {} if cache is None else cache
    for code in sorted(codes):
        if code not in cache:
            cache[code] = [
                row for row in _code_candidates(conn, code) if code in (row[1] or "").casefold()
            ]
        out.update((row[0], row) for row in cache[code])
    return list(out.values())


def _record_codes(rec: AnyRecord) -> set[str]:
    return _codes(" ".join(str(getattr(rec, a, None) or "") for a in ("value_text", "quote_orig")))


def _positive_words(rec: AnyRecord) -> set[str]:
    """A record's words with the value the user's contrast retires taken out of its quotes."""
    texts = [_english(rec)]
    for attr in ("quote_orig", "origin_quote", "keywords_ru", "keywords_en", "value_text"):
        v = getattr(rec, attr, None)
        if isinstance(v, str) and v:
            texts.append(hot.positive_text(v) if attr in ("quote_orig", "origin_quote") else v)
    return {w for t in texts for w in _words(t)}


def _lineage(conn: sqlite3.Connection, rid: str) -> set[str]:
    return forget._lineage_raw(conn, rid)  # the forget module's own lineage


def _users(conn: sqlite3.Connection, raw_id: str) -> set[str]:
    return forget._raw_users(conn, raw_id)


def _supersedes(conn: sqlite3.Connection, rid: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM edges WHERE from_id = ? AND etype = 'supersedes' LIMIT 1", (rid,)
    ).fetchone()
    return row is not None


def _hours(a: str | None, b: str | None) -> float:
    from datetime import datetime

    try:
        ta = datetime.fromisoformat((a or "").replace("Z", "+00:00"))
        tb = datetime.fromisoformat((b or "").replace("Z", "+00:00"))
    except ValueError:
        return 1e9
    return abs((tb - ta).total_seconds()) / 3600


class _Links:
    """Union-find over record and raw ids for one topic resolution."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def _topic(
    conn: sqlite3.Connection, named: Sequence[entities.Resolved], terms: Sequence[str]
) -> tuple[list[str], list[AnyRecord]]:
    """«забудь всё про <topic>» → (target ids: records and raw chunks, records).

    Seeds are records and raw messages whose own words cover a topic word (their words are
    Russian, the records English: the Russian side - quotes, keywords and the raw messages
    - carries the match; an old value the user's «не X, а Y» retires never counts). Strong
    links join a seed's matter: a record's own messages, the records learned from a message,
    one exchange (one turn, or the user's message and the Cube's reply right after it), the
    same code (a deposit or contract number) in another message, and two seed messages of one
    chat within hours. A matter is the topic when its seeds together cover the topic words
    (2/3, as an item match); «всё» takes every such matter. Never an entity (that is «всё
    про X») and never a newer value of the slot (A14) unless its own words name the topic."""
    want = max(1, -(-2 * len(terms) // 3))
    eids = [e.entity_id for e in named]
    about = set(_about(conn, eids)) if eids else None
    names = {w for a in forget._aliases_of(conn, eids) for w in _words(a)} if eids else set()
    history = (Status.SUPERSEDED, Status.RETRACTED)

    def fits(r: AnyRecord) -> bool:
        return r.kind is not RecordKind.ENTITY and (_usable(r) or r.status in history)

    recs = _records(conn, [*_prefix_records(conn, terms), *(about or ())])
    cover: dict[str, set[str]] = {}
    for rid, r in recs.items():
        if fits(r) and (about is None or rid in about):
            hit = _hits(terms, _positive_words(r))
            if hit:
                cover[rid] = hit
    rows: dict[str, Any] = {}
    for row in _prefix_raws(conn, terms):
        words = _words(hot.positive_text(row[1]))
        if names and not _hits(names, words):
            continue
        hit = _hits(terms, words)
        if hit:
            cover[row[0]] = hit
            rows[row[0]] = row
    if not cover:
        return [], []

    def raw_row(raw_id: str) -> Any:
        if raw_id not in rows:
            row = conn.execute(
                f"SELECT {_RAW_COLS} FROM raw_chunks c WHERE c.id = ? AND {_RAW_LIVE}",  # noqa: S608
                (raw_id,),
            ).fetchone()
            if row is None:
                return None
            rows[raw_id] = row
        return rows[raw_id]

    def exchange(row: Any) -> list[Any]:
        out = []
        if row[3]:
            out += conn.execute(
                f"SELECT {_RAW_COLS} FROM raw_chunks c WHERE c.turn_id = ? AND c.id != ?"  # noqa: S608
                f" AND {_RAW_LIVE}",
                (row[3], row[0]),
            ).fetchall()
        if row[5] in ("owner", "owner_message"):
            nxt = conn.execute(
                f"SELECT {_RAW_COLS} FROM raw_chunks c WHERE c.chat_id IS ? AND c.ts > ?"  # noqa: S608
                " AND c.status = 'active' ORDER BY c.ts LIMIT 1",
                (row[2], row[4]),
            ).fetchone()
            if nxt is not None and nxt[5] in _REPLY_KINDS and raw_row(nxt[0]) is not None:
                out.append(nxt)
        return out

    links = _Links()
    codes: dict[str, list[Any]] = {}
    seen: set[str] = set(cover)
    frontier = list(cover)
    for _depth in range(4):
        nxt: list[str] = []
        for node in frontier:
            linked: list[str] = []
            if node in recs:
                linked += [x for x in _lineage(conn, node) if raw_row(x) is not None]
                linked += [r[0] for r in _raws_with_codes(conn, _record_codes(recs[node]), codes)]
            elif raw_row(node) is not None:
                row = rows[node]
                users = _users(conn, node)
                recs.update(_records(conn, users - set(recs)))
                linked += [u for u in users if u in recs and fits(recs[u])]
                for p in exchange(row):
                    rows.setdefault(p[0], p)
                    linked.append(p[0])
                linked += [r[0] for r in _raws_with_codes(conn, _codes(row[1]), codes)]
            for x in linked:
                links.union(node, x)
                if x not in seen:
                    seen.add(x)
                    nxt.append(x)
        frontier = nxt
    seeds = sorted((rows[r] for r in cover if r in rows), key=lambda r: r[4] or "")
    for a, b in zip(seeds, seeds[1:], strict=False):
        if a[2] == b[2] and _hours(a[4], b[4]) <= 6:
            links.union(a[0], b[0])

    got: dict[str, set[str]] = {}
    for node, hit in cover.items():
        got.setdefault(links.find(node), set()).update(hit)
    topics = {root for root, hit in got.items() if len(hit) >= want}
    members = {n for n in seen if links.find(n) in topics}
    out: dict[str, AnyRecord] = {}
    for n in sorted(members):
        r = recs.get(n)
        if r is None or not fits(r):
            continue
        if _supersedes(conn, n) and n not in cover:
            continue  # A14: a newer value is another fact; only its own words bring it in
        out[n] = r
    raws = [n for n in sorted(members) if n in rows and _users(conn, n) <= set(out)]
    return [*out, *raws], list(out.values())


def _one_topic(conn: sqlite3.Connection, recs: Sequence[AnyRecord]) -> bool:
    """Tied candidates of one item forget are one thing to forget (no question) when they
    share a message or a code and are not rival values of one single-valued slot."""
    _plain, rival = _split_competing(conn, list(recs))
    if rival:
        return False
    links = _Links()
    keys: dict[str, str] = {}
    for r in recs:
        links.find(r.id)
        for k in [*(f"raw:{x}" for x in _lineage(conn, r.id)), *_record_codes(r)]:
            if k in keys:
                links.union(r.id, keys[k])
            else:
                keys[k] = r.id
    return len({links.find(r.id) for r in recs}) == 1


def _topic_target(
    conn: sqlite3.Connection,
    named: Sequence[entities.Resolved],
    rest: Sequence[str],
    query: str,
    turn: TurnCtx,
) -> _Target:
    ids, recs = _topic(conn, named, rest)
    if not ids:
        return _Target(missing=f"Не нашёл в памяти ничего про «{query}».")
    # namesakes («переезд Саши», two Sashas) whose records both came in are different people
    by_name: dict[str, list[entities.Resolved]] = {}
    for e in named:
        by_name.setdefault(_norm(e.display_name), []).append(e)
    got = {r.id for r in recs}
    for same in by_name.values():
        hit = [e for e in same if got & set(_about(conn, [e.entity_id]))]
        if len(hit) > 1:
            return _Target(ask=_ask_entities(conn, hit, turn))
    return _Target(ids=ids, scope=ForgetScope.ITEM, recs=recs, title=f"всё про «{query}»")


def _resolve_target(
    conn: sqlite3.Connection, target: str, scope: ForgetScope, turn: TurnCtx
) -> _Target:
    target = (target or "").strip()
    if scope is ForgetScope.CHAT:
        chat = target.removeprefix("chat:").strip()
        return _Target(ids=[chat], scope=scope, title=f"всё из чата {chat}")
    explicit = handles_in(target)
    if explicit:
        recs = list(_records(conn, explicit).values())
        known = {r.id for r in recs}
        raws = [i for i in explicit if i not in known]
        if raws:
            known |= {r["id"] for r in _raw_rows(conn, raws)}
        if len(known) < len(explicit):
            return _Target(missing="Такой записи нет — нечего забывать.")
        if scope is ForgetScope.ABOUT_ENTITY and (
            any(r.kind is not RecordKind.ENTITY for r in recs) or len(recs) != len(explicit)
        ):
            return _Target(missing="Это не человек и не вещь — «всё про» к нему не подходит.")
        return _Target(ids=list(explicit), scope=scope, recs=recs)

    phrase, whole = _forget_query(target)
    query, slot = _slot_cue(_strip_lead(phrase))
    query = _strip_lead(query)
    if not query:
        return _Target(missing="Не понял, что забыть.")
    named, rest = _named(conn, query)
    rest = [w for w in rest if w not in _DESCRIBE]
    if scope is ForgetScope.ABOUT_ENTITY or (named and not rest):
        if not named:
            return _Target(missing=f"Не знаю никого и ничего по имени «{query}».")
        if len(named) > 1:
            return _Target(ask=_ask_entities(conn, named, turn))
        e = named[0]
        rec = ledger.get_record(conn, e.entity_id)
        return _Target(
            ids=[e.entity_id],
            scope=ForgetScope.ABOUT_ENTITY,
            recs=[rec] if rec else [],
            title=f"всё про «{e.display_name}»",
        )
    if whole:  # «всё про <topic>»: every record and message of the topic
        return _topic_target(conn, named, rest, query, turn)
    found = _find(conn, query, include_history=True)
    if not found.scored:  # a topic word, not one fact («забудь про накопления»)
        return _topic_target(conn, named, rest, query, turn)
    if slot is not None:
        found.scored = sorted(
            ((s + (0.5 if _slot_ok(found.recs[r], slot) else 0.0), r) for s, r in found.scored),
            key=lambda x: (-x[0], x[1]),
        )
    best = found.scored[0][0]
    top = [rid for s, rid in found.scored if s == best]
    # «забудь» means the fact, not one copy: identical propositions are one target (the
    # forget takes every copy anyway); one question only when the candidates really differ:
    # rival values of one slot, or things from unrelated messages
    if len(top) > 1 and not forget.same_fact(conn, top):
        tied = [found.recs[i] for i in top]
        if not _one_topic(conn, tied):
            return _Target(ask=_ask_records(conn, tied, turn))
        return _Target(ids=top, scope=ForgetScope.ITEM, recs=tied)
    return _Target(
        ids=top, scope=ForgetScope.ITEM, recs=[found.recs[i] for i in top], one_fact=len(top) > 1
    )


def _affected(conn: sqlite3.Connection, payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(turn ids, session ids) the stop-use touched: lineage turns, served turns and the
    sessions that were served a forgotten id or live in an affected thread (§9.1c, §9.5)."""
    recs = list(payload.get("record_ids") or [])
    raws = list(payload.get("raw_ids") or [])
    threads = list(payload.get("thread_ids") or [])
    turns = set(payload.get("turn_ids") or [])
    sessions: set[str] = set()
    if recs:
        marks = ", ".join("?" for _ in recs)
        for t, s in conn.execute(
            f"SELECT turn_id, session_id FROM served WHERE item_id IN ({marks})",  # noqa: S608
            recs,
        ):
            if t:
                turns.add(t)
            if s:
                sessions.add(s)
    if raws:
        marks = ", ".join("?" for _ in raws)
        turns |= {
            r[0]
            for r in conn.execute(
                f"SELECT turn_id FROM raw_chunks WHERE id IN ({marks}) AND turn_id IS NOT NULL",  # noqa: S608
                raws,
            )
        }
    if threads:
        marks = ", ".join("?" for _ in threads)
        sessions |= {
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT session_id FROM served WHERE thread_id IN ({marks})"  # noqa: S608
                " AND session_id IS NOT NULL",
                threads,
            )
        }
    return sorted(turns), sorted(sessions)


def _what_line(t: _Target, n: int) -> str:
    if t.title:
        return t.title + (f" ({n} {_plural(n, 'запись', 'записи', 'записей')})" if n > 1 else "")
    if len(t.recs) == 1 or (t.recs and t.one_fact):
        return _q(_shown(t.recs[0]))
    return f"{n} {_plural(n, 'запись', 'записи', 'записей')}"


def forget_now(
    conn: sqlite3.Connection,
    target: str,
    *,
    turn: TurnCtx,
    because_quote: str,
    scope: ForgetScope = ForgetScope.ITEM,
    forever: bool = False,
) -> ForgetResult:
    """`api.forget`: resolve, check the user's naming words on a tainted turn, stop use at once
    (`forget.stop_use`). «насовсем» or a sensitive label makes the purge job due now; it
    runs in the worker pool after the call, never inside it (§9.1 F11).

    Tainted turn: no forget verb in the user's own words → `ask_first`; `forever`
    without the user's «насовсем» → `ask_first`, before the target is even resolved."""
    if _his_word_needed(turn):
        words = policy.turn_words(turn)
        if not _FORGET_VERB.search(words):
            return _result(
                outcome="ask_first",
                line="Забывать ты не просил — это из чужого текста. Забыть — скажешь сам?",
            )
        if forever and not _FOREVER_WORD.search(words):
            return _result(
                outcome="ask_first",
                line="Удалить насовсем, без «Вернуть»? Скажи «насовсем» сам — "
                "или просто перестану использовать.",
            )
    if scope is not ForgetScope.CHAT and _object_less(target):
        again = _thread_forget(conn, turn)
        if again is not None:  # «Забудь насовсем» after «верни»: the user's latest forget
            return _forget_again(conn, again, turn=turn, forever=forever)
    t = _resolve_target(conn, target, scope, turn)
    if t.missing:
        return _result(outcome="not_found", line=t.missing)
    if t.ask:
        return _result(outcome="ask_first", line=t.ask)
    if _his_word_needed(turn) and t.scope is not ForgetScope.CHAT:
        recs = t.recs or list(_records(conn, t.ids).values())
        if not recs or not _named_by_him(conn, recs, turn, because_quote):
            return _result(
                outcome="ask_first",
                line="Это пришло не из твоих слов. Что именно забыть — скажешь сам?",
            )
    if (
        t.scope is ForgetScope.CHAT
        and _his_word_needed(turn)
        and not policy.because_quote_ok(because_quote, turn, labels=Label.NONE, target=t.ids[0])
    ):
        return _result(
            outcome="ask_first",
            line="Это пришло не из твоих слов. Забыть всё из этого чата — скажешь сам?",
        )
    try:
        fid = forget.stop_use(conn, t.ids, scope=t.scope, forever=forever)
    except forget.ForgetNotFound:
        return _result(outcome="not_found", line="Это уже не используется — нечего забывать.")
    return _stopped(conn, fid, t, turn)


def _stopped(conn: sqlite3.Connection, fid: str, t: _Target, turn: TurnCtx) -> ForgetResult:
    """The result of a committed stop-use: what stopped, the turns and sessions it touched,
    and the turn's forget mark."""
    job = _job(conn, fid) or {"payload": {}, "not_before": None}
    body = job["payload"]
    record_ids = list(body.get("record_ids") or [])
    turns, sessions = _affected(conn, body)
    hot.note_commit(conn, turn, "forget", [fid])  # backs this turn's «забудь» only
    what_ = _what_line(t, len(record_ids))
    # §9.1 F11: never purge inside the call (the hot budget covers stop-use only); a due
    # purge job runs in the worker pool right after it, «удалил полностью» is purge_status's
    outcome = "stopped"
    if fid in forget.due(conn):
        line = f"Перестал использовать: {what_}. Удаляю полностью…"
    else:
        line = (
            f"Перестал использовать: {what_}. Лежит в корзине {TRASH_HOURS} ч — "
            "можно «Вернуть», потом удалю полностью."
        )
    return _result(
        outcome=outcome,
        forget_id=fid,
        record_ids=record_ids,
        turn_ids=turns,
        sessions_to_restart=sessions,
        line=line,
    )


def _object_less(target: str) -> bool:
    """The user's forget names nothing («Забудь насовсем», «Нет, всё-таки забудь это совсем»)."""
    phrase, _whole = _forget_query(target)
    query, _slot = _slot_cue(_strip_lead(phrase))
    return not _strip_lead(query)


def _thread_forget(conn: sqlite3.Connection, turn: TurnCtx) -> str | None:
    """The forget this thread last made or undid (its marks, newest first), while it is
    still the user's to act on: undone, or stopped and not yet purged. A chat-mode forget
    is not one the user can name by «это»."""
    for (fid,) in conn.execute(
        "SELECT item_id FROM served WHERE thread_id = ? AND via = 'write'"
        " AND item_kind IN ('forget', 'undo') ORDER BY id DESC LIMIT 20",
        (turn.thread_id,),
    ):
        job = _job(conn, fid)
        if job is None:
            continue
        body = job["payload"]
        if body.get("scope") == ForgetScope.CHAT.value:
            return None
        return fid if body.get("stage", "stopped") in ("undone", "stopped") else None
    return None


def _forget_again(
    conn: sqlite3.Connection, fid: str, *, turn: TurnCtx, forever: bool
) -> ForgetResult:
    """An object-less forget on this thread's last forget (keeps_true «верни» → «забудь
    насовсем»). Undone: the same targets, in the same scope, are forgotten again (a new
    forget; `forever` makes its purge due now). Still in the trash: «насовсем» makes the
    purge due now, the pool runs it after the call (§9.1 F11); without it, nothing changes.
    The user's words then named the target; this turn's words only point back at that forget."""
    job = _job(conn, fid) or {"payload": {}}
    body = job["payload"]
    record_ids = list(body.get("record_ids") or [])
    recs = [r for r in _records(conn, record_ids).values() if r.kind is not RecordKind.ENTITY]
    t = _Target(
        ids=list(body.get("target_ids") or []),
        scope=ForgetScope(body.get("scope") or ForgetScope.ITEM.value),
        recs=recs,
        one_fact=len(recs) > 1 and forget.same_fact(conn, [r.id for r in recs]),
    )
    if body.get("stage") == "undone":
        try:
            new = forget.stop_use(conn, t.ids, scope=t.scope, forever=forever)
        except forget.ForgetNotFound:
            return _result(outcome="not_found", line="Это уже не используется — нечего забывать.")
        return _stopped(conn, new, t, turn)
    if forever and fid not in forget.due(conn):
        _purge_now(conn, fid, body)
    return _stopped(conn, fid, t, turn)


def _purge_now(conn: sqlite3.Connection, fid: str, body: dict[str, Any]) -> None:
    """«насовсем» on a forget already in the trash: no «Вернуть» any more. The purge job is
    due at once (`not_before` = its stop time, so `purge_status` reads «удаляю») and the hot
    log lets go of its ids now, as a forever stop-use does."""
    at = str(body.get("stopped_at") or store.now())
    forget._locked(conn, lambda: forget._set_job(conn, fid, not_before=at, until=at))  # noqa: SLF001
    forget._scrub_hot_ops(  # noqa: SLF001
        conn, list(body.get("record_ids") or []) + list(body.get("raw_ids") or [])
    )


# whole words of an undo order; «наверное», «верно», «отменил встречу» are not one
_UNDO_WORDS = re.compile(
    r"(?<!\w)(?:верни|вернуть|вернёшь|вернешь|верните|возврати|отмени|отменить|отмените|"
    r"отмена|undo|restore|bring\s+(?:it\s+)?back|put\s+(?:it\s+)?back)(?!\w)",
    re.IGNORECASE,
)


def _trashed_words(conn: sqlite3.Connection, forget_id: str) -> dict[str, set[str]]:
    """The words each trashed record had before the forget masked it (its trash copy)."""
    out: dict[str, set[str]] = {}
    for t in forget._trash_rows(conn, forget_id, "memory"):
        kind, _, item = t["row_ref"].partition(":")
        body = t["payload"]
        if kind != "record" or not isinstance(body, dict):
            continue
        parts = [body.get("row") or {}, body.get("header") or {}, *(body.get("aliases") or [])]
        texts = [v for part in parts if isinstance(part, dict) for v in part.values()]
        out.setdefault(item, set()).update(
            w for v in texts if isinstance(v, str) for w in _words(v)
        )
    return out


def _names_back(
    conn: sqlite3.Connection, forget_id: str, recs: Sequence[AnyRecord], turn: TurnCtx, words: str
) -> bool:
    """Their words name every record that would come back: by name or topic (the forget rule,
    `_named_by_him`) or by the record's own words before the forget («верни бег»)."""
    mine = [w for w in _content_words(words) if not _UNDO_WORDS.fullmatch(w)]
    before = _trashed_words(conn, forget_id)
    return all(
        _named_by_him(conn, [rec], turn, words) or _covered(mine, before.get(rec.id, set())) > 0
        for rec in recs
    )


def undo(conn: sqlite3.Connection, forget_id: str, *, turn: TurnCtx) -> ForgetResult:
    """`api.undo_forget`: «Вернуть» while in the trash (§9.1 stage 2)."""
    job = _job(conn, forget_id)
    if job is None:
        return _result(outcome="not_found", line="Не нашёл такое забывание.")
    body = job["payload"]
    record_ids = list(body.get("record_ids") or [])
    stage = body.get("stage", "stopped")
    if stage == "undone":
        return _result(
            outcome="undone", forget_id=forget_id, record_ids=record_ids, line="Уже вернул."
        )
    if stage == "purged":
        return _result(
            outcome="purged",
            forget_id=forget_id,
            record_ids=record_ids,
            line="Уже удалил полностью — вернуть не могу.",
        )
    if stage != "stopped":
        return _result(
            outcome="stopped",
            forget_id=forget_id,
            record_ids=record_ids,
            line="Уже удаляю полностью — вернуть не могу.",
        )
    if _his_word_needed(turn):
        # the forget rule: an undo word from the user, and the user's words name what comes back
        words = policy.turn_words(turn)
        trashed = list(_records(conn, record_ids).values())
        if not _UNDO_WORDS.search(words) or not _names_back(conn, forget_id, trashed, turn, words):
            return _result(
                outcome="ask_first", forget_id=forget_id, line="Вернуть забытое — скажешь сам?"
            )
    try:
        forget.undo(conn, forget_id)
    except forget.ForgetError:
        return _result(outcome="not_found", line="Не нашёл такое забывание.")
    hot.note_commit(conn, turn, "undo", [forget_id])
    recs = _records(conn, record_ids)
    shown = [r for r in recs.values() if r.kind is not RecordKind.ENTITY]
    what_ = (
        _q(_shown(shown[0]))
        if len(shown) == 1
        else (f"{len(record_ids)} {_plural(len(record_ids), 'запись', 'записи', 'записей')}")
    )
    turns, sessions = _affected(conn, body)
    return _result(
        outcome="undone",
        forget_id=forget_id,
        record_ids=record_ids,
        turn_ids=turns,
        sessions_to_restart=sessions,
        line=f"Вернул: {what_}.",
    )


def purge_status(conn: sqlite3.Connection, forget_id: str) -> PurgeStatus:
    """`api.purge_status`: in the trash / «удаляю» / purged / undone, with the scope done."""
    from .api import PurgeStatus

    job = _job(conn, forget_id)
    if job is None:
        raise forget.ForgetNotFound(f"unknown forget {forget_id}")
    state = forget.status(conn, forget_id)
    done: list[str] = []
    path = forget.journal_path(conn)
    if path is not None and path.exists():
        for line in forget.read_journal(path):
            if line.forget_id == forget_id:
                done += [s for s in line.scope_done if s not in done]
    retry = job["not_before"] if state == "purging" and job["attempts"] else None
    return PurgeStatus(forget_id=forget_id, state=state, scope_done=done, retry_at=retry)  # type: ignore[arg-type]


_SCOPE_WORDS = {
    "ledger": "записи",
    "fts": "поиск",
    "raw": "исходные сообщения",
    "derived": "выводы и сводки",
    "suppressions": "запрет вернуться из старого",
    "vectors": "векторы",
    "memory": "база памяти",
    "hotlog": "журнал быстрых записей",
    "checkpoint": "файлы базы",
    "scan": "проверка, что не осталось ни слова",
}


def scope_words(done: Sequence[str]) -> str:
    """The purge scope in their words («записи, поиск, векторы»)."""
    return ", ".join(_SCOPE_WORDS.get(s, s) for s in done)


def status_line(st: PurgeStatus) -> str:
    return {
        "in_trash": f"В корзине: перестал использовать, удалю полностью через {TRASH_HOURS} ч.",
        "purging": "Удаляю полностью — ещё не закончил.",
        "purged": "Удалил полностью.",
        "undone": "Вернул — снова используется.",
    }[st.state]


def run_purges(conn: sqlite3.Connection) -> list[PurgeStatus]:
    """`api.run_purges`: due trash entries through the verified purge (retries back off)."""
    return [purge_status(conn, fid) for fid in forget.run_due(conn)]
