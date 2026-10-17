"""Cue parse by code: names and aliases, dates with the turn clock, scope hints, the
episode-hint pattern, sensitive-topic and send-verb lexicons; C3 partial constraints.

Spec: §10.2 step 1, §15.3.

Code only: regular expressions, the trigram name index and the policy lexicons. No model,
no network (`[T recall/test_hot_path::no_model_no_socket]`).
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date, datetime, timedelta

from pydantic import BaseModel, Field

from .. import policy
from ..model import LABEL_NAMES, Label, TurnCtx, TurnOrigin

NAME_FLOOR = 0.6  # trigram similarity a word needs to name a person («Сашу» ~ «Саша»: 0.67)
KIN_SCORE = 0.9  # a person named by a relation word («брат») rather than by a name


class Cue(BaseModel):
    entity_ids: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    scope: dict[str, str] = Field(default_factory=dict)
    episode_hint: bool = False
    sensitive_topics: list[str] = Field(default_factory=list)
    send_turn: bool = False
    constraints: dict[str, str] = Field(default_factory=dict)  # C3
    words: str = ""  # the words the finders search (the user's words or the task's order)
    name_scores: dict[str, float] = Field(default_factory=dict)
    relations: list[str] = Field(default_factory=list)  # relations the question asks about
    value_types: list[str] = Field(default_factory=list)  # where → place, when → date, …
    self_ref: bool = False  # first person («я», «мы», «my»): the user's own facts are the subject
    past: bool = False  # «был», «жила», «раньше», «used to»: the answer is not a plan
    asks_source: bool = False  # «кто сказал», "who told me": the answer is who said it
    ruled_out: list[str] = Field(default_factory=list)  # namesakes the question does not mean
    namesakes: dict[str, str] = Field(default_factory=dict)  # same-name people still asked


# ---------------------------------------------------------------- dates

_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7,
    "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH_RX = (
    r"(январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йяе]|июн[ьяе]|июл[ьяе]|август[ае]?"
    r"|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе]"
    r"|jan(?:uary)?|feb(?:ruary)?|march|mar|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})\s+{_MONTH_RX}\b")
_MONTH_DAY = re.compile(rf"\b{_MONTH_RX}\s+(\d{{1,2}})\b")
_IN_MONTH = re.compile(rf"\b(?:в|во|in|of|this|last)\s+{_MONTH_RX}\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DOTTED = re.compile(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b")
_AGO = re.compile(
    r"\b(\d+|пару|неделю|месяц|год)\s*(дн[еяй]+|день|недел[июь]*|месяц[аев]*|год[а]?|лет"
    r"|days?|weeks?|months?|years?)?\s*(назад|ago)\b"
)
_RELATIVE: tuple[tuple[re.Pattern[str], int, int], ...] = (
    (re.compile(r"\bпозавчера\b|\bday before yesterday\b"), -2, -2),
    (re.compile(r"\bпослезавтра\b|\bday after tomorrow\b"), 2, 2),
    (re.compile(r"\bвчера\b|\byesterday\b"), -1, -1),
    (re.compile(r"\bсегодня\b|\btoday\b|\btonight\b"), 0, 0),
    (re.compile(r"\bзавтра\b|\btomorrow\b"), 1, 1),
    (re.compile(r"\bнедавно\b|\brecently\b|\blately\b"), -30, 0),
)
_SEASONS = {"зим": (12, 2), "весн": (3, 5), "лет": (6, 8), "осен": (9, 11)}
_SEASON_RX = re.compile(r"\b(зимой|весной|летом|осенью|winter|spring|summer|autumn|fall)\b")
_SEASON_EN = {"winter": "зим", "spring": "весн", "summer": "лет", "autumn": "осен", "fall": "осен"}


def _month(token: str) -> int:
    t = token.casefold()
    for stem in sorted(_MONTHS, key=len, reverse=True):
        if t.startswith(stem):
            return _MONTHS[stem]
    return 0


def _span(a: date, b: date) -> str:
    return a.isoformat() if a == b else f"{a.isoformat()}..{b.isoformat()}"


def _month_span(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def turn_day(now: str | None) -> date:
    """The user's local calendar day by the turn clock (§15.1 «a day»)."""
    if not now:
        return datetime.now().astimezone().date()
    try:
        return datetime.fromisoformat(now.replace("Z", "+00:00")).astimezone().date()
    except ValueError:
        return datetime.now().astimezone().date()


AHEAD_MONTHS = 3  # a month name this close ahead may mean the coming one («где будет в ноябре»)


def parse_dates(text: str, *, now: str | None, ahead: bool = False) -> list[str]:
    """Day or `start..end` spans the words name, resolved with the turn clock. A bare month
    name means its last occurrence; with `ahead` (the words are not in the past tense) a
    month at most AHEAD_MONTHS ahead also gives its coming span, second, so a question
    about a plan («куда летит в ноябре») meets it and the past reading keeps first place."""
    t = text.casefold()
    today = turn_day(now)
    out: list[str] = []

    def add(a: date | None, b: date | None = None) -> None:
        if a is None:
            return
        s = _span(a, b or a)
        if s not in out:
            out.append(s)

    for rx, lo, hi in _RELATIVE:
        if rx.search(t):
            add(today + timedelta(days=lo), today + timedelta(days=hi))
    week0 = today - timedelta(days=today.weekday())
    if re.search(r"на прошлой неделе|last week", t):
        add(week0 - timedelta(days=7), week0 - timedelta(days=1))
    if re.search(r"на этой неделе|this week", t):
        add(week0, week0 + timedelta(days=6))
    if re.search(r"на следующей неделе|next week", t):
        add(week0 + timedelta(days=7), week0 + timedelta(days=13))
    if re.search(r"в прошлом месяце|last month", t):
        prev = today.replace(day=1) - timedelta(days=1)
        add(*_month_span(prev.year, prev.month))
    if re.search(r"в прошлом году|last year", t):
        add(date(today.year - 1, 1, 1), date(today.year - 1, 12, 31))
    for m in _AGO.finditer(t):
        n_txt, unit = m.group(1), (m.group(2) or m.group(1))
        n = int(n_txt) if n_txt.isdigit() else 2 if n_txt == "пару" else 1
        days = 1
        if re.match(r"недел|week", unit):
            days = 7
        elif re.match(r"месяц|month", unit):
            days = 30
        elif re.match(r"год|лет|year", unit):
            days = 365
        mid = today - timedelta(days=n * days)
        slack = max(1, n * days // 4)
        add(mid - timedelta(days=slack), min(today, mid + timedelta(days=slack)))
    for m in _ISO.finditer(t):
        add(_safe(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _DOTTED.finditer(t):
        y = int(m.group(3)) if m.group(3) else today.year
        add(_safe(y + 2000 if y < 100 else y, int(m.group(2)), int(m.group(1))))
    days_named: set[int] = set()
    for m in _DAY_MONTH.finditer(t):
        add(_safe(today.year, _month(m.group(2)), int(m.group(1))))
        days_named.add(_month(m.group(2)))
    for m in _MONTH_DAY.finditer(t):
        add(_safe(today.year, _month(m.group(1)), int(m.group(2))))
        days_named.add(_month(m.group(1)))
    for m in _IN_MONTH.finditer(t):
        mon = _month(m.group(1))
        if mon and mon not in days_named:
            year = today.year if mon <= today.month else today.year - 1
            add(*_month_span(year, mon))
            if ahead and 0 < (mon - today.month) % 12 <= AHEAD_MONTHS:
                add(*_month_span(today.year + (mon < today.month), mon))
    for m in _SEASON_RX.finditer(t):
        key = _SEASON_EN.get(m.group(1), m.group(1)[:4].rstrip("о"))
        key = next((k for k in _SEASONS if key.startswith(k)), "")
        if not key:
            continue
        a, b = _SEASONS[key]
        year = today.year if a <= today.month else today.year - 1
        start = date(year, a, 1)
        end = _month_span(start.year + (b < a), b)[1]
        add(start, min(end, today) if start <= today else end)
    return out


# ---------------------------------------------------------------- episode hints (C3)

_EPISODE_HINT = re.compile(
    r"помнишь|напомни,? как|тогда,? когда|в тот раз|того,? (?:кто|с кем|который)|ту,? (?:что|которую)"
    r"|тот (?:самый|раз)|той (?:встреч|поездк)|remember (?:when|that|the)|that time|the one (?:who|we)",
)
_EVENT_TYPES = (
    ("meeting", r"встрет|встреч|увидеться|meet"),
    ("dinner", r"ужин|обед|завтрак|кофе|dinner|lunch|coffee|breakfast"),
    ("trip", r"поездк|поехал|съездил|путешеств|командиров|отпуск|trip|travel|vacation"),
    ("call", r"созвон|звонил|звонок|позвонил|\bcall"),
    ("event", r"концерт|выставк|вечеринк|праздник|день рождени|concert|exhibition|party"),
)
_OUTCOMES = (
    ("cancelled", r"сорвал|не получил|не вышло|отменил|отмен|fell through|cancel|didn.t happen"),
    ("postponed", r"перенес|перенёс|отложил|postpon|moved to|rescheduled"),
    ("done", r"состоял|прош[её]л|прошл[аио]|получилось|happened|went well"),
)
_CAUSE = re.compile(r"(?:из-за|because of|due to|потому что|так как)\s+([^,.;!?]{2,40})")


def episode_constraints(text: str, entity_ids: list[str], dates: list[str]) -> dict[str, str]:
    """C3 partial constraints (§15.3): event type, outcome, cause, participant, vague time."""
    t = text.casefold()
    out: dict[str, str] = {}
    for name, rx in _EVENT_TYPES:
        if re.search(rx, t):
            out["event_type"] = name
            break
    for name, rx in _OUTCOMES:
        if re.search(rx, t):
            out["outcome"] = name
            break
    m = _CAUSE.search(t)
    if m:
        out["cause"] = m.group(1).strip()
    if entity_ids:
        out["participant"] = ",".join(entity_ids)
    if dates:
        out["time"] = dates[0]
    return out


# ---------------------------------------------------------------- parse


def _names(conn: sqlite3.Connection | None, text: str) -> dict[str, float]:
    if conn is None or not text.strip():
        return {}
    from . import lexical

    try:
        hits = lexical.search_names(conn, text, limit=8)
    except sqlite3.Error:
        return {}
    names = {eid: s for eid, s in hits if s >= NAME_FLOOR and eid != "self"}
    try:
        for eid in _possessive_names(conn, text):
            names[eid] = max(names.get(eid, 0.0), POSSESSIVE_SCORE)
    except sqlite3.Error:
        pass
    try:
        for eid, s in _same_full_name(conn, names).items():
            names.setdefault(eid, s)
    except sqlite3.Error:
        pass
    return names


# A Russian possessive adjective made from a first name («Лерин отпуск», «Костиной школе»,
# «мамин день рождения») is the name's stem + -ин/-ын + an adjective ending; trigrams of
# «костиной» never meet «костя». The stem with a first-declension nominative ending («кост» →
# «костя», «лер» → «лера») is looked up as an exact first word of a stored alias only.
_POSSESSIVE = re.compile(r"^([а-яё]{2,}?)(?:ин|ын)(?:|а|о|у|ы|ой|ого|ому|ым|ых|ыми|ую|ом|ою)$")
POSSESSIVE_SCORE = 0.95


def _possessive_names(conn: sqlite3.Connection, text: str) -> list[str]:
    noms: list[str] = []
    for tok in re.findall(r"[а-яё]+", text.casefold()):
        m = _POSSESSIVE.match(tok)
        if m:
            noms += [m.group(1) + end for end in ("а", "я") if m.group(1) + end not in noms]
    out: list[str] = []
    for nom in noms[:6]:
        for (eid,) in conn.execute(
            "SELECT DISTINCT a.entity_id FROM aliases a JOIN records r ON r.id = a.entity_id"
            " WHERE (a.alias_norm = ? OR (a.alias_norm > ? AND a.alias_norm < ?))"
            " AND r.status IN ('active', 'archived') LIMIT 4",
            (nom, nom + " ", nom + "!"),
        ):
            if eid != "self" and eid not in out:
                out.append(eid)
    return out


def _same_full_name(conn: sqlite3.Connection, names: dict[str, float]) -> dict[str, float]:
    """One person minted as several entities (a contact saved under a nickname, plus a chat
    member with the same full name in Latin script): a person the question resolved to brings
    the other person entities under the same full name, in either script, at the same score. A
    full name has two words or more, so a bare first name joins nobody. Read side only."""
    from .. import entities

    if not names:
        return {}
    ids = list(names)
    keys: dict[str, float] = {}
    for eid, name in conn.execute(
        f"SELECT id, display_name FROM entities WHERE etype = 'person' AND id IN ({policy._in(len(ids))})",  # noqa: S608, E501
        ids,
    ):
        norm = entities.normalize_alias(str(name or ""))
        if len(norm.split()) < 2:
            continue
        for k in {norm, *(entities.normalize_alias(t) for t in entities.transliterations(norm))}:
            keys[k] = max(keys.get(k, 0.0), names[eid])
    if not keys:
        return {}
    out: dict[str, float] = {}
    for eid, k in conn.execute(
        "SELECT a.entity_id, a.alias_norm FROM aliases a JOIN entities e ON e.id = a.entity_id"
        " WHERE e.etype = 'person' AND a.kind IN ('name', 'translit')"
        f" AND a.alias_norm IN ({policy._in(len(keys))})",  # noqa: S608
        list(keys),
    ):
        if eid not in names and eid != "self":
            out[eid] = max(out.get(eid, 0.0), keys[k])
    return out


def _topics(ws: list[str]) -> list[str]:
    hit = policy.label_topics(ws) if ws else Label.NONE
    return [n for i, n in enumerate(LABEL_NAMES) if hit & (1 << i)]


# ---------------------------------------------------------------- relations (the pet example)
# A natural question names what it asks for by a relation word («где живёшь», «брат», «кот»,
# «работаешь») and by its question word (where, when, who). These lexicons map such words,
# Russian and English, to the relation vocabulary of config/memory/relations.yaml, so the
# basket can fetch the asked-about entity's facts of that relation. `=w` is an exact word
# (short stems that would also prefix unrelated words), `a b` a phrase, else a prefix stem.
_REL_LEX: tuple[tuple[policy.Lexicon, tuple[str, ...]], ...] = tuple(
    (policy._lex(*words), rels)
    for words, rels in (
        (
            (
                "жив|=жил|=жила|=жили|прожива|снима|снимал|квартир|адрес|район|станци|метро",
                "=живет|=живешь|=живем|=живу|поселил|=однушк",
                "=live|=lives|=lived|=living|address|apartment|=flat|=rent|=rents|=rented",
                "neighbourhood|neighborhood|=home",
            ),
            ("lives_in", "home_address"),
        ),
        (("переезд|переед|переех|relocat|=move|=moves|=moved|=moving",), ("moves_to", "lives_in")),
        (
            ("рожден|=днюха|=днюху|=днюхи|днюх|=родился|=родилась|birthday|=born",),
            ("birthday", "born_in"),
        ),
        (("возраст|сколько лет|=age|how old",), ("age", "birthday")),
        (
            (
                "летит|летишь|летим|=лечу|летал|улета|прилета|полет|поездк|уезжа|уехал|уедет|=едет",
                "отпуск|путешеств|командиров|=trip|travel|=fly|=flies|flying|flight|vacation|holiday",
            ),
            ("travels_to", "visited", "plans_to"),
        ),
        (
            (
                "работа|работе|работу|работаю|работает|работал|компани|должност|начальн|руковод",
                "=шеф|=босс|коллег|=job|=works|=worked|employer|=boss|manager|colleague|position",
            ),
            ("works_at", "job_title", "works_on", "colleague_of"),
        ),
        (("зарплат|=оклад|salary|=paid",), ("salary",)),
        (
            (
                "учусь|учишь|=учится|=учиться|=учитесь|учил|учеб|универ|институт|=курс|=курсы|школ",
                "=study|studies|studying|school|universit|college|=course",
            ),
            ("studies_at", "studies", "enrolled_in"),
        ),
        (
            ("почт|=email|e-mail|телефон|=phone|=handle|=сайт|website",),
            ("has_email", "has_phone", "has_handle", "has_website"),
        ),
        (
            (
                "=брат|=брата|=брату|=братом|=брате|братья|=сестра|=сестры|=сестре|=сестру|сестрой",
                "brother|sister|sibling",
            ),
            ("sibling_of",),
        ),
        (
            (
                "=мама|=маме|=маму|=мамой|=мамы|=мать|=матери|=папа|=папе|=папу|=папой|=папы|=отец",
                "=отца|=отцу|родител|=сын|=сына|=сыну|=дочь|=дочка|=дочке|=дочери|ребен|=дети|=детей",
                "=mom|=mum|mother|=dad|father|parent|=son|daughter|=kid|=kids|child",
            ),
            ("parent_of", "child_of", "relative_of"),
        ),
        (
            ("=жена|=жене|=жену|=женой|=жены|=муж|=мужа|=мужу|=мужем|супруг|=wife|husband|spouse",),
            ("spouse_of", "partner_of"),
        ),
        (("девушк|=парень|=парня|=парню|partner|girlfriend|boyfriend",), ("partner_of",)),
        (("=друг|=друга|=другу|=другом|=друзья|=друзей|=друзьям|подруг|friend",), ("friend_of",)),
        (
            (
                "=кот|=кота|=коту|=котом|=коте|котик|котен|кошк|=кис|собак|=пес|=пса|=псу|=псом",
                "щен|питом|=cat|=cats|=dog|=dogs|=pet|=pets|puppy|kitten",
            ),
            ("has_pet",),
        ),
        (("аллерг|allerg",), ("allergic_to",)),
        (("болез|болеет|диагноз|=condition",), ("has_condition",)),
        (("лекарств|таблет|medicat|medicine|=pill|=pills",), ("takes_medication",)),
        (("встаю|просып|подъем|=wake|wakes|=woke",), ("wakes_at",)),
        (("язык|language|=speak|=speaks",), ("speaks", "native_language")),
        (("хобби|увлека|hobby|hobbies",), ("hobby", "interested_in")),
        (
            ("встрет|встреч|увидел|увидим|увидет|=meet|=meeting|=met",),
            ("plans_to", "wants_to_meet"),
        ),
    )
)
# the question word → the value types (relations.yaml `value_type`) an answer carries
_VALUE_LEX: tuple[tuple[policy.Lexicon, tuple[str, ...]], ...] = tuple(
    (policy._lex(*words), types)
    for words, types in (
        (
            ("=где|=куда|=откуда|=where|в каком районе|в каком городе|which city",),
            ("place", "address"),
        ),
        (
            ("=когда|какого числа|какое число|=дата|=дату|=даты|=when|what day|which day|=date",),
            ("date",),
        ),
        (("=кто|=кого|=кем|=кому|как зовут|как звали|=who|=whom|whose",), ("person",)),
        (("во сколько|в какое время|what time|at what time",), ("time",)),
        (("сколько стоит|сколько плат|=цена|=цену|how much|=cost|=price",), ("money", "number")),
    )
)
_SELF_LEX = policy._lex(
    "=я|=мне|=меня|=мной|=мой|=моя|=мое|=мои|=моего|=моей|=моих|=мы|=нам|=нас|=нами|=наш|=наша",
    "=наше|=наши|=нашего|=нашей|=наших|=i|=me|=my|=mine|=we|=us|=our|=ours|=myself",
)
_PAST_LEX = policy._lex(
    "=был|=была|=было|=были|=жил|=жила|=жили|снимал|раньше|прежн|прошл|до переезда",
    "=was|=were|=lived|used to|=before|previous|former",
)
# «кто сказал», «от кого узнал», "who told me", "according to": the question asks who said a
# thing, so its lines keep the words that name the sender or the relative behind the claim
_SOURCE_LEX = policy._lex(
    "кто сказал|кто говорил|кто рассказ|кто писал|кто написал|кто сообщ|кто передал|от кого"
    "|с чьих слов|чьи слова|по чьим словам|откуда я знаю|откуда ты знаешь|где я слышал",
    "who told|who said|who mentioned|who wrote|who texted|who messaged|who says|heard it from"
    "|heard that from|according to whom|whose words|where did i hear|where'd i hear",
)
# the relations through which a relation word names the user's person or pet («мой брат»)
_KIN = frozenset(
    (
        "sibling_of",
        "parent_of",
        "child_of",
        "relative_of",
        "spouse_of",
        "partner_of",
        "friend_of",
        "colleague_of",
        "has_pet",
    )
)


def relations_of(ws: list[str]) -> list[str]:
    """The relations a question's words ask about, in lexicon order, deduplicated."""
    out: list[str] = []
    for lex, rels in _REL_LEX:
        if policy._hits(ws, lex):
            out.extend(r for r in rels if r not in out)
    return out


def value_types_of(ws: list[str]) -> list[str]:
    out: list[str] = []
    for lex, types in _VALUE_LEX:
        if policy._hits(ws, lex):
            out.extend(t for t in types if t not in out)
    return out


def _kin_entities(conn: sqlite3.Connection | None, rels: list[str]) -> list[str]:
    """«брат», «кот», «руководитель»: the entities the user's own active facts link by that relation."""
    kin = [r for r in rels if r in _KIN]
    if conn is None or not kin:
        return []
    marks = ", ".join("?" for _ in kin)
    try:
        rows = conn.execute(
            "SELECT DISTINCT f.value_entity_id FROM facts f JOIN records r ON r.id = f.id"
            f" WHERE f.subject_id = 'self' AND f.relation IN ({marks})"  # noqa: S608
            " AND f.value_entity_id IS NOT NULL AND r.status = 'active'"
            " ORDER BY r.recorded_at DESC LIMIT 4",
            kin,
        ).fetchall()
        # the same link written from the other side ("X is my sister" → X sibling_of the
        # user): the person is the fact's subject
        rows += conn.execute(
            "SELECT DISTINCT f.subject_id FROM facts f JOIN records r ON r.id = f.id"
            f" WHERE f.value_entity_id = 'self' AND f.relation IN ({marks})"  # noqa: S608
            " AND f.subject_id IS NOT NULL AND r.status = 'active'"
            " ORDER BY r.recorded_at DESC LIMIT 4",
            kin,
        ).fetchall()
        if not rows and "has_pet" in kin:
            # «кот», "the dog" with no has-pet fact: the pets the user's own active facts are about
            rows = conn.execute(
                "SELECT DISTINCT f.subject_id FROM facts f JOIN records r ON r.id = f.id"
                " JOIN entities e ON e.id = f.subject_id WHERE e.etype = 'pet'"
                " AND r.status = 'active' AND r.author_tier IN ('user', 'user_confirmed')"
                " ORDER BY r.recorded_at DESC LIMIT 4"
            ).fetchall()
    except sqlite3.Error:
        return []
    return list(dict.fromkeys(r[0] for r in rows if r[0] and r[0] != "self"))


# a person named by a role or a kin word the relation table has no slot for («тётя»,
# «научрук», "landlord", "coach"), in either language: (question lexicon, the role's words in
# both languages, looked up in the user's own active facts that link a person)
_ROLES: tuple[tuple[policy.Lexicon, tuple[str, ...]], ...] = tuple(
    (policy._lex(*q), terms)
    for q, terms in (
        (("бабушк|бабул|=grandma|grandmother|granny",), ("grandmother", "grandma", "бабушк")),
        (("дедушк|=дед|=деда|=деду|=grandpa|grandfather",), ("grandfather", "grandpa", "дедушк")),
        (("=тетя|=тети|=тете|=тетю|=тетей|тетушк|=aunt|auntie",), ("aunt", "тет")),
        (("=дядя|=дяди|=дяде|=дядю|=дядей|=uncle",), ("uncle", "дяд")),
        (("двоюродн|=кузен|=кузина|=кузины|cousin",), ("cousin", "двоюродн", "кузен", "кузин")),
        (("племянн|nephew|=niece",), ("nephew", "niece", "племянн")),
        (
            ("=теща|=тещи|=тесть|=тестя|свекров|=свекор|=свекра|=зять|невестк|in-law",),
            ("in-law", "тещ", "тест", "свекр", "зят", "невестк"),
        ),
        (("крестн|godmother|godfather",), ("godmother", "godfather", "крестн")),
        (("=сосед|=соседа|=соседу|=соседи|соседк|neighbo",), ("neighbo", "сосед")),
        (
            ("арендодат|хозяин квартиры|хозяйка квартиры|landlord|landlady",),
            ("landlord", "landlady", "арендодат", "хозя"),
        ),
        (
            ("научрук|научный руковод|научного руковод|advisor|adviser|supervisor",),
            ("advisor", "adviser", "supervisor", "научрук", "руковод"),
        ),
        (
            ("преподава|=препод|учител|профессор|репетитор|teacher|professor|lecturer|tutor",),
            ("teacher", "professor", "lecturer", "tutor", "instructor", "преподав", "учител"),
        ),
        (("тренер|=coach|trainer",), ("coach", "trainer", "тренер")),
        (("наставн|mentor",), ("mentor", "наставн")),
        (
            ("одногрупп|однокурс|однокласс|classmate|groupmate",),
            ("classmate", "groupmate", "однокурс", "одногрупп", "однокласс"),
        ),
        (
            ("сосед по комнате|соседка по комнате|roommate|flatmate|housemate",),
            ("roommate", "flatmate", "housemate"),
        ),
        (
            ("начальн|руковод|=шеф|=босс|=boss|manager|team lead|тимлид",),
            ("boss", "manager", "team lead", "начальн", "руковод", "тимлид"),
        ),
        (("=няня|=няни|=няне|=nanny|babysitter",), ("nanny", "babysitter", "нян")),
    )
)
_ROLE_TEXT = (
    "(coalesce(f.text_en, '') || ' ' || coalesce(f.keywords_en, '') || ' ' ||"
    " coalesce(f.keywords_ru, '') || ' ' || coalesce(f.quote_orig, ''))"
)


def _like_forms(term: str) -> list[str]:
    """LIKE folds ASCII case only: a Cyrillic word start also as capitalised and with «ё»."""
    forms = {term, term.capitalize()}
    for i, ch in enumerate(term):
        if ch == "е":
            forms |= {(t[:i] + "ё" + t[i + 1 :]) for t in (term, term.capitalize())}
    return sorted(forms)


def _role_entities(conn: sqlite3.Connection | None, ws: list[str]) -> list[str]:
    """«моя тётя», "my landlord": the people the user's own active facts call by that role
    (the role's word as a word start in a fact's text, keywords or the user's quote; either
    language)."""
    terms = list(dict.fromkeys(t for lex, ts in _ROLES if policy._hits(ws, lex) for t in ts))
    if conn is None or not terms:
        return []
    starts = re.compile("(?:^|[^0-9a-zа-я])(?:" + "|".join(map(re.escape, terms)) + ")")
    pats = [f"%{f}%" for t in terms for f in _like_forms(t)]
    likes = " OR ".join(f"{_ROLE_TEXT} LIKE ?" for _ in pats)
    try:
        rows = conn.execute(
            f"SELECT f.subject_id, f.value_entity_id, {_ROLE_TEXT} FROM facts f"  # noqa: S608
            " JOIN records r ON r.id = f.id WHERE r.status = 'active'"
            f" AND r.author_tier IN ('user', 'user_confirmed') AND ({likes})"
            " ORDER BY r.recorded_at DESC LIMIT 60",
            pats,
        ).fetchall()
        out: list[str] = []
        for subj, val, text in rows:
            if not starts.search(str(text or "").casefold().replace("ё", "е")):
                continue
            out += [str(e) for e in (subj, val) if e and e != "self" and e not in out]
        if not out:
            return []
        people = {
            r[0]
            for r in conn.execute(
                f"SELECT id FROM entities WHERE etype = 'person' AND id IN ({policy._in(len(out))})",  # noqa: S608, E501
                out,
            )
        }
    except sqlite3.Error:
        return []
    return [e for e in out if e in people][:4]


_ABOUT_SQL = (
    "SELECT coalesce(f.text_en, '') || ' ' || coalesce(f.keywords_ru, '') || ' ' ||"
    " coalesce(f.keywords_en, '') || ' ' || coalesce(f.quote_orig, '') FROM facts f"
    " JOIN records r ON r.id = f.id WHERE (f.subject_id = ? OR f.value_entity_id = ?)"
    " AND r.status = 'active' UNION ALL SELECT coalesce(i.text_en, '') || ' ' ||"
    " coalesce(i.quote_orig, '') FROM impressions i JOIN records r ON r.id = i.id"
    " WHERE i.about_entity_id = ? AND r.status = 'active' LIMIT 60"
)


def _tell_namesakes(
    conn: sqlite3.Connection, text: str, names: dict[str, float], kin: list[str]
) -> tuple[dict[str, float], dict[str, str]]:
    """Two people under one first name (e.g. the user's own contact and a namesake from a chat)
    are told apart before anything is served about either: the question keeps only the ones
    it points at, by a name word only they carry (a surname, a nickname), by a kin or role
    word that links the user to them, or by its other words in what is known about them.
    When nothing points at one of them, all stay, returned as id → full name so that each
    line about them names whom it is about."""
    from .. import entities

    ids = list(names)
    try:
        people = {
            str(r[0]): str(r[1] or "")
            for r in conn.execute(
                f"SELECT id, display_name FROM entities WHERE etype = 'person' AND id IN ({policy._in(len(ids))})",  # noqa: S608, E501
                ids,
            )
        }
        groups = entities.namesake_groups(people)
        if not groups:
            return names, {}
        out = dict(names)
        standing: dict[str, str] = {}
        ws = policy.words(text)
        for group in groups:
            keep = _pick_namesakes(conn, text, ws, group, people, kin)
            if len(keep) > 1:
                standing.update({e: people[e] for e in keep})
            for e in group:
                if e in keep:
                    continue
                for x in list(out):  # the dropped one and its twins under the same full name
                    if x == e or (
                        x not in keep and entities.same_person_names(people.get(x), people[e])
                    ):
                        out.pop(x, None)
        return (out, standing) if out else (names, {})
    except sqlite3.Error:
        return names, {}


def _pick_namesakes(
    conn: sqlite3.Connection,
    text: str,
    ws: list[str],
    group: list[str],
    people: dict[str, str],
    kin: list[str],
) -> set[str]:
    from .. import entities
    from . import lexical

    first = set().union(*(entities.first_name_keys(people[e]) for e in group))

    def first_word(w: str) -> bool:
        return bool(entities.first_name_keys(w) & first) or any(
            policy._name_token_match(k, w) for k in first if len(k) >= 3
        )

    own: dict[str, set[str]] = {e: set() for e in group}
    for eid, alias in conn.execute(
        f"SELECT entity_id, alias_norm FROM aliases WHERE entity_id IN ({policy._in(len(group))})",  # noqa: S608
        group,
    ):
        own[str(eid)] |= set(str(alias or "").split())
    for e in group:
        own[e] |= set(entities.normalize_alias(people[e]).split())
        own[e] = {t.rstrip(".") for t in own[e] if len(t) >= 3 and not first_word(t)}
    others = [w for w in ws if not first_word(w)]
    # 1. a name word only some of them carry
    named = {
        e for e in group if any(policy._name_token_match(t, w) for t in own[e] for w in others)
    }
    if named and len(named) < len(group):
        return named
    # 2. a kin or role word that links the user to some of them ("my sister")
    linked = set(group) & set(kin)
    if linked and len(linked) < len(group):
        return linked
    # 3. the question's other words in what is known about each one
    name_like = set().union(*own.values())
    stems = [
        s
        for s in lexical.query_stems(" ".join(others))
        if len(s) >= 4 and not s.isdigit() and s not in name_like
    ]
    if not stems:
        return set(group)
    score: dict[str, int] = {}
    for e in group:
        toks = set(
            lexical.stem(" ".join(str(r[0]) for r in conn.execute(_ABOUT_SQL, (e, e, e)))).split()
        )
        score[e] = sum(
            1
            for s in stems
            if s in toks or (len(s) >= 5 and any(t.startswith(s) for t in toks if len(t) >= 5))
        )
    best = max(score.values())
    top = {e for e, v in score.items() if v == best}
    return top if best > 0 and len(top) == 1 else set(group)


def parse(turn: TurnCtx, *, now: str, conn: sqlite3.Connection | None = None) -> Cue:
    """The turn's cue. `conn` resolves names through the trigram alias index; without it the
    cue carries no entity ids (dates, scope, hints and lexicons still parse)."""
    text = policy.turn_words(turn)
    if turn.origin in (TurnOrigin.DUTY, TurnOrigin.EVENT):
        text = ""
    names = _names(conn, text)
    ws = policy.words(text) if text else []
    # a cause clause («из-за поездки», «потому что я улетал») is context, not what is asked
    rels = relations_of(policy.words(_CAUSE.sub(" ", text.casefold()))) if text else []
    kin = _kin_entities(conn, rels) + (_role_entities(conn, ws) if text else [])
    for eid in kin:
        names.setdefault(eid, KIN_SCORE)
    ruled_out: list[str] = []
    namesakes: dict[str, str] = {}
    if conn is not None and len(names) > 1:
        told, namesakes = _tell_namesakes(conn, text, names, kin)
        ruled_out = sorted(set(names) - set(told))
        names = told
    ids = sorted(names, key=lambda e: (-names[e], e))
    past = policy._hits(ws, _PAST_LEX)
    dates = parse_dates(text, now=now, ahead=not past) if text else []
    scope: dict[str, str] = {"thread": turn.thread_id}
    if ids:
        scope["people"] = ",".join(ids)
    if dates:
        scope["period"] = dates[0]
    if turn.origin is TurnOrigin.TASK:
        scope["task"] = "1"
    hint = bool(text and _EPISODE_HINT.search(text.casefold()))
    constraints = episode_constraints(text, ids, dates) if text else {}
    return Cue(
        entity_ids=ids,
        dates=dates,
        scope=scope,
        episode_hint=hint,
        sensitive_topics=_topics(policy.words(text)),
        send_turn=policy.is_send_turn(turn),
        constraints=constraints if (hint or len(constraints) >= 2) else {},
        words=text,
        name_scores=names,
        relations=rels,
        value_types=value_types_of(ws),
        self_ref=policy._hits(ws, _SELF_LEX),
        past=past,
        asks_source=bool(text) and policy._hits(ws, _SOURCE_LEX),
        ruled_out=ruled_out,
        namesakes=namesakes,
    )
