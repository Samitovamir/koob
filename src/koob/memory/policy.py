"""Policy: admit, process routing, use now, chat modes, never-store, label inheritance.

Spec: §2 (pipeline order), §5.1-§5.6.
Order is normative: admit -> store -> process with an allowed model -> use now -> rank.

- **Admit** (§5.1) runs before any write. A rejection writes one content-free `audit` line and
  nothing else. Never-store spans (secrets, passwords, keys, one-time codes, full card and bank
  numbers) never enter memory or a model: the chunk is dropped, or, in the user's own words,
  only the span becomes «[secret removed]».
- **Process** (§5.4): `route(read_set)` names the processors a job may use. The user's own words
  (`user` / `user_confirmed`) count as unlabeled; other people's chats and files follow the
  table, where the uncertain band alone (`unknown_sensitive`, no confident label) counts as
  unlabeled too; the cloud needs `labels_final`.
- **Use now** (§5.5): the user's own items are used wherever they apply, with a one-line note
  («учёл: аллергия на орехи»), never a card or a question gate. The direct-question rule binds
  only sensitive items from other people, chats and files whose labels name a topic (the
  uncertain band alone, once final, ranks like an unlabeled item). On a send turn sensitive
  items (and other people's precise places) stay visible but held «не отправлять без его слова».
- **Rank** never sees what use-now excluded: the finders apply `mask_clause` in SQL first and
  `decide` stays the second fence.

Code only: no model, no network, no logging of content.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import sqlite3
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from . import clock as _clockmod
from .model import (
    ISLAND_CHAT,
    LABEL_NAMES,
    SENSITIVE_DEFAULT,
    AdmitWindow,
    AuthorTier,
    ChatMode,
    EventKind,
    ForgetScope,
    Label,
    ReadRef,
    ReadSrc,
    Route,
    TurnCtx,
    TurnOrigin,
    label_from_names,
)

SECRET_MARK = "[secret removed]"
HOLD_LINE = "не отправлять без его слова"
BANNED_WINDOW_LINE = "Этот чат под полным запретом: следить за ним и запоминать из него я не буду."
ISLAND_MODE_LINE = "Наш с тобой разговор режима не получает: запреты — только для других чатов."
MODE_ORDER_LINE = "Режим чата меняю только по твоим словам: скажи сам, какой чат и какой режим."
MODE_FORGET_JOB = "mode_forget"  # §5.2 (v0.4 A3): the durable forget side of a mode change
MODE_BATCH = 4000  # ids per forget of a mode change (SQLite's 32 766 variables)
JOB_LEASE = timedelta(minutes=5)  # §18.1 step 3: a running job's lease, renewed per group

OWN_TIERS: frozenset[AuthorTier] = frozenset({AuthorTier.USER, AuthorTier.USER_CONFIRMED})
_OWN_SQL = "('user', 'user_confirmed')"
ALL_LABELS: Label = Label(sum(1 << i for i in range(len(LABEL_NAMES))))
_LABEL_BITS: tuple[Label, ...] = tuple(Label(1 << i) for i in range(len(LABEL_NAMES)))
# Labels whose topic is the item's own words: no fixed lexicon can name it (§5.5).
OPEN_TOPIC: Label = Label.UNKNOWN_SENSITIVE | Label.THIRD_PARTY_PRIVATE
# §5.5 the user's own items: under these labels, once the classifier has settled them, relevance
# ranks them. Every named topic (health, money, legal, intimate, identity documents, minors)
# keeps the applicability test: it goes out only where this turn applies to it.
OWN_RANKED: Label = OPEN_TOPIC
# §5.5 a question that names the subject of another person's message and asks that message's
# own words: at least CONTENT_MIN of the user's content words, and at least CONTENT_SHARE of
# them, are words of the message (see `_asks_content`).
CONTENT_MIN = 2
CONTENT_SHARE = 0.5
REPLY_WINDOW = timedelta(hours=24)  # §5.6: the peer's first message within 24 h of the send
PASSWORD_STORE_KINDS = frozenset(
    {"keychain", "password_manager", "1password", "bitwarden", "lastpass"}
)
FORBIDDEN_CAPTURES = frozenset({"bank", "medical", "private"})  # rule 7, `screen:<category>`


class PolicyRefused(ValueError):
    """A refusal the Cube says in one line (`str(exc)`), e.g. an intention on a banned chat."""


# ---------------------------------------------------------------- config (§5.4, §5.5)

_routing_override: dict[str, frozenset[Route]] | None = None
_sensitive_override: Label | None = None


def configure(
    *, routing: dict[str, list[str]] | None = None, sensitive_labels: list[str] | None = None
) -> None:
    """Tests and the host set the `memory.routing` / `memory.sensitive_labels` tables; None
    goes back to `config/settings.yaml`."""
    global _routing_override, _sensitive_override
    if routing is None:
        _routing_override = None
    else:
        table = {k: frozenset(Route(v) for v in vals) for k, vals in routing.items()}
        _routing_override = {**_default_routing(), **table}
    _sensitive_override = None if sensitive_labels is None else label_from_names(sensitive_labels)


def _default_routing() -> dict[str, frozenset[Route]]:
    local = frozenset({Route.LOCAL})
    return {"unlabeled": frozenset({Route.CLOUD}), **{n: local for n in LABEL_NAMES}}


def _settings_memory():  # noqa: ANN202 - MemoryCfg, imported lazily
    from koob.config import MemoryCfg

    try:
        from koob.config import settings

        return settings().memory
    except Exception:  # noqa: BLE001 - a broken settings file must not open the fences
        return MemoryCfg()


def routing_table() -> dict[str, frozenset[Route]]:
    if _routing_override is not None:
        return _routing_override
    cfg = _settings_memory()
    return {k: frozenset(Route(v) for v in vals) for k, vals in cfg.routing.items()}


def sensitive_mask() -> Label:
    """§5.5: every label but `location_precise` unless `memory.sensitive_labels` says otherwise."""
    if _sensitive_override is not None:
        return _sensitive_override
    try:
        return label_from_names(_settings_memory().sensitive_labels)
    except Exception:  # noqa: BLE001
        return SENSITIVE_DEFAULT


def bits(labels: Label) -> list[Label]:
    return [b for b in _LABEL_BITS if labels & b]


def names(labels: Label) -> list[str]:
    return [LABEL_NAMES[i] for i, b in enumerate(_LABEL_BITS) if labels & b]


# ---------------------------------------------------------------- text helpers

_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def norm(text: str) -> str:
    """NFKC, casefold, ё→е, collapsed whitespace."""
    t = unicodedata.normalize("NFKC", text or "").casefold().replace("ё", "е")
    return " ".join(t.split())


def words(text: str) -> list[str]:
    return _WORD.findall(norm(text))


def _lex(*entries: str) -> tuple[tuple[str, ...], frozenset[str], tuple[str, ...]]:
    """(prefix stems, exact words, phrases). `=word` is exact; an entry with a space is a phrase."""
    prefixes, exact, phrases = [], set(), []
    for raw in entries:
        for e in raw.split("|"):
            e = norm(e)
            if not e:
                continue
            if " " in e:
                phrases.append(e)
            elif e.startswith("="):
                exact.add(e[1:])
            else:
                prefixes.append(e)
    return tuple(prefixes), frozenset(exact), tuple(phrases)


Lexicon = tuple[tuple[str, ...], frozenset[str], tuple[str, ...]]


def _hits(ws: list[str], lex: Lexicon) -> bool:
    prefixes, exact, phrases = lex
    if any(w in exact or w.startswith(prefixes) for w in ws):
        return True
    if phrases:
        joined = " " + " ".join(ws) + " "
        return any(" " + p in joined for p in phrases)
    return False


# Topic lexicons per label (§5.5 direct question, §5.3 arrival labels). Strong hits set the
# label at arrival; a weak hit alone sets `unknown_sensitive` (fail-closed). Both count as a
# topic hit in the user's words.
_STRONG: dict[Label, Lexicon] = {
    Label.HEALTH: _lex(
        "врач|доктор|медиц|медсестр|болезн|болеет|болел|болен|больна|больн|болит|заболе|здоровь",
        "аллерг|лекарств|таблет|препарат|антибиот|диагноз|клиник|поликлин|госпитал|терапевт",
        "стоматолог|дантист|хирург|кардиолог|невролог|гинеколог|уролог|эндокринолог|онколог",
        "гастрит|ферритин|диабет|астм|мигрен|депресс|тревожн|психотерап|психиатр|психолог",
        "инфаркт|инсульт|беремен|аборт|гепатит|опухол|простуд|грипп|ковид|covid|кашл|насморк",
        "травм|перелом|лечени|лечит|лечу|лечат|симптом|вакцин|прививк|уколы|=укол|=вич|=рак",
        "health|doctor|medic|allerg|=pill|=pills|diagnos|hospital|clinic|symptom|=sick|illness",
        "disease|therap|surgery|pregnan|diabet|asthma|migraine|depress|anxiety|cancer",
        "prescription|dentist|vaccin|injur|gastrit|ferritin|=hiv",
    ),
    Label.MONEY: _lex(
        "деньг|денег|долг|зарплат|оклад|кредит|ипотек|=займ|заем|банк|оплат|заплат|платеж",
        "плачу|платит|=руб|рубл|доллар|=евро|бюджет|налог|сбережен|инвест|вклад|доход|расход",
        "финанс|потрат|=трата|=траты|стипенд|пенси|money|salary|=debt|=debts|=loan|=loans",
        "=bank|banking|payment|=pay|=paid|budget|=tax|=taxes|invest|saving|income|expense",
        "=owe|=owes|=price|=cost|dollar|tuition|scholarship",
        # rent and housing payments are money (they only add the label: privacy recall)
        "аренд|квартплат|коммуналк|=жкх|=rent|=rents|=rented|=renting|rental|=utilities",
        "utility bill|mortgage",
    ),
    Label.IDENTITY_DOCS: _lex(
        "паспорт|загранпаспорт|=виза|=визу|=визы|=визе|=визой|снилс|=инн|удостоверени",
        "водительск|гринкарт|грин-карт|=внж|гражданств|вид на жительство|свидетельство о рождении",
        "passport|=visa|=visas|=ssn|social security|green card|driver's license|drivers license",
        "residence permit|citizenship|birth certificate|=i-20|=i20|=ead|=sevis",
    ),
    Label.INTIMATE: _lex(
        "=секс|сексуал|интим|любовни|измен|развод|влюбл|ориентац|=гей|лесби|бисексу|контрацеп",
        "презерват|=sex|sexual|intimate|=affair|divorce|=lover|romantic",
    ),
    Label.LOCATION_PRECISE: _lex(
        # a flat, a floor or a city alone is not a precise place: with a number they are, and
        # `_PRECISE_ADDR` takes them there («квартира 12», «5 этаж», «apt 4B», «floor 3»)
        "адрес|домофон|подъезд|=улица|=улице|=улицу|=ул|проспект|переул|координат",
        "геолок|геопозиц|местоположен|address|=street|gate code|entrance",
        "coordinates|=zip|postcode|где живет|где живут|where he lives|where she lives",
    ),
    Label.THIRD_PARTY_PRIVATE: _lex(
        "секрет|по секрету|между нами|никому не|конфиденц|=тайна|=тайну|secret|confidential",
        "between us|=private",
    ),
    Label.LEGAL: _lex(
        "=суд|=суда|=суде|=судом|судебн|адвокат|юрист|=иск|=иска|полиц|арест|уголовн|прокурор",
        "нотариус|завещан|наследств|депорт|lawsuit|=court|lawyer|attorney|police|arrest",
        "criminal|=legal|=sue|=sued|deport|notary",
    ),
    Label.MINOR: _lex(
        "ребен|=дети|=детей|=детям|=детьми|=детях|детск|детсад|=сын|=сына|=сыну|=сыном|=сыне",
        "сынок|=дочь|=дочка|дочк|дочер|малыш|подрост|=внук|=внука|внуч|школьни|садик",
        "=kid|=kids|child|=son|=sons|daughter|=minor|toddler|=baby|=teen|teenager",
    ),
}
_WEAK: dict[Label, Lexicon] = {
    Label.HEALTH: _lex("анализ|рецепт|давлен|температур|=диета|=диету|=диеты|операци|=железо"),
    Label.MONEY: _lex("=счет|=счета"),
    Label.INTIMATE: _lex("отношени|свидан|бывш"),
    Label.LOCATION_PRECISE: _lex("=живет|=живут|=дом|=индекс"),
    # the noun «договор» (a contract), never the verb «договорились» (we agreed)
    Label.LEGAL: _lex("=договор|=договора|=договору|=договором|=договоре|=договоры|договорн|штраф"),
    Label.MINOR: _lex("=школа|=школу|=школе|=школы"),
}
_EMPTY: Lexicon = ((), frozenset(), ())

# Arrival precision for `minor` and `health` (the topic lexicons above still read the user's
# questions whole). `minor` is about a child's personal details: a child word sets it beside a
# detail about the child (school, health, safety, age, behaviour, a place, or another sensitive
# label of the same text); a bare mention («took the kids to the park») is a weak hit, and one's
# own childhood, things made for children or children in general («as a kid», «kids' books»,
# «a home for kids who need it») are no hit at all.
_CHILD_NOT = re.compile(
    r"\b(?:as|when i was|since i was|back when i was) (?:a )?(?:little |young )?(?:kid|child)\b"
    r"|\bchildhood\b|\bchildless\b|\bchildish\b|\bchild-?free\b"
    r"|\b(?:kids?|child|children)(?:'s|')? (?:books?|menus?|shows?|movies?|films?|tv|songs?|"
    r"music|toys?|clothes|section|games?|channel|channels|programs?|programmes?|charity)\b"
    r"|\bfor (?:the |all the |other |many |more )?(?:kids|children)\b"
    r"|\b(?:kids|children) (?:who|that) (?:need|needs|deserve)\b|\b(?:kids|children) in need\b"
    r"|\bкогда (?:я )?(?:был|была) (?:ребенком|маленьк\w*)|\bдля детей\b"
    r"|\bдетск\w* (?:книг\w*|мульт\w*|игрушк\w*|одежд\w*|магазин\w*|меню|песн\w*|фильм\w*)"
)
_CHILD_DETAIL: Lexicon = _lex(
    "school|kindergarten|preschool|daycare|nursery|teacher|=grade|=grades|homework|recess",
    "principal|detention|suspend|expell|report card|bully|bulli|=fight|=fights|fought|fighting",
    "tantrum|misbehav|=trouble|=hurt|=hurts|accident|injur|=lost|missing|kidnap|abus|stranger",
    "custody|=born|years old|year old|year-old|months old|month-old|=aged|autis|=adhd",
    "pediatric|paediatric|=fever|=scared|afraid|=bleeding|=broke|=crying|=cries|=allergic",
    "школ|садик|детсад|=ясли|учител|=классе|оценк|двойк|домашк|контрольн|подрал|драк",
    "хулига|истерик|обижа|травл|=бьет|избил|пропал|потерял|похит|опек|родился|родилась",
    "аутизм|=сдвг|логопед|=боится|температур|=плачет|упал|ушиб",
)
_CHILD_AGE = re.compile(r"(?i)\b\d{1,2}\s*(?:yo\b|y/o|-?years?\b|-?months?\b|лет\b|год|месяц)")
# English «health» names a field, a cause or a habit («work in mental health», «a race for
# mental health») as often as a condition: alone it is a weak hit. Figurative therapy («it's
# like therapy», «fun and therapeutic») is no hit at all.
_HEALTH_WORD = re.compile(r"\bhealth\w*")
_HEALTH_FIGURE = re.compile(
    r"\btherapeutic\b|\b(?:like|retail|kind of|sort of|a form of|my kind of) therapy\b"
    r"|\btherapy for (?:me|the soul|my soul)\b"
)


_ARRIVAL_RULED = frozenset({Label.HEALTH, Label.MINOR})  # labels set by the two rules below


def _arrival_health(t: str, ws: list[str]) -> tuple[bool, bool]:
    """(strong, weak) `health` at arrival from the normalised text `t` and its words."""
    lex = _STRONG[Label.HEALTH]
    if not _hits(ws, lex):
        return False, _hits(ws, _WEAK[Label.HEALTH])
    plain = words(_HEALTH_FIGURE.sub(" ", t))
    if not _hits(plain, lex):
        return False, _hits(ws, _WEAK[Label.HEALTH])
    if _hits(words(_HEALTH_WORD.sub(" ", " ".join(plain))), lex):
        return True, False
    return False, True


def _arrival_minor(t: str, text: str, other: Label) -> tuple[bool, bool]:
    """(strong, weak) `minor` at arrival: a child word with a detail about the child, or with
    another sensitive label of the same text (`other`), is strong; a bare mention is weak."""
    cw = words(_CHILD_NOT.sub(" ", t))
    if not _hits(cw, _STRONG[Label.MINOR]):
        return False, _hits(cw, _WEAK[Label.MINOR])
    if other or _hits(cw, _CHILD_DETAIL) or _hits(cw, _WEAK[Label.MINOR]):
        return True, False
    if _CHILD_AGE.search(text or ""):
        return True, False
    return False, True


# Situation domains for "applicability matches" of the user's own items (§5.5): an allergy
# applies to a food order, not to the weather.
_DOMAINS: dict[str, Lexicon] = {
    "food": _lex(
        "поесть|покушать|кушать|=еда|=еду|=еды|=едой|обед|ужин|завтрак|перекус|ресторан|=кафе",
        "=бар|пицц|суши|ролл|бургер|=торт|=торта|=торты|тортик|пирог|десерт|сладк|конфет",
        "шоколад|орех|арахис|молок|молочн|лактоз|глютен|=мясо|морепродукт|=яйца|фрукт|овощ",
        "продукт|рецепт|приготов|=меню|блюд|напит|=кофе|=чай|алкогол|=вино|=пиво|доставку еды",
        "=food|=eat|=eating|=meal|=meals|lunch|dinner|breakfast|restaurant|=cafe|pizza|sushi",
        "=cake|dessert|=nut|=nuts|peanut|=milk|dairy|lactose|gluten|=menu|recipe|=cook|grocer",
        "snack|=drink|=drinks",
    ),
    "travel": _lex(
        "поездк|путешеств|=лететь|=лечу|=летим|полет|перелет|=рейс|самолет|аэропорт|=отель",
        "гостиниц|хостел|=билет|=билеты|=поезд|вокзал|отпуск|командировк|travel|=trip|flight",
        "=fly|=plane|airport|hotel|ticket|=train|vacation",
    ),
    "sport": _lex(
        "спорт|трениров|фитнес|=бег|пробежк|плаван|бассейн|=йога|велосипед|=лыжи|футбол|теннис",
        "=gym|workout|fitness|=run|=running|=swim|swimming|=yoga|=bike|sport",
    ),
}

# §10.2 step 1 send-verb lexicon: an outward send or forward to another person.
_SEND_RU = re.compile(
    r"\b(?:напиш\w*|написать|отправ\w*|перешл\w*|пересл\w*|ответь\w*|ответить|передай\w*|"
    r"передать|сообщи\w*|предупреди\w*|поздрав\w*|пригласи\w*|скажи|сказать|напомни|спроси)"
    r"(?:\s+(\w+))?",
)
_SEND_EN = re.compile(
    r"\b(?:send|forward|reply|text|message|email|e-mail|tell|write to|dm)\b(?:\s+(\w+))?"
)
_SELF_TARGET = frozenset({"мне", "меня", "себе", "me", "myself"})
_LOOK = _lex("посмотр|глянь|взгляни|смотри|видишь|экран|открой|прочитай|проверь|=look|=see|screen")

_STOP_WORDS = (
    "меня мне тебя тебе себя него нему ними неё нее этот этого этом эта эти такой какой когда"
    " который которая очень просто можно нужно надо будет было были есть сейчас сегодня"
    " завтра вчера потом тоже только ещё всегда никогда запомни помни напиши скажи хочу"
    " where when what this that with have from about user they them your will would"
    " should there their likes prefers says said"
)
_STOP = frozenset(norm(w) for w in _STOP_WORDS.split())


_LAT = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        "a b v g d e e zh z i i k l m n o p r s t u f kh ts ch sh shch _ y _ e yu ya".split(),  # noqa: SIM905
        strict=True,
    )
)
_PHON = (("shch", "sh"), ("kh", "h"), ("ck", "k"), ("ph", "f"), ("x", "ks"), ("c", "k"),
         ("q", "k"), ("w", "v"), ("y", "i"), ("j", "i"), ("h", ""), ("_", ""))  # fmt: skip


def skeleton(word: str) -> str:
    """A rough cross-script skeleton: «Яндексе» and "Yandex", «Москве» and "Moscow" meet."""
    w = "".join(_LAT.get(ch, ch) for ch in word.casefold())
    for a, b in _PHON:
        w = w.replace(a, b)
    return w


def word_matches(word: str, skels: set[str]) -> bool:
    """`word` is one of `skels` (skeletons) up to an inflected ending (>= 4 shared letters)."""
    a = skeleton(word)
    if a in skels:
        return True
    for b in skels:
        n = 0
        for x, y in zip(a, b, strict=False):
            if x != y:
                break
            n += 1
        if n >= 4 and n >= min(len(a), len(b)) - 2:
            return True
    return False


def _stem(w: str) -> str:
    return w if len(w) <= 4 else w[: max(4, len(w) - 2)]


def _terms(text: str) -> set[str]:
    return {w for w in words(text) if len(w) >= 4 and w not in _STOP and not w.isdigit()}


def _overlap(item_terms: set[str], ws: list[str], *, exclude: Iterable[str] = ()) -> bool:
    excl = list(exclude)
    for t in item_terms:
        if any(_name_token_match(n, t) for n in excl):
            continue
        st = _stem(t)
        for w in ws:
            if len(w) >= 4 and w not in _STOP and (w.startswith(st) or t.startswith(_stem(w))):
                return True
    return False


def _name_token_match(tok: str, w: str) -> bool:
    """A name token against a word, tolerant to Russian inflection («Саша» / «Саши» / «Сашей»)."""
    if w == tok:
        return True
    if len(tok) < 3:
        return False
    root = tok if len(tok) <= 3 else tok[:-1]
    return w.startswith(root) and len(w) <= len(tok) + 3


def names_in(ws: list[str], aliases: Iterable[str]) -> bool:
    """True when their words name one of these names (every token of a multi-word alias)."""
    for alias in aliases:
        toks = [t for t in words(alias) if t]
        if toks and all(any(_name_token_match(t, w) for w in ws) for t in toks):
            return True
    return False


def topic_hit(
    label: Label, ws: list[str], *, item_terms: set[str] | None = None, exclude: Iterable[str] = ()
) -> bool:
    """§5.5: their words hit this label's topic lexicon (strong or weak). Open-topic labels
    (`unknown_sensitive`, `third_party_private`) also count the item's own words."""
    if _hits(ws, _STRONG.get(label, _EMPTY)) or _hits(ws, _WEAK.get(label, _EMPTY)):
        return True
    if label & OPEN_TOPIC and item_terms:
        return _overlap(item_terms, ws, exclude=exclude)
    return False


def turn_words(turn: TurnCtx | None) -> str:
    """The user's own words for this turn: the user's message, or the task's order (§1 terms)."""
    if turn is None or turn.origin is TurnOrigin.EVENT:
        return ""
    if turn.origin is TurnOrigin.OWNER:
        return turn.owner_words or ""
    return turn.task_words or turn.owner_words or ""


def is_send_turn(turn: TurnCtx | None) -> bool:
    """§5.5 send turns: the hooks adapter saw a send or forward to another person about to run
    (`turn.sends is True`), or their words ask for one (the send-verb lexicon). `None` leaves it
    to the lexicon; `False` never clears a lexicon hit."""
    if turn is not None and turn.sends is True:
        return True
    text = norm(turn_words(turn))
    if not text:
        return False
    for rx in (_SEND_RU, _SEND_EN):
        for m in rx.finditer(text):
            nxt = m.group(1)
            if nxt is None or nxt not in _SELF_TARGET:
                return True
    return False


# ---------------------------------------------------------------- never-store (§5.1 rule 3)

_PASS_KW = (
    r"(?:password|passwd|passphrase|pass ?code|pwd|pin(?:[ -]?code)?|пароль|пароля|паролем|"
    r"паролю|пин(?:[ -]?код)?|код доступа)"
)
_KEY_KW = (
    r"(?:api[ _-]?key|secret(?: key)?|access[ _-]?token|auth[ _-]?token|token|токен|"
    r"ключ(?: api| доступа)?|client[ _-]?secret)"
)
# a keyword starts a word or follows "_" (`DB_PASSWORD=`, `MY_API_KEY=`), never mid-word
_KW_LEFT = r"(?<![^\W_])"
_SHORT_KW = r"(?:pass|pw|pwd|пасс)"
_FOR = r"(?:\s+(?:от|для|к|на|в|from|for|to|of|on|in|at)\s+[^\s:=—–,;]+)?"
# Up to ten words between the keyword and the separator, commas included, never past the
# end of a sentence («пароль к роутеру на работе:», «the password to the router is»,
# «пароль от почты denis@example.com —», «пароль от вайфая на даче, если что: …»)
_GAP = r"(?:\s+[^\s:=—–;]*[^\s:=—–;.!?]){1,10}?"
_SEP0 = r"\s*(?:[:=—–]|\s-\s|\bis\b|\bэто\b)"
_SEP = rf"{_SEP0}\s*[\"'«“]?"
_VAL = r"([^\s\"'«»“”,;]{4,})"
# a status word between the separator and the value («пароль: теперь X», «is now X»)
_FILL = r"(?:(?:теперь|сейчас|now|новый|new|вот)\s+)*"
_PASS_SEP = re.compile(rf"(?i){_KW_LEFT}{_PASS_KW}{_FOR}{_SEP}{_FILL}{_VAL}")
_PASS_FAR = re.compile(rf"(?i){_KW_LEFT}{_PASS_KW}{_GAP}{_SEP}{_FILL}{_VAL}")
_PASS_BARE = re.compile(rf"(?i){_KW_LEFT}{_PASS_KW}{_FOR}\s+{_FILL}[\"'«“]?{_VAL}")
_PASS_SHORT = re.compile(rf"(?i){_KW_LEFT}{_SHORT_KW}\s*[:=]\s*[\"'«“]?{_VAL}")
# a quoted value is redacted whole, spaces and all («пароль: «мой кот Барсик 2019»»)
_QUOTED = (
    r"(?:«([^»\n]{1,120})»|\"([^\"\n]{1,120})\"|“([^”\n]{1,120})”|„([^“”\n]{1,120})[“”]"
    r"|'([^'\n]{1,120})')"
)
_PASS_QUOTED = [
    re.compile(rf"(?i){_KW_LEFT}{_PASS_KW}(?:{_FOR}(?:{_SEP0})?|{_GAP}{_SEP0})\s*{_FILL}{_QUOTED}"),
    re.compile(rf"(?i){_KW_LEFT}{_SHORT_KW}\s*[:=]\s*{_QUOTED}"),
]
# a change of password names the new one after «на» / «to» («поменял пароль на X»,
# «пароль от почты сменил на X», «changed the password to X», «password was changed to X»)
_PASS_KW_RX = re.compile(rf"(?i){_KW_LEFT}{_PASS_KW}")
_CHANGE = re.compile(
    r"(?i)(?<![^\W_])(?:помен|смен|измен|обнов|постав|установ|задал|задай|задам|сброс|"
    r"chang|re-?set|set\b|updat|switch)"
)
_TO_VAL = re.compile(
    r"(?i)(?<![^\W_])(?:на|to|into)\s+(?:[\"'«“]([^\"'»”\n]{1,120})[\"'»”]|([^\s\"'»”,;]{4,}))"
)
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)|\n")
# a bare «pass:» is also a test log («PASS: test_login», «pass: 12/12»), not only a password
_NOT_PASS_VALUE = re.compile(
    r"(?i)^(?:test|spec)\w*|::|(?:_test|\.test|_spec|\.spec)\b|^\d+(?:/\d+|%)$|^(?:true|false|yes)$"
)
# «boarding pass:» is a ticket, «I'll pass:» / «we pass:» a verb
_PASS_LEFT = re.compile(
    r"(?i)(?:\b(?:boarding|season|bus|ski|metro|day|press|backstage|i|we|you|they|will|would|"
    r"to|can|could|should|must|gonna|let's|lets)|'ll|’ll)\s+$"
)
_KEY_ANY = re.compile(rf"(?i){_KW_LEFT}{_KEY_KW}{_FOR}(?:{_SEP}|\s+[\"'«]?){_VAL}")
# env and config lines: an upper-case identifier ending in a secret word (`PGPASSWORD=`)
_ENV = re.compile(
    r"(?<![\w])[A-Z0-9_]*(?:PASSWORD|PASSWD|PASS|PWD|SECRET|TOKEN|API_?KEY)\s*[=:]\s*[\"']?"
    r"([^\s\"',;]{4,})"
)
_ADDRESS = re.compile(
    r"^(?:[\w.+-]+@[\w-]+(?:\.[\w-]+)+|[a-z][a-z0-9+.-]*://\S+|[/~.]\S*/\S*)[.,;:]?$", re.I
)
# one-time codes as services phrase them: up to five words between the keyword and the digits
# («Код для входа в Telegram: 52310», «Your verification code is 482193», «Ваш код: 4821»)
_OTP_KW = (
    r"(?:otp|2fa|one[- ]time (?:code|password)|verification code|confirmation code|login code|"
    r"sign[- ]?in code|sms code|security code|auth(?:entication)? code|access code|"
    r"одноразов\w*\s+код\w*|код\w*\s+(?:из\s+смс|из\s+sms|подтвержд\w*|для\s+входа|"
    r"верификац\w*|из\s+сообщения|авторизац\w*|безопасност\w*|доступа)|"
    r"(?:ваш|your)(?:\s+[^\s\d,.;:!?]+){0,2}?\s+(?:код|code))"
)
_OTP_GAP = r"(?:\s+[^\s\d:—–,.;!?-]+){0,5}?"
_KNOWN = [
    re.compile(p)
    for p in (
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?(?:-----END [A-Z0-9 ]*-----|\Z)",
        r"\bsk-ant-[A-Za-z0-9_\-]{10,}",
        r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}",
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}",
        r"\bglpat-[A-Za-z0-9_\-]{20,}",
        r"\bxox[abposr]-[A-Za-z0-9\-]{10,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bAIza[0-9A-Za-z_\-]{35}",
        r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}",
        r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
        r"\b\d{8,10}:AA[A-Za-z0-9_\-]{30,}",
        r"otpauth://\S+",
    )
]
_GROUP_KNOWN = [
    re.compile(p)
    for p in (
        r"(?i)\bbearer\s+([A-Za-z0-9._~+/\-]{16,}=*)",
        r"(?i)\b(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp|https?|ftp)://[^\s:@/]+:([^\s@/]+)@",
        rf"(?i)\b{_OTP_KW}{_OTP_GAP}\s*(?:[:—–-]|\bis\b|\bэто\b)?\s*(\d{{4,8}})\b",
        r"(?i)(?:^|[.!?\n]\s*)(?:код|code)\s*[:—–-]\s*(\d{4,8})\b",
        r"\bG-(\d{6})\b",
        r"(?i)(?<![\d-])(\d{4,8})\s*[—–-]?\s*(?:ваш|your|это)\s+(?:код|code)\b",
        r"(?i)(?:account (?:number|no\.?|#)|acct\.?|routing (?:number|no\.?)|номер\s+сч[её]та|"
        r"р/с|расч[её]тн\w*\s+сч[её]т\w*|лицев\w*\s+сч[её]т\w*)\s*[:#№—–-]?\s*(\d[\d -]{6,30}\d)",
    )
]
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_ACCOUNT20 = re.compile(r"(?<!\d)\d{20}(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
_TOKEN = re.compile(r"(?<![A-Za-z0-9_\-+=])[A-Za-z0-9_\-+=]{20,}(?![A-Za-z0-9_\-+=])")
_HEX = re.compile(r"^[0-9a-fA-F\-]+$")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1
    except ValueError:
        return False


def _entropy(s: str) -> float:
    n = len(s)
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return -sum(k / n * math.log2(k / n) for k in counts.values())


_PLACEHOLDERS = frozenset(
    [
        "required",
        "invalid",
        "incorrect",
        "expired",
        "changed",
        "reset",
        "wrong",
        "missing",
        "empty",
        "none",
        "null",
        "hidden",
        "protected",
        "optional",
        "needed",
        "redacted",
        "removed",
        "сменен",
        "изменен",
        "неверный",
        "неверен",
        "нужен",
        # «сбросил пароль от почты — теперь другой»: talk about a password, not the password
        "теперь",
        "сменил",
        "сбросил",
        "поменял",
        "обновил",
        "забыл",
        "другой",
        "новый",
        "старый",
        "такой",
        "простой",
        "сложный",
    ]
)


def _placeholder(tok: str) -> bool:
    """A masked value («********») or a status word after "password:" is no secret."""
    t = norm(tok).strip(".!,;")
    return not t.strip("*•x.-_") or t in _PLACEHOLDERS or t.startswith(("<", "{", "$"))


def _secretish(tok: str) -> bool:
    has_d = any(c.isdigit() for c in tok)
    has_a = any(c.isalpha() for c in tok)
    sym = any(c in "!@#$%^&*()+=?~[]{}|\\" for c in tok)
    mixed = any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok)
    return len(tok) >= 4 and ((has_d and has_a) or sym or (mixed and len(tok) >= 6))


def _short_pass_value(text: str, start: int, val: str) -> bool:
    """A value after a bare «pass:» / «pw:» is a password unless it is a test id, a score or
    a boolean, or the «pass» is a ticket («boarding pass:»)."""
    if _placeholder(val) or _ADDRESS.match(val) or _NOT_PASS_VALUE.search(val):
        return False
    return not _PASS_LEFT.search(text[max(0, start - 16) : start])


def _change_spans(text: str) -> list[tuple[int, int]]:
    """«поменял пароль на X», «пароль от почты сменил на X», «changed the password to X»:
    a change verb in the clause and «на/to» within four words after the keyword name the
    new password. «надо сменить пароль на почте» is a place, not a value: the value must look
    like a secret or be quoted."""
    out: list[tuple[int, int]] = []
    for kw in _PASS_KW_RX.finditer(text):
        ends = [m.start() for m in _SENTENCE_END.finditer(text, 0, kw.start())]
        clause = ends[-1] if ends else 0
        stop = _SENTENCE_END.search(text, kw.end())
        limit = stop.start() if stop else len(text)
        for to in _TO_VAL.finditer(text, kw.end(), limit):
            gap = text[kw.end() : to.start()]
            if len(gap.split()) > 4 or any(c in gap for c in ":=—–;"):
                break
            if not _CHANGE.search(text, clause, to.start()):
                continue
            g = 1 if to.group(1) is not None else 2
            val = to.group(g)
            if _placeholder(val) or _ADDRESS.match(val):
                continue
            if g == 1 or _secretish(val) or (val.isdigit() and len(val) >= 4):
                out.append(to.span(g))
    return out


def never_store_spans(text: str) -> list[tuple[int, int]]:
    """§5.1 rule 3: secret spans by patterns plus entropy: passwords, keys, tokens, one-time
    codes, full card and bank numbers, private keys. Sorted, merged `(start, end)` pairs."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    for rx in _KNOWN:
        spans.extend(m.span() for m in rx.finditer(text))
    for rx in _GROUP_KNOWN:
        spans.extend(m.span(1) for m in rx.finditer(text))
    pos = 0
    while (m := _PASS_SEP.search(text, pos)) is not None:
        if _PASS_KW_RX.match(m.group(1)):  # «пароль от роутера: пароль: X» - the value
            pos = m.start(1)  # is the next keyword; read on from it, never past X
            continue
        if not _placeholder(m.group(1)) and not _ADDRESS.match(m.group(1)):
            spans.append(m.span(1))
        pos = m.end()
    for rx in (_PASS_FAR, _PASS_BARE):
        for m in rx.finditer(text):
            val = m.group(1)
            if _ADDRESS.match(val) or _placeholder(val):
                continue  # «пароль от почты denis@example.com — …»: the address is no secret
            if _secretish(val) or (val.isdigit() and len(val) >= 4):
                spans.append(m.span(1))
    # «pass: sunflower» / «pw: sunflower» as «PASS: sunflower» is: any value, unless the
    # line is a test log or a ticket («PASS: test_login», «boarding pass: 12B»)
    for m in _PASS_SHORT.finditer(text):
        if _short_pass_value(text, m.start(), m.group(1)):
            spans.append(m.span(1))
    for rx in _PASS_QUOTED:
        for m in rx.finditer(text):
            g = next(i for i in range(1, 6) if m.group(i) is not None)
            if not _placeholder(m.group(g)):
                spans.append(m.span(g))
    spans.extend(_change_spans(text))
    for m in _ENV.finditer(text):
        val = m.group(1)
        if _placeholder(val) or _ADDRESS.match(val):
            continue
        if re.match(r"PASS\s*[=:]", m.group(0)) and not _short_pass_value(text, m.start(), val):
            continue
        spans.append(m.span(1))
    for m in _KEY_ANY.finditer(text):
        val = m.group(1)
        if len(val) >= 8 and _secretish(val):
            spans.append(m.span(1))
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn(digits):
            spans.append(m.span())
    spans.extend(m.span() for m in _ACCOUNT20.finditer(text))
    for m in _IBAN.finditer(text):
        if _iban_ok(m.group(0)):
            spans.append(m.span())
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        if _HEX.match(tok):
            continue  # commit ids, hashes and uuids are not secrets by shape alone
        if (
            sum(c.isdigit() for c in tok) >= 2
            and any(c.isupper() for c in tok)
            and any(c.islower() for c in tok)
            and _entropy(tok) >= 3.5
        ):
            spans.append(m.span())
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if s >= e:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged


def redact(text: str, spans: list[tuple[int, int]] | None = None) -> str:
    """Replace never-store spans with «[secret removed]»."""
    spans = never_store_spans(text) if spans is None else spans
    out, pos = [], 0
    for s, e in spans:
        out.append(text[pos:s])
        out.append(SECRET_MARK)
        pos = e
    out.append(text[pos:])
    return "".join(out)


# ---------------------------------------------------------------- labels (§5.3)

LabelClassifier = Callable[[str], dict[Label, float]]
_classifier: LabelClassifier | None = None
CLASSIFIER_HIGH = 0.5
CLASSIFIER_LOW = 0.2

_ETYPE_LABELS: dict[str, Label] = {
    "minor": Label.MINOR,
    "child": Label.MINOR,
    "doctor": Label.HEALTH,
    "clinic": Label.HEALTH,
    "hospital": Label.HEALTH,
    "medical": Label.HEALTH,
    "condition": Label.HEALTH,
    "medication": Label.HEALTH,
    "bank": Label.MONEY,
    "bank_account": Label.MONEY,
    "card": Label.MONEY,
    "loan": Label.MONEY,
    "lawyer": Label.LEGAL,
    "court": Label.LEGAL,
    "case": Label.LEGAL,
    "address": Label.LOCATION_PRECISE,
    "home": Label.LOCATION_PRECISE,
    "document": Label.IDENTITY_DOCS,
    "id_document": Label.IDENTITY_DOCS,
    "passport": Label.IDENTITY_DOCS,
    "visa": Label.IDENTITY_DOCS,
}
_MONEY_AMOUNT = re.compile(r"(?:[$€₽£]\s?\d|\d[\d\s.,]*\s?(?:[$€₽£]|руб|usd|eur|тыс|k\b))", re.I)
_PRECISE_ADDR = re.compile(
    r"(?i)(?:\b(?:ул\.?|улица|пр-т|проспект|пер\.?|переулок)\s+[^\s,]+[,\s]+(?:д\.?|дом)?\s*\d+"
    r"|\bкв\.?\s*\d+|\b\d{1,5}\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Ln|Dr)\b"
    r"|\bквартир\w*\s*(?:№\s*)?\d+|\b\d{1,3}\s*(?:-?(?:й|м|ом))?\s*этаж\w*|\bэтаж\w*\s*\d+"
    r"|\b(?:apartment|apt\.?|unit)\s*#?\s*\d+|\bfloor\s+\d+|\b\d{1,3}(?:st|nd|rd|th)\s+floor)"
)


def set_classifier(fn: LabelClassifier | None) -> None:
    """§5.3 the in-process classifier (per-label prototype vectors over the §10.3 embedder).
    Scores ≥ CLASSIFIER_HIGH set the label; scores in [LOW, HIGH) set `unknown_sensitive`."""
    global _classifier
    _classifier = fn


def classifier() -> LabelClassifier | None:
    """The installed in-process classifier (`set_classifier`), or None."""
    return _classifier


def classifier_ready() -> bool:
    """§5.3 fail-closed: True only when a classifier is installed and can score now. A
    classifier with a `ready()` method (memory/labeler.py) answers for itself: cold or missing
    embedder -> False, so no label becomes final on a guess."""
    fn = _classifier
    if fn is None:
        return False
    ready = getattr(fn, "ready", None)
    if ready is None:
        return True
    try:
        return bool(ready())
    except Exception:  # noqa: BLE001 - a broken classifier is a cold one
        return False


def _lexicon_pass(text: str, entity_types: list[str] | None) -> tuple[Label, bool]:
    """(labels, weak) from lexicons, entity types and patterns alone."""
    t = norm(text)
    ws = words(text)
    out = Label.NONE
    weak = False
    for lbl, lex in _STRONG.items():
        if lbl not in _ARRIVAL_RULED and _hits(ws, lex):
            out |= lbl
    for lbl, lex in _WEAK.items():
        if lbl not in _ARRIVAL_RULED and not out & lbl and _hits(ws, lex):
            weak = True
    for et in entity_types or []:
        out |= _ETYPE_LABELS.get(norm(et), Label.NONE)
    if _MONEY_AMOUNT.search(text or ""):
        out |= Label.MONEY
    if _PRECISE_ADDR.search(text or ""):
        out |= Label.LOCATION_PRECISE
    strong, soft = _arrival_health(t, ws)
    out |= Label.HEALTH if strong else Label.NONE
    weak = weak or (soft and not out & Label.HEALTH)
    strong, soft = _arrival_minor(t, text, out & ALL_LABELS & ~Label.MINOR)
    out |= Label.MINOR if strong else Label.NONE
    weak = weak or (soft and not out & Label.MINOR)
    return out, weak


def lexicon_label(text: str, *, entity_types: list[str] | None = None) -> Label:
    """§5.3 the floor without the classifier: lexicons, entity types and patterns; a weak hit
    alone -> UNKNOWN_SENSITIVE. (The label classifier calibrates against this floor.)"""
    out, weak = _lexicon_pass(text, entity_types)
    if weak and not out & sensitive_mask():
        out |= Label.UNKNOWN_SENSITIVE
    return out


def label(text: str, *, entity_types: list[str] | None = None) -> Label:
    """§5.3 at arrival, by code: lexicons, entity types, the in-process classifier;
    uncertain -> UNKNOWN_SENSITIVE."""
    out, weak = _lexicon_pass(text, entity_types)
    if _classifier is not None and text:
        for lbl, score in _classifier(text).items():
            if score >= CLASSIFIER_HIGH:
                out |= lbl
            elif score >= CLASSIFIER_LOW and not out & lbl:
                weak = True
    if weak and not out & sensitive_mask():
        out |= Label.UNKNOWN_SENSITIVE
    return out


def inherit(labels: list[Label]) -> Label:
    """§5.3: a derived item's labels are the OR of its read set's labels."""
    out = Label.NONE
    for x in labels:
        out |= x
    return out


def refine_labels(current: Label, proposed: Label, *, by_owner: bool = False) -> Label:
    """§5.3: workers may add labels, never remove them; only the user can (via «исправь»)."""
    return Label(proposed) if by_owner else Label(current | proposed)


# ---------------------------------------------------------------- store lookups

_NOW_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now() -> str:
    return _clockmod.now().astimezone(UTC).strftime(_NOW_FMT)


def _ts(value: str | None) -> datetime:
    if not value:
        return _clockmod.now()
    v = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _in(n: int) -> str:
    return ", ".join("?" for _ in range(n))


@dataclass(frozen=True)
class _Src:
    """What policy needs to know about one read-set member or memory item."""

    id: str
    src: ReadSrc
    usable: bool
    labels: Label
    final: bool
    own: bool
    system: bool = False  # the Cube's own text in a clean turn, or a worker's output
    subjects: tuple[str, ...] = ()
    text: str = ""
    quote: str | None = None


def _tier_own(tier: str | None) -> bool:
    return tier in ("user", "user_confirmed")


def _appl_text(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    return " ".join(str(v) for k, v in data.items() if isinstance(v, str) and k != "for_whom")


def _load(conn: sqlite3.Connection, ids: list[str], *, depth: int = 0) -> dict[str, _Src]:
    """Records, then raw chunks, then derived items (resolved through their read sets)."""
    out: dict[str, _Src] = {}
    ids = list(dict.fromkeys(ids))
    if not ids:
        return out
    for start in range(0, len(ids), 400):
        chunk = ids[start : start + 400]
        for r in conn.execute(
            "SELECT r.id, r.status, r.labels, r.labels_final, r.author_tier, r.author_entity_id,"
            " r.origin_quote, f.subject_id, f.value_entity_id, f.text_en AS f_text, f.keywords_ru,"
            " f.keywords_en, f.quote_orig AS f_quote, f.value_text, f.applicability,"
            " i.about_entity_id, i.text_en AS i_text, i.quote_orig AS i_quote, l.text_en AS l_text"
            " FROM records r LEFT JOIN facts f ON f.id = r.id"
            " LEFT JOIN impressions i ON i.id = r.id LEFT JOIN loops l ON l.id = r.id"
            f" WHERE r.id IN ({_in(len(chunk))})",
            chunk,
        ):
            quote = r["f_quote"] or r["i_quote"] or r["origin_quote"]
            text = " ".join(
                x
                for x in (
                    quote,
                    r["keywords_ru"],
                    r["keywords_en"],
                    r["f_text"],
                    r["i_text"],
                    r["l_text"],
                    r["value_text"],
                    _appl_text(r["applicability"]),
                )
                if x
            )
            subjects = tuple(
                s
                # the subject a direct question must name; never the speaker (asking about
                # Дима's health must not surface Дима's claim about Саша)
                for s in (r["subject_id"], r["value_entity_id"], r["about_entity_id"])
                if s
            )
            out[r["id"]] = _Src(
                id=r["id"],
                src=ReadSrc.RECORD,
                usable=r["status"] in ("active", "archived"),
                labels=Label(r["labels"]),
                final=bool(r["labels_final"]),
                own=_tier_own(r["author_tier"]),
                system=r["author_tier"] == "system",
                subjects=subjects,
                text=text,
                quote=quote or r["keywords_ru"] or r["f_text"] or r["i_text"] or r["l_text"],
            )
    rest = [i for i in ids if i not in out]
    for start in range(0, len(rest), 400):
        chunk = rest[start : start + 400]
        for r in conn.execute(
            "SELECT id, status, labels, labels_final, author_tier, author_entity_id, text"
            f" FROM raw_chunks WHERE id IN ({_in(len(chunk))})",
            chunk,
        ):
            out[r["id"]] = _Src(
                id=r["id"],
                src=ReadSrc.RAW,
                usable=r["status"] == "active",
                labels=Label(r["labels"]),
                final=bool(r["labels_final"]),
                own=_tier_own(r["author_tier"]),
                system=r["author_tier"] == "system",
                subjects=(r["author_entity_id"],) if r["author_entity_id"] else (),
                text=r["text"] or "",
                quote=r["text"],
            )
    rest = [i for i in ids if i not in out]
    for start in range(0, len(rest), 400):
        chunk = rest[start : start + 400]
        for r in conn.execute(
            f"SELECT id, valid, labels, content, read_set FROM derived WHERE id IN ({_in(len(chunk))})",
            chunk,
        ):
            try:
                members = [
                    str(m[0]) if isinstance(m, list) else str(m) for m in json.loads(r["read_set"])
                ]
            except (ValueError, TypeError, IndexError):
                members = []
            srcs = _load(conn, members, depth=depth + 1) if depth < 2 and members else {}
            known = [srcs.get(m) for m in members]
            complete = bool(members) and all(s is not None for s in known)
            labels = Label(r["labels"]) | inherit([s.labels for s in known if s is not None])
            out[r["id"]] = _Src(
                id=r["id"],
                src=ReadSrc.DERIVED,
                usable=bool(r["valid"]),
                labels=labels if complete else labels | Label.UNKNOWN_SENSITIVE,
                final=complete and all(s.final for s in known if s is not None),
                own=complete and all(s.own for s in known if s is not None),
                text=r["content"] or "",
                quote=None,
            )
    return out


def labels_of(conn: sqlite3.Connection, read_set: list[ReadRef]) -> Label:
    """§5.3: the OR of a read set's labels, as a derived item built from it must carry.
    An unknown member counts as `unknown_sensitive` (fail-closed)."""
    found = _load(conn, [r.id for r in read_set])
    return inherit(
        [found[r.id].labels if r.id in found else Label.UNKNOWN_SENSITIVE for r in read_set]
    )


def _entity_names(conn: sqlite3.Connection, entity_ids: Iterable[str]) -> list[str]:
    ids = list(dict.fromkeys(entity_ids))
    if not ids:
        return []
    q = _in(len(ids))
    rows = conn.execute(
        f"SELECT display_name AS n FROM entities WHERE id IN ({q})"
        f" UNION SELECT alias FROM aliases WHERE entity_id IN ({q})"
        f" UNION SELECT alias_norm FROM aliases WHERE entity_id IN ({q})",
        ids * 3,
    ).fetchall()
    return [r[0] for r in rows if r[0]]


# ---------------------------------------------------------------- process routes (§5.4)


def _route_labels(labels: Label, final: bool) -> Label:
    """The labels a route is read from. Final labels that are only the uncertain band
    (`unknown_sensitive` with no named label and no reserved bit: the classifier scored in
    [LOW, HIGH) or a lexicon hit was weak, and nothing was confident) route as unlabeled, so
    anyone's message the classifier is unsure about is read where the user's own words are. A
    confident label, with or without the uncertain bit beside it, or a reserved bit keeps the
    table (local only by default). Before the labelling pass the bit is the lexicon floor's
    fail-closed guess and keeps the table too; the cloud needs final labels anyway."""
    if final and int(labels) == int(Label.UNKNOWN_SENSITIVE):
        return Label.NONE
    return labels


def _uncertain_only(labels: Label, final: bool, sens_mask: Label) -> bool:
    """Final labels whose sensitive part is only the uncertain band: the classifier finished,
    scored in [LOW, HIGH) and named no topic, so the item is not known to be sensitive. A
    confident label beside the bit, a reserved bit, or labels still pending say otherwise."""
    if not final or int(labels) >> len(LABEL_NAMES):
        return False
    return int(labels & sens_mask) == int(Label.UNKNOWN_SENSITIVE)


def _routes_for(labels: Label) -> set[Route]:
    table = routing_table()
    if not labels:
        return set(table.get("unlabeled", frozenset()))
    out = {Route.LOCAL, Route.CLOUD}
    for i, b in enumerate(_LABEL_BITS):
        if labels & b:
            out &= set(table.get(LABEL_NAMES[i], frozenset({Route.LOCAL})))
    if int(labels) >> len(LABEL_NAMES):  # reserved bits: fail-closed like unknown_sensitive
        out &= set(table.get("unknown_sensitive", frozenset({Route.LOCAL})))
    return out


def route(conn: sqlite3.Connection, read_set: list[ReadRef]) -> set[Route]:
    """§5.4: processors allowed for the union of the read set's labels; cloud needs
    labels_final. An empty set means the job waits (then parks), never downgrades.

    The user's own words (`user` / `user_confirmed`) count as unlabeled whatever their labels,
    also beside the Cube's own `system` text (its reply in a clean turn): there only the
    Cube's labels follow the table, so a plain reply never pulls the user's words local. A job
    with anyone else's text follows the table on every member's labels (the user's words give
    the context another person's reply is read in). A final union that is only the uncertain
    band routes as unlabeled (`_route_labels`); a confident label on any member keeps the
    whole job on the table. A missing or unusable member allows nothing (fail-closed)."""
    if not read_set:
        return _routes_for(Label.NONE)
    found = _load(conn, [r.id for r in read_set])
    members = [m for m in (found.get(r.id) for r in read_set) if m is not None]
    if len(members) != len(read_set) or any(not m.usable for m in members):
        return set()
    ours = all(m.own or m.system for m in members)
    # The UNION of the labels picks the route (§5.4), not the intersection of per-member
    # routes: a HEALTH row next to a plain one is a HEALTH job (local), never a park.
    union = inherit([m.labels for m in members if not (ours and m.own)])
    final = all(m.final for m in members)
    allowed = set(_routes_for(_route_labels(union, final)))
    if not final:
        allowed.discard(Route.CLOUD)  # the cloud never does the labelling
    return allowed


def route_each(conn: sqlite3.Connection, read_set: list[ReadRef]) -> dict[str, set[Route]]:
    """`route(conn, [ref])` for every member, from one load: each message's own route."""
    found = _load(conn, [r.id for r in read_set])
    out: dict[str, set[Route]] = {}
    for r in read_set:
        m = found.get(r.id)
        if m is None or not m.usable:
            out[r.id] = set()
            continue
        allowed = set(_routes_for(Label.NONE if m.own else _route_labels(m.labels, m.final)))
        if not m.final:
            allowed.discard(Route.CLOUD)
        out[r.id] = allowed
    return out


def ours(conn: sqlite3.Connection, read_set: list[ReadRef]) -> set[str]:
    """The members `route` reads as ours: the user's own words and the Cube's own text. A
    per-message split (scheduler) keeps them apart from anyone else's text, so the union rule
    above still decides every part a processor reads."""
    found = _load(conn, [r.id for r in read_set])
    return {i for i, m in found.items() if m.own or m.system}


# ---------------------------------------------------------------- use now (§5.5)


@dataclass(frozen=True)
class Use:
    """One use-now verdict. `note` is the one-line «учёл: …» an answer carries when it uses one
    of the user's own sensitive items; `hold` marks an item the brain sees but must not put into
    an outward message without their word in this turn («не отправлять без его слова»)."""

    allowed: bool
    reason: str
    note: str | None = None
    hold: bool = False

    @property
    def hold_line(self) -> str | None:
        return HOLD_LINE if self.hold else None


def _domains(ws: list[str]) -> set[str]:
    return {name for name, lex in _DOMAINS.items() if _hits(ws, lex)}


def label_topics(ws: list[str]) -> Label:
    """The labels whose topic lexicons (strong or weak) the words hit; no classifier."""
    out = Label.NONE
    for lbl in _LABEL_BITS:
        if _hits(ws, _STRONG.get(lbl, _EMPTY)) or _hits(ws, _WEAK.get(lbl, _EMPTY)):
            out |= lbl
    return out


_label_topics = label_topics  # the old private name, kept for callers in flight


_NOTE_LEAD = re.compile(
    r"^(?:запомни|помни|учти|кстати|ещё|еще|и)\s*[:,—-]?\s*|^(?:у меня|я|мой|моя|мне|my|i am|i'm|i)\s+",
    re.I,
)


def note_for(quote: str | None) -> str | None:
    """«учёл: аллергия на орехи» from the item's own words, ≤ 48 characters."""
    if not quote:
        return None
    q = " ".join(quote.split()).strip(" .!;")
    for _ in range(3):
        q2 = _NOTE_LEAD.sub("", q).strip()
        if q2 == q:
            break
        q = q2
    if not q:
        return None
    if len(q) > 48:
        cut = q[:48].rsplit(" ", 1)[0]
        q = (cut or q[:48]).rstrip(" ,;") + "…"
    return f"учёл: {q[0].lower() + q[1:]}"


def _turn_domains(ws: list[str]) -> set[str]:
    return _domains(ws) | {f"label:{n}" for n in names(label_topics(ws))}


def _applies(item: _Src, ws: list[str], turn_dom: set[str] | None = None) -> bool:
    """§5.5 applicability of the user's own item to this turn: a shared situation domain (an
    allergy and a food order), their words on the item's sensitive topic, or the item's own words."""
    if not ws:
        return False
    item_dom = _domains(words(item.text)) | {f"label:{n}" for n in names(item.labels)}
    if turn_dom is None:
        turn_dom = _turn_domains(ws)
    if item_dom & turn_dom:
        return True
    return _overlap(_terms(item.text), ws)


_NOT_WHO = ("place", "org", "matter", "thing")  # entity types that are what, not who


def _topic_exclude(
    conn: sqlite3.Connection,
    subjects: tuple[str, ...],
    ws: list[str],
    targets: frozenset[str],
    subject_names: list[str],
) -> list[str]:
    """The names an open-topic overlap may not count: a name alone unlocks nothing (§5.5).
    When their words name a person (or pet) among the item's subjects, only the named people's
    names stay out: a place or an organisation the item is about («Сплин», "Seoul") is then
    what the user asks about, and an alias the extractor gave it from a plain word («билеты») is
    no name of whom the user asks. Without a named person every subject name stays out."""
    ids = [s for s in dict.fromkeys(subjects) if s]
    if len(ids) < 2:
        return subject_names
    try:
        what = {
            r[0]
            for r in conn.execute(
                f"SELECT id FROM entities WHERE id IN ({_in(len(ids))})"  # noqa: S608
                f" AND etype IN ({_in(len(_NOT_WHO))})",
                [*ids, *_NOT_WHO],
            )
        }
    except sqlite3.Error:
        return subject_names
    people = [
        e for e in ids if e not in what and (e in targets or names_in(ws, _entity_names(conn, [e])))
    ]
    return _entity_names(conn, people) if people else subject_names


# question words that carry no content of their own (shorter ones never count, see `_content_words`)
_ASK_TEXT = (
    "does which whose whom many much kind sort type both ever doing done tell know"
    " почему зачем откуда сколько какая какие каком какую чего кого кому чем кто что"
)
_ASK_WORDS = frozenset(norm(w) for w in _ASK_TEXT.split())


def _content_words(text: str, ws: list[str], exclude: Iterable[str]) -> list[str]:
    """The user's content words, one per stem: four letters or more, no stop or question word, no
    name. A name is a word of `exclude` (the subject's names) or a capitalised word inside a
    sentence (other people the user names); a name alone is no content (§5.5)."""
    toks = (text or "").split()
    proper: set[str] = set()
    for k, tok in enumerate(toks):
        if k and tok[:1].isupper() and not toks[k - 1].endswith((".", "!", "?")):
            letters = [ch for ch in tok if ch.isalpha()]
            if len(letters) >= 3 and all(ch.isupper() for ch in letters):
                continue  # an acronym («LGBTQ», «ADHD») is a topic, not a name
            proper.update(words(tok))
    names_ = [t for name in exclude for t in words(name)]
    out: dict[str, str] = {}
    for w in ws:
        if len(w) < 4 or w.isdigit() or w in _STOP or w in _ASK_WORDS or w in proper:
            continue
        if any(_name_token_match(n, w) for n in names_):
            continue
        out.setdefault(_stem(w), w)
    return list(out.values())


def _asks_content(
    item: _Src, turn: TurnCtx, ws: list[str], exclude: list[str], missing: Label
) -> bool:
    """§5.5 a direct question by content: their words ask this message's own words, so the
    label does not need its topic word in the question («how did Kim get hurt?» asks the
    message «I got hurt at work» whether the classifier or the word «injury» set `health`).

    Only a raw message whose labels are final qualifies (its labels came from that very text);
    the open-topic labels keep their own test (it already is the message's own words). The
    caller has checked the subject is named and the turn sends nothing."""
    if item.src is not ReadSrc.RAW or not item.final or not missing or missing & OPEN_TOPIC:
        return False
    asked = _content_words(turn_words(turn), ws, exclude)
    if len(asked) < CONTENT_MIN:
        return False
    terms = _terms(item.text)
    hit = [w for w in asked if any(t.startswith(_stem(w)) or w.startswith(_stem(t)) for t in terms)]
    return len(hit) >= CONTENT_MIN and len(hit) >= CONTENT_SHARE * len(asked)


def _asked(held: Label, ws: list[str], item: _Src, names_: list[str]) -> bool:
    terms = _terms(item.text)
    return all(topic_hit(b, ws, item_terms=terms, exclude=names_) for b in bits(held))


def _decide(
    conn: sqlite3.Connection,
    item: _Src | None,
    turn: TurnCtx,
    ws: list[str],
    send: bool,
    sens_mask: Label,
    turn_dom: set[str] | None = None,
    targets: frozenset[str] = frozenset(),
) -> Use:
    if item is None:
        return Use(False, "missing")
    if not item.usable:
        return Use(False, "not_usable")
    sens = item.labels & sens_mask
    held = sens | (Label.NONE if item.own else item.labels & Label.LOCATION_PRECISE)
    note = None
    if not sens:
        allowed, reason = True, "not_sensitive"
    elif item.own:
        if not ws:
            # neutral turn (core view) or an event: the user's own items may sit in the core and fire
            allowed, reason = True, "own_neutral"
        elif _applies(item, ws, turn_dom):
            allowed, reason, note = True, "own_applies", note_for(item.quote)
        elif item.final and not sens & ~OWN_RANKED:
            # the classifier finished and named no topic (unsure, or someone else's matter
            # in the user's words): no proof the item does not apply, so whether it fits this
            # turn is a ranking question, never a denial; send turns still hold it. A label
            # still pending is not yet known and keeps the applicability test until it lands.
            allowed, reason, note = True, "own_ranked", note_for(item.quote)
        else:
            allowed, reason = False, "own_not_applicable"
    else:
        subject_names = _entity_names(conn, item.subjects)
        terms = _terms(item.text)
        excl = _topic_exclude(conn, item.subjects, ws, targets, subject_names)
        missing = Label.NONE
        for b in bits(sens):
            if not topic_hit(b, ws, item_terms=terms, exclude=excl):
                missing |= b
        topic = not missing
        # the question's target resolved to its entity (aliases, nicknames, the other script)
        # names the subject as well as the user's literal words do; the topic test stays (§5.5)
        named = names_in(ws, subject_names) or bool(targets & set(item.subjects))
        allowed = topic and named
        reason = "direct_question" if allowed else ("name_only" if named else "not_asked")
        if (
            named
            and not allowed
            and not send
            and not _uncertain_only(item.labels, item.final, sens_mask)
            and _asks_content(item, turn, ws, excl, missing)
        ):
            # the user names whom the message is about and asks its own words: a direct
            # question about that message, never on a send turn
            allowed, reason = True, "direct_content"
        if not allowed and _uncertain_only(item.labels, item.final, sens_mask):
            # another person's item the classifier finished unsure about: not known to be
            # sensitive, so relevance ranks it like an unlabeled one (the same reading as its
            # route) even though the name and topic test did not pass; the basket ranks it
            # among the clean lines by its score, and send turns still hold it
            allowed, reason = True, "uncertain_ranked"
    if not allowed:
        return Use(False, reason)
    hold = bool(send and held and not _asked(held, ws, item, _entity_names(conn, item.subjects)))
    return Use(True, reason, note=note, hold=hold)


def decide_many(conn: sqlite3.Connection, item_ids: list[str], turn: TurnCtx) -> dict[str, Use]:
    """`decide` for a candidate list in one pass (the basket's step 3)."""
    found = _load(conn, item_ids)
    ws = words(turn_words(turn))
    send = is_send_turn(turn)
    mask = sensitive_mask()
    dom = _turn_domains(ws) if ws else set()
    return {i: _decide(conn, found.get(i), turn, ws, send, mask, dom) for i in item_ids}


def decide_targeted(
    conn: sqlite3.Connection, item_ids: list[str], turn: TurnCtx, targets: Iterable[str]
) -> dict[str, Use]:
    """`decide_many` with the question's target already resolved to entity ids (the cue's name
    index): «direct» is decided by who the question is about, not by which spelling of the
    name their words use («Шурик», "Kim", a declined form). Other people's items about that
    target count as named; their topic test is unchanged, so a name alone still unlocks
    nothing sensitive."""
    found = _load(conn, item_ids)
    ws = words(turn_words(turn))
    send = is_send_turn(turn)
    mask = sensitive_mask()
    dom = _turn_domains(ws) if ws else set()
    tg = frozenset(t for t in targets if t and t != "self")
    return {i: _decide(conn, found.get(i), turn, ws, send, mask, dom, tg) for i in item_ids}


def decide(conn: sqlite3.Connection, item_id: str, turn: TurnCtx) -> Use:
    """§5.5 verdict with its note and send-turn hold."""
    return decide_many(conn, [item_id], turn)[item_id]


def use_now(conn: sqlite3.Connection, item_id: str, turn: TurnCtx) -> bool:
    """§5.5: the user's own sensitive items where applicability matches (with a note); others'
    sensitive items only on a direct question (topic lexicon + subject); send turns mark
    them «не отправлять без его слова»; the user's own items fire; pinned categories. The
    second fence after the SQL label mask."""
    return decide(conn, item_id, turn).allowed


def decide_retired(conn: sqlite3.Connection, item_id: str, turn: TurnCtx) -> Use:
    """§5.5 for a record that is no longer usable (superseded, retracted, needs review), as
    if it were: «откуда это» on an old handle keeps the same fence."""
    import dataclasses

    item = _load(conn, [item_id]).get(item_id)
    if item is not None:
        item = dataclasses.replace(item, usable=True)
    ws = words(turn_words(turn))
    dom = _turn_domains(ws) if ws else set()
    return _decide(conn, item, turn, ws, is_send_turn(turn), sensitive_mask(), dom)


def allowed(conn: sqlite3.Connection, item_ids: list[str], turn: TurnCtx) -> list[str]:
    """The ids ranking may see, in their order: use-now runs before rank (§2)."""
    verdicts = decide_many(conn, item_ids, turn)
    return [i for i in item_ids if verdicts[i].allowed]


def label_mask(turn: TurnCtx) -> Label:
    """§5.5, §10.2: labels the finders may return this turn (applied in SQL).

    Every non-sensitive label, plus each sensitive label whose topic their words hit; the
    open-topic labels whenever the user says anything (their topic is the item's own words).
    The user's own items pass the SQL whatever their labels (see `mask_clause`): applicability
    is checked by `decide`. A superset of what `decide` allows, except another person's raw
    message opened by its own words (`_asks_content`): the basket's raw finder brings those
    past this mask when the question names someone, and `decide` rules on each."""
    sens = sensitive_mask()
    ws = words(turn_words(turn))
    mask = ALL_LABELS & ~sens
    if ws:
        mask |= label_topics(ws) & sens
        mask |= OPEN_TOPIC & sens
    return Label(mask)


def mask_clause(
    turn: TurnCtx, *, alias: str = "records", tiered: bool = True
) -> tuple[str, list[int]]:
    """The finders' SQL fence: `(labels & ~mask) = 0` or one of the user's own items. Tables
    without `author_tier` (derived) pass `tiered=False`: stricter there, never looser."""
    mask = int(label_mask(turn))
    if not tiered:
        return (f"(({alias}.labels & ~?) = 0)", [mask])
    return (f"(({alias}.labels & ~?) = 0 OR {alias}.author_tier IN {_OWN_SQL})", [mask])


def because_quote_ok(quote: str, turn: TurnCtx, *, labels: Label, target: str | None) -> bool:
    """§5.5: a substring of the user's own words that hits the label's topic or names the target."""
    q = norm(quote)
    own = norm(turn_words(turn))
    if len(q) < 2 or not own or q not in own:
        return False
    qw = words(q)
    if target and names_in(qw, [target]):
        return True
    lbls = bits(Label(labels))
    return bool(lbls) and all(topic_hit(b, qw) for b in lbls)


def may_send(
    conn: sqlite3.Connection, item_id: str, turn: TurnCtx, because_quote: str | None
) -> bool:
    """§5.5 send turns: an item enters an outward message only when it is allowed and either
    not held or `because_quote` (their words in this turn) asks for its held labels."""
    found = _load(conn, [item_id])
    item = found.get(item_id)
    verdict = decide(conn, item_id, turn)
    if item is None or not verdict.allowed:
        return False
    if not verdict.hold:
        return True
    if not because_quote:
        return False
    held = (item.labels & sensitive_mask()) | (
        Label.NONE if item.own else item.labels & Label.LOCATION_PRECISE
    )
    q = norm(because_quote)
    if len(q) < 2 or q not in norm(turn_words(turn)):
        return False
    return _asked(held, words(q), item, _entity_names(conn, item.subjects))


# ---------------------------------------------------------------- chat modes (§5.2)


def chat_mode(conn: sqlite3.Connection, chat_id: str) -> ChatMode:
    """§5.2: `normal` unless `chat_policy` says otherwise (new contacts are normal)."""
    row = conn.execute("SELECT mode FROM chat_policy WHERE chat_id = ?", (chat_id,)).fetchone()
    return ChatMode(row[0]) if row else ChatMode.NORMAL


# the words that name each mode in the user's order
_MODE_WORDS: dict[ChatMode, Lexicon] = {
    ChatMode.BANNED: _lex(
        "запрет|запрещ|забан|=бан|блокир|заблок|не запоминай|ничего не запоминай|=ban|banned",
        "=block|blocked|stop remembering",
    ),
    ChatMode.NO_DIG: _lex("не копа|не копай|не следи|не читай|no dig|=no_dig|don't dig"),
    ChatMode.NORMAL: _lex(
        "сними|снять|разбан|разблок|обычн|можно копать|можно следить|отмени запрет|=unban",
        "unblock|=normal|=lift|lift the",
    ),
}


def _chat_names(conn: sqlite3.Connection, chat_id: str) -> list[str]:
    """What names a chat in their words: its id's tail (`tg:boss` → «boss») and the names of
    the people who write in it or whose platform id it is."""
    tail = chat_id.split(":", 1)[-1]
    names = [chat_id, tail]
    ids = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT author_entity_id FROM raw_chunks"
            " WHERE chat_id = ? AND author_entity_id IS NOT NULL",
            (chat_id,),
        )
    }
    ids |= {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT e.id FROM entities e, json_each(e.stable_ids) j"
            " WHERE json_extract(j.value, '$.value') IN (?, ?)",
            (chat_id, tail),
        )
    }
    if ids:
        names += [
            r[0]
            for r in conn.execute(
                f"SELECT alias FROM aliases WHERE entity_id IN ({_in(len(ids))})",  # noqa: S608
                list(ids),
            )
        ]
    return names


def mode_ordered(conn: sqlite3.Connection, turn: TurnCtx, chat_id: str, mode: ChatMode) -> bool:
    """§13.2 (the forget rule `commands._his_word_needed` applied to modes): in a turn that read
    foreign text or is not the user's message, a mode change stands only when the user's own
    words name the mode and the chat. A clean turn of the user's: their word is the order."""
    if not (turn.is_tainted or turn.origin is not TurnOrigin.OWNER):
        return True
    ws = words(turn_words(turn))
    if not ws or not _hits(ws, _MODE_WORDS[ChatMode(mode)]):
        return False
    return names_in(ws, _chat_names(conn, chat_id))


def set_mode(conn: sqlite3.Connection, chat_id: str, mode: ChatMode) -> int:
    """§5.2: returns the new policy_version; the forget side runs from the job.

    Bumps `meta.policy_version`, writes the chat's row, one content-free `events` row (`mode`)
    and, to `banned` or `no_dig`, a `mode_forget` job (v0.4 A3: chat, mode, policy_version),
    all in one transaction, so it applies before the next admission and a crash leaves the
    job for `api.startup`. Setting the mode a chat already has changes nothing and returns its
    version. A ban drops the chat's admit windows. The island never takes a mode (A11)."""
    mode = ChatMode(mode)
    if chat_id.strip().casefold() == ISLAND_CHAT:
        raise PolicyRefused(ISLAND_MODE_LINE)
    own_tx = not conn.in_transaction
    if own_tx:
        conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT mode, policy_version FROM chat_policy WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        current = ChatMode(row[0]) if row else ChatMode.NORMAL
        if current is mode and (row is not None or mode is ChatMode.NORMAL):
            pv = (
                int(row[1])
                if row
                else int(conn.execute("SELECT policy_version FROM meta WHERE id = 1").fetchone()[0])
            )
            if own_tx:
                conn.execute("COMMIT")
            return pv
        ts = _now()
        pv, mv = conn.execute(
            "UPDATE meta SET policy_version = policy_version + 1, updated_at = ? WHERE id = 1"
            " RETURNING policy_version, memory_version",
            (ts,),
        ).fetchone()
        conn.execute(
            "INSERT INTO chat_policy (chat_id, mode, policy_version, updated_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT (chat_id) DO UPDATE SET mode = excluded.mode,"
            " policy_version = excluded.policy_version, updated_at = excluded.updated_at",
            (chat_id, mode.value, pv, ts),
        )
        if mode is ChatMode.BANNED:
            conn.execute("DELETE FROM admit_windows WHERE chat_id = ?", (chat_id,))
        conn.execute(
            "INSERT INTO events (ts, kind, memory_version, item_ids) VALUES (?, ?, ?, '[]')",
            (ts, EventKind.MODE.value, mv),
        )
        if mode is not ChatMode.NORMAL:
            payload = {"chat_id": chat_id, "mode": mode.value, "policy_version": int(pv)}
            conn.execute(
                "INSERT INTO jobs (id, jkind, payload, state, attempts, not_before,"
                " idempotency_key, created_at, updated_at) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?, ?)",
                (
                    f"job-{secrets.token_hex(8)}",
                    MODE_FORGET_JOB,
                    json.dumps(payload, ensure_ascii=False),
                    ts,
                    f"{MODE_FORGET_JOB}:{chat_id}:{int(pv)}",
                    ts,
                    ts,
                ),
            )
        if own_tx:
            conn.execute("COMMIT")
        return int(pv)
    except BaseException:
        if own_tx and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _claim_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Take a `mode_forget` job: `queued`, or `running` with an ended lease (a crash). A live
    lease belongs to another process and is left alone (§18.1 step 3)."""
    from . import store as _store  # the lease names its holder

    now = _now()
    row = conn.execute(
        f"UPDATE jobs SET state = 'running', not_before = ?, updated_at = ?, {_store.HOLD_SQL}"
        " WHERE id = ? AND jkind = ? AND (state = 'queued'"
        " OR (state = 'running' AND (not_before IS NULL OR not_before <= ?))) RETURNING payload",
        (_lease(now), now, _store.lease_holder(), job_id, MODE_FORGET_JOB, now),
    ).fetchone()
    return json.loads(row[0]) if row else None


def _lease(now: str) -> str:
    return (_ts(now) + JOB_LEASE).strftime(_NOW_FMT)


def run_mode_forget(conn: sqlite3.Connection, job_id: str) -> list[str] | None:
    """§5.2 (v0.4 A3): stop-use for one mode change, from its `mode_forget` job; marks it
    `done`. Returns the job's forget ids (empty when nothing was left or a later mode change
    superseded it), or None when the job is not there to take (done, or another process
    holds its lease).

    The ids are picked now (`mode_scope`, active rows only), so a rerun after a crash takes
    only what the first run did not stop. They go to the forget pipeline in batches of
    `MODE_BATCH`, each its own forget, so a chat of any length stays under SQLite's
    variable limit; the lease is renewed per batch."""
    from . import forget  # forget imports policy

    body = _claim_job(conn, job_id)
    if body is None:
        return None
    chat_id, mode = str(body["chat_id"]), ChatMode(body["mode"])
    row = conn.execute(
        "SELECT mode, policy_version FROM chat_policy WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    current = row is not None and row[0] == mode.value and int(row[1]) == body["policy_version"]
    fids: list[str] = list(body.get("forget_ids") or [])
    if current:
        ids = mode_scope(conn, chat_id, mode)
        for start in range(0, len(ids), MODE_BATCH):
            fids.append(
                forget.stop_use(
                    conn, ids[start : start + MODE_BATCH], scope=ForgetScope.CHAT, forever=True
                )
            )
            body["forget_ids"] = fids
            now = _now()
            conn.execute(
                "UPDATE jobs SET payload = ?, not_before = ?, updated_at = ? WHERE id = ?",
                (json.dumps(body, ensure_ascii=False), _lease(now), now, job_id),
            )
    else:
        body["superseded"] = True
    now = _now()
    conn.execute(
        "UPDATE jobs SET state = 'done', payload = ?, updated_at = ? WHERE id = ?",
        (json.dumps(body, ensure_ascii=False), now, job_id),
    )
    return fids


def finish_mode_forgets(conn: sqlite3.Connection) -> int:
    """§18.1 step 5: run every unfinished `mode_forget` job (oldest first) before any turn.
    Returns how many ran. `api.startup` calls this; a live lease is skipped."""
    now = _now()
    jobs = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM jobs WHERE jkind = ? AND (state = 'queued'"
            " OR (state = 'running' AND (not_before IS NULL OR not_before <= ?)))"
            " ORDER BY created_at, id",
            (MODE_FORGET_JOB, now),
        )
    ]
    return sum(run_mode_forget(conn, job_id) is not None for job_id in jobs)


def _shingles(ws: list[str], n: int = 8) -> set[bytes]:
    return {
        hashlib.blake2b(" ".join(ws[i : i + n]).encode(), key=_SALT, digest_size=12).digest()
        for i in range(len(ws) - n + 1)
    }


def mode_scope(conn: sqlite3.Connection, chat_id: str, mode: ChatMode) -> list[str]:
    """§5.2: the memory-side ids a mode change hands to the forget pipeline.

    - `banned`: the chat's raw; the raw of Cube turns whose `turn_sources` include it, except
      the user's own words that share no 8-word shingle with the chat's raw; records grounded
      in them.
    - `no_dig`: the same, minus what the user asked for: raw inside the chat's admit windows
      and turns where the read was `requested` stay.
    - `normal`: nothing."""
    mode = ChatMode(mode)
    if mode is ChatMode.NORMAL:
        return []
    rows = conn.execute(
        "SELECT id, ts, author_tier, text, turn_id FROM raw_chunks"
        " WHERE chat_id = ? AND status = 'active'",
        (chat_id,),
    ).fetchall()
    keep = set() if mode is ChatMode.BANNED else _requested_raw(conn, chat_id, rows)
    raw_ids: list[str] = []
    chat_shingles: set[bytes] = set()
    for r in rows:
        chat_shingles |= _shingles(words(r["text"] or ""))
        if r["id"] not in keep:
            raw_ids.append(r["id"])
    req_col = "1" if mode is ChatMode.BANNED else "max(requested) = 0"
    turns = [
        r[0]
        for r in conn.execute(
            f"SELECT turn_id FROM turn_sources WHERE chat_id = ? GROUP BY turn_id HAVING {req_col}",
            (chat_id,),
        )
    ]
    for start in range(0, len(turns), 400):
        chunk = turns[start : start + 400]
        for r in conn.execute(
            "SELECT id, author_tier, text FROM raw_chunks WHERE status = 'active'"
            f" AND turn_id IN ({_in(len(chunk))})",
            chunk,
        ):
            if _tier_own(r["author_tier"]) and not (
                _shingles(words(r["text"] or "")) & chat_shingles
            ):
                continue  # the user's own words in that turn stay (§5.2)
            raw_ids.append(r["id"])
    raw_ids = list(dict.fromkeys(raw_ids))
    record_ids: list[str] = []
    for start in range(0, len(raw_ids), 400):
        chunk = raw_ids[start : start + 400]
        q = _in(len(chunk))
        record_ids.extend(
            r[0]
            for r in conn.execute(
                f"SELECT id FROM facts WHERE source_raw_id IN ({q})"
                f" UNION SELECT target_record_id FROM justifications WHERE member_record_id IN ({q})",
                chunk * 2,
            )
        )
    return raw_ids + [r for r in dict.fromkeys(record_ids) if r not in raw_ids]


# ---------------------------------------------------------------- admit windows (§5.6)


def open_window(conn: sqlite3.Connection, window: AdmitWindow) -> None:
    """§5.6: a direct-reply window (by action_key) or an intention window.

    Direct reply: `after_ts` is the send's time; `cube_message_id` (v0.4 F4) is the sent
    message's platform id, so a message whose `reply_to` is that id is admitted; a `reason`
    of `reply:<id>` (rows written before migration 0002) works the same; otherwise the peer's
    first message within 24 h. Intention: that sender's messages after `after_ts` until the
    intention is done or cancelled (or `until_ts`). On a `banned` chat: `PolicyRefused` with
    one line."""
    if chat_mode(conn, window.chat_id) is ChatMode.BANNED:
        raise PolicyRefused(BANNED_WINDOW_LINE)
    cols = ["id", "chat_id", "sender_id", "after_ts", "until_ts", "reason", "action_key"]
    cols += ["intention_id", "created_at"]
    vals: list[str | None] = [
        window.id,
        window.chat_id,
        window.sender_id,
        window.after_ts,
        window.until_ts,
        window.reason,
        window.action_key,
        window.intention_id,
        _now(),
    ]
    update = "sender_id, after_ts, until_ts, reason"
    if _has_cube_message_id(conn):
        cols.append("cube_message_id")
        vals.append(window.cube_message_id)
        update += ", cube_message_id"
    sets = ", ".join(f"{c} = excluded.{c}" for c in update.split(", "))
    conn.execute(
        f"INSERT INTO admit_windows ({', '.join(cols)}) VALUES ({_in(len(cols))})"  # noqa: S608
        f" ON CONFLICT (id) DO UPDATE SET {sets}",
        vals,
    )


def _has_cube_message_id(conn: sqlite3.Connection) -> bool:
    """Whether `admit_windows.cube_message_id` exists (migration 0002)."""
    return any(r[1] == "cube_message_id" for r in conn.execute("PRAGMA table_info(admit_windows)"))


def close_windows(conn: sqlite3.Connection, *, intention_id: str, ts: str | None = None) -> int:
    """§5.6: an intention done or cancelled closes its windows."""
    cur = conn.execute(
        "UPDATE admit_windows SET until_ts = ? WHERE intention_id = ? AND until_ts IS NULL",
        (ts or _now(), intention_id),
    )
    return cur.rowcount


def _windows(conn: sqlite3.Connection, chat_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT w.*, i.state AS istate FROM admit_windows w"
        " LEFT JOIN intentions i ON i.id = w.intention_id WHERE w.chat_id = ?",
        (chat_id,),
    ).fetchall()


def _requested_raw(conn: sqlite3.Connection, chat_id: str, rows: list[sqlite3.Row]) -> set[str]:
    """Raw of the exchanges the user asked for in this chat (§5.2 «не копать» keeps them, v0.4
    A11): (a) within each admit window, or within 24 h after a send the user ordered (`actions`
    to this chat), the user's and the Cube's messages plus the peer's first reply; an intention
    window keeps what came in while it was open (raw keeps no `reply_to`, so the 24 h rule
    stands in); (b) raw of a turn whose `turn_sources` row for this chat has `requested=1`,
    whatever its author; (c) the user's own words that share no 8-word shingle with the chat's
    other raw."""
    spans: list[tuple[datetime, datetime, bool]] = []
    far = datetime.max.replace(tzinfo=UTC)
    for w in _windows(conn, chat_id):
        after = _ts(w["after_ts"])
        until = _ts(w["until_ts"]) if w["until_ts"] else far
        if w["intention_id"] is None:
            spans.append((after, min(until, after + REPLY_WINDOW), True))
        else:
            spans.append((after, until, False))
    for a in conn.execute(
        "SELECT reserved_at FROM actions WHERE stable_target = ? AND state IN ('ok', 'unknown')",
        (chat_id,),
    ):
        after = _ts(a[0])
        spans.append((after, after + REPLY_WINDOW, True))
    asked = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT turn_id FROM turn_sources WHERE chat_id = ? AND requested = 1",
            (chat_id,),
        )
    }
    keep: set[str] = {r["id"] for r in rows if r["turn_id"] is not None and r["turn_id"] in asked}
    shingles = {r["id"]: _shingles(words(r["text"] or "")) for r in rows}
    seen = Counter(sh for v in shingles.values() for sh in v)  # rows holding each shingle
    for r in rows:
        if _tier_own(r["author_tier"]) and all(seen[sh] == 1 for sh in shingles[r["id"]]):
            keep.add(r["id"])
    ordered = sorted(rows, key=lambda r: _ts(r["ts"]))
    for after, until, reply in spans:
        peer_taken = False
        for r in ordered:
            t = _ts(r["ts"])
            if t < after or t > until:
                continue
            if reply and r["author_tier"] == "other":
                if peer_taken:
                    continue
                peer_taken = True
            keep.add(r["id"])
    return keep


def _match_window(
    conn: sqlite3.Connection,
    chat_id: str,
    sender_id: str | None,
    reply_to: str | None,
    ts: datetime,
) -> sqlite3.Row | None:
    for w in _windows(conn, chat_id):
        after = _ts(w["after_ts"])
        if ts < after or (w["until_ts"] and ts > _ts(w["until_ts"])):
            continue
        if w["intention_id"] is not None:
            if w["istate"] in ("done", "cancelled"):
                continue
            if w["sender_id"] and sender_id != w["sender_id"]:
                continue
            return w
        if reply_to:
            # v0.4 F4: the sent message's id; `reply:<id>` stays for rows before migration 0002
            sent = w["cube_message_id"] if "cube_message_id" in w.keys() else None  # noqa: SIM118
            reason = str(w["reason"])
            old = reason[len("reply:") :] if reason.startswith("reply:") else None
            if reply_to in (sent, old):
                return w
        if w["sender_id"] and sender_id and sender_id != w["sender_id"]:
            continue
        if ts <= after + REPLY_WINDOW and not _peer_spoke(conn, chat_id, after, ts):
            return w
    return None


def _peer_spoke(conn: sqlite3.Connection, chat_id: str, after: datetime, before: datetime) -> bool:
    """Whether the peer's first message after the send is already in (then this one is not
    the first)."""
    for r in conn.execute(
        "SELECT ts FROM raw_chunks WHERE chat_id = ? AND author_tier = 'other' AND status = 'active'",
        (chat_id,),
    ):
        t = _ts(r[0])
        if after <= t < before:
            return True
    return False


# ---------------------------------------------------------------- turn reads (§5.1)

_SALT = secrets.token_bytes(16)
_READS: OrderedDict[str, dict[str, set[bytes]]] = OrderedDict()
_READS_MAX_TURNS = 64


def note_read(turn_id: str, chat_id: str, text: str) -> None:
    """The hooks adapter hands each read's text here, in RAM only: salted 8-word shingle
    hashes, never written, dropped by `end_turn`. `admit` uses them to keep the user's own
    words from carrying a banned read (§5.1)."""
    per = _READS.setdefault(turn_id, {})
    _READS.move_to_end(turn_id)
    per.setdefault(chat_id, set()).update(_shingles(words(text)))
    while len(_READS) > _READS_MAX_TURNS:
        _READS.popitem(last=False)


def end_turn(turn_id: str) -> None:
    _READS.pop(turn_id, None)


def record_turn_source(
    conn: sqlite3.Connection, turn_id: str, source_kind: str, chat_id: str, *, requested: bool
) -> None:
    """§5.1: one `turn_sources` row, from tool input and platform metadata only.
    `requested` = their words asked for this read; a request anywhere in the turn wins."""
    conn.execute(
        "INSERT INTO turn_sources (turn_id, source_kind, chat_id, requested, created_at)"
        " VALUES (?, ?, ?, ?, ?) ON CONFLICT (turn_id, source_kind, chat_id)"
        " DO UPDATE SET requested = max(requested, excluded.requested)",
        (turn_id, source_kind, chat_id, int(bool(requested)), _now()),
    )


def _poisoned(conn: sqlite3.Connection, turn_id: str) -> list[tuple[str, ChatMode]]:
    """Chats this turn read that bar it: banned, or `no_dig` read in the background."""
    rows = conn.execute(
        "SELECT s.chat_id, max(s.requested) AS req, p.mode FROM turn_sources s"
        " JOIN chat_policy p ON p.chat_id = s.chat_id"
        " WHERE s.turn_id = ? AND p.mode <> 'normal' GROUP BY s.chat_id",
        (turn_id,),
    ).fetchall()
    return [
        (r["chat_id"], ChatMode(r["mode"])) for r in rows if r["mode"] == "banned" or not r["req"]
    ]


# ---------------------------------------------------------------- admit (§5.1)


# §5.1 rule 3 across two messages: «пароль от вайфая:» in one message and «Barsik2019!»
# alone in the next. Admission sees one message at a time, so a message that ends on a password
# keyword leaves a tiny per-chat flag: a few minutes, a few messages, never any content. A lone
# value that follows is a secret: the user's own is redacted, another person's is dropped.
_ASK_TAIL = re.compile(
    rf"(?i){_KW_LEFT}(?:{_PASS_KW}|{_SHORT_KW}|{_KEY_KW})(?:\s+[^\s:=—–;.!?]+){{0,4}}?"
    r"\s*(?P<sep>[:=—–]|\bis\b|\bэто\b|\bтакой\b)?\s*[?!.…]*\s*$"
)
_LONE = re.compile(
    r"^\s*(?:(?:вот|держи|лови|here|it's|its)[:,]?\s+)?(?:[\"'«“]([^\"'»”\n]{1,120})[\"'»”]"
    r"|(\S{4,}))\s*$",
    re.I,
)
_EXPECT: OrderedDict[str, tuple[datetime, int, bool]] = OrderedDict()  # key → (at, left, strong)
_EXPECT_MAX = 64
_EXPECT_TTL = timedelta(minutes=5)
_EXPECT_MSGS = 3


def _ask_key(conn: sqlite3.Connection, chat_id: str | None, turn: TurnCtx | None) -> str | None:
    """One flag per chat (or island thread) of one memory.db."""
    chat = chat_id or (f"thread:{turn.thread_id}" if turn is not None else None)
    if chat is None:
        return None
    row = conn.execute("PRAGMA database_list").fetchone()
    db = row[2] if row is not None and row[2] else f"mem:{id(conn)}"
    return f"{db}\x00{chat}"


def _note_secret_ask(
    key: str | None, text: str, spans: list[tuple[int, int]], when: datetime
) -> None:
    """Arm the flag after a message that ends on a password keyword with no value after it;
    count down (and drop) it otherwise. «пароль от вайфая:» arms a strong flag (any lone
    word that follows is the value), «какой пароль от вайфая?» a weak one (only a value that
    looks like a secret)."""
    if key is None:
        return
    m = _ASK_TAIL.search(text or "")
    if m and not any(e > m.start() for _s, e in spans):
        _EXPECT[key] = (when, _EXPECT_MSGS, m.group("sep") is not None)
        _EXPECT.move_to_end(key)
        while len(_EXPECT) > _EXPECT_MAX:
            _EXPECT.popitem(last=False)
        return
    held = _EXPECT.get(key)
    if held is None:
        return
    if spans or held[1] <= 1:
        del _EXPECT[key]
    else:
        _EXPECT[key] = (held[0], held[1] - 1, held[2])


def _expected_secret(key: str | None, text: str, when: datetime) -> list[tuple[int, int]]:
    """The span of a lone value that answers an armed password keyword in this chat."""
    held = _EXPECT.get(key) if key is not None else None
    if held is None:
        return []
    at, _left, strong = held
    if abs(when - at) > _EXPECT_TTL:
        del _EXPECT[key]  # type: ignore[arg-type]
        return []
    m = _LONE.match(text or "")
    if m is None:
        return []
    g = 1 if m.group(1) is not None else 2
    val = m.group(g)
    if _placeholder(val) or _ADDRESS.match(val):
        return []
    if g == 1 or _secretish(val) or (val.isdigit() and len(val) >= 4) or strong:
        return [m.span(g)]
    return []


def clear_secret_asks() -> None:
    """Forget every armed flag (tests; a host restart starts empty anyway)."""
    _EXPECT.clear()


@dataclass(frozen=True)
class Admission:
    """`admit_ex`'s verdict. `text` is what may be stored: the user's own words only in a turn
    that read a barred chat, with never-store spans replaced. `intention_id` scopes a no_dig
    intention window's message into that matter only."""

    admit: bool
    reason: str | None = None
    text: str | None = None
    keep_only_own_words: bool = False
    window_id: str | None = None
    intention_id: str | None = None


def _audit_reject(conn: sqlite3.Connection, reason: str) -> None:
    conn.execute(
        "INSERT INTO audit (ts, event, record_id, actor) VALUES (?, ?, NULL, 'policy')",
        (_now(), f"admit.reject.{reason}"),
    )


def _reject(conn: sqlite3.Connection, reason: str) -> Admission:
    _audit_reject(conn, reason)
    return Admission(False, reason)


def _merge(*lexes: Lexicon) -> Lexicon:
    return (
        tuple(p for lx in lexes for p in lx[0]),
        frozenset().union(*(lx[1] for lx in lexes)),
        tuple(p for lx in lexes for p in lx[2]),
    )


# a look verb alone is not the user's word about the bank; the user's words name the category too
_CAPTURE_TOPIC: dict[str, Lexicon] = {
    "bank": _merge(
        _STRONG[Label.MONEY],
        _lex(
            "баланс|выписк|=карта|=карту|=карте|=карты|карточк|=счет|=счета|=счету|=счетом",
            "перевод|транзакц|сбер|тинькоф|т-банк|альфа|=втб|account|balance|=card|statement",
            "transaction|transfer",
        ),
    ),
    "medical": _merge(
        _STRONG[Label.HEALTH],
        _lex("анализ|рецепт|медкарт|справк|=lab|=labs|=results|test results"),
    ),
    "private": _lex("приватн|инкогнито|private|incognito"),
}


def _forbidden_capture(kind: str) -> str | None:
    """`screen:bank`, `screen: bank`, `capture:medical-portal` → the rule-7 category."""
    head, sep, rest = kind.partition(":")
    if not sep or head.strip() not in ("screen", "capture"):
        return None
    cat = rest.strip()
    return next((c for c in sorted(FORBIDDEN_CAPTURES) if cat.startswith(c)), None)


def admit_ex(
    conn: sqlite3.Connection,
    *,
    source_kind: str,
    chat_id: str | None,
    text: str,
    author_tier: AuthorTier,
    turn: TurnCtx | None,
    sender_id: str | None = None,
    reply_to: str | None = None,
    ts: str | None = None,
) -> Admission:
    """§5.1 in full; `admit` is its (bool, reason) view. Nothing here writes content."""
    tier = AuthorTier(author_tier)
    own = tier in OWN_TIERS
    kind = norm(source_kind)
    when = _ts(ts or (turn.clock if turn is not None else None))

    # rule 7: password managers and the Keychain never; bank, medical, private captures only
    # on the user's word
    if kind in PASSWORD_STORE_KINDS:
        return _reject(conn, "never_store")
    category = _forbidden_capture(kind)
    if category is not None:
        ws = words(turn.owner_words) if turn is not None else []
        asked = (
            turn is not None
            and turn.origin is TurnOrigin.OWNER
            and _hits(ws, _LOOK)
            and _hits(ws, _CAPTURE_TOPIC[category])
        )
        if not asked:
            return _reject(conn, "capture_forbidden")

    # chat modes and admit windows (§5.2, §5.6)
    window = None
    if chat_id:
        mode = chat_mode(conn, chat_id)
        if mode is ChatMode.BANNED:
            return _reject(conn, "chat_banned")
        if mode is ChatMode.NO_DIG:
            window = _match_window(conn, chat_id, sender_id, reply_to, when)
            if window is None:
                return _reject(conn, "no_dig_outside_window")

    # a Cube turn is admitted on the union of its sources
    body = text or ""
    keep_only_own = False
    if turn is not None:
        barred = _poisoned(conn, turn.turn_id)
        if barred:
            if not own:
                worst = "banned" if any(m is ChatMode.BANNED for _, m in barred) else "no_dig"
                return _reject(conn, f"turn_read_{worst}")
            keep_only_own = True
            body = (turn.owner_words or "").strip() or body
            mine = _shingles(words(body))
            reads = _READS.get(turn.turn_id, {})
            if mine:
                for chat, _mode in barred:
                    seen = reads.get(chat)
                    if seen is None or mine & seen:
                        # an unregistered read cannot be checked: fail closed for 8+ words
                        return _reject(conn, "own_words_share_read")

    # never-store (§5.1 rule 3); G-S3: a lone value right after «пароль от вайфая:» in this chat
    spans = never_store_spans(body)
    ctx_key = _ask_key(conn, chat_id, turn)
    if not spans:
        spans = _expected_secret(ctx_key, body, when)
    _note_secret_ask(ctx_key, body, spans, when)
    reason = None
    if spans:
        if not own:
            return _reject(conn, "never_store")
        body = redact(body, spans)
        reason = "secret_redacted"

    return Admission(
        True,
        reason,
        text=body,
        keep_only_own_words=keep_only_own,
        window_id=window["id"] if window is not None else None,
        intention_id=window["intention_id"] if window is not None else None,
    )


def admit(
    conn: sqlite3.Connection,
    *,
    source_kind: str,
    chat_id: str | None,
    text: str,
    author_tier: AuthorTier,
    turn: TurnCtx | None,
    sender_id: str | None = None,
    reply_to: str | None = None,
    ts: str | None = None,
) -> tuple[bool, str | None]:
    """§5.1: (admit, content-free reason). Runs before any write."""
    a = admit_ex(
        conn,
        source_kind=source_kind,
        chat_id=chat_id,
        text=text,
        author_tier=author_tier,
        turn=turn,
        sender_id=sender_id,
        reply_to=reply_to,
        ts=ts,
    )
    return a.admit, a.reason
