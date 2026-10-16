"""Entity resolution and aliases: stable ids -> aliases -> model only for the ambiguous;
code transliteration at creation; merge and split.

Spec: §10.3 (aliases in every script), §13.3, §16.1.
"""

from __future__ import annotations

import itertools
import json
import re
import sqlite3
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .model import (
    Alias,
    AliasKind,
    AuthorTier,
    EdgeType,
    Entity,
    EventKind,
    OpKind,
    ReadRef,
    Status,
)

_USER_TIERS = (AuthorTier.USER, AuthorTier.USER_CONFIRMED)
_NAME_KINDS = (AliasKind.NAME, AliasKind.DIMINUTIVE, AliasKind.NICKNAME)

# ---------------------------------------------------------------- normalization and scripts


def normalize_alias(text: str) -> str:
    """NFKC + casefold + collapsed whitespace (the `aliases.alias_norm` key)."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


_CODE_TOKEN = re.compile(r"[\w-]+")
_PAREN = re.compile(r"\([^)]*\)")


def _code_token(tok: str) -> bool:
    digits = sum(ch.isdigit() for ch in tok)
    letters = sum(ch.isalpha() for ch in tok)
    return digits >= 3 or (digits > 0 and letters > 0)


def code_like(name: str | None) -> bool:
    """Guarantee 2: a code, a contract or protocol number or a canary («ТГКЛСФ-5104»,
    «PBUUUQX-5104», «6034-5094») is never an entity name. True when the name's first word, or
    at least half of its words, is a code (three digits, or letters and digits in one word);
    «Релиз 1 июля» is a name."""
    toks = _CODE_TOKEN.findall(_PAREN.sub(" ", name or ""))
    if not toks:
        return False
    codes = [t for t in toks if _code_token(t)]
    return _code_token(toks[0]) or 2 * len(codes) >= len(toks)


def _is_cyr(ch: str) -> bool:
    return "Ѐ" <= ch <= "ӿ"


def script_of(text: str) -> str:
    cyr = any(_is_cyr(c) for c in text)
    lat = any("a" <= c <= "z" for c in text.casefold())
    if cyr and lat:
        return "mixed"
    return "cyrillic" if cyr else "latin" if lat else "other"


# Cyrillic -> Latin: each pattern has its options, the first one being the base form.
_RU_OPTS: list[tuple[str, tuple[str, ...]]] = [
    ("щ", ("shch",)),
    ("ий$", ("iy", "y", "ii", "i")),
    ("ия", ("iya", "ia")),
    ("ья", ("ya", "ia")),
    ("кс", ("ks", "x")),
    ("^е", ("e", "ye")),
    ("х", ("kh", "h")),
    ("й", ("y", "i")),
    ("ё", ("yo", "e")),
]
_RU1 = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        [
            "a",
            "b",
            "v",
            "g",
            "d",
            "e",
            "yo",
            "zh",
            "z",
            "i",
            "y",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "r",
            "s",
            "t",
            "u",
            "f",
            "kh",
            "ts",
            "ch",
            "sh",
            "shch",
            "",
            "y",
            "",
            "e",
            "yu",
            "ya",
        ],
        strict=True,
    )
)
# Latin -> Cyrillic, longest first; word-final forms marked with $.
_LAT: list[tuple[str, tuple[str, ...]]] = [
    ("shch", ("щ",)),
    ("sch", ("щ",)),
    ("iy$", ("ий",)),
    ("ia$", ("ия", "иа")),
    ("ey$", ("ей",)),
    ("ay$", ("ай",)),
    ("oy$", ("ой",)),
    ("sh", ("ш",)),
    ("zh", ("ж",)),
    ("kh", ("х",)),
    ("ts", ("ц",)),
    ("tz", ("ц",)),
    ("ch", ("ч",)),
    ("yu", ("ю",)),
    ("ju", ("ю",)),
    ("ya", ("я",)),
    ("ja", ("я",)),
    ("yo", ("ё",)),
    ("jo", ("ё",)),
    ("ye", ("е",)),
    ("y$", ("ий", "и")),
    ("x", ("кс",)),
]
_LAT1 = dict(
    zip(
        "abcdefghijklmnopqrstuvwyz",
        [
            "а",
            "б",
            "к",
            "д",
            "е",
            "ф",
            "г",
            "х",
            "и",
            "й",
            "к",
            "л",
            "м",
            "н",
            "о",
            "п",
            "к",
            "р",
            "с",
            "т",
            "у",
            "в",
            "в",
            "ы",
            "з",
        ],
        strict=True,
    )
)
_MAX_VARIANTS = 12


def _segments(
    word: str, table: list[tuple[str, tuple[str, ...]]], single: dict[str, str]
) -> list[tuple[str, ...]]:
    segs: list[tuple[str, ...]] = []
    i = 0
    while i < len(word):
        for pat, opts in table:
            start, end = pat.startswith("^"), pat.endswith("$")
            core = pat.strip("^$")
            if (start and i != 0) or not word.startswith(core, i):
                continue
            if end and i + len(core) != len(word):
                continue
            segs.append(opts)
            i += len(core)
            break
        else:
            ch = word[i]
            segs.append((single.get(ch, ch),))
            i += 1
    return segs


def _code_translit(text: str) -> list[str]:
    norm = normalize_alias(text)
    sc = script_of(norm)
    if sc not in ("cyrillic", "latin"):
        return []
    table, single = (_RU_OPTS, _RU1) if sc == "cyrillic" else (_LAT, _LAT1)
    words = [_segments(w, table, single) for w in norm.split()]
    per_word = [
        list(
            dict.fromkeys(
                "".join(p) for p in itertools.islice(itertools.product(*w), _MAX_VARIANTS)
            )
        )
        for w in words
    ]
    out = [" ".join(c) for c in itertools.islice(itertools.product(*per_word), _MAX_VARIANTS)]
    if "ё" in norm:
        out.append(norm.replace("ё", "е"))
    return out


# Conventional spellings and diminutive families: (Russian full name, Latin spellings of the
# full name, Russian diminutives). Diminutives' Latin forms come from code transliteration.
_NAMES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "Александр",
        ("Alexander", "Aleksandr", "Alexandr", "Alex"),
        ("Саша", "Шура", "Саня", "Санёк"),
    ),
    ("Александра", ("Alexandra", "Aleksandra"), ("Саша", "Шура")),
    ("Алексей", ("Alexei", "Alexey", "Aleksei", "Aleksey"), ("Лёша", "Алёша", "Лёха")),
    ("Дмитрий", ("Dmitry", "Dmitri", "Dmitriy"), ("Дима", "Митя")),
    ("Михаил", ("Mikhail", "Michael"), ("Миша",)),
    ("Евгений", ("Evgeny", "Yevgeny", "Eugene"), ("Женя",)),
    ("Евгения", ("Evgenia", "Eugenia"), ("Женя",)),
    ("Сергей", ("Sergey", "Sergei"), ("Серёжа", "Серёга")),
    ("Андрей", ("Andrey", "Andrei", "Andrew"), ("Андрюша",)),
    ("Николай", ("Nikolai", "Nikolay", "Nicholas"), ("Коля",)),
    ("Пётр", ("Pyotr", "Petr", "Peter"), ("Петя",)),
    ("Павел", ("Pavel", "Paul"), ("Паша",)),
    ("Владимир", ("Vladimir",), ("Вова", "Володя")),
    ("Константин", ("Konstantin", "Constantine"), ("Костя",)),
    ("Илья", ("Ilya", "Ilia"), ("Илюша",)),
    ("Иван", ("Ivan",), ("Ваня",)),
    ("Юрий", ("Yuri", "Yury", "Yuriy"), ("Юра",)),
    ("Георгий", ("Georgy", "George"), ("Гоша", "Жора")),
    ("Григорий", ("Grigory", "Gregory"), ("Гриша",)),
    ("Максим", ("Maxim", "Maksim"), ("Макс",)),
    ("Артём", ("Artem", "Artyom"), ("Тёма",)),
    ("Роман", ("Roman",), ("Рома",)),
    ("Виктор", ("Viktor", "Victor"), ("Витя",)),
    ("Вячеслав", ("Vyacheslav",), ("Слава",)),
    ("Владислав", ("Vladislav",), ("Влад", "Слава")),
    ("Станислав", ("Stanislav",), ("Стас",)),
    ("Анатолий", ("Anatoly", "Anatoliy"), ("Толя",)),
    ("Борис", ("Boris",), ("Боря",)),
    ("Екатерина", ("Ekaterina", "Yekaterina", "Catherine", "Katherine"), ("Катя",)),
    ("Елена", ("Elena", "Yelena", "Helen"), ("Лена",)),
    ("Мария", ("Maria", "Mariya", "Mary"), ("Маша",)),
    ("Наталья", ("Natalia", "Natalya", "Natalie"), ("Наташа",)),
    ("Анастасия", ("Anastasia", "Anastasiya"), ("Настя",)),
    ("Татьяна", ("Tatiana", "Tatyana"), ("Таня",)),
    ("Ольга", ("Olga",), ("Оля",)),
    ("Юлия", ("Yulia", "Yuliya", "Julia"), ("Юля",)),
    ("Анна", ("Anna", "Ann"), ("Аня",)),
    ("Дарья", ("Darya", "Daria"), ("Даша",)),
    ("Ксения", ("Ksenia", "Kseniya", "Xenia"), ("Ксюша",)),
    ("Светлана", ("Svetlana",), ("Света",)),
    ("Ирина", ("Irina",), ("Ира",)),
    ("Виктория", ("Viktoria", "Victoria"), ("Вика",)),
    ("Елизавета", ("Elizaveta", "Elizabeth"), ("Лиза",)),
    ("Софья", ("Sofya", "Sofia", "Sophia"), ("Соня",)),
    ("Надежда", ("Nadezhda",), ("Надя",)),
    ("Любовь", ("Lyubov",), ("Люба",)),
    ("Галина", ("Galina",), ("Галя",)),
)


def transliterations(name: str) -> list[str]:
    """Саша <-> Sasha, by code (§10.3), both ways, plus the conventional spelling of a full
    name (Александр <-> Alexander). Diminutive families are `name_family`'s."""
    norm = normalize_alias(name)
    out = list(_code_translit(name))
    for full, latin, _dims in _NAMES:
        nf, nl = normalize_alias(full), [normalize_alias(x) for x in latin]
        if norm == nf:
            out += latin
        elif norm in nl:
            out.append(full)
    seen, res = {norm}, []
    for t in out:
        k = normalize_alias(t)
        if k and k not in seen:
            seen.add(k)
            res.append(t)
    return res


def name_family(name: str) -> list[str]:
    """Full name and diminutives in both scripts («Саша» -> «Александр», «Александра», Alexander,
    «Шура», ...): weaker than a transliteration; `resolve` ranks them below exact names."""
    norm = normalize_alias(name)
    keys = {norm, *(normalize_alias(t) for t in _code_translit(name))}
    out: list[str] = []
    for full, latin, dims in _NAMES:
        members = [full, *latin, *dims]
        member_keys = {normalize_alias(m) for m in members} | {
            normalize_alias(t) for d in dims for t in _code_translit(d)
        }
        if keys & member_keys:
            out += members + [t for d in dims for t in _code_translit(d)[:1]]
    seen, res = {norm}, []
    for t in out:
        k = normalize_alias(t)
        if k not in seen:
            seen.add(k)
            res.append(t)
    return res


_RU_ENDINGS = (
    "ами",
    "ями",
    "ой",
    "ей",
    "ёй",
    "ою",
    "ею",
    "ом",
    "ем",
    "ём",
    "ам",
    "ям",
    "ах",
    "ях",
    "у",
    "ю",
    "е",
    "ы",
    "и",
    "а",
    "я",
)


def _ru_lemmas(norm: str) -> set[str]:
    """Nominative guesses for an inflected Russian name («Саше» -> «Саша», «Илье» -> «Илья»)."""
    if " " in norm or script_of(norm) != "cyrillic":
        return set()
    out: set[str] = set()
    for end in _RU_ENDINGS:
        if norm.endswith(end) and len(norm) - len(end) >= 2:
            stem = norm[: -len(end)]
            out.update({stem, stem + "а", stem + "я", stem + "й", stem + "ь"})
    out.discard(norm)
    return out


# Russian diminutive and affectionate suffixes, longest first («Сашенька», «Илюша», «Илюха»,
# «Шурик», «Сашка»). With a case ending stripped first, a declined diminutive and its name meet
# at one root: «Илюхе» → «илюх» → «ил» ~ «Илья» → «иль» → «ил»; «Сашке» → «сашк» → «саш» ~
# «Саша» → «саш».
_DIM_SUFFIXES = (
    "еньк",
    "оньк",
    "ушк",
    "юшк",
    "ечк",
    "очк",
    "ичк",
    "юш",
    "уш",
    "юх",
    "ух",
    "ик",
    "ек",
    "ок",
    "к",
    "ня",
)


def name_roots(token: str) -> set[str]:
    """Roots of one Cyrillic name token by general morphology (case ending, then a
    diminutive suffix, then a final soft sign): no table, so any name meets its forms."""
    norm = normalize_alias(token).replace("ё", "е")
    if not norm or " " in norm or script_of(norm) != "cyrillic":
        return set()
    cands = {norm} | {
        norm[: -len(end)] for end in _RU_ENDINGS if norm.endswith(end) and len(norm) - len(end) >= 2
    }
    out: set[str] = set()
    for c in cands:
        out.add(c.rstrip("ь"))
        for suf in _DIM_SUFFIXES:
            if c.endswith(suf) and len(c) - len(suf) >= 2:
                out.add(c[: -len(suf)].rstrip("ь"))
                break
    return {r for r in out if len(r) >= 2}


def family_of(token: str) -> list[str]:
    """The name families (full name, Latin spellings, diminutives) one of whose members
    shares a root of three letters or more with the token (Шурик ~ Шура → Александр)."""
    roots = {r for r in name_roots(token) if len(r) >= 3}
    if not roots:
        return []
    out: list[str] = []
    for full, latin, dims in _NAMES:
        if any(roots & {r for r in name_roots(m) if len(r) >= 3} for m in (full, *dims)):
            out += [full, *latin, *dims]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- alias writes (§13.3)


def put_alias(
    conn: sqlite3.Connection,
    ts: str,
    entity_id: str,
    text: str,
    kind: AliasKind,
    *,
    tier: AuthorTier,
    confirmed: bool,
    with_translit: bool = False,
    alias_id: str | None = None,
) -> bool:
    """Insert an alias (and its FTS name row) or upgrade an unconfirmed one to confirmed.
    Returns whether anything changed. Never touches another entity's alias."""
    from . import ledger

    changed = False
    forms = [(text, kind, alias_id)]
    if with_translit and kind in _NAME_KINDS:
        forms += [(t, AliasKind.TRANSLIT, None) for t in transliterations(text)]
    for alias, akind, aid in forms:
        norm = normalize_alias(alias)
        if not norm:
            continue
        row = conn.execute(
            "SELECT id, confirmed FROM aliases WHERE entity_id = ? AND alias_norm = ?",
            (entity_id, norm),
        ).fetchone()
        if row is not None:
            if confirmed and not row["confirmed"]:
                conn.execute(
                    "UPDATE aliases SET confirmed = 1, tier = ?, updated_at = ? WHERE id = ?",
                    (AuthorTier(tier).value, ts, row["id"]),
                )
                changed = True
            continue
        new = aid or ledger.new_id()
        conn.execute(
            "INSERT INTO aliases (id, entity_id, alias, alias_norm, script, kind, tier, confirmed,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new,
                entity_id,
                alias,
                norm,
                script_of(alias),
                AliasKind(akind).value,
                AuthorTier(tier).value,
                int(confirmed),
                ts,
                ts,
            ),
        )
        fts = conn.execute("SELECT fts_rowid FROM aliases WHERE id = ?", (new,)).fetchone()[0]
        conn.execute("INSERT INTO fts_names (rowid, name) VALUES (?, ?)", (fts, alias))
        changed = True
    return changed


def bind_in(
    ctx: Any,
    entity_id: str,
    text: str,
    kind: AliasKind,
    *,
    tier: AuthorTier,
    platform: bool,
    touch: bool = True,
) -> bool:
    """Bind inside a commit: confirmed only from their word or platform metadata (§13.3);
    anything else is stored as an unconfirmed candidate. Returns whether anything changed."""
    from . import ledger

    confirmed = AuthorTier(tier) in _USER_TIERS or platform
    changed = put_alias(
        ctx.conn,
        ctx.ts,
        entity_id,
        text,
        kind,
        tier=tier,
        confirmed=confirmed,
        with_translit=confirmed,
    )
    if changed and touch:
        ledger.touch_entity(ctx, entity_id, tokens=[text])
    return changed


def bind_alias(conn: sqlite3.Connection, alias: Alias, *, tier: AuthorTier, platform: bool) -> bool:
    """§13.3: binds only from user / user_confirmed or platform metadata; else a candidate.
    Returns True when the alias is bound (confirmed)."""
    from . import ledger

    bound = AuthorTier(tier) in _USER_TIERS or platform

    def fn(ctx: Any) -> tuple[bool, list[str]]:
        changed = bind_in(
            ctx, alias.entity_id, alias.alias, alias.kind, tier=tier, platform=platform
        )
        return bound, [alias.entity_id] if changed else []

    return ledger.run_in_commit(
        conn, fn, op_kind=OpKind.LINK, target=alias.entity_id, author_tier=tier
    )


# ---------------------------------------------------------------- a message's author (§13.3)

_TG_ID = re.compile(r"^tg:(\d+)$")
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")
_HANDLE = re.compile(r"^@?[a-z][a-z0-9]*(?:[._][a-z0-9]+)*$")


def sender_stable_ids(sender_id: str | None) -> list[tuple[str, str]]:
    """The stable ids a message's platform metadata carries for its sender: a Telegram user
    id (`tg:<id>`) or an e-mail address. A group or channel peer, a handle or a name is none."""
    s = (sender_id or "").strip()
    if m := _TG_ID.match(s):
        return [("tg_user_id", m.group(1))]
    if _EMAIL.match(s):
        return [("email", s.casefold())]
    return []


def sender_name(sender: str | None) -> str | None:
    """The display name in a message's platform metadata. A bare username reads as its words
    («@anna_k» -> «Anna K», «oleg.petrov» -> «Oleg Petrov»); an id, a code or an empty name is
    no name (the person stays unnamed rather than get a stand-in)."""
    s = unicodedata.normalize("NFKC", sender or "").strip()
    if not s or ":" in s or code_like(s) or not any(ch.isalpha() for ch in s):
        return None
    if _HANDLE.match(s):
        return " ".join(p.capitalize() for p in re.split(r"[._]+", s.lstrip("@")) if p) or None
    return s.lstrip("@").strip() or None


def author_entity(
    conn: sqlite3.Connection,
    *,
    sender_id: str | None,
    sender: str | None,
    source: str | None = None,
) -> str | None:
    """§13.3: the person a message comes from, by the stable id in its platform metadata (the
    sender id of that person's own message), never by the name alone. A known contact
    resolves to its entity; an unknown sender with a name is minted once, as a person holding
    that id (a code write, `platform`). No stable id, no name, or a refused mint: None."""
    from . import ledger, store
    from .model import Proposal, ProposalOp

    sids = sender_stable_ids(sender_id)
    for kind, value in sids:
        row = conn.execute(
            "SELECT s.entity_id FROM entity_stable_ids s JOIN records r ON r.id = s.entity_id"
            " WHERE s.kind = ? AND s.value = ? AND r.status IN ('active', 'archived')"
            " ORDER BY r.recorded_at, r.id LIMIT 1",
            (kind, value),
        ).fetchone()
        if row is not None:
            return str(row[0])
    name = sender_name(sender)
    if not sids or name is None or conn.in_transaction:
        return None
    eid = ledger.new_id()
    raw = unicodedata.normalize("NFKC", sender or "").strip()
    aliases = (
        [{"alias": raw, "kind": AliasKind.HANDLE.value}]
        if raw and normalize_alias(raw) != normalize_alias(name)
        else []
    )
    parts = [p for p in re.split(r"[\s.]+", name) if len(p) >= 2 and p.isalpha()]
    if len(parts) > 1:  # «Ира Лосева» is also «Ира» and «Лосева» in the user's words
        aliases += [{"alias": p, "kind": AliasKind.NAME.value} for p in parts]
    rec = {
        "kind": "entity",
        "id": eid,
        "etype": "person",
        "display_name": name,
        "author_tier": AuthorTier.OTHER.value,
        "labels_final": True,  # a platform name is not the message's content
        "stable_ids": [
            {"kind": k, "value": v, "tier": "platform", "source": source} for k, v in sids
        ],
    }
    op = ProposalOp(
        op=OpKind.ADD,
        group=eid,
        payload={"record": rec, "aliases": aliases, "platform": True},
    )
    try:
        res = store.commit(conn, Proposal(ops=[op], author_tier=AuthorTier.OTHER), ledger.apply_op)
    except store.OpRejected:  # §9.8: a forgotten person's id, from a message with no date
        return None
    return eid if res.ok else None


# ---------------------------------------------------------------- resolution (§16.1)


@dataclass(frozen=True)
class Resolved:
    entity_id: str
    display_name: str
    etype: str
    confidence: float
    token: ReadRef
    unconfirmed: bool
    via: str
    stable_ids: list[dict[str, Any]] = field(default_factory=list)
    # the alias rows (alias_norm) the hit rests on; G-C1 tells a founding name from a later one
    matched: frozenset[str] = frozenset()


_STABLE = [
    (re.compile(r"^(?:m:)(.+)$"), "id"),
    (re.compile(r"^tg:(\d+)$"), "tg_user_id"),
    (re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$"), "email"),
    (re.compile(r"^https?://\S+$"), "url"),
    (re.compile(r"^(?:/|~/)\S*$"), "path"),
]
# confidence per (via, confirmed)
_SCORE = {
    "alias": (0.95, 0.5),
    "translit": (0.9, 0.45),
    "inflection": (0.85, 0.45),
    "family": (0.6, 0.35),
}


def _entity_rows(conn: sqlite3.Connection, ids: list[str], kind: str) -> dict[str, sqlite3.Row]:
    if not ids:
        return {}
    marks = ", ".join("?" for _ in ids)
    params: list[Any] = list(ids)
    etype = ""
    if kind and kind != "any":
        etype, params = " AND e.etype = ?", [*ids, kind]
    rows = conn.execute(
        "SELECT e.id, e.etype, e.display_name, e.stable_ids, r.version FROM entities e"
        f" JOIN records r ON r.id = e.id WHERE e.id IN ({marks})"  # noqa: S608
        f" AND r.status IN ('active', 'archived'){etype}",
        params,
    ).fetchall()
    return {r["id"]: r for r in rows}


def _resolved(
    row: sqlite3.Row, conf: float, unconfirmed: bool, via: str, matched: Iterable[str] = ()
) -> Resolved:
    return Resolved(
        entity_id=row["id"],
        display_name=row["display_name"],
        etype=row["etype"],
        confidence=conf,
        token=ReadRef(id=row["id"], version=row["version"]),
        unconfirmed=unconfirmed,
        via=via,
        stable_ids=json.loads(row["stable_ids"]),
        matched=frozenset(matched),
    )


def resolve_rows(conn: sqlite3.Connection, text: str, kind: str) -> list[Resolved]:
    """Stable ids first, then aliases (exact, transliterated, inflected, diminutive family).
    Only usable entities. `kind` filters `etype`; "" or "any" does not. Unconfirmed bindings
    are flagged: hands ask before using them (§13.3). The model step for the ambiguous is the
    caller's (§16.1)."""
    q = text.strip()
    if not q:
        return []
    for pattern, skind in _STABLE:
        m = pattern.match(q)
        if not m:
            continue
        if skind == "id":
            ids = [m.group(1)]
        else:
            value = (
                m.group(1) if skind == "tg_user_id" else (q.casefold() if skind == "email" else q)
            )
            ids = [  # the stable-id index (migration 0002, G9): no JSON scan per entity
                r[0]
                for r in conn.execute(
                    "SELECT entity_id FROM entity_stable_ids WHERE kind = ? AND value = ?"
                    " ORDER BY entity_id",
                    (skind, value),
                )
            ]
        rows = _entity_rows(conn, ids, kind)
        found = [_resolved(r, 1.0, False, "stable_id") for r in rows.values()]
        if found or skind in ("id", "tg_user_id"):
            return sorted(found, key=lambda r: (r.display_name, r.entity_id))
    norm = normalize_alias(q)
    tiers: dict[str, str] = {norm: "alias"}
    for t in transliterations(q):
        tiers.setdefault(normalize_alias(t), "translit")
    lemmas = _ru_lemmas(norm)
    for lem in lemmas:
        tiers.setdefault(lem, "inflection")
        for t in transliterations(lem):
            tiers.setdefault(normalize_alias(t), "inflection")
    for base in [q, *lemmas]:
        for t in name_family(base):
            tiers.setdefault(normalize_alias(t), "family")
    keys = list(tiers)
    matches: list[sqlite3.Row] = []
    for i in range(0, len(keys), 400):
        chunk = keys[i : i + 400]
        marks = ", ".join("?" for _ in chunk)
        matches += conn.execute(
            f"SELECT entity_id, alias_norm, confirmed FROM aliases WHERE alias_norm IN ({marks})",  # noqa: S608
            chunk,
        ).fetchall()
    rows = _entity_rows(conn, list(dict.fromkeys(m["entity_id"] for m in matches)), kind)
    best: dict[str, dict[bool, tuple[float, str]]] = {}
    hit: dict[tuple[str, bool], set[str]] = {}
    for m in matches:
        if m["entity_id"] not in rows:
            continue
        hit.setdefault((m["entity_id"], bool(m["confirmed"])), set()).add(m["alias_norm"])
        via = tiers[m["alias_norm"]]
        conf = _SCORE[via][0 if m["confirmed"] else 1]
        slot = best.setdefault(m["entity_id"], {})
        if conf > slot.get(bool(m["confirmed"]), (0.0, ""))[0]:
            slot[bool(m["confirmed"])] = (conf, via)
    out = []
    for eid, slot in best.items():
        sure = True in slot
        conf, via = slot[sure]
        out.append(_resolved(rows[eid], conf, not sure, via, hit.get((eid, sure), ())))
    return sorted(out, key=lambda r: (-r.confidence, r.display_name, r.entity_id))


def candidates(conn: sqlite3.Connection, text: str, kind: str) -> list[tuple[str, float, ReadRef]]:
    """(entity_id, confidence, token) for §16.1 resolve."""
    return [(r.entity_id, r.confidence, r.token) for r in resolve_rows(conn, text, kind)]


# ---------------------------------------------------------------- merge and split

# (table, column) pairs that hold an entity id; each is a content column of its record.
_REFS = (
    ("facts", "subject_id"),
    ("facts", "value_entity_id"),
    ("facts", "speaker_entity_id"),
    ("impressions", "about_entity_id"),
    ("decisions", "matter_id"),
    ("work_states", "matter_id"),
    ("loops", "owner"),
)


def _repoint(ctx: Any, old: str, new: str, record_ids: list[str] | None) -> list[str]:
    """Move references from `old` to `new` (all records, or only `record_ids`)."""
    conn = ctx.conn
    scope, params = "", []
    if record_ids is not None:
        scope = f" AND id IN ({', '.join('?' for _ in record_ids)})"
        params = list(record_ids)
    moved: list[str] = []
    for table, col in _REFS:
        ids = [
            r[0]
            for r in conn.execute(f"SELECT id FROM {table} WHERE {col} = ?{scope}", [old, *params])
        ]  # noqa: S608
        if ids:
            conn.execute(
                f"UPDATE {table} SET {col} = ? WHERE {col} = ?{scope}", [new, old, *params]
            )  # noqa: S608
            moved += ids
    conn.execute(
        f"UPDATE records SET author_entity_id = ? WHERE author_entity_id = ?{scope}",
        [new, old, *params],
    )  # noqa: S608
    ep_scope = scope.replace("id IN", "episode_id IN")
    eps = [
        r[0]
        for r in conn.execute(
            f"SELECT episode_id FROM episode_links WHERE target_id = ?{ep_scope}", [old, *params]
        )
    ]  # noqa: S608
    if eps:
        conn.execute(
            f"UPDATE OR IGNORE episode_links SET target_id = ? WHERE target_id = ?{ep_scope}",
            [new, old, *params],
        )  # noqa: S608
        conn.execute(f"DELETE FROM episode_links WHERE target_id = ?{ep_scope}", [old, *params])  # noqa: S608
        moved += eps
    e_scope = scope.replace("id IN", "from_id IN")
    conn.execute(
        f"UPDATE OR IGNORE edges SET to_id = ? WHERE to_id = ? AND etype IN ('about', 'mentions'){e_scope}",  # noqa: S608
        [new, old, *params],
    )
    conn.execute(
        f"DELETE FROM edges WHERE to_id = ? AND etype IN ('about', 'mentions'){e_scope}",
        [old, *params],  # noqa: S608
    )
    return list(dict.fromkeys(moved))


def _entity(conn: sqlite3.Connection, eid: str) -> sqlite3.Row:
    from .ledger import CommitRejected

    row = conn.execute(
        "SELECT e.*, r.status, r.author_tier, r.labels FROM entities e JOIN records r ON r.id = e.id"
        " WHERE e.id = ?",
        (eid,),
    ).fetchone()
    if row is None:
        raise CommitRejected("not_found", eid)
    return row


def merge_in(ctx: Any, keep_id: str, drop_id: str) -> list[str]:
    """Inside a commit: aliases and stable ids move to `keep`, references are repointed (a
    content change of each record), `drop` is superseded by `keep`."""
    from . import ledger

    conn = ctx.conn
    keep, drop = _entity(conn, keep_id), _entity(conn, drop_id)
    if drop["status"] not in ("active", "archived"):
        raise ledger.CommitRejected("bad_op", f"{drop_id} is not usable")
    tokens = [drop["display_name"]]
    for a in conn.execute("SELECT * FROM aliases WHERE entity_id = ?", (drop_id,)).fetchall():
        have = conn.execute(
            "SELECT id, confirmed FROM aliases WHERE entity_id = ? AND alias_norm = ?",
            (keep_id, a["alias_norm"]),
        ).fetchone()
        if have is None:
            conn.execute(
                "UPDATE aliases SET entity_id = ?, updated_at = ? WHERE id = ?",
                (keep_id, ctx.ts, a["id"]),
            )
            tokens.append(a["alias"])
            continue
        if a["confirmed"] and not have["confirmed"]:
            conn.execute(
                "UPDATE aliases SET confirmed = 1, tier = ? WHERE id = ?", (a["tier"], have["id"])
            )
        conn.execute("DELETE FROM fts_names WHERE rowid = ?", (a["fts_rowid"],))
        conn.execute("DELETE FROM aliases WHERE id = ?", (a["id"],))
    sids = json.loads(keep["stable_ids"])
    seen = {(s["kind"], s["value"]) for s in sids}
    sids += [s for s in json.loads(drop["stable_ids"]) if (s["kind"], s["value"]) not in seen]
    conn.execute(
        "UPDATE entities SET stable_ids = ? WHERE id = ?",
        (json.dumps(sids, ensure_ascii=False), keep_id),
    )
    conn.execute("UPDATE records SET labels = labels | ? WHERE id = ?", (drop["labels"], keep_id))
    moved = [m for m in _repoint(ctx, drop_id, keep_id, None) if m not in (keep_id, drop_id)]
    ledger.bump(ctx, moved)
    ledger.set_status(ctx, drop_id, Status.SUPERSEDED)
    ledger.add_edge(ctx, keep_id, drop_id, EdgeType.SUPERSEDES)
    ledger.touch_entity(ctx, keep_id, tokens=tokens)
    ledger.invalidate(ctx, moved)
    ledger.mark_pending_updates(ctx, moved)
    return [keep_id, drop_id, *moved]


def split_in(
    ctx: Any,
    entity_id: str,
    record_ids: list[str],
    *,
    display_name: str | None = None,
    new_id: str | None = None,
) -> tuple[str, list[str]]:
    """Inside a commit: a new entity takes the given records (their references to
    `entity_id` move); it gets the name side (name and transliterations), never the stable
    ids, which stay until the user says whose they are."""
    from . import ledger

    conn = ctx.conn
    src = _entity(conn, entity_id)
    nid = new_id or ledger.new_id()
    new = Entity(
        id=nid,
        etype=src["etype"],
        display_name=display_name or src["display_name"],
        author_tier=AuthorTier(src["author_tier"]),
        recorded_at=ctx.ts,
    )
    ledger.insert_record(ctx, new)
    ledger.add_edge(ctx, nid, entity_id, EdgeType.DERIVED_FROM)
    moved = _repoint(ctx, entity_id, nid, record_ids) if record_ids else []
    missing = set(record_ids) - set(moved)
    if missing:
        raise ledger.CommitRejected(
            "bad_op", f"{len(missing)} records do not reference {entity_id}"
        )
    ledger.bump(ctx, moved)
    ledger.invalidate(ctx, moved)
    ledger.mark_pending_updates(ctx, moved)
    return nid, [nid, *moved]


def merge(conn: sqlite3.Connection, keep_id: str, drop_id: str) -> None:
    """Trusted code path (the user's fix): merge as its own commit, or inside the open transaction."""
    from . import ledger

    ledger.run_in_commit(
        conn,
        lambda ctx: (None, merge_in(ctx, keep_id, drop_id)),
        op_kind=OpKind.MERGE,
        target=keep_id,
        event_kind=EventKind.FIX,
        author_tier=AuthorTier.USER_CONFIRMED,
    )


def split(conn: sqlite3.Connection, entity_id: str, record_ids: list[str]) -> str:
    """Trusted code path (the user's fix): the new entity's id."""
    from . import ledger

    return ledger.run_in_commit(
        conn,
        lambda ctx: split_in(ctx, entity_id, record_ids),
        op_kind=OpKind.SPLIT,
        target=entity_id,
        event_kind=EventKind.FIX,
        author_tier=AuthorTier.USER_CONFIRMED,
    )
