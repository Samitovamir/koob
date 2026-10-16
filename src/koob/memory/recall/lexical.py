"""Stemmed RU/EN BM25 over `fts_records`, trigram names over `fts_names`, contentless raw FTS.
PyStemmer per token by script (Cyrillic -> russian, Latin -> english), at index and query.

Spec: §3.2 (FTS), §10.3 (four layers, never stemming alone), §9.1 (secure delete), G9.

Layers here: Snowball stems (exact), a stem-prefix pass for alternations the stemmer splits
(переехал / переезд), bilingual keywords and the original quote in the `keywords` column,
aliases in every script on entity rows, and trigram names for irregular and misspelt forms.
The embedder (vec) bridges the rest.

Semantics shared by the finders:
- forgotten and purged items are never returned (their FTS rows are gone too, §9.1);
- the label fence mirrors `policy.mask_clause`: `(labels & ~mask) = 0`, or one of the user's
  own items (`policy.decide` checks those later); `label_mask=-1` lets everything through;
- the list order is the rank (usable items first, then relevance); the score is relevance,
  higher is better.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Callable, Iterable
from functools import lru_cache

import Stemmer

from ..model import RecordKind, Status

_TOKEN = re.compile(r"[^\W_]+")
_CYR = re.compile(r"[а-яё]")
_LAT = re.compile(r"[a-z]")

_GONE_SQL = "('forgotten', 'purged')"
_USABLE = frozenset({Status.ACTIVE.value, Status.ARCHIVED.value})
_OWN_SQL = "('user', 'user_confirmed')"  # = policy._OWN_SQL (the user's own items pass the fence)

# bm25 column weights: text, keywords (bilingual keywords + original quote), aliases
_WEIGHTS = (1.0, 1.5, 2.0)
_EXACT_BONUS = 0.5  # a row holding every query stem exactly scores 1.5x its bm25
_MIN_PREFIX_STEM = 5  # stems shorter than this get no prefix pass (too noisy)
_NAME_FLOOR = 0.5  # trigram Dice similarity a name needs to count
_FAMILY_SCORE = 0.8  # a diminutive or declined form of the alias («Илюха» ~ «Илья», «Шурик» ~ «Саша»)
_BATCH = 250  # records per write transaction in reindex_all (write-lock budget)

# Query-only stop words: grammar that carries no topic. The index keeps every token.
# fmt: off
_STOP = frozenset((
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "from", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "that", "this", "these", "those",
    "he", "she", "they", "them", "his", "her", "their", "we", "you", "i", "me", "my", "your",
    "do", "does", "did", "have", "has", "had", "will", "would", "what", "when", "where",
    "which", "who", "whom", "how", "about", "into", "also", "just", "very", "not", "no",
    "и", "а", "но", "или", "в", "во", "на", "с", "со", "к", "ко", "о", "об", "от", "до", "по",
    "за", "из", "у", "же", "ли", "бы", "не", "ни", "да", "то", "это", "этот", "эта", "эти",
    "как", "что", "чтобы", "так", "там", "тут", "где", "когда", "кто", "который", "которая",
    "которые", "он", "она", "оно", "они", "его", "ее", "её", "их", "им", "ей", "ему", "мне",
    "меня", "мы", "нас", "вы", "вас", "ты", "тебя", "тебе", "я", "себя", "свой", "своя",
    "быть", "было", "была", "были", "будет", "есть", "нет", "для", "про", "при", "без",
    "над", "под", "после", "перед", "через", "между", "только", "уже", "ещё", "еще", "очень",
    "мой", "моя", "мои", "твой", "твоя", "наш", "ваш", "там", "вот", "ну", "помнишь",
))
# fmt: on


@lru_cache(maxsize=2)
def _stemmer(lang: str) -> Stemmer.Stemmer:
    return Stemmer.Stemmer(lang)


@lru_cache(maxsize=65536)
def _stem_token(token: str) -> str:
    """One case-folded token -> its Snowball stem by script (Cyrillic wins in mixed tokens).
    Snowball folds «ё» to «е» itself, so both spellings meet."""
    if _CYR.search(token):
        return _stemmer("russian").stemWord(token)
    if _LAT.search(token):
        return _stemmer("english").stemWord(token)
    return token


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def stem(text: str) -> str:
    """The FTS form of a text: NFKC, casefold, letters-and-digits tokens, each stemmed by its
    script; joined by spaces. The same function serves index, query and forget (§9.1)."""
    if not text:
        return ""
    return " ".join(_stem_token(t) for t in _tokens(text))


def query_stems(query: str) -> list[str]:
    """Distinct topic stems of a question (stop words and one-letter tokens dropped)."""
    out: dict[str, None] = {}
    for t in _tokens(query):
        if t in _STOP or (len(t) < 2 and not t.isdigit()):
            continue
        out[_stem_token(t)] = None
    return list(out)


_query_stems = query_stems  # the old private name, kept for callers in flight

_WORD_CASED = re.compile(r"[^\W\d_]+")
_SENTENCE_END = re.compile(r"[.!?…]\s*$")


def _name_tokens(text: str) -> list[str]:
    """Capitalized words past a sentence's first word («у Бублика», «Bublik's»): the proper
    names of a question, whatever their language."""
    out: list[str] = []
    for m in _WORD_CASED.finditer(unicodedata.normalize("NFKC", text or "")):
        tok = m.group(0)
        before = text[: m.start()]
        if len(tok) < 3 or not tok[0].isupper() or tok.casefold() in _STOP:
            continue
        if not before.strip() or _SENTENCE_END.search(before):
            continue
        out.append(tok)
    return out


@lru_cache(maxsize=512)
def _translit_stems(token: str) -> tuple[str, ...]:
    from .. import entities

    return tuple(
        dict.fromkeys(_stem_token(_fold(t.casefold())) for t in entities.transliterations(token))
    )


def name_stems(text: str) -> set[str]:
    """Stems of the question's proper names and of their transliterations
    (Bublik → бублик, Сокол → sokol), so a name meets the record in the other script."""
    out: set[str] = set()
    for tok in _name_tokens(text):
        out.add(_stem_token(tok.casefold()))
        out.update(_translit_stems(tok))
    return {s for s in out if len(s) >= 3}


def _match_stems(query: str) -> list[str]:
    """The question's topic stems plus its proper names in the other script."""
    stems = query_stems(query)
    return list(dict.fromkeys([*stems, *sorted(name_stems(query) - set(stems))]))


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _match_expr(stems: Iterable[str]) -> str:
    """One FTS5 expression: a long stem as its prefix (so alternations the stemmer splits
    meet: перееха- / переезд, встрет- / встреч), a short stem exactly (prefixes of 3-4
    letters are too noisy). One pass: a stem and its prefix share one doclist."""
    parts: dict[str, None] = {}
    for s in stems:
        if len(s) >= _MIN_PREFIX_STEM and not s.isdigit():
            parts[_quote(s[: max(4, len(s) - 2)]) + "*"] = None
        else:
            parts[_quote(s)] = None
    return " OR ".join(parts)


def _fence(alias: str) -> str:
    return f"(({alias}.labels & ~?) = 0 OR {alias}.author_tier IN {_OWN_SQL})"


def _top(
    conn: sqlite3.Connection,
    fts_sql: str,
    owner_sql: str,
    expr: str,
    label_mask: int,
    limit: int,
    bonus: Callable[[sqlite3.Row | tuple], float] | None = None,
) -> list[tuple[str, float]]:
    """FTS first (top-k by bm25, no join: the join over every match is what costs at 10^5),
    then the label fence on those rows only; k grows ×4 while the fence leaves fewer than
    `limit` and the index has more. Usable first, then relevance."""
    k = max(limit * 3, 100)
    while True:
        try:
            rows = conn.execute(fts_sql, (expr, k)).fetchall()
        except sqlite3.OperationalError:  # a malformed expression never breaks a turn
            return []
        if not rows:
            return []
        score = {r[0]: -float(r[1]) * (bonus(r) if bonus else 1.0) for r in rows}
        keep: list[tuple[str, float, bool]] = []
        rowids = list(score)
        for i in range(0, len(rowids), 500):
            chunk = rowids[i : i + 500]
            marks = ", ".join("?" for _ in chunk)
            for rowid, oid, st in conn.execute(
                owner_sql.format(marks=marks), [*chunk, int(label_mask)]
            ):
                keep.append((oid, score[rowid], st in _USABLE))
        if len(keep) >= limit or len(rows) < k:
            keep.sort(key=lambda x: (not x[2], -x[1], x[0]))
            return [(oid, s) for oid, s, _ in keep[:limit]]
        k *= 4


def search_records(
    conn: sqlite3.Connection, query: str, *, label_mask: int, limit: int = 50
) -> list[tuple[str, float]]:
    """Stemmed BM25 over `fts_records` (OR of the question's topic stems, long ones as
    prefixes); a row holding a stem exactly scores up to 1.5×. Every status but
    forgotten/purged comes back (usable first): «забудь» may aim at history, and the
    basket's `use_now` is the second fence."""
    stems = query_stems(query)
    if not stems or limit <= 0:
        return []
    w = ", ".join(str(x) for x in _WEIGHTS)
    fts_sql = (
        f"SELECT rowid, bm25(fts_records, {w}) AS s, text, keywords, aliases FROM fts_records"
        " WHERE fts_records MATCH ? ORDER BY s LIMIT ?"
    )
    owner_sql = (
        "SELECT r.fts_rowid, r.id, r.status FROM records r WHERE r.fts_rowid IN ({marks})"
        f" AND r.status NOT IN {_GONE_SQL} AND {_fence('r')}"
    )
    want = set(stems)

    def exact(row: sqlite3.Row | tuple) -> float:
        toks = set(f"{row[2]} {row[3]} {row[4]}".split())
        return 1.0 + _EXACT_BONUS * len(want & toks) / len(want)

    return _top(
        conn, fts_sql, owner_sql, _match_expr(_match_stems(query)), label_mask, limit, exact
    )


def search_raw(
    conn: sqlite3.Connection, query: str, *, limit: int = 50, label_mask: int = -1
) -> list[tuple[str, float]]:
    """Raw chunk ids by stemmed BM25 over the contentless `fts_raw`: active chunks only,
    never quarantined or junk. `label_mask` is the same fence as `search_records`."""
    stems = query_stems(query)
    if not stems or limit <= 0:
        return []
    fts_sql = (
        "SELECT rowid, bm25(fts_raw) AS s FROM fts_raw WHERE fts_raw MATCH ? ORDER BY s LIMIT ?"
    )
    owner_sql = (
        "SELECT c.fts_rowid, c.id, c.status FROM raw_chunks c WHERE c.fts_rowid IN ({marks})"
        f" AND c.status = 'active' AND c.quarantined = 0 AND c.junk = 0 AND {_fence('c')}"
    )
    return _top(conn, fts_sql, owner_sql, _match_expr(_match_stems(query)), label_mask, limit)


# ---------------------------------------------------------------- names (trigram)


def _fold(token: str) -> str:
    return token.replace("ё", "е")


def _trigrams(token: str) -> set[str]:
    return {token[i : i + 3] for i in range(len(token) - 2)}


def _dice(a: set[str], b: set[str]) -> float:
    return 2.0 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def _name_forms(query: str) -> list[str]:
    """Each topic token and its stem (folded «ё»), three letters or more."""
    out: dict[str, None] = {}
    for t in _tokens(query):
        if t in _STOP:
            continue
        for f in (_fold(t), _fold(_stem_token(t))):
            if len(f) >= 3:
                out[f] = None
    return list(out)


# Short first-declension names (Аня, Оля, Ира, Ася, Анна): the Snowball stem of «Ани/Аню/Ане»
# is two letters and a trigram of «ани» never meets «аня». A query token that is such a
# name's base plus a case ending scores 1.0; the base is the alias minus its final «а/я».
_SHORT_ENDS = ("ою", "ею", "ой", "ей", "а", "я", "ы", "и", "е", "у", "ю")
_SHORT_NOM = ("а", "я")
_SHORT_ALIAS = (3, 4)  # alias lengths the rule covers (longer names meet by trigrams)


def _short_bases(query: str) -> set[str]:
    """Bases (2-3 letters) of Cyrillic query tokens that end in a first-declension case
    ending: «ани», «аню», «ане», «аней» → «ан»; «оли» → «ол»."""
    out: set[str] = set()
    for t in _tokens(query):
        if t in _STOP or not _CYR.search(t):
            continue
        t = _fold(t)
        for end in _SHORT_ENDS:
            n = len(t) - len(end)
            if t.endswith(end) and n in (2, 3) and _CYR.fullmatch(t[0]):
                out.add(t[:n])
    return out


def _short_match(tok: str, bases: set[str]) -> bool:
    """An alias token like «аня» whose base («ан») a query token declined."""
    return len(tok) in _SHORT_ALIAS and tok[-1] in _SHORT_NOM and tok[:-1] in bases


def _name_expr(forms: Iterable[str]) -> str:
    """OR of the forms' trigrams; each «е» also as «ё», since `fts_names` keeps the spelling."""
    grams: set[str] = set()
    for f in forms:
        for g in _trigrams(f):
            grams.add(g)
            if "е" in g:
                grams.add(g.replace("е", "ё"))
    return " OR ".join(_quote(g) for g in sorted(grams))


def search_names(
    conn: sqlite3.Connection, query: str, *, limit: int = 20
) -> list[tuple[str, float]]:
    """Entity ids whose aliases look like a word of the query: trigram candidates from
    `fts_names`, scored by the best Dice similarity between a query form (token or stem) and
    an alias token; below 0.5 is dropped. Usable entities only."""
    from .. import entities

    forms = _name_forms(query)
    bases = _short_bases(query)
    # candidate nominatives of short names: only to fetch alias rows, never Dice-scored
    noms = [b + e for b in bases for e in _SHORT_NOM]
    # A declined diminutive («Илюхе», «Сашке», «Шурик») meets its name by its
    # root and by the name's family, so the alias rows of the family are fetched too
    roots: set[str] = set()
    family: set[str] = set()
    for tok in _name_tokens(query):
        roots |= entities.name_roots(tok)
        family |= {_fold(m.casefold()) for m in entities.family_of(tok)}
    if not (forms or noms) or limit <= 0:
        return []
    try:
        rows = conn.execute(
            "SELECT a.entity_id, a.alias FROM fts_names"
            " JOIN aliases a ON a.fts_rowid = fts_names.rowid"
            " JOIN records r ON r.id = a.entity_id"
            " WHERE fts_names MATCH ? AND r.status IN ('active', 'archived')"
            " ORDER BY bm25(fts_names) LIMIT 400",
            (_name_expr([*forms, *noms, *family, *(r for r in roots if len(r) >= 3)]),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    grams = [(f, _trigrams(f)) for f in forms]
    best: dict[str, float] = {}
    for eid, alias in rows:
        for tok in _tokens(alias):
            tok = _fold(tok)
            if bases and _short_match(tok, bases):
                best[eid] = 1.0
                continue
            if (family and tok in family) or (roots and roots & entities.name_roots(tok)):
                best[eid] = max(best.get(eid, 0.0), _FAMILY_SCORE)
            tg = _trigrams(tok)
            for f, fg in grams:
                s = 1.0 if f == tok else _dice(fg, tg)
                if s > best.get(eid, 0.0):
                    best[eid] = s
    hits = [(e, s) for e, s in best.items() if s >= _NAME_FLOOR]
    hits.sort(key=lambda h: (-h[1], h[0]))
    return hits[:limit]


# ---------------------------------------------------------------- indexing


def index_record(conn: sqlite3.Connection, record_id: str) -> None:
    """A record's FTS rows, stemmed, in the caller's transaction (§3.2): `text` (its English
    text fields), `keywords` (bilingual keywords and the original quote), `aliases` (an
    entity's aliases in every script). An entity also refreshes its `fts_names` rows (raw
    spelling: trigram matching folds case itself). Forgotten or purged: rows removed."""
    from .. import ledger

    row = conn.execute(
        "SELECT kind, status, fts_rowid FROM records WHERE id = ?", (record_id,)
    ).fetchone()
    if row is None:
        return
    kind = RecordKind(row[0])
    fts_rowid = row[2]
    conn.execute("DELETE FROM fts_records WHERE rowid = ?", (fts_rowid,))
    gone = row[1] in (Status.FORGOTTEN.value, Status.PURGED.value)
    if gone:
        if kind is RecordKind.ENTITY:
            conn.execute(
                "DELETE FROM fts_names WHERE rowid IN"
                " (SELECT fts_rowid FROM aliases WHERE entity_id = ?)",
                (record_id,),
            )
        return
    text, keywords, aliases = ledger._fts_parts(conn, record_id, kind)
    conn.execute(
        "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, ?, ?)",
        (fts_rowid, stem(text), stem(keywords), stem(aliases)),
    )
    if kind is RecordKind.ENTITY:
        for a_rowid, alias in conn.execute(
            "SELECT fts_rowid, alias FROM aliases WHERE entity_id = ?", (record_id,)
        ).fetchall():
            conn.execute("DELETE FROM fts_names WHERE rowid = ?", (a_rowid,))
            conn.execute("INSERT INTO fts_names (rowid, name) VALUES (?, ?)", (a_rowid, alias))


def index_raw(conn: sqlite3.Connection, raw_id: str) -> None:
    """A raw chunk's `fts_raw` row (stemmed text), in the caller's transaction; removed when
    the chunk is not active or has no text. For raw's `store_raw` and for `reindex_all`."""
    row = conn.execute(
        "SELECT fts_rowid, status, text FROM raw_chunks WHERE id = ?", (raw_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return
    conn.execute("DELETE FROM fts_raw WHERE rowid = ?", (row[0],))
    if row[1] == "active" and row[2]:
        conn.execute("INSERT INTO fts_raw (rowid, text) VALUES (?, ?)", (row[0], stem(row[2])))


def reindex_all(conn: sqlite3.Connection) -> int:
    """Rebuild `fts_records` (and `fts_names`) from the ledger and `fts_raw` from the raw
    archive, e.g. after a stemmer change (G12). Inside a caller's transaction it runs inline;
    otherwise in background write transactions of 250 items. Returns the rows written
    (records plus raw chunks)."""
    from .. import store

    rec_ids = [
        r[0]
        for r in conn.execute(
            f"SELECT id FROM records WHERE status NOT IN {_GONE_SQL} ORDER BY id"
        ).fetchall()
    ]
    raw_ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM raw_chunks WHERE status = 'active' AND text IS NOT NULL ORDER BY id"
        ).fetchall()
    ]
    jobs = [(index_record, i) for i in rec_ids] + [(index_raw, i) for i in raw_ids]
    if conn.in_transaction:
        for fn, i in jobs:
            fn(conn, i)
        return len(jobs)
    for start in range(0, len(jobs), _BATCH):
        with store.write_tx(conn, priority="background"):
            for fn, i in jobs[start : start + _BATCH]:
                fn(conn, i)
    return len(jobs)
