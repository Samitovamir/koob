"""Text matching for the graders (README: Matching).

Text is NFKC, casefolded, e-diaeresis folded, whitespace collapsed. A value is a string
(substring), a typed matcher (`date`, `money`, `time`, expanded to every common written form)
or a list of alternatives (any one counts).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

_WS = re.compile(r"\s+")

RU_MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]  # fmt: skip
RU_MONTHS_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]  # fmt: skip
EN_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]  # fmt: skip
EN_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return _WS.sub(" ", t).strip()


@lru_cache(maxsize=4096)
def _compile(alt_key: str, kind: str) -> tuple[re.Pattern[str], ...]:
    if kind == "str":
        return (re.compile(re.escape(normalize(alt_key))),)
    raise AssertionError(kind)


def _date_patterns(value: str) -> list[str]:
    value = str(value)
    m = re.fullmatch(r"(?:(\d{4})-|--)(\d{2})-(\d{2})", value)
    if not m:
        raise ValueError(f"bad date {value!r}: use 2026-05-17 or --05-17")
    year, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
    day = rf"(?<!\d)0?{d}(?!\d)"
    mon = mo - 1
    months = {RU_MONTHS[mon], RU_MONTHS_NOM[mon], EN_MONTHS[mon], EN_ABBR[mon] + r"\.?"}
    mon_alt = "|".join(
        sorted(re.escape(x).replace(r"\\.", r"\.") if "\\" not in x else x for x in months)
    )
    pats = [
        rf"{day}\s*(?:{mon_alt})",
        rf"(?:{mon_alt})\s*{day}",
        rf"{day}\s*[./]\s*0?{mo}(?!\d)",
        rf"(?<!\d)0?{mo}\s*[/]\s*{day}",
    ]
    if year:
        y = year
        pats += [
            rf"{y}-0?{mo}-0?{d}(?!\d)",
            rf"{day}\s*[./]\s*0?{mo}\s*[./]\s*(?:{y}|{y[2:]})(?!\d)",
        ]
    else:
        pats += [rf"(?<!\d)(?:\d{{4}}-)?0?{mo}-0?{d}(?!\d)"]
    return pats


# Currency marks as they appear in normalised text. With `cur` given, an amount marked with
# ANOTHER currency is not this amount ("$50" is not {money: 50, cur: RUB}); an unmarked amount
# still counts (the README's 52 000 / 52k forms carry no sign).
CURRENCY_MARKS: dict[str, tuple[str, ...]] = {
    "RUB": ("₽", "руб", "rub", "rur", r"р(?![а-яa-z])", r"тр(?![а-яa-z])"),
    "USD": (r"\$", "usd", "долл", "dollar", "бакс", "buck"),
    "EUR": ("€", "eur", "евро"),
    "GBP": ("£", "gbp", "фунт", "pound"),
    "KZT": ("₸", "kzt", "рублей"),
    "UAH": ("₴", "uah", "грн", "гривн"),
    "CNY": ("¥", "cny", "rmb", "юан", "yuan"),
    "JPY": ("jpy", "иен", "йен", "yen"),
    "TRY": ("₺", "try", "лир", "lira"),
}
_SIGNS = {"₽", r"\$", "€", "£", "₸", "₴", "¥", "₺"}


def _amount(amount: Any) -> tuple[int, str]:
    """`{money: 3.5}`, `3.50`, `"1 200,50"` -> (1200, "50"); whole amounts -> (n, "00")."""
    t = str(amount).replace(" ", "").replace(" ", "").replace(",", ".")
    m = re.fullmatch(r"(\d+)(?:\.(\d{1,2}))?", t)
    if not m:
        raise ValueError(f"bad money {amount!r}: use 52000 or 3.50")
    return int(m.group(1)), (m.group(2) or "00").ljust(2, "0")


def _int_form(n: int) -> str:
    """The integer part as written: 1200000, 1 200 000, 1,200,000, 1.200.000, 1'200'000."""
    digits = str(n)
    if n < 1000:
        return digits
    head, rest = digits[: len(digits) % 3 or 3], digits[len(digits) % 3 or 3 :]
    groups = [rest[i : i + 3] for i in range(0, len(rest), 3)]
    grouped = head + "".join(rf"[\s,.']?{g}" for g in groups)
    return rf"(?:{digits}|{grouped})"


def _cur_guard(cur: str | None) -> tuple[str, str, str]:
    """(before, after, word) lookarounds that refuse an amount marked with another currency;
    `word` guards the currency word between units and cents («3 доллара 50 центов»)."""
    if not cur:
        return "", "", ""
    others = [t for c, toks in CURRENCY_MARKS.items() if c != cur.upper() for t in toks]
    alt = "|".join(others)
    signs = [t for t in others if t in _SIGNS]
    before = "".join(rf"(?<!{t})(?<!{t}\s)" for t in signs)
    return before, rf"(?!\s?(?:{alt}))", rf"(?!(?:{alt}))"


def _money_patterns(amount: Any, cur: str | None) -> list[str]:
    n, cents = _amount(amount)
    whole = _int_form(n)
    # the cents as written: "50" (also "5" when it ends in 0); a whole amount may show ",00"
    if cents == "00":  # and not the head of a bigger number: 1 200 in «1 200 000»
        frac = r"(?:[.,]0{1,2})?(?![.,]?\d)(?![\s,.']\d{3}(?!\d))"
    else:
        frac = rf"[.,]{cents[0]}{'0?' if cents[1] == '0' else cents[1]}(?!\d)"
    before, after, word = _cur_guard(cur)
    lead = rf"{before}(?<![\d.,])(?<!\d[\s,.'])"  # not the tail of a bigger number either
    pats = [rf"{lead}{whole}{frac}{after}"]
    if cents != "00":  # «3 доллара 50 центов», «1 200 руб. 50 коп.»
        c = cents.lstrip("0") or "0"
        pats.append(rf"{lead}{whole}\s?{word}[^\d\s]{{1,10}}\s?0?{c}\s?(?:коп|цент|cent|¢)")
    if n >= 1000 and cents == "00" and n % 100 == 0:  # 52k, «52 тыс», 3,5k, «3.5 тыс»
        k = f"{n / 1000:g}".replace(".", "[.,]")
        pats.append(rf"{lead}{k}\s*(?:k|к|тыс(?:яч\w*|\.)?|тр)(?![а-яa-z]){after}")
    if n >= 1_000_000 and cents == "00" and n % 100_000 == 0:
        mm = f"{n / 1_000_000:g}".replace(".", "[.,]")
        pats.append(rf"{lead}{mm}\s*(?:m|млн|mln)(?![а-яa-z]){after}")
    return pats


def _time_patterns(value: str) -> list[str]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value))
    if not m:
        raise ValueError(f'bad time {value!r}: use "16:00"')
    h, mi = int(m.group(1)), int(m.group(2))
    pats = [rf"(?<!\d)0?{h}\s*[:.\-]\s*{mi:02d}(?!\d)"]
    if mi == 0:
        pats.append(rf"(?<![\d:.])0?{h}\s*(?:ч|час|часов|h)(?![а-яa-z])")
        if h > 12:
            pats.append(rf"(?<!\d){h - 12}\s*(?:pm|p\.m\.)")
        elif h == 12:
            pats.append(r"(?<!\d)12\s*(?:pm|p\.m\.)")
        else:
            pats.append(rf"(?<!\d){h}\s*(?:am|a\.m\.)")
    return pats


@lru_cache(maxsize=2048)
def _typed(kind: str, a: str, b: str) -> tuple[re.Pattern[str], ...]:
    if kind == "date":
        pats = _date_patterns(a)
    elif kind == "money":
        pats = _money_patterns(a, b or None)
    else:
        pats = _time_patterns(a)
    return tuple(re.compile(p) for p in pats)


def compile_value(v: Any) -> list[re.Pattern[str]]:
    """One value or a list of alternatives -> patterns over normalised text."""
    if isinstance(v, list):
        out: list[re.Pattern[str]] = []
        for x in v:
            out += compile_value(x)
        return out
    if isinstance(v, dict):
        for kind in ("date", "money", "time"):
            if kind in v:
                return list(_typed(kind, str(v[kind]), str(v.get("cur") or "")))
        raise ValueError(f"bad typed matcher {v!r}")
    return list(_compile(str(v), "str"))


def find(text: str, value: Any) -> bool:
    """True when any alternative of `value` occurs in `text`."""
    t = normalize(text)
    return any(p.search(t) for p in compile_value(value))


# A value shown as history is not shown as current. The mark sits in the value's
# own clause (a line, cut at . ! ? ;): a past word before it («раньше 14 мая», «earlier note
# said 14 May»), or a retired word after it («14 мая - неверно», «Thursday → Friday»). The
# direction matters: in «Москва, раньше Тверь» the first city is shown as current.
HISTORY_BEFORE = ("раньше", "ранее", "прежде", "прежн", "было", "бывш", "вместо", "до этого", "previously", "earlier", "formerly", "used to", "was ", "were ", "instead of", "old:", "before:")  # fmt: skip
HISTORY_AFTER = ("неверн", "ошибочн", "ошибк", "неправд", "устарел", "отмен", "перепутал", "не так", "wrong", "incorrect", "mistaken", "outdated", "superseded", "retired", "no longer", "not anymore", "(old)", "→", "->")  # fmt: skip
_CLAUSE = re.compile(r"(?<=[.!?;…])\s+")


def clauses(text: str) -> list[str]:
    """The text's clauses, normalised: lines, cut after sentence punctuation."""
    out = []
    for line in text.splitlines():
        out += [c for c in (normalize(x) for x in _CLAUSE.split(line)) if c]
    return out


# A value its own clause negates or overrides is not shown as current either, for every stale
# check: the negation word stands right before the value, at most two small words
# apart («без кинзы», «уже не в Твери», "not in Boulder", "isn't Boulder", «вместо кинзы»);
# or a cessative frames it («больше не живёшь в Твери», "no longer live in Boulder", "don't
# live in Boulder anymore") with no comma and no polarity-flipping verb between («больше не
# забывай кинзу» affirms it); or «уже нет» / "not anymore" follows it. A negated negation («не
# без кинзы»), a negative question («ты не в Твери?») and a negation of another word («не
# забудь кинзу», "Boulder, not Denver") leave the value shown as current.
_NEG = r"не|без|нет|ни|not|no|without|never|nor|вместо|instead of|rather than"
_FILL = r"(?:в|во|на|из|у|с|со|к|ко|от|для|про|о|об|по|за|in|at|on|from|to|for|with|by|the|a|an|about)"  # fmt: skip
_TAIL = r"[^\w\s]*$"  # «, $, an opening quote between the frame and the value
_WORDS = r"((?:[^\s,;:—–-]+\s+){0,3})"  # a short frame: no comma, no dash
_DIRECT = re.compile(rf"(?<!\w)({_NEG}|\w+n['’]t)\s+(?:{_FILL}\s+){{0,2}}{_TAIL}")
_DOUBLE = re.compile(r"(?<!\w)(?:не|not|no|never)\s+$")
_CEASE = re.compile(rf"(?<!\w)(?:(?:больше|уже)\s+не|no longer)\s+{_WORDS}{_TAIL}")
_NOT_ANYMORE = re.compile(rf"(?<!\w)(?:not|never|\w+n['’]t)\s+{_WORDS}{_TAIL}")
_ANYMORE = re.compile(r"\s*,?\s*any\s?more\b")
_NEG_AFTER = re.compile(r"\s*[,—–-]?\s*(?:(?:уже|больше)\s+нет|no longer|not any\s?more)\b")
_FLIP = ("забы", "забуд", "пропус", "пропуск", "избега", "отказ", "теря", "игнор", "forget", "miss", "skip", "avoid", "ignor", "lose", "drop", "omit", "neglect", "refus")  # fmt: skip
_WORD_HEAD = re.compile(r"\w*$")
_WORD_REST = re.compile(r"\w*")
_LEAD = re.compile(r"[^\w]*")


def _negated(clause: str, m: re.Match[str]) -> bool:
    """The value's own clause negates or overrides it (see above)."""
    if clause.rstrip().endswith("?"):
        return False
    start = m.start() + _LEAD.match(m.group()).end()
    before = clause[:start]
    before = before[: _WORD_HEAD.search(before).start()]  # a value that starts mid-word
    pos = max(0, len(before) - 120)
    d = _DIRECT.search(before, pos)
    if d and not _DOUBLE.search(before, 0, d.start()):
        return True
    for frame in (_CEASE, _NOT_ANYMORE):
        f = frame.search(before, pos)
        if not f or any(k in f.group(1) for k in _FLIP):
            continue
        if frame is _CEASE:
            return True
        after = clause[m.end() :]
        if _ANYMORE.match(after, _WORD_REST.match(after).end()):
            return True
    after = clause[m.end() :]
    return bool(_NEG_AFTER.match(after, _WORD_REST.match(after).end()))


def _marked(clause: str, m: re.Match[str]) -> bool:
    before, after = clause[: m.start()], clause[m.end() :]
    return (
        any(k in before for k in HISTORY_BEFORE)
        or any(k in after for k in HISTORY_AFTER)
        or _negated(clause, m)
    )


def shown_as_current(text: str, value: Any, *, history: bool = True) -> bool:
    """True when some occurrence of `value` is shown as current: its own clause neither negates
    it nor (with `history`, the `allow: history` reading) marks it as history. A value the
    clause cut splits (found in the whole text, in no clause) counts as current."""
    pats = compile_value(value)
    mark = _marked if history else _negated
    seen = False
    for cl in clauses(text):
        for p in pats:
            for m in p.finditer(cl):
                seen = True
                if not mark(cl, m):
                    return True
    return not seen and find(text, value)


def describe(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(describe(x) for x in value)
    if isinstance(value, dict):
        return ",".join(f"{k}={v}" for k, v in value.items())
    return str(value)
