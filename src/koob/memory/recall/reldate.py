"""Relative day words in a line, resolved against the day the line was said.

A message dated `2023-05-08` that says «I lost my job yesterday» reads as
«I lost my job yesterday (`2023-05-07`)»: whoever answers from the line gets the day itself and
never has to count back from the message date. Only what can be pinned to a day, a week, a
weekend, a month or a year is resolved; vague words («the other day», "a few days ago",
«на днях») and words for the day it was said («today», «сегодня») stay as they are.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

_NUM = {
    "a": 1, "an": 1, "one": 1, "a couple of": 2, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "один": 1, "одну": 1, "пару": 2, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
}  # fmt: skip
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_SEASONS = {"spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11)}
_NUMS = "|".join(sorted((re.escape(k) for k in _NUM), key=len, reverse=True))

_PHRASE = re.compile(
    r"\b(?:"
    r"(?P<dby>the day before yesterday|позавчера)"
    r"|(?P<dat>the day after tomorrow|послезавтра)"
    r"|(?P<yday>yesterday|last night|вчера)"
    r"|(?P<tmrw>tomorrow|завтра)"
    rf"|(?P<n>\d+|{_NUMS}) (?P<unit>days?|weeks?|months?|years?"
    r"|дн(?:я|ей|ь)|недел[ьиюя]|месяц(?:а|ев)?|лет|год(?:а)?) (?:ago|назад)"
    r"|(?P<dir>last|next) (?P<what>week(?:end)?|month|year"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday|spring|summer|autumn|fall)"
    r"|на (?P<rwdir>прошлой|следующей) неделе"
    r"|в (?P<rdir>прошлом|следующем) (?P<rwhat>месяце|году)"
    r")\b(?! \(\d)",
    re.IGNORECASE,
)


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _month(d: date, shift: int) -> str:
    m = d.year * 12 + d.month - 1 + shift
    return f"{m // 12:04d}-{m % 12 + 1:02d}"


def _unit(u: str) -> str:
    u = u.casefold()
    if u.startswith(("day", "дн", "ден")):
        return "day"
    if u.startswith(("week", "недел")):
        return "week"
    if u.startswith(("month", "месяц")):
        return "month"
    return "year"


def _resolve(m: re.Match[str], said: date) -> str | None:
    g = m.groupdict()
    if g["dby"]:
        return (said - timedelta(days=2)).isoformat()
    if g["dat"]:
        return (said + timedelta(days=2)).isoformat()
    if g["yday"]:
        return (said - timedelta(days=1)).isoformat()
    if g["tmrw"]:
        return (said + timedelta(days=1)).isoformat()
    if g["n"]:
        raw = g["n"].casefold()
        n = int(raw) if raw.isdigit() else _NUM.get(raw)
        if not n or n > 400:
            return None
        unit = _unit(g["unit"])
        if unit == "day":
            return (said - timedelta(days=n)).isoformat()
        if unit == "week":
            return f"week of {_monday(said - timedelta(weeks=n)).isoformat()}"
        if unit == "month":
            return f"~{_month(said, -n)}"
        return f"~{said.year - n}"
    if g["rwdir"] or g["rdir"]:
        step = -1 if (g["rwdir"] or g["rdir"]).casefold().startswith("прошл") else 1
        if g["rwdir"]:
            return f"week of {(_monday(said) + timedelta(weeks=step)).isoformat()}"
        if g["rwhat"].casefold().startswith("месяц"):
            return _month(said, step)
        return str(said.year + step)
    step = -1 if g["dir"].casefold() == "last" else 1
    what = g["what"].casefold()
    if what == "week":
        return f"week of {(_monday(said) + timedelta(weeks=step)).isoformat()}"
    if what == "weekend":
        saturday = _monday(said) + timedelta(days=5)
        if step < 0 or said.weekday() >= 5:
            saturday += timedelta(weeks=step)
        return f"weekend of {saturday.isoformat()}"
    if what == "month":
        return _month(said, step)
    if what == "year":
        return str(said.year + step)
    if what in _WEEKDAYS:  # the nearest such day before (last) or after (next) the day said
        k = _WEEKDAYS.index(what)
        if step < 0:
            return (said - timedelta(days=(said.weekday() - k) % 7 or 7)).isoformat()
        return (said + timedelta(days=(k - said.weekday()) % 7 or 7)).isoformat()
    first, last = _SEASONS[what]
    if step < 0:
        return f"{what} {said.year if said.month > last else said.year - 1}"
    return f"{what} {said.year if said.month < first else said.year + 1}"


def resolve_relative(text: str, said: str | None) -> str:
    """`text` with each relative day word followed by the day it names, counted from `said`
    (the line's date, `YYYY-MM-DD...`); without a valid date the text stays as it is."""
    if not said or not text:
        return text
    try:
        day = date.fromisoformat(str(said)[:10])
    except ValueError:
        return text

    def sub(m: re.Match[str]) -> str:
        try:
            got = _resolve(m, day)
        except (ValueError, OverflowError):
            got = None
        return f"{m.group(0)} ({got})" if got else m.group(0)

    return _PHRASE.sub(sub, text)
