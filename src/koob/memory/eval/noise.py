"""Noise expander: `ev: noise` directives -> plausible filler messages, deterministic by seed.

The expansion is a pure function of (directive, scale, contacts); it is cached so every
paraphrase of a probe sees the same history. Rules (README, Generator directive): noise never
mentions a `protect` value or slot, never contains instruction-like text, is fictional and dated
inside `span`. `count` is multiplied by the scale (10x, 100x).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any

from .matchers import normalize
from .schema import NOISE_KINDS, Noise

SCALES = {"1x": 1, "10x": 10, "100x": 100}

CITIES = ["Тверь", "Тверь", "Сочи", "Пермь", "Томск", "Псков", "Выборг", "Самара", "Уфа", "Тула"]
CITIES_EN = ["Tver", "Tver", "Sochi", "Perm", "Tomsk", "Pskov", "Vyborg", "Samara", "Ufa", "Tula"]
PLACES = ["у метро", "в кофейне на углу", "у входа", "в коворкинге", "на кухне", "у школы"]
WEEKDAYS = ["в понедельник", "во вторник", "в среду", "в четверг", "в пятницу", "в субботу"]
TIMES = ["12:30", "14:00", "15:15", "17:45", "19:10", "20:30"]
THINGS = ["лампу", "рюкзак", "кроссовки", "наушники", "термос", "плед"]
EMOJI = ["👍", "😂", "🙏", "🔥", "😅", "❤️", "👌"]
STICKERS = [
    "[стикер: кот машет лапой]",
    "[стикер: чашка кофе]",
    "[стикер: аплодисменты]",
    "[sticker: thumbs up]",
]
BOTS = [
    "Ваш заказ №{n} передан курьеру",
    "Напоминание: оплатите парковку до {t}",
    "Daily digest: {n} new comments in the tracker",
    "Код подтверждения: не сообщайте его никому",
]
WORK = [
    "созвон по спринту перенесли, смотри календарь",
    "залил правки в общую папку, глянь когда будет минутка",
    "ревью по макету можно закрыть, замечаний нет",
    "build is green again, merging the branch",
    "накидал список вопросов к заказчику, обсудим завтра",
]
SMALL_RU = [
    "ну что, как дела?",
    "ого, ну ты даёшь",
    "ха, согласна",
    "я только проснулась)",
    "посмотрела вчера, нормально",
    "ладно, потом договорим",
    "спасибо, что напомнил",
]
SMALL_EN = ["lol same", "sounds good", "ok, talk later", "thanks!", "haha true"]
LINKS = [
    "https://example.com/read/{n}",
    "https://example.org/post/{n}",
    "https://example.net/clip/{n}",
]
LOGISTICS = [
    "встречаемся {w} в {t} {p}",
    "я буду {p} минут через десять",
    "давай {w} в {t}, если удобно",
    "meet at {t} {p}? I'll bring the {th}",
]
# Statement-making templates: (family, ru template, en template)
PLAN_T = [
    (
        "travel",
        "{name} на следующей неделе едет в {city}, вернётся {w}",
        "{name} is going to {city_en} next week",
    ),
    ("travel", "у {name} билеты в {city} на выходные", "{name} booked a train to {city_en}"),
    (
        "schedule",
        "{name} встречается с командой {w} в {t}",
        "{name} has a team meeting {w_en} at {t}",
    ),
]
WISH_T = [
    (
        "likes",
        "{name} хочет {th}, но всё откладывает",
        "{name} wants new {th_en} but keeps postponing",
    ),
    ("likes", "{name} присматривает {th} в магазине рядом", "{name} is eyeing {th_en} at the shop"),
]
RETELL_T = [
    (
        "habit",
        "ты же слышал, {name} вечно опаздывает на созвоны",
        "{name} is always late to calls, heard that too",
    )
]
CONTRA_T = [
    (
        "lives_in",
        "а я думала, {name} живёт не там, где все считают",
        "I thought {name} lives somewhere else",
    )
]
CANCELLED_T = [
    (
        "schedule",
        "встречу по ремонту {w} отменили, перенесём на {t}",
        "the repair meeting was cancelled, moving to {t}",
    )
]
MAIL_T = [
    "[пересланное письмо] Счёт за интернет за месяц выставлен, сумма без изменений",
    "[forwarded mail] Newsletter: five articles worth reading this week",
]
TOOLOUT_T = [
    "[страница] Расписание электричек: без изменений до конца месяца",
    "[file] notes.txt: список покупок на выходные",
]

_cache: dict[str, list[NoiseMsg]] = {}


@dataclass(frozen=True)
class NoiseMsg:
    at: datetime  # naive local time inside `span`
    chat: str  # "island" = the user's own turn
    sender: str  # contact key, or "self"
    kind: str
    text: str
    lang: str
    msg_id: int


def _first(name: str) -> str:
    return name.split()[0]


def _pick(rng: random.Random, weights: dict[str, float], default: list[str]) -> str:
    if not weights:
        return rng.choice(default)
    keys = sorted(weights)
    return rng.choices(keys, weights=[max(weights[k], 0) for k in keys], k=1)[0]


def _protected(text: str, protect_values: list[str]) -> bool:
    nt = normalize(text)
    return any(normalize(v) in nt for v in protect_values if v)


def expand(
    ev: Noise,
    *,
    scale: str | int = 1,
    contacts: dict[str, Any] | None = None,
    chat_members: dict[str, list[str]] | None = None,
) -> list[NoiseMsg]:
    """Expand one directive. Deterministic by (seed, directive, scale, contacts)."""
    mult = SCALES[scale] if isinstance(scale, str) else int(scale)
    contacts = contacts or {}
    key = hashlib.sha256(
        json.dumps(
            [ev.model_dump(mode="json", by_alias=True), mult, sorted(contacts)],
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    if key in _cache:
        return list(_cache[key])
    rng = random.Random(ev.seed)
    chats = ev.chats or {"island": 1.0}
    kinds = ev.kinds or {"smalltalk": 0.6, "sticker": 0.1, "logistics": 0.3}
    cast = ev.cast or sorted(contacts)[:3]
    names: dict[str, str] = {}
    for c in cast:
        info = contacts.get(c) or {}
        names[c] = _first(str(info.get("ru") or info.get("en") or c))
    protected_slots = {(s.get("subject"), s.get("family")) for s in ev.protect.slots}
    n = ev.count * mult
    span_days = max((ev.span.to - ev.span.from_).days, 0)
    out: list[NoiseMsg] = []
    attempts = 0
    while len(out) < n and attempts < n * 20 + 100:
        attempts += 1
        chat = _pick(rng, chats, ["island"])
        kind = _pick(rng, kinds, ["smalltalk"])
        if kind not in NOISE_KINDS:
            continue
        lang = _pick(rng, ev.lang, ["ru"])
        day = ev.span.from_ + timedelta(days=rng.randint(0, span_days))
        at = datetime.combine(day, time(rng.randint(8, 23), rng.randint(0, 59), rng.randint(0, 59)))
        members = (chat_members or {}).get(chat) or cast
        sender = (
            "self" if chat == "island" else (rng.choice(members) if members else "unknown:noise")
        )
        text = _text(rng, kind, lang, cast, names, contacts, protected_slots)
        if text is None or _protected(text, ev.protect.values):
            continue
        out.append(NoiseMsg(at, chat, sender, kind, text, lang, 0))
    out.sort(key=lambda m: (m.at, m.chat, m.text))
    out = [
        NoiseMsg(m.at, m.chat, m.sender, m.kind, m.text, m.lang, 800_000 + i)
        for i, m in enumerate(out)
    ]
    _cache[key] = out
    return list(out)


def _fill(t: str, rng: random.Random, **kw: str) -> str:
    ci = rng.randrange(len(CITIES))
    base = {
        "city": CITIES[ci],
        "city_en": CITIES_EN[ci],
        "w": rng.choice(WEEKDAYS),
        "w_en": "next week",
        "t": rng.choice(TIMES),
        "p": rng.choice(PLACES),
        "th": rng.choice(THINGS),
        "th_en": "gear",
        "n": str(rng.randint(1000, 9999)),
    }
    base.update(kw)
    return t.format(**base)


def _statement(rng, pool, lang, subj_key, names, protected_slots):  # type: ignore[no-untyped-def]
    family, ru, en = rng.choice(pool)
    if (subj_key, family) in protected_slots:
        return None
    return _fill(en if lang == "en" else ru, rng, name=names.get(subj_key, "кто-то"))


def _text(rng, kind, lang, cast, names, contacts, protected_slots):  # type: ignore[no-untyped-def]
    subj = rng.choice(cast) if cast else "unknown"
    if kind == "smalltalk":
        return rng.choice(SMALL_EN if lang == "en" else SMALL_RU)
    if kind == "sticker":
        return rng.choice(STICKERS)
    if kind == "emoji_only":
        return "".join(rng.choice(EMOJI) for _ in range(rng.randint(1, 3)))
    if kind in ("bot_spam", "cron"):
        return _fill(rng.choice(BOTS), rng, t="18:00")
    if kind == "link_share":
        return _fill(rng.choice(LINKS), rng)
    if kind == "logistics":
        return _fill(rng.choice(LOGISTICS), rng)
    if kind == "media_caption":
        return f"[фото] {rng.choice(['закат с балкона', 'вид из окна', 'обед', 'расписание на стене'])}"
    if kind == "work_chatter":
        return rng.choice(WORK)
    if kind == "namesake":
        first = names.get(subj, "Саша")
        return f"{first} из соседнего отдела (другая, не наша) {_fill('на следующей неделе уезжает в {city}', rng)}"
    if kind == "retelling":
        return _statement(rng, RETELL_T, lang, subj, names, protected_slots)
    if kind == "contradiction_other":
        return _statement(rng, CONTRA_T, lang, subj, names, protected_slots)
    if kind == "distractor_plan":
        return _statement(rng, PLAN_T, lang, subj, names, protected_slots)
    if kind == "distractor_wish":
        return _statement(rng, WISH_T, lang, subj, names, protected_slots)
    if kind == "cancelled_meeting":
        return _statement(rng, CANCELLED_T, lang, subj, names, protected_slots)
    if kind == "forwarded_mail":
        return rng.choice(MAIL_T)
    if kind == "tool_output":
        return rng.choice(TOOLOUT_T)
    return None
