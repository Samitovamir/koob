"""One line of memory data, safe inside a fence (§13; foreign text is data).

Every fenced read path - the MCP tools' `<memory-data>`, the basket's `<memory-recall>` and
the core view's `<memory-core>` - runs item text through `clean`: quarantined spans cut
(§6.2 step 6), `<memory-…>` tags defused, no line breaks. A stranger's text then cannot close
the fence, open a new one or start a fake `[m:…]` line.

The user's own words are the user's tier: the quarantine cut applies to foreign tiers only,
as ingest quarantines only foreign raws. The user's «Запомни: …» is the thing to keep, so it
is shown as said; fence tags and handles are defused for every tier. An order to forget
or cancel (`order_spans`) is cut for every tier: its own sentence names what was dropped."""

from __future__ import annotations

import re
from typing import Any

_FENCE_TAG = re.compile(r"<\s*/?\s*memory-", re.IGNORECASE)
_HANDLE = re.compile(r"\[\s*m\s*:", re.IGNORECASE)
_CUT = "[…]"
_OWN = frozenset({"user", "user_confirmed"})  # policy.OWN_TIERS as plain values


def is_own(tier: Any) -> bool:
    """The author tier is the user's own (`user`, `user_confirmed`); None or any other is foreign."""
    return str(getattr(tier, "value", tier) or "") in _OWN


def clean(text: str | None, *, handles: bool = False, tier: Any = None) -> str:
    """One line of data: quarantined spans cut, fence tags defused, whitespace collapsed.
    `handles=True` also defuses `[m:` inside the text, for item text that sits next to the
    item's own handle (a basket or core line); MCP output keeps handles, since its `what`
    and `where_from` lines carry real ones. `tier` is the text's author tier: the user's own
    tier is never quarantined; any other tier, or none given, is."""
    from ..ingest import quarantine_spans

    body = str(text or "")
    for s, e in reversed(order_spans(body)):  # an order to forget is not memory
        body = body[:s] + _CUT + body[e:]
    if not is_own(tier):
        for s, e in reversed(quarantine_spans(body)):
            body = body[:s] + _CUT + body[e:]
    body = _FENCE_TAG.sub("‹memory-", body)
    if handles:
        body = _HANDLE.sub("‹m:", body)
    return " ".join(body.split())


# ---------------------------------------------------------------- the user's orders to the memory
#
# Forget and cancel in the colloquial forms people use, Russian and English
# («выкинь из головы», «не держи это в голове», «вычеркни», "scratch that", "stop keeping it
# in mind", "drop it", «отмени», "cancel that"). One lexicon serves the command parser
# (`commands.forget_now`: every form is «забудь» with its object kept), the safety-net
# detector (`writers.hot.detect`) and every fenced read (`clean`: the order's own sentence is
# never served back as memory, since it names what the user asked to drop). A negated form is
# the opposite order («не забудь», "don't forget", "don't cancel") and never matches.

_O = r"(?:\s+[^\s.!?;]+){0,5}?"  # a short object inside the Russian phrase («не держи ЭТО в голове»)
_NEG = r"(?<!\bне\s)(?<!\bnot\s)(?<!n't\s)(?<!\bnever\s)(?<!\bнельзя\s)"
# "I forget", "you forget": a statement, not an order
_SUBJ = r"(?<!\bi\s)(?<!\bwe\s)(?<!\byou\s)(?<!\bthey\s)(?<!\bhe\s)(?<!\bshe\s)"
_FORGET_FORMS = (
    r"{neg}(?<!\w)(?:по)?забудь(?:те)?(?:-ка)?(?!\w)",
    r"(?<!\w)(?:выкинь|выкиньте|выбрось|выбросьте)(?P<o1>{o})\s+из\s+(?:головы|памяти|башки)(?!\w)",
    r"(?<!\w)не\s+(?:держи|держите|храни|храните)(?P<o2>{o})\s+в\s+(?:голове|памяти|башке)(?!\w)",
    r"(?<!\w)(?:перестань|хватит|больше\s+не)\s+(?:держать|хранить)(?P<o3>{o})\s+в\s+(?:голове|памяти)"
    r"(?!\w)",
    r"(?<!\w)(?:сотри|сотрите|удали|удалите|вычеркни|вычеркните)(?P<o4>{o})\s+из\s+(?:памяти|головы)"
    r"(?!\w)",
    r"{neg}(?<!\w)вычеркни(?:те)?(?:-ка)?(?!\w)",
    r"(?<!\w)не\s+(?:запоминай|запоминайте|помни|помните)(?!\w)",
    r"(?<!\w)(?:больше\s+не\s+вспоминай|перестань\s+помнить)(?!\w)",
    r"{neg}{subj}(?<!\w)forget(?!\w)",
    r"{neg}(?<!\w)(?:strike|scratch)(?:\s+out)?(?=\s+(?:that|this|it|what|everything|all)\b)",
    r"(?<![\w'])(?:stop|quit)\s+keeping(?P<o5>{o})\s+in\s+(?:mind|memory|your\s+head)(?!\w)",
    r"(?<!\w)(?:don'?t|do\s+not|no\s+need\s+to)\s+keep(?P<o6>{o})\s+in\s+(?:mind|memory|your\s+head)"
    r"(?!\w)",
    r"{neg}(?<!\w)drop(?=\s+(?:it|that|this|what)\b)",
    r"(?<!\w)(?:put|get)(?P<o7>{o})\s+out\s+of\s+your\s+(?:head|mind)(?!\w)",
    r"{neg}(?<!\w)(?:erase|wipe|delete|remove)(?P<o8>{o})\s+from\s+(?:your\s+)?memory(?!\w)",
    r"{neg}(?<!\w)disregard(?=\s+(?:that|this|it|what|everything)\b)",
)
_CANCEL_FORMS = (
    r"{neg}(?<!\w)отмени(?:те)?(?:-ка)?(?!\w)",
    r"{neg}(?<!\w)cancel(?!\w)",
    r"{neg}(?<!\w)call(?P<o9>(?:\s+(?:it|that|this|the\s+[^\s.!?;]+))?)\s+off(?!\w)",
)


def _fill(form: str) -> str:
    return form.replace("{neg}", _NEG).replace("{subj}", _SUBJ).replace("{o}", _O)


_FORGET_ORDER = re.compile("|".join(_fill(f) for f in _FORGET_FORMS), re.IGNORECASE)
_CANCEL_ORDER = re.compile("|".join(_fill(f) for f in _CANCEL_FORMS), re.IGNORECASE)
_ANY_ORDER = re.compile("|".join(_fill(f) for f in (*_FORGET_FORMS, *_CANCEL_FORMS)), re.IGNORECASE)
ORDER = _ANY_ORDER  # any forget or cancel form (the safety net locates its clause with it)
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)|\n")


def forget_order(text: str | None) -> bool:
    """The user's words order the memory to forget, in any of its forms («выкинь из головы»)."""
    return bool(text) and bool(_FORGET_ORDER.search(text))


def cancel_order(text: str | None) -> bool:
    """The user's words cancel a plan («отмени встречу», "cancel that", "call it off")."""
    return bool(text) and bool(_CANCEL_ORDER.search(text))


def order_kind(text: str | None) -> str | None:
    """`forget`, `cancel` (a cancel with no forget form in the same text) or None."""
    if forget_order(text):
        return "forget"
    return "cancel" if cancel_order(text) else None


def canon_order(text: str | None) -> str:
    """The user's order with every forget or cancel form rewritten to the plain verb («забудь»,
    "forget") and the object it held kept in place: «не держи это в голове» → «забудь это»,
    "stop keeping the gate code in mind" → "forget the gate code"."""

    def plain(m: re.Match[str]) -> str:
        inner = next((g for g in m.groups() if g), "")
        verb = "forget" if _LATIN.search(m.group(0)[:3]) else "забудь"
        return verb + inner

    return _ANY_ORDER.sub(plain, text or "")


def order_spans(text: str | None) -> list[tuple[int, int]]:
    """The sentences that carry the user's forget or cancel order, whole (the target may stand
    before the verb: «Про давление забудь»), merged and sorted."""
    body = text or ""
    spans: list[tuple[int, int]] = []
    for m in _ANY_ORDER.finditer(body):
        s = 0
        for e in _SENTENCE_END.finditer(body, 0, m.start()):
            s = e.end()
        while s < m.start() and body[s].isspace():
            s += 1
        end = _SENTENCE_END.search(body, m.end())
        e = end.end() if end else len(body)
        if spans and s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans
