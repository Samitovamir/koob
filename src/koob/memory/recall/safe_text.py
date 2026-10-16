"""One line of memory data, safe inside a fence (§13; foreign text is data).

Every fenced read path - the MCP tools' `<memory-data>`, the basket's `<memory-recall>` and
the core view's `<memory-core>` - runs item text through `clean`: quarantined spans cut
(§6.2 step 6), `<memory-…>` tags defused, no line breaks. A stranger's text then cannot close
the fence, open a new one or start a fake `[m:…]` line.

The user's own words are the user's tier: the quarantine cut applies to foreign tiers only,
as ingest quarantines only foreign raws. The user's «Запомни: …» is what the user asked to
keep, so it is shown as said; fence tags and handles are defused for every tier."""

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
    if not is_own(tier):
        for s, e in reversed(quarantine_spans(body)):
            body = body[:s] + _CUT + body[e:]
    body = _FENCE_TAG.sub("‹memory-", body)
    if handles:
        body = _HANDLE.sub("‹m:", body)
    return " ".join(body.split())
