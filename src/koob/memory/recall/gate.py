"""Gate: threshold, items already in the core view, visible conversation or this thread's
served set at the same version; dedupe by entity; at most one assoc item; budget.

Spec: §10.2 steps 5-6.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .fusion import RRF_K

MIN_SCORE = 1.0 / (RRF_K + 30)  # a lane rank past 30 is noise
HIGH_SCORE = 2.0 / (RRF_K + 4)  # past the session budget: top of two lanes, or strong + more
MAX_ITEMS = 6  # one basket; keeps the median near 300 tokens (§10.2 step 6)
# A broad question ("how are things with the diploma", "where are we with the move") strongly
# matches more records than one basket holds: the basket then grows with it, up to this, inside
# the token budget; the policy, stale and served checks run on every item as before.
MAX_ITEMS_BROAD = 10
PER_ENTITY = 3  # one person never fills the basket
SESSION_BUDGET = 8000
TURN_BUDGET = 1500


@dataclass
class Cand:
    """A candidate after hydration: what the gate and the renderer need."""

    id: str
    kind: str
    version: int
    text: str
    line: str = ""
    tokens: int = 0
    entity: str | None = None  # the subject the item is about (dedupe by entity)
    slot: str | None = None  # subject + relation: one value per slot
    lanes: set[str] = field(default_factory=set)
    strong: bool = False  # grounded evidence: a named entity, scope, episode fit, coverage
    assoc: bool = False
    dkey: str | None = None  # assoc goal id (§15.1 «the same goal»)


def served_same_version(
    conn: sqlite3.Connection, thread_id: str, versions: dict[str, int]
) -> set[str]:
    """Ids this thread already holds at the same version (core, basket, tool, write,
    update): re-serving them only spends tokens."""
    out: set[str] = set()
    ids = list(versions)
    for i in range(0, len(ids), 400):
        chunk = ids[i : i + 400]
        marks = ", ".join("?" for _ in chunk)
        for item_id, ver in conn.execute(
            f"SELECT item_id, version FROM served WHERE thread_id = ? AND item_id IN ({marks})",  # noqa: S608
            [thread_id, *chunk],
        ):
            if versions.get(item_id) == ver:
                out.add(item_id)
    return out


def breadth_cap(
    ranked: list[tuple[str, float]], cands: dict[str, Cand], max_items: int = MAX_ITEMS
) -> int:
    """How many items this question may take: `max_items`, or as many as its candidates at
    the high gate (top of two lanes, or strong and more) when more than that many match, so
    a broad question gets its lines and a narrow one keeps its short basket; never past
    MAX_ITEMS_BROAD. Association items do not widen it."""
    high = 0
    for item_id, score in ranked:
        c = cands.get(item_id)
        if c is None or c.assoc or score < HIGH_SCORE:
            continue
        if c.strong or len(c.lanes) >= 2:
            high += 1
    return min(max(max_items, high), max(max_items, MAX_ITEMS_BROAD))


def gate(
    conn: sqlite3.Connection,
    ranked: list[tuple[str, float]],
    *,
    thread_id: str,
    budget_tokens: int = TURN_BUDGET,
    cands: dict[str, Cand] | None = None,
    session_used: int = 0,
    session_budget: int = SESSION_BUDGET,
    blocked_goals: set[str] | None = None,
    max_items: int = MAX_ITEMS,
) -> list[str]:
    """The ids that go into the block, best first. Without `cands` only the score threshold
    and the served set apply (every item then counts as one token)."""
    cands = cands or {}
    versions = {i: cands[i].version for i, _ in ranked if i in cands}
    already = served_same_version(conn, thread_id, versions) if versions else set()
    blocked_goals = blocked_goals or set()
    out: list[str] = []
    used = 0
    slots: set[str] = set()
    per_entity: dict[str, int] = {}
    assoc_taken = False
    if cands and max_items == MAX_ITEMS:
        max_items = breadth_cap(ranked, cands, max_items)
    for item_id, score in ranked:
        c = cands.get(item_id)
        strong = c.strong if c else False
        multi = len(c.lanes) >= 2 if c else False
        if score < MIN_SCORE or not (strong or multi or c is None):
            continue
        if item_id in already:
            continue
        if c is not None:
            if c.assoc:
                if assoc_taken or (c.dkey and c.dkey in blocked_goals):
                    continue
            else:
                if c.slot and c.slot in slots:
                    continue
                if c.entity and per_entity.get(c.entity, 0) >= PER_ENTITY:
                    continue
        cost = c.tokens if c else 1
        if used + cost > budget_tokens:
            continue
        if session_used + used + cost > session_budget and score < HIGH_SCORE:
            continue  # past the session budget only items above the higher gate
        out.append(item_id)
        used += cost
        if c is not None:
            if c.assoc:
                assoc_taken = True
            if c.slot:
                slots.add(c.slot)
            if c.entity:
                per_entity[c.entity] = per_entity.get(c.entity, 0) + 1
        if len(out) >= max_items + (1 if assoc_taken else 0):
            break
    return out
