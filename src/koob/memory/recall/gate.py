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
# A broad question ("where are we with the diploma", "where are we with the move") strongly matches
# more records than one basket holds: the basket then grows with it, up to this, inside the token
# budget; the policy, stale and served checks run on every item as before.
MAX_ITEMS_BROAD = 10
# Recall first: a candidate one lane found, that policy and truth allow, is ranked, not
# dropped. Grounded lines (strong, or found by two lanes) are said first; the rest fill the
# basket after them, best first, up to this many lines and this many tokens in all.
MAX_LINES = 8
WEAK_BUDGET = 600
PER_ENTITY = 3  # a person the question does not name never fills the basket
# An item allowed only under the uncertain band ranks as if its score were this much of
# itself; at 1.0 it ranks by relevance like an unlabeled line (§5.5: the uncertain band is
# used now as unlabeled), and a lower weight would let a weaker clean line displace it.
UNSURE_WEIGHT = 1.0
SESSION_BUDGET = 8000
TURN_BUDGET = 1500


@dataclass(frozen=True)
class Width:
    """How much one basket may hold: grounded items, all lines, and the token budgets of the
    grounded lines and of every line once weaker ones fill in."""

    items: int = MAX_ITEMS
    lines: int = MAX_LINES
    tokens: int = TURN_BUDGET
    fill_tokens: int = WEAK_BUDGET


NARROW = Width()
# A question that asks to recall (what someone said or did, when something happened, which
# things): its answer may sit in any of many dated lines, so the basket holds more of them.
RECALL = Width(items=15, lines=15, tokens=900, fill_tokens=900)


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
    # some lane found it above that lane's floor; a meaning-only hit below the vector floor
    # is no evidence, so a basket of only such hits stays empty
    evidence: bool = True
    # no evidence, but above the noise floor: said after every evidenced line, in the room
    # they leave, and never alone
    faint: bool = False
    low: bool = False  # scoped to another person or period: ranked after the grounded lines
    # another person's item allowed only because its sole label is the classifier's uncertain
    # band: ranked among the clean lines by its score times `UNSURE_WEIGHT`
    unsure: bool = False
    dkey: str | None = None  # assoc goal id (§15.1 «the same goal»)
    # the raw messages the item stands for: a raw message itself, a record the messages it
    # was derived from. Once the block holds every one of them, the item repeats what is
    # already said and only fills room left at the end (`gate`); empty = never a repeat, and
    # a single-valued slot's value always takes its place (it keeps older values out)
    sources: frozenset[str] = frozenset()


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


def _rank_unsure(
    ranked: list[tuple[str, float]], cands: dict[str, Cand]
) -> list[tuple[str, float]]:
    """`ranked` with each `unsure` candidate moved to where its score times `UNSURE_WEIGHT`
    would rank among the clean ones; an exact tie goes to the clean line. The clean order and
    every score are unchanged, so the floors and gates read the real score."""
    unsure = [(i, s) for i, s in ranked if (c := cands.get(i)) is not None and c.unsure]
    if not unsure:
        return ranked
    out: list[tuple[str, float]] = []
    k = 0
    for item_id, score in ranked:
        c = cands.get(item_id)
        if c is not None and c.unsure:
            continue
        while k < len(unsure) and unsure[k][1] * UNSURE_WEIGHT > score:
            out.append(unsure[k])
            k += 1
        out.append((item_id, score))
    return out + unsure[k:]


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
    width: Width | None = None,
    subjects: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """The ids that go into the block, best first. Grounded candidates (strong, or found by
    two lanes) come first, up to `max_items` inside `budget_tokens` (an `unsure` one ranks
    among the clean ones by its weighted score, `_rank_unsure`); the other evidenced
    candidates then fill up to MAX_LINES lines and WEAK_BUDGET tokens in all, and the faint
    ones (no evidence, above the noise floor) take what room is left once something else is
    in, so what the finders found is ranked rather than lost. Without `cands` only the score
    threshold and the served set apply (every item then counts as one token). `width` sets
    the caps (a recall question gets RECALL); items about `subjects`, the people the question
    names, are the question's matter rather than crowding, so PER_ENTITY limits only the
    others. A candidate whose `sources` the lines already taken hold in full (a fact next to
    the message it was read from) gives its place to the next candidate and is offered again
    only after every other line, in the room still left."""
    cands = cands or {}
    versions = {i: cands[i].version for i, _ in ranked if i in cands}
    already = served_same_version(conn, thread_id, versions) if versions else set()
    blocked_goals = blocked_goals or set()
    out: list[str] = []
    used = 0
    slots: set[str] = set()
    per_entity: dict[str, int] = {}
    assoc_taken = False
    held: set[str] = set()  # the messages the taken lines stand for
    repeats: list[tuple[str, float]] = []
    line_cap, fill_tokens = MAX_LINES, WEAK_BUDGET
    if width is not None:
        max_items, line_cap, fill_tokens = width.items, width.lines, width.fill_tokens
        budget_tokens = min(budget_tokens, width.tokens)
    if cands and max_items == MAX_ITEMS:
        max_items = breadth_cap(ranked, cands, max_items)
    grounded: list[tuple[str, float]] = []
    rest: list[tuple[str, float]] = []
    faint: list[tuple[str, float]] = []
    for item_id, score in _rank_unsure(ranked, cands):
        if score < MIN_SCORE or item_id in already:
            continue
        c = cands.get(item_id)
        if c is None or (not c.low and (c.strong or len(c.lanes) >= 2)):
            grounded.append((item_id, score))
        elif c.evidence:
            rest.append((item_id, score))
        elif c.faint:
            faint.append((item_id, score))

    def take(
        item_id: str, score: float, line_cap: int, token_cap: int, repeat_ok: bool = False
    ) -> bool:
        nonlocal used, assoc_taken
        c = cands.get(item_id)
        if c is not None:
            if c.sources and not c.slot and c.sources <= held and not repeat_ok:
                repeats.append((item_id, score))
                return False
            if c.assoc:
                if assoc_taken or (c.dkey and c.dkey in blocked_goals):
                    return False
            else:
                if c.slot and c.slot in slots:
                    return False
                if (
                    c.entity
                    and c.entity not in subjects
                    and per_entity.get(c.entity, 0) >= PER_ENTITY
                ):
                    return False
        cost = c.tokens if c else 1
        if used + cost > token_cap:
            return False
        if session_used + used + cost > session_budget and score < HIGH_SCORE:
            return False  # past the session budget only items above the higher gate
        out.append(item_id)
        used += cost
        if c is not None:
            if c.assoc:
                assoc_taken = True
            if c.slot:
                slots.add(c.slot)
            if c.entity:
                per_entity[c.entity] = per_entity.get(c.entity, 0) + 1
            held.update(c.sources)
        return len(out) >= line_cap + (1 if assoc_taken else 0)

    def full() -> bool:  # every line of the basket's width is taken
        return len(out) >= line_cap + (1 if assoc_taken else 0)

    for item_id, score in grounded:
        if take(item_id, score, max_items, budget_tokens):
            return out
    for k, (item_id, score) in enumerate(rest + faint):
        if k >= len(rest) and not out:
            break  # faint hits fill room beside evidenced lines, never a basket alone
        if full():
            break
        if take(item_id, score, line_cap, min(budget_tokens, fill_tokens)):
            break
    for item_id, score in list(repeats):  # repeats of what is said: the room still left
        if full():
            break
        if take(item_id, score, line_cap, min(budget_tokens, fill_tokens), repeat_ok=True):
            break
    return out
