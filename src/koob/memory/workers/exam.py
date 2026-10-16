"""Nightly exam sampled from the raw archive, judged by a cross-family model under the same
routing table.

Spec: §5.4, §18.

- **Sample.** `n` raw chunks of the archive (active, not quarantined, not junk) that have at
  least one usable fact extracted from them (`facts.source_raw_id`).
- **Probe.** Per fact a code-built question (the subject's name and the slot) goes through
  recall (`Recaller`, default: lexical search on a duty turn of its own thread). The fact
  counts as kept when its exact value comes back.
- **Code first, judge for free text.** A short value (≤ 4 words) is checked in code only. A
  longer free-text value goes to the `Judge` with the exact expected text; the judge must be
  of another model family than the extractor (`EXTRACTOR_FAMILY`) and obeys the routing table
  (`policy.route` over the raw and the fact): a judge whose route the table forbids is not
  called and the probe counts as `skipped_route`. Without a judge such probes are skipped too.
- **Output.** Numbers only (`dict[str, float]`) and one content-free audit line; never a
  line for the user. No live model in tests: they inject fakes.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Callable
from typing import Protocol

from .. import ledger, policy
from ..model import ReadRef, ReadSrc, Route, TurnCtx, TurnOrigin

EXTRACTOR_FAMILY = "claude"
_SAME_FAMILY = frozenset({"claude", "anthropic"})
PASS = 0.5  # a judge score at or above this counts as kept
SHORT_WORDS = 4


class Judge(Protocol):
    """A cross-family grader (local Qwen in production, a fake in tests)."""

    family: str
    route: Route

    def grade(self, *, question: str, expected: str, answer: str) -> float: ...


Recaller = Callable[[sqlite3.Connection, str], list[str]]

_judge: list[Judge | None] = [None]
_recaller: list[Recaller | None] = [None]


def use_judge(judge: Judge | None) -> None:
    if judge is not None and judge.family.casefold() in _SAME_FAMILY:
        raise ValueError("the exam judge must be of another model family than the extractor")
    _judge[0] = judge


def use_recaller(fn: Recaller | None) -> None:
    _recaller[0] = fn


def _search(conn: sqlite3.Connection, question: str) -> list[str]:
    from .. import commands

    turn = TurnCtx(
        turn_id="exam", thread_id="memory-exam", origin=TurnOrigin.DUTY, owner_words=question
    )
    items = commands.search(conn, question, turn=turn, include_archived=True, limit=10)
    return [i.text for i in items]


def _sample(conn: sqlite3.Connection, n: int, rng: random.Random) -> list[str]:
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT c.id FROM raw_chunks c JOIN facts f ON f.source_raw_id = c.id"
            " JOIN records r ON r.id = f.id WHERE c.status = 'active' AND c.quarantined = 0"
            " AND c.junk = 0 AND c.text IS NOT NULL AND r.status IN ('active', 'archived')"
            " ORDER BY c.id"
        )
    ]
    return rng.sample(ids, min(n, len(ids)))


def _facts(conn: sqlite3.Connection, raw_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT f.id, r.version, f.relation, f.value_text, f.value_norm, f.text_en,"
        " coalesce(e.display_name, f.subject_id) AS subject, c.rev"
        " FROM facts f JOIN records r ON r.id = f.id JOIN raw_chunks c ON c.id = f.source_raw_id"
        " LEFT JOIN entities e ON e.id = f.subject_id"
        " WHERE f.source_raw_id = ? AND r.status IN ('active', 'archived') ORDER BY f.id",
        (raw_id,),
    ).fetchall()


def code_kept(value: str, answers: list[str]) -> bool:
    """The exact value is back: its normalized dates and numbers, or the text itself."""
    joined = " ".join(answers).casefold()
    if value.casefold() in joined:
        return True
    want = ledger.value_tokens(value)
    return bool(want) and want <= ledger.value_tokens(joined)


def run_exam(
    conn: sqlite3.Connection,
    *,
    n: int = 20,
    judge: Judge | None = None,
    recall: Recaller | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    from .scheduler import doctor_line

    judge = judge if judge is not None else _judge[0]
    if judge is not None and judge.family.casefold() in _SAME_FAMILY:
        raise ValueError("the exam judge must be of another model family than the extractor")
    recall = recall or _recaller[0] or _search
    out = {k: 0.0 for k in ("sampled", "probes", "code_checked", "judged", "skipped_route")}
    kept = 0
    for raw_id in _sample(conn, n, random.Random(seed)):
        out["sampled"] += 1
        for f in _facts(conn, raw_id):
            value = (f["value_text"] or "").strip()
            if not value:
                continue
            question = f"{f['subject']} {f['relation'].replace('_', ' ')}"
            short = len(value.split()) <= SHORT_WORDS
            if not short:
                refs = [
                    ReadRef(id=raw_id, version=int(f["rev"]), src=ReadSrc.RAW),
                    ReadRef(id=f["id"], version=int(f["version"])),
                ]
                if judge is None or judge.route not in policy.route(conn, refs):
                    out["skipped_route"] += 1
                    continue
            out["probes"] += 1
            answers = recall(conn, question)
            if short:
                out["code_checked"] += 1
                kept += code_kept(value, answers)
                continue
            out["judged"] += 1
            assert judge is not None
            score = judge.grade(question=question, expected=value, answer="\n".join(answers))
            kept += score >= PASS
    out["kept"] = kept / out["probes"] if out["probes"] else 0.0
    doctor_line(conn, f"exam:kept={out['kept']:.2f}:probes={int(out['probes'])}")
    return out


__all__ = ["EXTRACTOR_FAMILY", "Judge", "code_kept", "run_exam", "use_judge", "use_recaller"]
