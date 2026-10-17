"""Evidence recall of a LoCoMo context, with no model call.

Every LoCoMo question names the dialog turns that hold its answer (`evidence`, ids such as
`D3:7`). A context holds a piece of evidence when it shows that message itself or an item whose
provenance points to it: the raw chunk stored for the message, or a record derived from it
(`edges.derived_from` from the record to the raw chunk). Items are read off a context by their
`[m:<id>]` handles; the plain RAG control reports the dialog ids of the messages it picked, and
the full-context control holds every message by construction.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from typing import Any

_HANDLE = re.compile(r"\[m:([A-Za-z0-9_-]+)\]")
_DIA = re.compile(r"D:?(\d+):(\d+)")  # the release also spells a few ids `D:11:26` or `D8:6; D9:17`


def norm_evidence(evidence: Iterable[str]) -> tuple[str, ...]:
    """The question's evidence as clean `D<session>:<turn>` ids, in order, without repeats."""
    out: list[str] = []
    for e in evidence:
        for s, t in _DIA.findall(str(e)):
            d = f"D{int(s)}:{int(t)}"
            if d not in out:
                out.append(d)
    return tuple(out)


def handles(text: str | None) -> list[str]:
    """The `[m:<id>]` handles of a context, in order of first appearance."""
    return list(dict.fromkeys(_HANDLE.findall(text or "")))


def dia_of_ref(source_ref: str | None, sample_id: str) -> str | None:
    """`msg:<sample>:<dia>` (the arrival rows' `msg_id`, as the raw chunk stores it) to `<dia>`."""
    head = f"msg:{sample_id}:"
    if not source_ref or not source_ref.startswith(head):
        return None
    got = norm_evidence([source_ref[len(head) :]])
    return got[0] if got else None


def provenance(conn: sqlite3.Connection, sample_id: str) -> dict[str, set[str]]:
    """Item id -> the dialog ids it stands for: each raw chunk its own message, each record the
    messages it was derived from."""
    raws: dict[str, str] = {}
    for rid, ref in conn.execute(
        "SELECT id, source_ref FROM raw_chunks WHERE source_ref LIKE ?", (f"msg:{sample_id}:%",)
    ):
        d = dia_of_ref(ref, sample_id)
        if d:
            raws[rid] = d
    out: dict[str, set[str]] = {rid: {d} for rid, d in raws.items()}
    for src, dst in conn.execute("SELECT from_id, to_id FROM edges WHERE etype = 'derived_from'"):
        if dst in raws:
            out.setdefault(src, set()).add(raws[dst])
    return out


def covered(item_ids: Iterable[str], prov: dict[str, set[str]]) -> set[str]:
    """The dialog ids a context's items stand for."""
    got: set[str] = set()
    for i in item_ids:
        got |= prov.get(i, set())
    return got


def recall_row(evidence: tuple[str, ...], got: set[str]) -> dict[str, Any]:
    """How much of one question's evidence a context holds."""
    hit = [e for e in evidence if e in got]
    return {"n": len(evidence), "hit": len(hit), "missing": [e for e in evidence if e not in got]}


def summarize(rows: list[dict[str, Any]], systems: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Per system over questions that name evidence: evidence-item recall (held / named), the
    share of questions holding all of it and holding any of it, and mean context tokens."""
    out: dict[str, dict[str, Any]] = {}
    for s in systems:
        rs = [r for r in rows if (r.get("recall") or {}).get(s) and r["recall"][s]["n"]]
        n = sum(r["recall"][s]["n"] for r in rs)
        hit = sum(r["recall"][s]["hit"] for r in rs)
        toks = [r["tokens"][s] for r in rs if s in (r.get("tokens") or {})]
        out[s] = {
            "questions": len(rs),
            "evidence": n,
            "item_recall": round(100 * hit / n, 1) if n else None,
            "all_held": round(
                100 * sum(r["recall"][s]["hit"] == r["recall"][s]["n"] for r in rs) / len(rs), 1
            )
            if rs
            else None,  # fmt: skip
            "any_held": round(100 * sum(r["recall"][s]["hit"] > 0 for r in rs) / len(rs), 1)
            if rs
            else None,
            "tokens_mean": round(sum(toks) / len(toks)) if toks else None,
        }
    return out


def format_recall(summary: dict[str, dict[str, Any]]) -> str:
    head = f"{'system':<12} {'q':>4} {'ev':>4} {'item%':>6} {'all%':>6} {'any%':>6} {'tok':>6}"
    lines = [head]
    for s, v in summary.items():
        lines.append(
            f"{s:<12} {v['questions']:>4} {v['evidence']:>4} {v['item_recall'] or 0:>6} "
            f"{v['all_held'] or 0:>6} {v['any_held'] or 0:>6} {v['tokens_mean'] or 0:>6}"
        )
    return "\n".join(lines)


__all__ = [
    "covered",
    "dia_of_ref",
    "format_recall",
    "handles",
    "norm_evidence",
    "provenance",
    "recall_row",
    "summarize",
]
