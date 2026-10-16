"""Goal events enqueue a cross-scope search at once; a usefulness gate (>= 0.8); links
stored as `derived(dkind='assoc_link')` with read set and grounds.

Spec: §15.1. Covers assoc (jobs, gate, links) and basket (`active_links`, the read the
basket's `assoc` finder makes: precomputed links only, never a search at turn time).
"""

from __future__ import annotations

import sqlite3


def on_goal_event(conn: sqlite3.Connection, record_id: str) -> str:
    """Enqueue an assoc job; returns the job id."""
    raise NotImplementedError("assoc")


def active_links(
    conn: sqlite3.Connection,
    entity_ids: list[str],
    *,
    label_mask: int = -1,
    limit: int = 5,
) -> list[str]:
    """Valid `assoc_link` ids whose read set touches one of `entity_ids` (cue entities and
    the turn's candidate records), or whose goal (`dkey`) is one of them; newest first.
    One indexed lookup in `read_deps` (§7.2 reverse index); labels fenced like any finder
    (`derived` has no author tier, so stricter, never looser)."""
    ids = [i for i in dict.fromkeys(entity_ids) if i]
    if not ids or limit <= 0:
        return []
    out: list[str] = []
    for i in range(0, len(ids), 400):
        chunk = ids[i : i + 400]
        marks = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT d.id FROM derived d WHERE d.dkind = 'assoc_link' AND d.valid = 1"
            f" AND (d.labels & ~?) = 0 AND (d.dkey IN ({marks}) OR d.id IN (SELECT holder_id"
            f" FROM read_deps WHERE holder_kind = 'derived' AND item_id IN ({marks})))"
            " ORDER BY d.built_at DESC, d.id LIMIT ?",
            [int(label_mask), *chunk, *chunk, limit],
        ).fetchall()
        out.extend(r[0] for r in rows if r[0] not in out)
    return out[:limit]
