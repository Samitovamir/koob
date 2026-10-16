"""RRF (k = 60) plus bonuses (scope, pinned, review penalty); `assoc` fused separately.

Spec: §10.2 step 4.
"""

from __future__ import annotations

RRF_K = 60

# Bonuses are in RRF units: SCOPE equals one more lane's top rank (§10.2: scope hits get a
# fusion bonus); a pinned category a little less; an item flagged `review` (G3) loses half.
SCOPE_BONUS = 1.0 / (RRF_K + 1)
PINNED_BONUS = 0.5 / (RRF_K + 1)
REVIEW_PENALTY = -0.5 / (RRF_K + 1)


def rrf(
    lanes: dict[str, list[tuple[str, float]]], *, bonuses: dict[str, float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each lane is `(id, lane score)` best first; an id scores
    `sum 1 / (k + rank)` over the lanes that found it (rank from 1, duplicates in a lane
    count once), plus its bonus. Ties break by id, so the order is deterministic.
    A bonus alone never creates an item: only ids some lane found are ranked."""
    score: dict[str, float] = {}
    for hits in lanes.values():
        seen: set[str] = set()
        rank = 0
        for item_id, _ in hits:
            if item_id in seen:
                continue
            seen.add(item_id)
            rank += 1
            score[item_id] = score.get(item_id, 0.0) + 1.0 / (RRF_K + rank)
    for item_id, b in (bonuses or {}).items():
        if item_id in score:
            score[item_id] += b
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))
