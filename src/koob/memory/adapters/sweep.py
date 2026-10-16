"""Sweep of `koob_messages` rows with no turn (`tgbot.record_outgoing`, `windows.say`,
`write_report`) by the `meta.sweep_watermark` id; a scheduler tick runs it.

Spec: §6.2 step 1, §9.4 (re-ingest after restore). The archiving itself lives in
`ingest.sweep`, reached as `api.sweep` from the memory host's job tick; this module is the
reader: one page of the conversation store above a watermark, nothing written.
"""

from __future__ import annotations

from typing import Any

from .conversation import ConversationStore

PAGE = 500  # rows per page, as `rows_without_turn`'s default


def sweep(conversation: ConversationStore, *, watermark: int) -> tuple[int, list[dict[str, Any]]]:
    """(new watermark, rows) for `turn_id IS NULL` rows above the watermark: one page in id
    order; the new watermark is the highest row id read (the old one when there is none).
    The caller archives the rows and only then moves `meta.sweep_watermark` (`ingest.sweep`),
    so a crash in between reads the same page again (ingestion dedupes by content hash)."""
    rows = list(conversation.rows_without_turn(after_id=int(watermark), limit=PAGE))
    top = int(watermark)
    for row in rows:
        rid = row.get("row_id", row.get("id"))
        if rid is not None:
            top = max(top, int(rid))
    return top, rows
