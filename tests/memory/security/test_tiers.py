"""the memory spec §6.2 step 4: tiers of the Cube's own conversation.
Spec marker: `[T security/test_tiers::<case>]`."""

from __future__ import annotations

from koob.memory import api

TS = "2026-10-12T09:00:00.000000Z"
SAID = "у Саши день рождения 12 мая"


def _turn(mem, turn_id, rows):
    return api.ingest_turn(
        turn_id=turn_id, thread_id="thread-1", origin="owner", status="done", rows=rows, store=mem
    )


def _tiers(mem, turn_id):
    q = "SELECT source_kind, author_tier FROM raw_chunks WHERE turn_id = ?"
    return dict(mem.execute(q, (turn_id,)).fetchall())


def test_cube_restatement_not_user(mem):
    """§6.2: the Cube restating the user is system tier (tool when tainted), never user."""
    clean = [
        {"role": "owner", "ts": TS, "text": f"запомни: {SAID}"},
        {"role": "cube", "ts": TS, "text": f"Запомнил: {SAID}.", "author_tier": "user"},
    ]
    _turn(mem, "turn-1", clean)
    assert _tiers(mem, "turn-1") == {"owner": "user", "cube": "system"}
    tainted = [
        {"role": "owner", "ts": TS, "text": "что пишет Дима про день рождения Саши?"},
        {
            "role": "tool", "ts": TS, "tool": "tg_read", "source": "telegram", "chat_id": "tg:777",
            "msg_id": 9, "author_tier": "other", "text": f"Дима: {SAID}, не забудь",
        },
        {"role": "cube", "ts": TS, "text": f"Дима напоминает: {SAID}. Учёл."},
    ]  # fmt: skip
    _turn(mem, "turn-2", tainted)
    assert _tiers(mem, "turn-2") == {"owner": "user", "telegram": "other", "cube": "tool"}
    q = "SELECT count(*) FROM raw_chunks WHERE source_kind != 'owner' AND author_tier = 'user'"
    assert mem.execute(q).fetchone()[0] == 0
