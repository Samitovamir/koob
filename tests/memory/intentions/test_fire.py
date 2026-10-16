"""The memory spec §5.5 and §12: the user's own reminders and loops fire whatever their labels;
a due loop becomes overdue and is surfaced once. Spec marker: `[T intentions/test_fire::<case>]`."""

from __future__ import annotations

from koob.memory import api, intentions
from koob.memory.model import AuthorTier, IntentionState, Label, Loop, LoopState

from ..races._kit import TS0, seed, ulid
from ._kit import on_date, reminder, state

FRIDAY = "2026-10-16"
TUESDAY = "2026-10-13T08:00:00.000000Z"
SATURDAY = "2026-10-17T08:00:00.000000Z"


def _loop_state(conn, lid: str) -> str:
    return conn.execute("SELECT state FROM loops WHERE id = ?", (lid,)).fetchone()[0]


def test_own_health_loop_fires(mem):
    """§5.5: the user's own reminder fires whatever its labels («записаться к врачу до пятницы»)."""
    quote = "напомни записаться к врачу до пятницы"
    rem = reminder("doctor", on_date(TUESDAY), quote=quote, labels=Label.HEALTH)
    loop = Loop(
        id=ulid("loop:doctor"),
        author_tier=AuthorTier.USER,
        recorded_at=TS0,
        labels=Label.HEALTH,
        labels_final=True,
        origin_quote="записаться к врачу до пятницы",
        owner="user",
        text_en="Book a doctor's appointment by Friday",
        due=FRIDAY,
    )
    seed(mem, rem, loop)

    fired = api.on_event("clock", "date:2026-10-13", {"ts": TUESDAY}, store=mem)
    assert [f.intention_id for f in fired] == [rem.id]
    assert state(mem, rem.id) == IntentionState.DONE  # telling the user is the whole action
    assert _loop_state(mem, loop.id) == LoopState.OPEN  # not due yet

    api.on_event("clock", "date:2026-10-17", {"ts": SATURDAY}, store=mem)
    assert _loop_state(mem, loop.id) == LoopState.OVERDUE
    api.on_event("clock", "date:2026-10-17", {"ts": SATURDAY}, store=mem)  # a later tick
    texts = [n["text"] for n in api.recorded_notifications(store=mem)]
    assert texts == [
        "Напоминание: записаться к врачу до пятницы",
        "Просрочено: записаться к врачу до пятницы",
    ]
    # Unclear is asked once: only the transition reports True.
    assert intentions.set_loop(mem, loop.id, LoopState.UNCLEAR) is True
    assert intentions.set_loop(mem, loop.id, LoopState.UNCLEAR) is False
    assert intentions.loop_line(mem, loop.id) == "Неясно, сделано ли: записаться к врачу до пятницы"
