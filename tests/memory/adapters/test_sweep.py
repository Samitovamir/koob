"""the memory spec §6.2 step 1: rows with no turn, by `meta.sweep_watermark`.
Spec marker: `[T adapters/test_sweep::<case>]`."""

from __future__ import annotations

from koob.memory import api, forget

TS = "2026-10-12T09:00:00.000000Z"


def _mark(mem) -> int:
    return mem.execute("SELECT sweep_watermark FROM meta WHERE id = 1").fetchone()[0]


def test_no_turn_rows_ingested(mem):
    """§6.2: rows with no turn are swept by the watermark; turn rows are left to the after-turn adapter."""
    rows = [
        {"row_id": 10, "role": "cube", "source": "bot", "ts": TS, "text": "Напоминаю: в 18:00 созвон с Ильёй"},
        {"row_id": 11, "role": "cube", "ts": TS, "text": "Отчёт за неделю готов: три отклика, одно интервью"},
        {"row_id": 12, "role": "owner", "turn_id": "turn-7", "ts": TS, "text": "это строка хода, не для sweep"},
    ]  # fmt: skip
    assert api.sweep(rows=rows, store=mem) == 2
    got = dict(mem.execute("SELECT source_kind, turn_id FROM raw_chunks").fetchall())
    assert got == {"bot": None, "report": None}
    assert _mark(mem) == 11
    # rows at or below the watermark are not read again
    again = [{"row_id": 9, "role": "cube", "ts": TS, "text": "старая строка ниже отметки воды"}]
    assert api.sweep(rows=again, store=mem) == 0 and _mark(mem) == 11


class _Conv:
    def __init__(self, rows):
        self.rows = rows

    def rows_without_turn(self, *, after_id: int, limit: int = 500):
        return [r for r in self.rows if r["id"] > after_id][:limit]


def test_sweep_reads_the_wired_conversation(mem, monkeypatch):
    """`rows=None`: the wired koob.db adapter's `koob_messages` rows (id, role, created_at)."""
    conv = _Conv(
        [
            {"id": 3, "thread_id": 1, "role": "assistant", "text": "Написал Диме, что ты будешь к семи", "created_at": TS},
            {"id": 5, "thread_id": 1, "role": "assistant", "text": "Готово: отчёт по учёбе за неделю", "created_at": TS},
        ]
    )  # fmt: skip
    monkeypatch.setattr(forget._W, "conversation", conv)
    assert api.sweep(store=mem) == 2 and _mark(mem) == 5
    assert api.sweep(store=mem) == 0
