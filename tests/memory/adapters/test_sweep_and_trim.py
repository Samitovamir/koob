"""The routine trim of old recall blocks (MEMORY.md §11) and the sweep page reader (§6.2 step
1). Offline, tmp_path only."""

from __future__ import annotations

import json

from koob.assistant import compact
from koob.memory.adapters import sweep


def _user(content) -> str:
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def _basket(n: int) -> str:
    return (
        f'<memory-recall turn="t{n}">\n<memory-data>\n[m:x{n}] факт {n}\n'
        "</memory-data>\n</memory-recall>"
    )


def test_trim_recall_blocks_keeps_the_newest_and_every_update(tmp_path):
    path = tmp_path / "s.jsonl"
    upd = "<memory-update>\n[m:a] теперь 14 мая\n</memory-update>"
    answer = {"type": "assistant", "message": {"content": [{"type": "text", "text": "ок"}]}}
    lines = [
        _user(f"{upd}\n\n{_basket(1)}\n\nвопрос 1"),
        json.dumps(answer) + "\n",
        _user([{"type": "text", "text": f"{_basket(2)}\n\nвопрос 2"}]),
        _user(f"{_basket(3)}\n\nвопрос 3"),
        _user(f"{_basket(4)}\n\nвопрос 4"),
    ]
    path.write_text("".join(lines), encoding="utf-8")
    assert compact.trim_recall_blocks(path, keep=2) > 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    out = json.dumps(rows, ensure_ascii=False)
    assert "факт 1" not in out and "факт 2" not in out
    assert "факт 3" in out and "факт 4" in out
    assert "теперь 14 мая" in out and "вопрос 1" in out and "вопрос 2" in out
    assert rows[0]["message"]["content"].startswith("<memory-update>")
    assert rows[1] == answer
    assert rows[2]["message"]["content"][0]["text"] == "вопрос 2"
    assert compact.trim_recall_blocks(path, keep=2) == 0  # nothing old left


def test_trim_recall_blocks_leaves_a_file_without_blocks_alone(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(_user("<memory_recall>\nзаметки\n</memory_recall>\n\nвопрос"), "utf-8")
    before = path.read_bytes()
    assert compact.trim_recall_blocks(path, keep=0) == 0
    assert path.read_bytes() == before
    assert compact.trim_recall_blocks(tmp_path / "missing.jsonl") == 0


class _Conv:
    def __init__(self, rows):
        self.rows = rows
        self.asked = []

    def rows_without_turn(self, *, after_id: int, limit: int = 500):
        self.asked.append((after_id, limit))
        return [r for r in self.rows if r["id"] > after_id][:limit]


def test_sweep_reads_one_page_above_the_watermark():
    conv = _Conv([{"id": 3, "role": "cube", "text": "a"}, {"id": 7, "role": "cube", "text": "b"}])
    top, rows = sweep.sweep(conv, watermark=3)
    assert top == 7 and [r["id"] for r in rows] == [7]
    assert sweep.sweep(conv, watermark=7) == (7, [])
    assert conv.asked[0] == (3, sweep.PAGE)
