"""Acceptance for the memory spec §17 (G8): «что ты помнишь про X», «откуда это»,
«забудь», «Вернуть» work on the ledger from the first stored record, in plain Russian, and
stop-use is immediate. Spec marker: `[T inspect/test_commands]`."""

from __future__ import annotations

from koob.memory import api, commands, forget
from koob.memory.model import ForgetScope

from ._world import MOVE, first_record, status, turn


def test_commands(mem):
    """§17: what / where-from / forget / undo work from the first stored record; stop-use is
    immediate and «удалил полностью» only after a verified purge."""
    first_record(mem)

    # what: the record, with its date and source, in Russian
    lines = commands.what(mem, "Саша", turn("что ты помнишь про Сашу"))
    text = "\n".join(lines)
    assert "Саша переезжает в Тверь в ноябре" in text
    assert "12 октября 2026" in text
    assert "ты сам" in text
    assert f"m:{MOVE}" in text

    # where from: quote, chat, date, who said it
    lines = commands.where_from(mem, f"m:{MOVE}", turn("откуда ты это знаешь"))
    text = "\n".join(lines)
    assert "«Саша переезжает в Тверь в ноябре»" in text
    assert "ты сам" in text and "12 октября 2026" in text
    assert "разговор с Кубом" in text

    # forget: stop-use at once, said as «перестал использовать»
    res = api.forget(
        f"m:{MOVE}",
        turn=turn("забудь про переезд Саши", n=2),
        because_quote="переезд Саши",
        store=mem,
    )
    assert res.outcome == "stopped"
    assert res.line and res.line.startswith("Перестал использовать")
    assert MOVE in res.record_ids
    assert status(mem, MOVE) == "forgotten"
    answer = api.what("Саша", turn=turn("что ты помнишь про Сашу", n=3), store=mem)
    assert MOVE not in [i.id for i in answer.items]
    assert api.purge_status(res.forget_id, store=mem).state == "in_trash"

    # «Вернуть» while in the trash
    back = api.undo_forget(res.forget_id, turn=turn("верни", n=4), store=mem)
    assert back.outcome == "undone"
    assert status(mem, MOVE) == "active"
    assert MOVE in [i.id for i in api.what("Саша", turn=turn("что про Сашу", n=5), store=mem).items]

    # «насовсем»: due now, purged in the pool after the call (F11), then «удалил полностью»
    gone = api.forget(
        f"m:{MOVE}",
        turn=turn("забудь переезд Саши насовсем", n=6),
        because_quote="переезд Саши",
        forever=True,
        scope=ForgetScope.ITEM,
        store=mem,
    )
    assert gone.outcome == "stopped", gone
    assert "Удаляю полностью" in (gone.line or "")
    assert api.purge_status(gone.forget_id, store=mem).state == "purging"
    api.run_purges(store=mem)
    assert forget.status(mem, gone.forget_id) == "purged"
    assert api.purge_status(gone.forget_id, store=mem).state == "purged"
