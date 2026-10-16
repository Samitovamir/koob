"""the memory spec §12: the bot outbox carries a read set; delivery re-validates it.
Spec marker: `[T intentions/test_outbox::<case>]`."""

from __future__ import annotations

from koob.memory import api, intentions
from koob.memory.model import ReadRef

from ._kit import DAY, DUE, address_action, deps, fix_value, on_date, reminder, seed_dima


def _deliver(conn, row: dict, sent: list[str], iid: str) -> bool:
    """What the bot outbox does at delivery (§7.4): stale → hold and re-compose, else send."""
    if intentions.outbox_stale(conn, row["read_set"]):
        action, ok = intentions.compose(conn, intentions.action_of(conn, iid))
        assert ok
        refs = [ReadRef(**r) for r in action["read_set"]]
        row.update(text=action["text_en"], read_set=intentions.stamp_outbox(conn, row["id"], refs))
        return False
    sent.append(row["text"])
    intentions.release_outbox(conn, row["id"])
    return True


def test_corrected_value_never_delivered(mem):
    """§12: a bot outbox row whose read set went stale is held and re-composed, never sent."""
    addr = seed_dima(mem)
    iid = intentions.arm(mem, reminder("addr", on_date(), action=address_action(mem, addr)))
    api.on_event("clock", DAY, {"ts": DUE}, store=mem)
    action = intentions.action_of(mem, iid)
    refs = [ReadRef(**r) for r in action["read_set"]]
    row = {"id": "ob-1", "text": action["text_en"]}
    row["read_set"] = intentions.stamp_outbox(mem, "ob-1", refs)  # queued while Telegram is out
    assert deps(mem, "outbox", "ob-1") == [(addr.id, refs[0].version)]

    fix_value(mem, addr.id, "addr2", "Мира 7")  # corrected while the row waits
    sent: list[str] = []
    assert _deliver(mem, row, sent, iid) is False  # held, re-composed
    assert _deliver(mem, row, sent, iid) is True
    assert sent == ["Адрес: Мира 7"]
    assert deps(mem, "outbox", "ob-1") == []
