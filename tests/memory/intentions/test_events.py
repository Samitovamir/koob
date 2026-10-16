"""the memory spec §12 (v0.4): event keys and payloads per source, arming through the
facade, read sets of intention actions. Spec marker: `[T intentions/test_events::<case>]`."""

from __future__ import annotations

import json

from koob.memory import api, intentions, ledger, policy, store
from koob.memory.model import (
    Assertion,
    ChatMode,
    IntentionState,
    OpKind,
    Proposal,
    ProposalOp,
    RecordKind,
    Status,
    date_event_key,
    message_event_key,
    record_event_key,
)

from ..races._kit import seed, version_of
from ._kit import (
    DAY,
    DIMA,
    DUE,
    address_action,
    deps,
    fix_value,
    from_dima,
    on_date,
    reminder,
    seed_dima,
    state,
    tg_message,
)


def test_event_key_formats():
    assert message_event_key("telegram", "tg:777", "m-42") == "telegram:tg:777:m-42"
    assert record_event_key("01J", 7) == "record:01J:7"
    assert date_event_key("2026-10-13") == "date:2026-10-13"


def _windows(conn, iid: str) -> list[tuple[str, str | None]]:
    return [
        (r[0], r[1])
        for r in conn.execute(
            "SELECT chat_id, until_ts FROM admit_windows WHERE intention_id = ?", (iid,)
        )
    ]


def test_arm_through_facade(mem):
    """`api.arm_intention`: armed; banned chat refused with one line; no_dig opens a window."""
    seed(mem, DIMA)
    plain = reminder("plain", on_date())
    res = api.arm_intention(plain, store=mem)
    assert res.outcome == "committed" and res.record_ids == [plain.id]
    assert state(mem, plain.id) == IntentionState.ARMED
    again = api.arm_intention(plain, store=mem)  # arming twice is one intention
    assert again.outcome == "committed"
    assert mem.execute("SELECT count(*) FROM intentions").fetchone()[0] == 1

    policy.set_mode(mem, "tg:888", ChatMode.BANNED)
    banned = reminder("banned", on_date(), action={"tool": "tg_send", "stable_target": "tg:888"})
    res = api.arm_intention(banned, store=mem)
    assert res.outcome == "rejected" and res.line and "\n" not in res.line
    assert mem.execute("SELECT 1 FROM records WHERE id = ?", (banned.id,)).fetchone() is None
    assert api.recorded_denials(store=mem)[-1]["tool"] == "memory_arm"

    policy.set_mode(mem, "tg:777", ChatMode.NO_DIG)
    watch = reminder("watch", from_dima())
    assert api.arm_intention(watch, store=mem).outcome == "committed"
    assert _windows(mem, watch.id) == [("tg:777", None)]
    key, payload = tg_message("m-1")
    fired = api.on_event("telegram", key, payload, store=mem)
    assert [f.intention_id for f in fired] == [watch.id]
    assert state(mem, watch.id) == IntentionState.DONE
    (window,) = _windows(mem, watch.id)
    assert window[1] is not None  # done closes its window (§5.6)


def _retract(conn, record_id: str) -> None:
    def apply(ctx, op):
        ledger.set_status(ctx, record_id, Status.RETRACTED)
        return [record_id]

    assert store.commit(conn, Proposal(ops=[ProposalOp(op=OpKind.FLAG, group="x")]), apply).ok


def test_action_read_set_recomposed(mem):
    """A fix to a value inside `action.read_set` makes the fire re-compose the action from
    current values; the old value is never sent."""
    addr = seed_dima(mem)
    iid = intentions.arm(mem, reminder("addr", on_date(), action=address_action(mem, addr)))
    assert deps(mem, "intention", iid) == [(addr.id, version_of(mem, addr.id))]

    new = fix_value(mem, addr.id, "addr2", "Мира 7")
    (fire,) = api.on_event("clock", DAY, {"ts": DUE}, store=mem)
    assert fire.intention_id == iid
    action = intentions.action_of(mem, iid)
    assert action["text_en"] == "Адрес: Мира 7"
    ref = {"id": new.id, "version": version_of(mem, new.id), "src": "record"}
    assert action["read_set"] == [ref]
    assert deps(mem, "intention", iid) == [(new.id, version_of(mem, new.id))]
    notes = [n["text"] for n in api.recorded_notifications(store=mem)]
    assert "Ленина" not in json.dumps([action, notes], ensure_ascii=False)
    assert state(mem, iid) == IntentionState.FIRED  # a hands action waits for its outcome
    assert intentions.settle(mem, iid, "ok") and state(mem, iid) == IntentionState.DONE

    # A value gone with no successor: held for the user, the old text leaves the action.
    gone = intentions.arm(mem, reminder("gone", on_date(), action=address_action(mem, new)))
    _retract(mem, new.id)
    fired = api.on_event("clock", DAY, {"ts": DUE}, store=mem)
    assert [f.intention_id for f in fired] == [gone]
    assert state(mem, gone) == IntentionState.NEEDS_ATTENTION
    held = intentions.action_of(mem, gone)
    assert held["held"] is True and "text_en" not in held
    assert "Мира" not in json.dumps(held, ensure_ascii=False)
    assert "проверь" in api.recorded_notifications(store=mem)[-1]["text"]


def test_remember_reminder_arms_through_arm(mem, turn):
    """§12: «напомни …» ("remind me …") arrives through
    `remember(kind='intention', assertion.trigger=…)`, which arms it; the date then fires it once
    with one plain line."""
    words = "напомни купить билеты 13 октября"
    t = turn.model_copy(update={"owner_words": words})
    a = Assertion(text_en="Buy the tickets", trigger=on_date())
    res = api.remember(
        words, turn=t, because_quote=words, kind=RecordKind.INTENTION, assertion=a, store=mem
    )
    assert res.outcome == "committed", res.line
    (iid,) = res.record_ids
    assert state(mem, iid) == IntentionState.ARMED
    fired = api.on_event("clock", DAY, {"ts": DUE}, store=mem)
    assert [f.intention_id for f in fired] == [iid]
    last = api.recorded_notifications(store=mem)[-1]["text"]
    assert last == "Напоминание: купить билеты 13 октября"
