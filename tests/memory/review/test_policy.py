"""Adversarial review: chat modes and foreign orders (§5.2, §13.2, G1, G11; orders found in
foreign text are never executed as the user's own).

Each review test's docstring names its finding in the design notes. The tests after them pin
the v0.4 mode-change shape (§5.2: durable `mode_forget` job, `island` refused)."""

from __future__ import annotations

import contextlib
import json

from koob.memory import api, policy
from koob.memory.model import ChatMode, TurnCtx, TurnOrigin

from ..forget import _seed as s
from ..policy.conftest import Mem

clock = s.clock  # fixture: a manual clock wired into forget


def _turn(words: str, n: int = 1, **flags) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        message_id=f"msg-{n}",
        clock="2026-10-12T09:00:00.000000Z",
        **flags,
    )


def test_a_tainted_turn_cannot_lift_a_ban_his_words_did_not_name(mem, clock):
    """R-M1 (P1, G1/G11): `api.set_chat_mode` refuses only speakerphone turns. In a turn that
    read a page or a mail («see what this contact writes»), an injected «set tg:boss to normal»
    lifts the user's full ban: background reading and storage of that chat resume. The forget
    side has the user's-words check (`commands._his_word_needed`); the mode change does not."""
    api.set_chat_mode(
        "tg:boss", ChatMode.BANNED, turn=_turn("полный запрет на чат с боссом"), store=mem
    )
    tainted = _turn("глянь, что пишет Глеб", n=2, foreign_in_chat=True)
    with contextlib.suppress(policy.PolicyRefused):
        api.set_chat_mode("tg:boss", ChatMode.NORMAL, turn=tainted, store=mem)
    assert policy.chat_mode(mem, "tg:boss") is ChatMode.BANNED


def test_ban_of_a_long_chat_forgets_its_memory(mem, clock):
    """R-M2 (P1, G1/G5): a ban runs the forget pipeline over every raw chunk of the chat
    (one per message, §6.2). `forget._plan`, `_delete_vectors` and `_purge_rows` bind every
    id in one `IN (...)` list (only `_ids_in` chunks by 400), so a chat of more than 32 766
    messages (SQLite's variable limit; a family chat over a few years) raises «too many SQL
    variables»: the mode says «banned» but nothing of the chat is forgotten."""
    n = 32_800
    rows = [
        (f"raw-{i}", f"сообщение номер {i}", f"h-{i}", "2026-10-01T09:00:00.000000Z")
        for i in range(n)
    ]
    mem.executemany(
        "INSERT INTO raw_chunks (id, source_kind, chat_id, author_tier, ts, text, content_hmac,"
        " policy_version, created_at, updated_at) VALUES (?, 'telegram', 'tg:family', 'other',"
        " ?4, ?2, ?3, 0, ?4, ?4)",
        rows,
    )
    with contextlib.suppress(Exception):
        api.set_chat_mode(
            "tg:family", ChatMode.BANNED, turn=_turn("полный запрет на семейный чат"), store=mem
        )
    left = mem.execute(
        "SELECT count(*) FROM raw_chunks WHERE chat_id = 'tg:family' AND status = 'active'"
    ).fetchone()[0]
    assert left == 0


def test_a_bank_capture_needs_his_word_about_the_bank(mem):
    """The assistant must not look at bank, medical or private windows without the user's word
    about them. Guards against `admit_ex` letting a `screen:bank` / `screen:medical` capture in
    whenever the user's words hold any look verb («посмотри», «экран», «проверь»). «Посмотри,
    какая погода» while the bank window is on screen would store the user's balance."""
    turn = _turn("посмотри, какая завтра погода")
    decision = api.admit(
        "screen:bank",
        chat_id=None,
        text="Баланс счёта 1 234 567 ₽, последний платёж Ozon",
        author_tier="tool",
        turn=turn,
        store=mem,
    )
    assert not decision.admit
    # the category is normalised (`screen: bank`); the user's word about the bank lets it in
    spaced = api.admit("screen: Bank", chat_id=None, text="Баланс 1 234 ₽", author_tier="tool",
                       turn=turn, store=mem)  # fmt: skip
    assert not spaced.admit
    asked = _turn("посмотри баланс на карте", n=2)
    ok = api.admit("screen:bank", chat_id=None, text="Баланс 1 234 ₽", author_tier="tool",
                   turn=asked, store=mem)  # fmt: skip
    assert ok.admit


# ---------------------------------------------------------------- v0.4 mode changes (§5.2)


def _dima(mem) -> Mem:
    db = Mem(mem)
    db.raw("r-dima", text="Дима: живём в отеле Морской", chat="tg:dima")
    db.fact("f-hotel", subject="e-dima", text_en="Dima stays at the Morskoy hotel",
            tier="other", source_raw_id="r-dima")  # fmt: skip
    return db


def _status(mem, table: str, rid: str) -> str:
    return mem.execute(f"SELECT status FROM {table} WHERE id = ?", (rid,)).fetchone()[0]  # noqa: S608


def test_his_words_naming_chat_and_mode_change_it_in_a_tainted_turn(mem, clock):
    """The other direction: in a tainted turn the user's own words that name the chat and the
    mode are their order; naming only one of them is not."""
    db = Mem(mem)
    db.entity("e-boss", "Олег")
    db.raw("r-boss", text="Олег: созвон в 10", chat="tg:boss", author="e-boss")
    api.set_chat_mode("tg:boss", ChatMode.BANNED, turn=_turn("полный запрет на чат с Олегом"),
                      store=mem)  # fmt: skip
    for words in ("сними запрет", "глянь, что пишет Олег"):
        with contextlib.suppress(policy.PolicyRefused):
            api.set_chat_mode(
                "tg:boss", ChatMode.NORMAL, turn=_turn(words, n=2, tainted=True), store=mem
            )
        assert policy.chat_mode(mem, "tg:boss") is ChatMode.BANNED
    order = _turn("глянь письмо и сними запрет с чата Олега", n=3, tainted=True)
    api.set_chat_mode("tg:boss", ChatMode.NORMAL, turn=order, store=mem)
    assert policy.chat_mode(mem, "tg:boss") is ChatMode.NORMAL


def test_the_island_never_takes_a_mode(mem):
    """A11: `island` (the Cube's own conversation) is refused as a mode target."""
    for chat in ("island", " Island "):
        refused = False
        try:
            api.set_chat_mode(chat, ChatMode.BANNED, turn=_turn("запрет на island"), store=mem)
        except policy.PolicyRefused:
            refused = True
        assert refused, chat
    assert mem.execute("SELECT count(*) FROM chat_policy").fetchone()[0] == 0


def test_mode_change_commits_its_job_and_the_job_finishes_it(mem):
    """A3: the mode row, its `events` row and a `mode_forget` job commit together; a crash
    before stop-use leaves the job, and running it (startup step 5) forgets the chat."""
    _dima(mem)
    version = policy.set_mode(mem, "tg:dima", ChatMode.BANNED)  # then the process died
    job = mem.execute(
        "SELECT id, state, payload FROM jobs WHERE jkind = ?", (policy.MODE_FORGET_JOB,)
    ).fetchone()
    assert job["state"] == "queued"
    assert json.loads(job["payload"]) == {
        "chat_id": "tg:dima", "mode": "banned", "policy_version": version,
    }  # fmt: skip
    assert mem.execute("SELECT count(*) FROM events WHERE kind = 'mode'").fetchone()[0] == 1
    assert _status(mem, "records", "f-hotel") == "active"
    assert policy.finish_mode_forgets(mem) == 1
    assert policy.finish_mode_forgets(mem) == 0  # done is done
    assert mem.execute("SELECT state FROM jobs WHERE id = ?", (job["id"],)).fetchone()[0] == "done"
    assert _status(mem, "records", "f-hotel") != "active"
    assert _status(mem, "raw_chunks", "r-dima") != "active"


def test_a_live_lease_is_not_taken(mem):
    """§18.1 step 3: a `running` mode job whose lease has not ended belongs to another
    process; one whose lease ended is taken again."""
    _dima(mem)
    policy.set_mode(mem, "tg:dima", ChatMode.BANNED)
    mem.execute("UPDATE jobs SET state = 'running', not_before = '2999-01-01T00:00:00.000000Z'")
    assert policy.finish_mode_forgets(mem) == 0
    assert _status(mem, "records", "f-hotel") == "active"
    mem.execute("UPDATE jobs SET not_before = '2000-01-01T00:00:00.000000Z'")
    assert policy.finish_mode_forgets(mem) == 1
    assert _status(mem, "records", "f-hotel") != "active"


def test_a_superseded_mode_job_forgets_nothing(mem):
    """A ban the user lifted before its job ran: the job closes without forgetting (the user's
    latest word stands)."""
    _dima(mem)
    policy.set_mode(mem, "tg:dima", ChatMode.BANNED)
    policy.set_mode(mem, "tg:dima", ChatMode.NORMAL)
    assert policy.finish_mode_forgets(mem) == 1
    assert _status(mem, "records", "f-hotel") == "active"
