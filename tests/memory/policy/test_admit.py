"""the memory spec §5.1: admission runs before any write; a Cube turn is admitted on the
union of its sources. Spec marker: `[T policy/test_admit::<case>]`."""

from __future__ import annotations

from koob.memory import api, policy
from koob.memory.model import AuthorTier, ChatMode

from .conftest import owner

GLEB = "Глеб: верни, пожалуйста, мой серый свитер. Я дома в субботу, домофон GRAVEN-8813 (ГРАВЕН-8813)."


def _ban(conn, chat="tg:gleb", mode=ChatMode.BANNED):
    policy.set_mode(conn, chat, mode)


def test_banned_read_turn_keeps_only_own_words(db):
    """§5.1: a turn that read a banned chat keeps only the user's own words; answer and tool rows
    rejected."""
    conn = db.conn
    _ban(conn)
    turn = owner("Прочитай, что мне сегодня написал Глеб.", turn_id="t-e2")
    api.record_turn_source("t-e2", "tg_read", "tg:gleb", requested=True, store=conn)
    policy.note_read("t-e2", "tg:gleb", GLEB)
    before = db.table_counts()

    mine = api.admit(
        "cube_turn",
        chat_id="island",
        text=turn.owner_words,
        author_tier=AuthorTier.USER,
        turn=turn,
        store=conn,
    )
    assert mine.admit and mine.keep_only_own_words

    for tier, text in (
        (
            AuthorTier.SYSTEM,
            "Глеб просит вернуть серый свитер, он дома в субботу, домофон GRAVEN-8813.",
        ),
        (AuthorTier.TOOL, GLEB),
    ):
        d = api.admit(
            "cube_turn", chat_id="island", text=text, author_tier=tier, turn=turn, store=conn
        )
        assert not d.admit and d.reason == "turn_read_banned"

    # a user message quoting 8+ words of the banned read is not "the user's own words"
    quoting = owner(
        "перескажи: верни, пожалуйста, мой серый свитер. Я дома в субботу, домофон", turn_id="t-e2"
    )
    d = api.admit(
        "cube_turn",
        chat_id="island",
        text=quoting.owner_words,
        author_tier=AuthorTier.USER,
        turn=quoting,
        store=conn,
    )
    assert not d.admit and d.reason == "own_words_share_read"

    after = db.table_counts()
    assert {t: after[t] - before[t] for t in after if after[t] != before[t]} == {"audit": 3}
    joined = " ".join(r[0] for r in conn.execute("SELECT event FROM audit"))
    assert "GRAVEN" not in joined and "tg:gleb" not in joined


def test_own_words_text_is_his_words_only(db):
    """In a barred turn the stored text is `owner_words` (fenced foreign text stripped)."""
    conn = db.conn
    _ban(conn)
    turn = owner("Ладно, отдам ему свитер в субботу.", turn_id="t-3")
    policy.record_turn_source(conn, "t-3", "tg_read", "tg:gleb", requested=True)
    policy.note_read("t-3", "tg:gleb", GLEB)
    a = policy.admit_ex(
        conn,
        source_kind="cube_turn",
        chat_id="island",
        text="Ладно, отдам ему свитер в субботу.\n> [forwarded] " + GLEB,
        author_tier=AuthorTier.USER,
        turn=turn,
    )
    assert a.admit and a.text == "Ладно, отдам ему свитер в субботу."


def test_unregistered_read_fails_closed_for_long_own_words(db):
    """No shingle record of the barred read: 8+ own words cannot be checked, so they wait out."""
    conn = db.conn
    _ban(conn)
    policy.record_turn_source(conn, "t-4", "tg_read", "tg:gleb", requested=False)
    short = owner("Что там у Глеба?", turn_id="t-4")
    long = owner(
        "Перескажи мне подробно всё, что он писал за последнюю неделю, пожалуйста", turn_id="t-4"
    )
    assert policy.admit(
        conn,
        source_kind="cube_turn",
        chat_id="island",
        text=short.owner_words,
        author_tier=AuthorTier.USER,
        turn=short,
    ) == (True, None)
    assert policy.admit(
        conn,
        source_kind="cube_turn",
        chat_id="island",
        text=long.owner_words,
        author_tier=AuthorTier.USER,
        turn=long,
    ) == (False, "own_words_share_read")


def test_banned_chat_message_rejected(db):
    """Rule 1: a banned chat's own messages never enter, whoever wrote them."""
    conn = db.conn
    _ban(conn)
    for tier in (AuthorTier.OTHER, AuthorTier.USER):
        assert policy.admit(
            conn,
            source_kind="telegram",
            chat_id="tg:gleb",
            text="привет",
            author_tier=tier,
            turn=None,
        ) == (False, "chat_banned")
    # the ban's own durable `mode_forget` job (v0.4 A3) is the only job; admission adds none
    assert db.count("raw_chunks") == 0
    assert db.count("jobs", "jkind != ?", (policy.MODE_FORGET_JOB,)) == 0


def test_no_dig_read_background_vs_requested(db):
    """A background read of a «не копать» chat bars the turn; a read the user asked for does not."""
    conn = db.conn
    _ban(conn, "tg:olga", ChatMode.NO_DIG)
    bg = owner("Что нового?", turn_id="t-bg")
    policy.record_turn_source(conn, "t-bg", "tg_read", "tg:olga", requested=False)
    assert policy.admit(
        conn,
        source_kind="cube_turn",
        chat_id="island",
        text="Ольга писала…",
        author_tier=AuthorTier.SYSTEM,
        turn=bg,
    ) == (False, "turn_read_no_dig")

    asked = owner("Прочитай, что написала Ольга", turn_id="t-req")
    policy.record_turn_source(conn, "t-req", "tg_read", "tg:olga", requested=True)
    assert policy.admit(
        conn,
        source_kind="cube_turn",
        chat_id="island",
        text="Ольга пишет…",
        author_tier=AuthorTier.SYSTEM,
        turn=asked,
    ) == (True, None)


def test_clean_turn_admitted(db, turn):
    policy.record_turn_source(db.conn, turn.turn_id, "tg_read", "tg:friend", requested=False)
    assert policy.admit(
        db.conn,
        source_kind="cube_turn",
        chat_id="island",
        text="ответ",
        author_tier=AuthorTier.SYSTEM,
        turn=turn,
    ) == (True, None)


def test_capture_rule7_needs_his_word(db):
    """Rule 4: bank, medical and private-window captures only on the user's word."""
    conn = db.conn
    quiet = owner("", turn_id="t-c")
    assert policy.admit(
        conn,
        source_kind="screen:bank",
        chat_id=None,
        text="Баланс…",
        author_tier=AuthorTier.TOOL,
        turn=quiet,
    ) == (False, "capture_forbidden")
    assert policy.admit(
        conn,
        source_kind="screen:medical",
        chat_id=None,
        text="…",
        author_tier=AuthorTier.TOOL,
        turn=None,
    ) == (False, "capture_forbidden")
    asked = owner("посмотри, что у меня на экране в банке", turn_id="t-c2")
    assert policy.admit(
        conn,
        source_kind="screen:bank",
        chat_id=None,
        text="Баланс…",
        author_tier=AuthorTier.TOOL,
        turn=asked,
    ) == (True, None)
    assert policy.admit(
        conn,
        source_kind="screen",
        chat_id=None,
        text="редактор кода",
        author_tier=AuthorTier.TOOL,
        turn=quiet,
    ) == (True, None)
