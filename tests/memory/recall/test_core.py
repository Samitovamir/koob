"""Acceptance tests for the memory spec §10.1 (the core view, "pocket") and its
§7.6 diff for a resumed session. Spec marker: `[T recall/test_core::<case>]`."""

from __future__ import annotations

from datetime import timedelta

import pytest

from koob.memory import api, store
from koob.memory import clock as _clock
from koob.memory.model import (
    AuthorTier,
    Label,
    Lesson,
    Loop,
    OpKind,
    StableId,
    TurnCtx,
    WorkState,
    WorkStateBody,
)
from koob.memory.recall import core_view
from koob.memory.writers import hot

from .._ledger import add, commit, entity, fact, op, open_seeded, pref, ts


@pytest.fixture
def conn(tmp_path):
    c = open_seeded(tmp_path)
    yield c
    c.close()


def _day(n: int) -> str:
    return (_clock.now().date() + timedelta(days=n)).isoformat()


def _ok(res) -> None:
    assert res.ok, res


def _seed(c) -> None:
    _ok(add(c, fact("f-home", "self", "lives_in", "Moscow", text="Lives in Moscow")))
    _ok(add(c, pref("p-tea", "self", "prefers", "green tea", text="Prefers green tea")))
    _ok(add(c, entity("e-dima", "Dima")))
    _ok(add(c, fact("f-dima-job", "e-dima", "works_at", "Yandex", text="Dima works at Yandex")))
    loop = Loop(
        id="l-cv",
        owner="user",
        text_en="Send the CV to Dima",
        due=_day(3),
        author_tier=AuthorTier.USER,
        recorded_at=ts(),
        origin_quote="напомни отправить резюме Диме",
    )
    _ok(add(c, loop))
    lesson = Lesson(
        id="les-1",
        rule_en="Ask before booking flights",
        helpful=3,
        author_tier=AuthorTier.USER,
        recorded_at=ts(),
        origin_quote="спрашивай перед покупкой билетов",
    )
    _ok(add(c, lesson))
    ws = WorkState(
        id="w-1",
        matter_id="career",
        state=WorkStateBody(goal="Land a security internship", next_step="Finish the cover letter"),
        author_tier=AuthorTier.USER,
        recorded_at=ts(),
        origin_quote="работаю над стажировкой",
    )
    _ok(add(c, ws))


def _supersede(c, old: str, new_rec) -> None:
    _ok(
        commit(
            c,
            op(OpKind.SUPERSEDE, target=old, group=old, record=new_rec.model_dump(mode="json")),
        )
    )


def test_sections_in_cache_stable_order(conn):
    """§10.1: About you, People, Working now, Lessons, What changed; each with its read set."""
    _seed(conn)
    view = api.core(thread_id="t1", session_id="s1", store=conn)
    text = view.text
    assert text.startswith(core_view.OPEN) and text.endswith(core_view.CLOSE)
    heads = [text.index(f"## {core_view.TITLES[n]}") for n in core_view.SECTIONS[:4]]
    assert heads == sorted(heads)
    for needle in (
        "Lives in Moscow",
        "Prefers green tea",
        "Dima",
        "Dima works at Yandex",
        "Send the CV to Dima",
        f"due {_day(3)}",
        "Finish the cover letter",
        "Ask before booking flights",
        "[m:f-home]",
    ):
        assert needle in text
    about = {r.id: r.version for r in view.sections["about_you"]}
    assert (
        about["f-home"]
        == conn.execute("SELECT version FROM records WHERE id = 'f-home'").fetchone()[0]
    )
    assert {"e-dima", "f-dima-job"} <= {r.id for r in view.sections["people"]}
    assert 0 < view.tokens <= core_view.BUDGET_TOKENS
    served = {r[0] for r in conn.execute("SELECT item_id FROM served WHERE via = 'core'")}
    assert {"f-home", "p-tea", "e-dima", "l-cv", "les-1", "w-1"} <= served


def test_rebuild_is_byte_identical(conn):
    """Cache stability: an unchanged ledger rebuilds the same bytes and writes nothing."""
    _seed(conn)
    first = api.core(thread_id="t1", session_id="s1", store=conn).text
    mv = store.memory_version(conn)
    rows = conn.execute("SELECT id, rev FROM derived ORDER BY id").fetchall()
    assert {r[0] for r in rows} >= {"core:about_you", "core:people", "core:working_now"}
    again = api.core(thread_id="t2", session_id="s2", store=conn).text
    assert again == first
    assert core_view.build(conn)["about_you"] in first
    assert store.memory_version(conn) == mv
    assert conn.execute("SELECT id, rev FROM derived ORDER BY id").fetchall() == rows
    deps = conn.execute(
        "SELECT item_id FROM read_deps WHERE holder_kind = 'derived' AND holder_id = 'core:about_you'"
    ).fetchall()
    assert {"f-home", "p-tea"} <= {r[0] for r in deps}


def test_policy_neutral_turn(conn):
    """§10.1, §5.5: no sensitive items except the user's own; others' precise places stay out."""
    _seed(conn)
    mine = fact(
        "f-allergy", "self", "allergic_to", "nuts", text="Allergic to nuts", labels=Label.HEALTH
    )
    _ok(add(conn, mine))
    theirs = fact(
        "f-dima-ill",
        "e-dima",
        "allergic_to",
        "penicillin",
        text="Dima is allergic to penicillin",
        labels=Label.HEALTH,
        tier=AuthorTier.OTHER,
        author_entity_id="e-dima",
        speaker_entity_id="e-dima",
    )
    _ok(add(conn, theirs))
    text = api.core(thread_id="t1", session_id="s1", store=conn).text
    # the user's own health item no longer rides on every turn (the basket answers a direct
    # question); others' sensitive items never enter, as before
    assert "Allergic to nuts" not in text and "Lives in Moscow" in text
    assert "penicillin" not in text and "f-dima-ill" not in text


def test_frozen_entity_left_out(conn):
    """A frozen entity and its records are not in the core while frozen."""
    _seed(conn)
    hot.freeze(conn, ["e-dima"], cues=["test"])
    text = api.core(thread_id="t1", session_id="s1", store=conn).text
    assert "Yandex" not in text and "[m:e-dima]" not in text
    assert "Lives in Moscow" in text


def test_people_stable_id_hints_only(conn):
    """§10.1: stable-id hints, never the raw ids of others' accounts."""
    tg = StableId(kind="tg_user_id", value="987654321", tier=AuthorTier.USER)
    _ok(add(conn, entity("e-ilya", "Ilya", stable_ids=[tg])))
    text = api.core(thread_id="t1", session_id="s1", store=conn).text
    assert "Ilya" in text and "reach: tg_user_id" in text
    assert "987654321" not in text


def test_token_budget_enforced(conn):
    """~6k tokens: a large ledger is trimmed per section and in total."""
    _seed(conn)
    for i in range(300):
        long = f"Fact number {i} about the owner with a fairly long English description " * 3
        _ok(add(conn, pref(f"p-{i:03d}", "self", "prefers", f"thing {i}", text=long)))
    view = api.core(thread_id="t1", session_id="s1", store=conn)
    assert view.tokens <= core_view.BUDGET_TOKENS
    assert core_view.tokens(view.text) <= core_view.BUDGET_TOKENS
    assert "Send the CV to Dima" in view.text  # later sections keep their own budget


def test_windows_and_lesson_rank(conn):
    """Working now: loops due within 7 days (overdue kept); Lessons: top 10 by helpful − harmful."""
    for lid, due in (("l-soon", _day(2)), ("l-late", _day(20)), ("l-over", _day(-1))):
        loop = Loop(
            id=lid,
            owner="user",
            text_en=f"Loop {lid}",
            due=due,
            author_tier=AuthorTier.USER,
            recorded_at=ts(),
            origin_quote=f"напомни {lid}",
        )
        _ok(add(conn, loop))
    for i in range(12):
        lesson = Lesson(
            id=f"les-{i:02d}",
            rule_en=f"Lesson {i:02d}",
            helpful=i,
            harmful=0 if i else 5,
            author_tier=AuthorTier.USER,
            recorded_at=ts(),
            origin_quote=f"урок {i}",
        )
        _ok(add(conn, lesson))
    text = api.core(thread_id="t1", session_id="s1", store=conn).text
    assert "Loop l-soon" in text and "Loop l-late" not in text
    assert "Loop l-over" in text and "(overdue)" in text
    assert "Lesson 11" in text and "Lesson 00" not in text and "Lesson 01" not in text
    assert text.index("Lesson 11") < text.index("Lesson 02")


def test_what_changed_last(conn):
    """§10.1: What changed recently (7 days, one line each), the last section."""
    _seed(conn)
    _supersede(
        conn, "f-home", fact("f-home2", "self", "lives_in", "Berlin", text="Lives in Berlin")
    )
    text = api.core(thread_id="t1", session_id="s1", store=conn).text
    tail = text[text.index("## What changed recently") :]
    assert text.index("## Lessons") < text.index("## What changed recently")
    assert 'now "Lives in Berlin"' in tail and "Moscow" not in tail  # never the old value
    about = text[: text.index("## People")]
    assert "Lives in Berlin" in about and "Moscow" not in about
    later = core_view.compute(conn, today=_clock.now().date() + timedelta(days=9))
    assert not [s for s in later if s.name == "what_changed" and s.lines]


def test_diff_on_resume(conn):
    """§10.1: a resumed session with a changed core gets the diff (≤ 1k tokens) or a restart."""
    _seed(conn)
    api.core(thread_id="thread-1", session_id="s1", store=conn)
    same = api.core_diff(thread_id="thread-1", session_id="s1", store=conn)
    assert same.block is None and not same.restart
    _supersede(
        conn, "f-home", fact("f-home2", "self", "lives_in", "Berlin", text="Lives in Berlin")
    )
    upd = api.core_diff(thread_id="thread-1", session_id="s1", store=conn)
    assert upd.block is not None and not upd.restart
    assert upd.block.startswith(core_view.UPDATE_OPEN)
    assert '[m:f-home] outdated: now "Lives in Berlin"' in upd.block
    assert "Moscow" not in upd.block  # the handle names it, never the value
    assert core_view.tokens(upd.block) <= core_view.DIFF_TOKENS
    assert "f-home" in upd.item_ids
    # delivered: the session now holds the new lines
    again = api.core_diff(thread_id="thread-1", session_id="s1", store=conn)
    assert again.block is None and not again.restart
    for i in range(120):
        text = f"New fact {i} about the owner, long enough to count for something real"
        _ok(add(conn, pref(f"p-n{i:03d}", "self", "prefers", f"new {i}", text=text)))
    big = api.core_diff(thread_id="thread-1", session_id="s1", store=conn)
    assert big.restart and big.block is None


def test_forgotten_never_restated(conn):
    """§7.6: forgotten ids leave the core and never get an update line."""
    _seed(conn)
    api.core(thread_id="thread-1", session_id="s1", store=conn)
    words = TurnCtx(
        turn_id="turn-f",
        thread_id="thread-1",
        session_id="s1",
        owner_words="забудь про зелёный чай",
        message_id="msg-f",
    )
    res = api.forget("m:p-tea", turn=words, because_quote="зелёный чай", store=conn)
    assert res.outcome == "stopped"
    text = api.core(thread_id="thread-2", session_id="s2", store=conn).text
    assert "green tea" not in text
    upd = api.core_diff(thread_id="thread-1", session_id="s1", store=conn)
    assert "p-tea" not in (upd.block or "") and "green tea" not in (upd.block or "")


def test_correction_rebuilds_before_next_turn(conn):
    """G9: a fix rebuilds the core (hot hook); the ledger's rebuild job also heals a section."""
    _seed(conn)
    api.core(thread_id="t1", session_id="s1", store=conn)
    _ok(commit(conn, op(OpKind.RETRACT, target="f-home", group="f-home")))
    row = conn.execute("SELECT valid FROM derived WHERE id = 'core:about_you'").fetchone()
    assert row["valid"] == 0
    hot._core_changed(conn, ["f-home"])
    row = conn.execute("SELECT valid, content FROM derived WHERE id = 'core:about_you'").fetchone()
    assert row["valid"] == 1 and "Moscow" not in row["content"]
    # the queued rebuild job runs through the registered runner and is a no-op now
    assert store.run_queued_jobs(conn, core_view.REBUILD_JOB) >= 1
    _ok(add(conn, pref("p-coffee", "self", "prefers", "coffee", text="Prefers coffee")))
    api.core(thread_id="t1", session_id="s1", store=conn)  # rebuilt fresh, holds coffee
    _ok(commit(conn, op(OpKind.RETRACT, target="p-tea", group="p-tea")))
    assert conn.execute("SELECT valid FROM derived WHERE id = 'core:about_you'").fetchone()[0] == 0
    store.run_queued_jobs(conn, core_view.REBUILD_JOB)
    row = conn.execute("SELECT valid, content FROM derived WHERE id = 'core:about_you'").fetchone()
    assert row["valid"] == 1 and "green tea" not in row["content"] and "coffee" in row["content"]
    # a section that became empty keeps no stale row
    _ok(commit(conn, op(OpKind.RETRACT, target="p-coffee", group="p-coffee")))
    store.run_queued_jobs(conn, core_view.REBUILD_JOB)
    assert conn.execute("SELECT 1 FROM derived WHERE id = 'core:about_you'").fetchone() is None
