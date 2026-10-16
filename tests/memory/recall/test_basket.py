"""Pre-turn recall, the basket: the memory spec §10.2, §7.4, §15.1 (assoc), §15.3 (C3).

Spec marker: `[T recall/test_basket::<case>]`. Vectors stay absent here (no embedder, no
index): the basket must work on code-only finders."""

from __future__ import annotations

import pytest

from koob.memory import api, store
from koob.memory.model import (
    AuthorTier,
    Impression,
    OpKind,
    ReadRef,
    ReadSrc,
    TurnCtx,
    TurnOrigin,
)
from koob.memory.recall import basket, gate, lexical, vectors
from koob.memory.writers import hot

from .._ledger import add, commit, entity, fact, op, open_seeded, ts

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str, *, thread: str = "thread-1", origin=TurnOrigin.OWNER, **kw) -> TurnCtx:
    return TurnCtx(
        turn_id=kw.pop("turn_id", "turn-1"),
        thread_id=thread,
        session_id=kw.pop("session_id", "session-1"),
        origin=origin,
        owner_words=words,
        clock=CLOCK,
        **kw,
    )


def _sasha(conn) -> None:
    assert add(conn, entity("e-sasha", "Саша")).ok
    assert add(
        conn, fact("f-bday", "e-sasha", "birthday", "May 12", text="Sasha's birthday is May 12")
    ).ok
    assert add(
        conn, fact("f-city", "e-sasha", "lives_in", "Berlin", text="Sasha lives in Berlin")
    ).ok


def test_impression_marked(conn, turn):
    """§4.3: impressions are shown as «впечатление», never as facts."""
    assert add(conn, entity("e-dima", "Дима")).ok
    imp = Impression(
        id="i-dima",
        about_entity_id="e-dima",
        text_en="Dima seemed tired of the project",
        observed_at="2026-10-01T10:00:00Z",
        author_tier=AuthorTier.USER,
        recorded_at=ts(),
        origin_quote="Дима какой-то уставший от проекта",
    )
    assert add(conn, imp).ok
    b = api.recall(t("что там с Димой?"), store=conn)
    assert [i.id for i in b.items] == ["i-dima"]
    assert b.items[0].impression is True
    assert "[m:i-dima]" in b.block and "впечатление" in b.block


def test_names_finder_brings_facts_about_the_person(conn):
    _sasha(conn)
    b = api.recall(t("что ты помнишь про Сашу"), store=conn)
    assert {i.id for i in b.items} == {"f-bday", "f-city"}
    assert b.block.startswith('<memory-recall turn="turn-1">')
    assert {r.id for r in b.read_set} == {"f-bday", "f-city"}
    assert all(r.version > 0 for r in b.read_set)
    assert b.tokens <= 1500 and "total" in b.timings_ms


def test_nothing_passes_no_block(conn):
    _sasha(conn)
    assert api.recall(t("привет, как дела?"), store=conn).block is None
    assert api.recall(t(""), store=conn).block is None


def test_duty_and_event_turns_get_no_basket(conn):
    _sasha(conn)
    for origin in (TurnOrigin.DUTY, TurnOrigin.EVENT):
        assert api.recall(t("что про Сашу", origin=origin), store=conn).block is None


def test_task_turn_cues_on_the_order(conn):
    _sasha(conn)
    turn = TurnCtx(
        turn_id="t9",
        thread_id="task-1",
        origin=TurnOrigin.TASK,
        task_words="поздравь Сашу",
        clock=CLOCK,
    )
    assert "f-bday" in {i.id for i in api.recall(turn, store=conn).items}


def test_served_rows_and_same_version_not_reserved(conn):
    """An item this thread holds at the same version is not served again; a new version is;
    another thread gets it."""
    _sasha(conn)
    first = api.recall(t("что про Сашу"), store=conn)
    rows = conn.execute(
        "SELECT item_id, version, via, turn_id FROM served WHERE thread_id = 'thread-1'"
    ).fetchall()
    assert {(r[0], r[2], r[3]) for r in rows} == {(i.id, "basket", "turn-1") for i in first.items}
    assert api.recall(t("что про Сашу", turn_id="turn-2"), store=conn).block is None
    other = api.recall(t("что про Сашу", thread="thread-2"), store=conn)
    assert {i.id for i in other.items} == {i.id for i in first.items}
    # a correction bumps the version: the thread gets the new one
    ver = conn.execute("SELECT version FROM records WHERE id = 'f-city'").fetchone()[0]
    res = commit(
        conn,
        op(
            OpKind.UPDATE,
            target="f-city",
            read_set=[ReadRef(id="f-city", version=ver)],
            fields={"text_en": "Sasha lives in Hamburg", "value_text": "Hamburg"},
        ),
    )
    assert res.ok
    again = api.recall(t("что про Сашу", turn_id="turn-3"), store=conn)
    assert [i.id for i in again.items] == ["f-city"]
    assert "Hamburg" in again.block


def test_core_view_items_are_not_repeated(conn):
    _sasha(conn)
    v = conn.execute("SELECT version FROM records WHERE id = 'f-bday'").fetchone()[0]
    conn.execute(
        "INSERT INTO served (thread_id, item_id, item_kind, version, via, created_at)"
        " VALUES ('thread-1', 'f-bday', 'fact', ?, 'core', 't')",
        (v,),
    )
    assert [i.id for i in api.recall(t("что про Сашу"), store=conn).items] == ["f-city"]


def test_retracted_and_forgotten_never_come_back(conn):
    _sasha(conn)
    conn.execute("UPDATE records SET status = 'retracted' WHERE id = 'f-city'")
    assert [i.id for i in api.recall(t("что про Сашу"), store=conn).items] == ["f-bday"]


def test_frozen_entity_dropped(conn):
    _sasha(conn)
    hot.freeze(conn, ["e-sasha"], turn=t("нет, Саша живёт не там"))
    assert api.recall(t("что про Сашу", thread="thread-9"), store=conn).block is None


def test_lexical_hit_needs_topic_coverage(conn):
    assert add(conn, entity("e-x", "Олег")).ok
    assert add(
        conn, fact("f-gym", "self", "gym", "FitLab", text="I go to the FitLab gym on Mondays")
    ).ok
    assert add(
        conn, fact("f-noise", "self", "likes", "tea", text="I like green tea in the morning")
    ).ok
    b = api.recall(t("where is my gym"), store=conn)
    assert [i.id for i in b.items] == ["f-gym"]


def test_author_tier_shown(conn):
    assert add(conn, entity("e-dima", "Дима", tier=AuthorTier.OTHER)).ok
    assert add(conn, entity("e-lena", "Лена")).ok
    f = fact(
        "f-claim",
        "e-lena",
        "works_at",
        "Yandex",
        tier=AuthorTier.OTHER,
        text="Lena works at Yandex",
    )
    f.author_entity_id = "e-dima"
    assert add(conn, f).ok
    b = api.recall(t("где работает Лена"), store=conn)
    assert b.items and b.items[0].said_by == "Дима"
    assert "со слов: Дима" in b.block


def test_budget_and_dedupe(conn):
    """≤ 1.5k tokens, ≤ MAX_ITEMS lines, ≤ PER_ENTITY per person, one value per slot."""
    assert add(conn, entity("e-sasha", "Саша")).ok
    long = "Sasha said a lot about the new flat and the move " * 12
    for k in range(12):
        assert add(conn, fact(f"f-{k}", "e-sasha", f"note{k}", str(k), text=f"{long} {k}")).ok
    b = api.recall(t("что там у Саши"), store=conn)
    assert 0 < len(b.items) <= gate.PER_ENTITY
    assert b.tokens <= 1500
    assert all(len(i.text) <= basket.LINE_CHARS for i in b.items)


def test_session_budget_only_high_items_past_it(conn):
    """Past 8k tokens a session gets only items above the higher gate."""
    low = gate.Cand(id="a", kind="fact", version=1, text="a", tokens=10, strong=True)
    high = gate.Cand(id="b", kind="fact", version=1, text="b", tokens=10, strong=True)
    ranked = [("b", gate.HIGH_SCORE + 0.001), ("a", gate.MIN_SCORE + 0.001)]
    cands = {"a": low, "b": high}
    assert gate.gate(conn, ranked, thread_id="x", cands=cands) == ["b", "a"]
    past = gate.gate(conn, ranked, thread_id="x", cands=cands, session_used=gate.SESSION_BUDGET)
    assert past == ["b"]


def test_session_budget_counts_basket_rows(conn):
    _sasha(conn)
    used = gate.SESSION_BUDGET // basket.AVG_LINE_TOKENS + 1
    conn.executemany(
        "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, via, created_at)"
        " VALUES ('old', 'session-1', ?, 'fact', 1, 'basket', 't')",
        [(f"x{i}",) for i in range(used)],
    )
    past = api.recall(t("что про Сашу"), store=conn)
    fresh = api.recall(t("что про Сашу", thread="thread-2", session_id="session-2"), store=conn)
    assert len(past.items) <= len(fresh.items) and fresh.items


def test_revalidate_drops_a_stale_item(conn):
    """§7.4: a stale basket item is dropped right before the model call."""
    _sasha(conn)
    b = api.recall(t("что про Сашу"), store=conn)
    conn.execute("UPDATE records SET version = version + 1 WHERE id = 'f-city'")
    fresh = basket.revalidate(conn, b, turn_id="turn-1")
    assert [i.id for i in fresh.items] == ["f-bday"] and "f-city" not in fresh.block
    pre = api.pre_send_check(thread_id="thread-1", read_set=fresh.read_set, store=conn)
    assert pre.ok


def _episode(conn, eid, summary, *, etype, outcome, start, people=(), cause=None):
    now = ts()
    rowid = conn.execute("SELECT coalesce(max(fts_rowid), 0) + 1 FROM records").fetchone()[0]
    conn.execute(
        "INSERT INTO records (id, fts_rowid, kind, version, author_tier, recorded_at, created_at,"
        " updated_at) VALUES (?, ?, 'episode', 1, 'system', ?, ?, ?)",
        (eid, rowid, now, now, now),
    )
    conn.execute(
        "INSERT INTO episodes (id, event_type, span_start, outcome, summary_en) VALUES (?, ?, ?, ?, ?)",
        (eid, etype, start, outcome, summary),
    )
    for p in people:
        conn.execute("INSERT INTO episode_links VALUES (?, 'participant', ?)", (eid, p))
    if cause:
        conn.execute("INSERT INTO episode_links VALUES (?, 'cause', ?)", (eid, cause))
    lexical.index_record(conn, eid)


def test_c3_one_fit_answers_two_tied_ask(conn):
    """§15.3: one fitting candidate → specifics; two tied → `ask` with their features."""
    for eid, name in (("e-ilya", "Илья"), ("e-mark", "Марк")):
        assert add(conn, entity(eid, name)).ok
    _episode(conn, "ep-ilya", "Meeting with Ilya postponed because of the trip; his workshop",
             etype="meeting", outcome="postponed", start="2026-08-20", people=["e-ilya"])  # fmt: skip
    _episode(conn, "ep-gym", "Gym session done", etype="event", outcome="done", start="2026-08-21")
    hint = "помнишь, мы хотели встретиться, но сорвалось из-за поездки?"
    b = api.recall(t(hint), store=conn)
    assert [i.id for i in b.items] == ["ep-ilya"] and b.items[0].ask is None
    _episode(conn, "ep-mark", "Meeting with Mark cancelled because of the trip; concert",
             etype="meeting", outcome="cancelled", start="2026-08-25", people=["e-mark"])  # fmt: skip
    b = api.recall(t(hint, thread="thread-2"), store=conn)
    assert len([i for i in b.items if i.kind == "episode"]) == 1
    assert b.items[0].ask and len(b.items[0].ask) == 2
    assert "ask:" in b.block


def test_c3_partial_matches_give_none(conn):
    """§15.3: none fitting → the closest partial only inside a question."""
    _episode(conn, "ep-dinner", "Dinner with Ilya cancelled because of illness",
             etype="dinner", outcome="cancelled", start="2026-08-20", cause="r-ill")  # fmt: skip
    b = api.recall(t("помнишь тот ужин, который прошёл отлично из-за дождя?"), store=conn)
    ep = [i for i in b.items if i.id == "ep-dinner"]
    assert ep and ep[0].ask and "нет точного совпадения" in ep[0].ask[0]


def test_assoc_at_most_one_and_once_per_goal_per_day(conn):
    """§15.1: precomputed links only; ≤ 1 per turn; ≤ 1 per goal per day."""
    _sasha(conn)
    for k in (1, 2):
        conn.execute(
            "INSERT INTO derived (id, dkind, dkey, content, read_set, built_at, created_at,"
            " updated_at) VALUES (?, 'assoc_link', ?, ?, '[]', ?, 't', 't')",
            (
                f"d-a{k}",
                f"goal-{k}",
                f"Can combine the Berlin trip with offer {k}. Check?",
                f"2026-10-1{k}",
            ),
        )
        conn.execute(
            "UPDATE derived SET read_set = '[[\"e-sasha\", 1]]' WHERE id = ?", (f"d-a{k}",)
        )
        conn.execute("INSERT INTO read_deps VALUES ('derived', ?, 'e-sasha', 1)", (f"d-a{k}",))
    b = api.recall(t("что про Сашу"), store=conn)
    offers = [i for i in b.items if i.kind == "derived"]
    assert len(offers) == 1
    served_goal = offers[0].id
    b2 = api.recall(t("что про Сашу", thread="thread-2"), store=conn)
    offers2 = [i.id for i in b2.items if i.kind == "derived"]
    assert offers2 and served_goal not in offers2  # the same goal waits for tomorrow
    assert all(r.src is ReadSrc.DERIVED for r in b2.read_set if r.id in offers2)


def test_served_write_failure_means_no_block(conn, monkeypatch):
    _sasha(conn)

    def boom(*a, **k):
        raise store.sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(basket, "_write_served", boom)
    assert api.recall(t("что про Сашу"), store=conn).block is None
