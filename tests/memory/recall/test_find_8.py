"""Finder misses covered by the memory recall tests.

- a question about the past gets the superseded value of the asked relation as a line marked
  past (G3: never as current); the read set holds the user's source words, not the superseded
  record, so the pre-send check still refuses the record as current; a retracted value
  (proven wrong) is never shown, and a question about now gets no past line;
- a person the question resolves to brings the other entities under the same full name
  (the user's nickname for a chat member minted twice); a bare first name joins nobody;
- a move in effect answers «where does she live now»; a move ahead does not;
- a pet word with no has-pet fact reaches the user's pets; a pet's vet visit is no person's
  health record, while a person's health item keeps the policy's topic test.

Synthetic data only; names and wordings differ from the dev probes on purpose."""

from __future__ import annotations

import pytest

from koob.memory import policy, store
from koob.memory.model import AuthorTier, Label, ReadRef, ReadSrc, TurnCtx, TurnOrigin
from koob.memory.recall import basket, cue, vectors

from .._ledger import add, entity, fact, open_seeded, raw

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str) -> TurnCtx:
    return TurnCtx(
        turn_id="turn-1", thread_id="thread-1", session_id="session-1",
        origin=TurnOrigin.OWNER, owner_words=words, message_id="msg-1", clock=CLOCK,
    )  # fmt: skip


def _retire(conn, rid: str, status: str, when: str) -> None:
    conn.execute("UPDATE records SET status = ?, retired_at = ? WHERE id = ?", (status, when, rid))
    conn.commit()


def _moved(conn):
    raw(conn, "r-old", "Живём пока в Кунцево, у парка", ts_="2025-11-02T10:00:00Z")
    raw(conn, "r-new", "Всё, перебрались в Липки", ts_="2026-06-01T10:00:00Z")
    assert add(
        conn,
        fact("f-old", "self", "lives_in", "Кунцево", text="The owner lives in Kuntsevo.",
             quote_orig="Живём пока в Кунцево", source_raw_id="r-old",
             labels=Label.LOCATION_PRECISE),
    ).ok  # fmt: skip
    assert add(
        conn,
        fact("f-new", "self", "lives_in", "Липки", text="The owner lives in Lipki.",
             quote_orig="перебрались в Липки", source_raw_id="r-new", valid_from="2026-06-01",
             labels=Label.LOCATION_PRECISE),
    ).ok  # fmt: skip
    _retire(conn, "f-old", "superseded", "2026-06-01T10:00:00.000000Z")


@pytest.mark.parametrize("words", ["Где мы раньше жили?", "What was my address before the move?"])
def test_past_question_gets_the_superseded_value_marked_past(conn, words):
    _moved(conn)
    b = basket.assemble(conn, t(words))
    assert b.block and "Kuntsevo" in b.block, (words, b.block)
    line = next(i for i in b.items if "Kuntsevo" in i.text)
    assert line.text.startswith("в прошлом, до 2026-06-01"), line.text
    # the read set carries the user's words then, never the superseded record as current
    assert all(r.id != "f-old" for r in b.read_set)
    assert store.pre_send_check(conn, thread_id="thread-1", read_set=b.read_set).ok
    bad = [ReadRef(id="f-old", version=1, src=ReadSrc.RECORD)]
    assert not store.pre_send_check(conn, thread_id="thread-1", read_set=bad).ok


def test_question_about_now_gets_no_past_line(conn):
    _moved(conn)
    b = basket.assemble(conn, t("Где мы сейчас живём?"))
    assert "Kuntsevo" not in (b.block or "") and "Кунцево" not in (b.block or "")


def test_retracted_value_is_never_a_past_line(conn):
    _moved(conn)
    _retire(conn, "f-old", "retracted", "2026-06-01T10:00:00.000000Z")
    b = basket.assemble(conn, t("Где мы раньше жили?"))
    assert "Kuntsevo" not in (b.block or "")


def test_past_period_keeps_only_the_values_that_overlap_it(conn):
    _moved(conn)
    assert "Kuntsevo" in (basket.assemble(conn, t("Где мы жили в январе?")).block or "")
    assert "Kuntsevo" not in (basket.assemble(conn, t("Где мы жили в августе?")).block or "")


def _verdicts(conn, ids: list[str], words: str) -> dict[str, policy.Use]:
    turn = t(words)
    rows = basket._hydrate(conn, ids, CLOCK[:10])
    c = cue.parse(turn, now=CLOCK, conn=conn)
    verdicts = policy.decide_many(conn, ids, turn)
    basket._recheck_targets(conn, turn, verdicts, c, rows, CLOCK[:10])
    return verdicts


def _alias(conn, eid: str, alias: str) -> None:
    from koob.memory import entities
    from koob.memory.model import AliasKind

    assert entities.put_alias(
        conn, CLOCK, eid, alias, AliasKind.NICKNAME, tier=AuthorTier.USER, confirmed=True
    )
    conn.commit()


def test_nickname_reaches_the_same_full_name_minted_twice(conn):
    assert add(conn, entity("e-his", "Лёва Гришин")).ok
    _alias(conn, "e-his", "Гриня")
    assert add(conn, entity("e-chat", "Лёва Гришин", tier=AuthorTier.OTHER)).ok
    assert add(conn, entity("e-other", "Лёва Петров", tier=AuthorTier.OTHER)).ok
    assert add(
        conn,
        fact("f-oslo", "e-chat", "travels_to", "Oslo", text="Lyova Grishin is flying to Oslo.",
             quote_orig="лечу в Осло на конференцию", tier=AuthorTier.OTHER,
             author_entity_id="e-chat", labels=Label.THIRD_PARTY_PRIVATE),
    ).ok  # fmt: skip
    c = cue.parse(t("Куда Гриня летит?"), now=CLOCK, conn=conn)
    assert "e-chat" in c.entity_ids and "e-other" not in c.entity_ids, c.entity_ids
    assert _verdicts(conn, ["f-oslo"], "Куда Гриня летит?")["f-oslo"].allowed


def test_bare_first_name_joins_nobody(conn):
    assert add(conn, entity("e-short", "Лёва")).ok
    assert add(conn, entity("e-full", "Лёва Гришин", tier=AuthorTier.OTHER)).ok
    assert cue._same_full_name(conn, {"e-short": 1.0}) == {}


def _vera(conn, valid_from=None):
    assert add(conn, entity("e-vera", "Вера Зотова")).ok
    assert add(
        conn,
        fact("f-move", "e-vera", "moves_to", "Тверь", text="Vera Zotova is moving to Tver.",
             quote_orig="переезжаю в Тверь", tier=AuthorTier.OTHER, author_entity_id="e-vera",
             valid_from=valid_from, labels=Label.THIRD_PARTY_PRIVATE),
    ).ok  # fmt: skip


def test_move_in_effect_answers_where_she_lives_now(conn):
    _vera(conn)
    assert _verdicts(conn, ["f-move"], "Где теперь живёт Вера?")["f-move"].allowed


def test_move_ahead_does_not_answer_where_she_lives_now(conn):
    _vera(conn, valid_from="2027-01-10")
    assert not _verdicts(conn, ["f-move"], "Где теперь живёт Вера?")["f-move"].allowed


def _vet(conn, subject: str, name: str, etype: str):
    assert add(conn, entity(subject, name, etype=etype)).ok
    assert add(
        conn,
        fact(f"f-{subject}", subject, "other", "3 ноября", text="A vet check on 3 November.",
             quote_orig="приём 3 ноября", labels=Label.HEALTH),
    ).ok  # fmt: skip


def test_pet_word_reaches_his_pet_and_its_vet_visit(conn):
    _vet(conn, "e-pet", "Пушок", "pet")
    c = cue.parse(t("Псу когда надо к врачу?"), now=CLOCK, conn=conn)
    assert "e-pet" in c.entity_ids
    b = basket.assemble(conn, t("Псу когда надо к врачу?"))
    assert "3 November" in (b.block or ""), b.block


def test_a_persons_health_item_keeps_its_topic_test(conn):
    _vet(conn, "e-man", "Олег", "person")
    turn = t("Что там у Олега на неделе?")
    rows = basket._hydrate(conn, ["f-e-man"], CLOCK[:10])
    c = cue.parse(turn, now=CLOCK, conn=conn)
    v = policy.decide_many(conn, ["f-e-man"], turn)
    basket._recheck(rows, v, {"f-e-man": {"names"}}, c, [], set(), basket._pets(conn, rows))
    assert not v["f-e-man"].allowed
