"""Another person's fact reached by its relation, in any language.

- an open-topic item (third-party private, unknown) of a person the question names is asked
  for when the question asks that fact's relation («куда летит» → travels_to), so a Russian
  question meets an English record; a question word alone («где») asks it only when the fact
  holds in the asked period (none named: on the turn's day);
- a relation word is no topic word for health, money or minors; another relation, or a place
  named without its person, unlocks nothing;
- a bare month name a few months ahead also gives its coming span (not in the past tense);
- the topic overlap of another person's item excludes the names of the people the question
  names, not a place's or an organisation's alias made from a plain word.

Synthetic data only; names and wordings differ from the dev probes on purpose."""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import AuthorTier, FKind, Label, TurnCtx, TurnOrigin
from koob.memory.recall import basket, cue, vectors

from .._ledger import add, entity, fact, open_seeded

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


def _verdicts(conn, ids: list[str], words: str) -> dict[str, policy.Use]:
    turn = t(words)
    rows = basket._hydrate(conn, ids, CLOCK[:10])
    c = cue.parse(turn, now=CLOCK, conn=conn)
    verdicts = policy.decide_many(conn, ids, turn)
    basket._recheck_targets(conn, turn, verdicts, c, rows, CLOCK[:10])
    return verdicts


def _nika(conn, *, labels=Label.THIRD_PARTY_PRIVATE):
    assert add(conn, entity("e-nika", "Ника Орлова")).ok
    assert add(conn, entity("e-lis", "Lisbon", etype="place")).ok
    assert add(
        conn,
        fact("f-trip", "e-nika", "travels_to", "Lisbon", text="Nika Orlova is flying to Lisbon.",
             keywords_en="Lisbon, trip, cousins", quote_orig="off to Lisbon to see my cousins",
             value_entity_id="e-lis", tier=AuthorTier.OTHER, author_entity_id="e-nika",
             fkind=FKind.PLAN, valid_from="2026-12-04", valid_to="2026-12-11", labels=labels),
    ).ok  # fmt: skip


@pytest.mark.parametrize(
    "words",
    ["Куда Ника улетает в декабре?", "Что у Орловой с поездкой?", "where is Nika flying?"],
)
def test_relation_word_asks_another_persons_fact_in_any_language(conn, words):
    _nika(conn)
    v = _verdicts(conn, ["f-trip"], words)["f-trip"]
    assert v.allowed, (words, v)


def test_question_word_asks_it_only_in_its_period(conn):
    _nika(conn)
    assert _verdicts(conn, ["f-trip"], "Где Ника будет в декабре?")["f-trip"].allowed
    # no period named: «where is she now», and a December trip is not now
    assert not _verdicts(conn, ["f-trip"], "В каком городе теперь Ника?")["f-trip"].allowed


def test_another_relation_or_a_bare_name_unlocks_nothing(conn):
    _nika(conn)
    for words in ("Где живёт Ника?", "Как там Ника?", "Что нового у Ники?"):
        assert not _verdicts(conn, ["f-trip"], words)["f-trip"].allowed, words
    # the place named without its person: the question is not about her
    assert not _verdicts(conn, ["f-trip"], "Что там в Lisbon?")["f-trip"].allowed


def test_relation_word_is_no_topic_word_for_health(conn):
    _nika(conn, labels=Label.THIRD_PARTY_PRIVATE | Label.HEALTH)
    assert not _verdicts(conn, ["f-trip"], "Куда Ника улетает в декабре?")["f-trip"].allowed


def test_month_ahead_gives_its_coming_span_unless_past():
    now = "2026-10-12T09:00:00Z"
    assert cue.parse_dates("в декабре", now=now, ahead=True) == [
        "2025-12-01..2025-12-31",
        "2026-12-01..2026-12-31",
    ]
    assert cue.parse_dates("в январе", now=now, ahead=True)[-1] == "2027-01-01..2027-01-31"
    assert cue.parse_dates("в декабре", now=now) == ["2025-12-01..2025-12-31"]
    assert cue.parse_dates("в марте", now=now, ahead=True) == ["2026-03-01..2026-03-31"]
    c = cue.parse(t("Где мы были в декабре?"), now=now)
    assert c.past and c.dates == ["2025-12-01..2025-12-31"]


def _alias(conn, eid: str, alias: str) -> None:
    conn.execute(
        "INSERT INTO aliases (id, entity_id, alias, alias_norm, script, kind, tier, confirmed,"
        " created_at, updated_at) VALUES (?, ?, ?, ?, 'cyrillic', 'name', 'other', 0, ?, ?)",
        (f"a-{eid}-{alias}", eid, alias, alias.casefold(), CLOCK, CLOCK),
    )
    conn.commit()


def _gleb(conn):
    assert add(conn, entity("e-gleb", "Глеб")).ok
    assert add(conn, entity("e-band", "Кино", etype="org")).ok
    _alias(conn, "e-band", "абонемент")  # an extractor's alias made from a plain word
    assert add(
        conn,
        fact("f-pass", "e-gleb", "other", "Кино, абонемент", text="Gleb bought a season pass.",
             keywords_ru="абонемент, концерты", quote_orig="Купил абонемент на Кино!",
             value_entity_id="e-band", tier=AuthorTier.OTHER, author_entity_id="e-gleb",
             labels=Label.UNKNOWN_SENSITIVE),
    ).ok  # fmt: skip


def test_named_person_keeps_the_items_other_words_as_its_topic(conn):
    _gleb(conn)
    for words in ("Глеб писал про абонемент?", "Что там Глеб с абонементом?"):
        v = policy.decide_targeted(conn, ["f-pass"], t(words), ["e-gleb"])["f-pass"]
        assert v.allowed and v.reason == "direct_question", words


def test_a_name_alone_still_unlocks_nothing(conn):
    _gleb(conn)
    assert not policy.decide_targeted(conn, ["f-pass"], t("Как там Глеб?"), ["e-gleb"])[
        "f-pass"
    ].allowed
    # the organisation named without the person: every subject name stays out of the topic
    for words in ("Как там Кино?", "Что с абонементом?"):
        out = policy.decide_targeted(conn, ["f-pass"], t(words), ["e-band"])
        assert not out["f-pass"].allowed, words
