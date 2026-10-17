"""Use now for another person's item whose only label is the classifier's uncertain band.

Final labels that are only `unknown_sensitive` mean the item is not known to be sensitive: it is
ranked by relevance like an unlabeled one, with no name or direct-question test. A confident
label, alone or beside the uncertain bit, keeps the direct question; so does a label still
pending. Send turns still hold the uncertain item.
"""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import AuthorTier, Label, TurnCtx, TurnOrigin
from koob.memory.recall import vectors

from .._ledger import add, entity, fact, open_seeded

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str, **kw) -> TurnCtx:
    return TurnCtx(
        turn_id="turn-1",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=CLOCK,
        **kw,
    )


def _denis_fact(conn, labels: Label, *, final: bool = True) -> None:
    assert add(conn, entity("e-denis", "Denis Smith")).ok
    assert add(
        conn,
        fact(
            "f-loan",
            "e-denis",
            "other",
            "a loan for the bakery",
            text="Denis Smith took a loan to open a bakery",
            quote_orig="взял кредит, открываю пекарню",
            tier=AuthorTier.OTHER,
            author_entity_id="e-denis",
            labels=labels,
            labels_final=final,
        ),
    ).ok


def _use(conn, words: str, **kw) -> policy.Use:
    return policy.decide_many(conn, ["f-loan"], t(words, **kw))["f-loan"]


def test_uncertain_only_fact_of_another_person_ranks_without_a_name(conn):
    _denis_fact(conn, Label.UNKNOWN_SENSITIVE)
    use = _use(conn, "what was that about opening a bakery?")
    assert use.allowed and use.reason == "uncertain_ranked"


def test_uncertain_only_fact_that_passes_the_name_and_topic_test_is_a_direct_question(conn):
    """The name and topic test passes: the item is allowed as a direct question, not as an
    uncertain item ranked by relevance."""
    _denis_fact(conn, Label.UNKNOWN_SENSITIVE)
    use = _use(conn, "did Denis Smith open the bakery?")
    assert use.allowed and use.reason == "direct_question"


@pytest.mark.parametrize("label", [Label.HEALTH, Label.MONEY])
def test_confident_label_of_another_person_still_needs_a_direct_question(conn, label):
    _denis_fact(conn, label)
    assert not _use(conn, "what was that about opening a bakery?").allowed
    assert _use(conn, "how is Denis Smith doing?").reason == "name_only"


def test_money_fact_of_another_person_opens_on_a_direct_question(conn):
    _denis_fact(conn, Label.MONEY)
    use = _use(conn, "did Denis Smith borrow money?")
    assert use.allowed and use.reason == "direct_question"


@pytest.mark.parametrize("label", [Label.HEALTH, Label.MONEY, Label.THIRD_PARTY_PRIVATE])
def test_uncertain_bit_beside_a_confident_label_stays_protected(conn, label):
    _denis_fact(conn, label | Label.UNKNOWN_SENSITIVE)
    use = _use(conn, "what was that about opening a bakery?")
    assert not use.allowed and use.reason == "not_asked"


def test_pending_uncertain_label_keeps_the_direct_question(conn):
    _denis_fact(conn, Label.UNKNOWN_SENSITIVE, final=False)
    assert not _use(conn, "what was that about opening a bakery?").allowed


def test_uncertain_only_fact_is_held_on_a_send_turn_his_words_do_not_ask_for(conn):
    _denis_fact(conn, Label.UNKNOWN_SENSITIVE)
    use = _use(conn, "send the team the weekly update", sends=True)
    assert use.allowed and use.hold
    assert not _use(conn, "send the team a note about the bakery", sends=True).hold
