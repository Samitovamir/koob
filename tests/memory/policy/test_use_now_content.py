"""Direct question by content: another person's message with a confident label opens when their
words name whom it is about and ask its own words, whether the classifier or a topic word of
the message set the label.

The guarantee stays: an indirect question, a send turn, a question about someone else or about
another matter, a question with one content word, a label still pending and a derived record
all keep the topic test (§5.5).
"""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import AuthorTier, Label, TurnCtx, TurnOrigin
from koob.memory.recall import basket, lexical, vectors
from koob.memory.recall import cue as cue_mod

from .._ledger import add, entity, fact, open_seeded, raw

CLOCK = "2026-10-12T09:00:00.000000Z"
CLASS_TEXT = "Last Saturday I went to a pottery class with two friends from work, so relaxing."
ASK = "When did Denis Smith go to a pottery class with friends?"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    assert add(c, entity("e-denis", "Denis Smith")).ok
    assert add(c, entity("e-mark", "Mark Brown")).ok
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


def _message(conn, rid: str, text: str, labels: Label, *, final: bool = True, by="e-denis"):
    raw(conn, rid, text, tier="other", labels=int(labels))
    conn.execute(
        "UPDATE raw_chunks SET author_entity_id = ?, labels_final = ?, fts_rowid = rowid"
        " WHERE id = ?",
        (by, int(final), rid),
    )
    lexical.index_raw(conn, rid)


def _use(conn, rid: str, words: str, **kw) -> policy.Use:
    return policy.decide_many(conn, [rid], t(words, **kw))[rid]


def test_question_naming_the_sender_and_asking_the_message_words_opens_it(conn):
    _message(conn, "r-class", CLASS_TEXT, Label.MINOR)
    use = _use(conn, "r-class", ASK)
    assert use.allowed and use.reason == "direct_content"


def test_indirect_question_about_the_sender_stays_closed(conn):
    _message(conn, "r-class", CLASS_TEXT, Label.MINOR)
    for words in (
        "What is new with Denis Smith?",
        "How was the week for Denis Smith?",
        "Denis Smith",
    ):
        assert not _use(conn, "r-class", words).allowed, words


def test_send_turn_never_opens_a_message_by_its_words(conn):
    _message(conn, "r-class", CLASS_TEXT, Label.MINOR)
    assert not _use(conn, "r-class", ASK, sends=True).allowed
    assert not _use(
        conn, "r-class", "Send Denis Smith a note on the pottery class with friends"
    ).allowed


def test_question_about_another_person_stays_closed(conn):
    _message(conn, "r-class", CLASS_TEXT, Label.MINOR)
    use = _use(conn, "r-class", "When did Mark Brown go to a pottery class with friends?")
    assert not use.allowed and use.reason == "not_asked"


def test_question_about_another_matter_stays_closed(conn):
    _message(conn, "r-class", CLASS_TEXT, Label.MINOR)
    for words in ("When did Denis Smith go to a cooking class?", "Where does Denis Smith work?"):
        use = _use(conn, "r-class", words)
        assert not use.allowed and use.reason == "name_only", words


def test_label_the_message_words_carry_opens_when_asked_its_words(conn):
    text = "After the pottery class with friends I went to see my doctor about the pain."
    _message(conn, "r-doctor", text, Label.HEALTH)
    use = _use(conn, "r-doctor", ASK)
    assert use.allowed and use.reason == "direct_content"
    for words in ("How is Denis Smith feeling?", "Where does Denis Smith work?"):
        assert not _use(conn, "r-doctor", words).allowed, words


def test_money_amount_opens_only_for_a_question_asking_the_message(conn):
    _message(conn, "r-paid", "The pottery class with friends was $120 for the month.", Label.MONEY)
    assert _use(conn, "r-paid", ASK).reason == "direct_content"
    assert not _use(conn, "r-paid", "What does Denis Smith spend on hobbies?").allowed
    assert not _use(conn, "r-paid", ASK, sends=True).allowed


def test_acronym_in_the_question_counts_as_content(conn):
    _message(conn, "r-conf", "Two days ago I went to an LGBTQ conference downtown.", Label.INTIMATE)
    use = _use(conn, "r-conf", "When did Denis Smith go to the LGBTQ conference?")
    assert use.allowed and use.reason == "direct_content"
    for words in ("Is Denis Smith LGBTQ?", "What conference did Mark Brown attend?"):
        assert not _use(conn, "r-conf", words).allowed, words


def test_pending_label_keeps_the_topic_test(conn):
    _message(conn, "r-class", CLASS_TEXT, Label.MINOR, final=False)
    assert not _use(conn, "r-class", ASK).allowed


def test_derived_record_keeps_the_topic_test(conn):
    assert add(
        conn,
        fact(
            "f-class",
            "e-denis",
            "other",
            "a pottery class with friends",
            text="Denis Smith went to a pottery class with friends",
            quote_orig=CLASS_TEXT,
            tier=AuthorTier.OTHER,
            author_entity_id="e-denis",
            labels=Label.MINOR,
            labels_final=True,
        ),
    ).ok
    assert not _use(conn, "f-class", ASK).allowed


def test_other_names_in_the_question_are_no_content(conn):
    _message(conn, "r-thanks", "Thanks, Mark! The pottery was great.", Label.INTIMATE)
    assert not _use(conn, "r-thanks", "Did Denis Smith thank Mark for the class?").allowed


def _cue(**kw) -> cue_mod.Cue:
    return cue_mod.Cue.model_construct(**{"entity_ids": [], "send_turn": False, **kw})


def test_raw_lane_brings_fenced_messages_only_for_a_named_question(conn):
    _message(conn, "r-class", CLASS_TEXT, Label.MINOR)
    words = "pottery class friends"
    mask = int(policy.label_mask(t(ASK)))
    assert "r-class" not in dict(basket._raw_lane(conn, words, mask, _cue()))
    named = basket._raw_lane(conn, words, mask, _cue(entity_ids=["e-denis"]))
    assert "r-class" in dict(named)
    sending = basket._raw_lane(conn, words, mask, _cue(entity_ids=["e-denis"], send_turn=True))
    assert "r-class" not in dict(sending)
