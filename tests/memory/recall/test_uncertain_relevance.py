"""Relevance for another person's item that use-now ranks under the uncertain band alone.

Such an item answers a question that names nobody, or one that names whom it is about; a
question about one person does not get another person's uncertain fact or message.
"""

from __future__ import annotations

from koob.memory import policy
from koob.memory.model import ReadSrc
from koob.memory.recall import basket, lexical
from koob.memory.recall import cue as cue_mod

RANKED = policy.Use(True, "uncertain_ranked")


def _rows() -> dict[str, dict]:
    return {
        "f-anna": {"src": ReadSrc.RECORD, "subject_id": "e-anna"},
        "f-denis": {"src": ReadSrc.RECORD, "subject_id": "e-denis"},
        "r-anna": {"src": ReadSrc.RAW, "author_entity_id": "e-anna", "text": "moving in August"},
        "r-note": {"src": ReadSrc.RAW, "author_entity_id": "e-anna", "text": "Denis flies Friday"},
    }


def _run(words: str, targets: list[str]) -> dict[str, policy.Use]:
    verdicts = {i: RANKED for i in _rows()}
    cue = cue_mod.Cue(words=words, entity_ids=targets)
    basket._off_target(_rows(), verdicts, cue, lexical.name_stems(words))
    return verdicts


def test_uncertain_items_about_someone_else_stay_out_of_a_question_naming_a_person():
    v = _run("Where is Denis flying in November?", ["e-denis"])
    assert v["f-denis"].allowed and v["r-note"].allowed
    assert v["f-anna"].reason == "other_subject" and v["r-anna"].reason == "other_subject"


def test_uncertain_items_rank_freely_when_the_question_names_nobody():
    v = _run("who is moving in August?", [])
    assert all(u.allowed for u in v.values())


def test_named_items_and_other_verdicts_are_untouched():
    rows = _rows()
    verdicts = {"f-anna": policy.Use(True, "not_sensitive")}
    basket._off_target(rows, verdicts, cue_mod.Cue(words="x", entity_ids=["e-denis"]), set())
    assert verdicts["f-anna"].allowed


def test_a_diminutive_names_the_person_it_matches_best():
    """«Denny» is one person's alias and only a looser form of another's name: the looser
    match is not whom the question asks about."""
    rows = _rows()
    verdicts = {i: RANKED for i in rows}
    cue = cue_mod.Cue(
        words="where is Denny flying?",
        entity_ids=["e-denis", "e-anna"],
        name_scores={"e-denis": 1.0, "e-anna": 0.8},
    )
    basket._off_target(rows, verdicts, cue, set())
    assert verdicts["f-denis"].allowed and not verdicts["f-anna"].allowed


def test_a_looser_match_still_answers_when_the_question_asks_its_relation():
    rows = {
        "f-trip": {"src": ReadSrc.RECORD, "subject_id": "e-anna", "kind": "fact",
                   "relation": "travels_to"},
        "f-club": {"src": ReadSrc.RECORD, "subject_id": "e-anna", "kind": "fact",
                   "relation": "attends"},
    }  # fmt: skip
    verdicts = {i: RANKED for i in rows}
    cue = cue_mod.Cue(
        words="where is Denny flying?",
        entity_ids=["e-denis", "e-anna"],
        name_scores={"e-denis": 1.0, "e-anna": 0.8},
        relations=["travels_to"],
    )
    basket._off_target(rows, verdicts, cue, set(), "2026-10-12")
    assert verdicts["f-trip"].allowed and not verdicts["f-club"].allowed


def test_own_raw_whose_dated_facts_lie_outside_the_asked_period_is_not_asked():
    row = {"src": ReadSrc.RAW, "author_tier": "user", "text": "С сентября живу в общежитии",
           "labels": 0, "labels_final": True}  # fmt: skip
    learned = {"subjects": {"self"}, "relations": {"lives_in"}, "words": ["dormitory"],
               "values": ["общежитие"], "periods": [("2026-09-01", None)]}  # fmt: skip
    words = "где я жила в феврале, в общежитии?"
    stems = lexical.query_stems(words)
    feb = cue_mod.Cue(words=words, dates=["2026-02-01..2026-02-28"], self_ref=True)
    assert not basket._raw_asked(row, feb, stems, set(), learned)
    autumn = cue_mod.Cue(words=words, dates=["2026-10-01..2026-10-31"], self_ref=True)
    assert basket._raw_asked(row, autumn, stems, set(), learned)


def _namesake_rows() -> dict[str, dict]:
    return {
        "f-trip": {"src": ReadSrc.RECORD, "subject_id": "e-alex-gray", "kind": "fact",
                   "relation": "travels_to", "value_type": "place",
                   "valid_from": "2026-11-03"},
        "f-move": {"src": ReadSrc.RECORD, "subject_id": "e-alex-stone", "kind": "fact",
                   "relation": "moves_to", "value_type": "place",
                   "valid_from": "2026-08-01"},
        "f-denis": {"src": ReadSrc.RECORD, "subject_id": "e-denis"},
    }  # fmt: skip


def _namesake_cue(words: str, **kw) -> cue_mod.Cue:
    return cue_mod.Cue(
        words=words,
        entity_ids=["e-alex-gray", "e-alex-stone"],
        name_scores={"e-alex-gray": 0.9, "e-alex-stone": 0.9},
        namesakes={"e-alex-gray": "Alex Gray", "e-alex-stone": "Alex Stone"},
        **kw,
    )


def test_a_name_two_namesakes_share_keeps_only_facts_whose_relation_it_asks():
    """«Al» fits Alex Gray and Alex Stone equally: Gray's November trip does not answer
    where an Alex lives now, Stone's move made in August does."""
    rows = _namesake_rows()
    verdicts = {i: RANKED for i in rows}
    cue = _namesake_cue("which city does Al live in now?", value_types=["place"])
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01")
    assert verdicts["f-move"].allowed
    assert verdicts["f-trip"].reason == "namesake_unasked"
    assert verdicts["f-denis"].reason == "other_subject"


def test_a_namesake_fact_answers_when_the_question_asks_its_relation():
    rows = _namesake_rows()
    verdicts = {i: RANKED for i in rows}
    cue = _namesake_cue("where is Al flying in November?", relations=["travels_to"])
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01")
    assert verdicts["f-trip"].allowed and not verdicts["f-move"].allowed


def test_namesakes_told_apart_by_the_question_keep_the_named_rule():
    """Once a surname picks one Alex, the cue holds no namesakes and the person it names
    keeps every ranked fact about them."""
    rows = _namesake_rows()
    verdicts = {i: RANKED for i in rows}
    cue = cue_mod.Cue(
        words="what about Alex Gray?", entity_ids=["e-alex-gray"], name_scores={"e-alex-gray": 1.0}
    )
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01")
    assert verdicts["f-trip"].allowed and not verdicts["f-move"].allowed


def test_a_question_naming_two_different_people_is_not_a_namesake_tie():
    rows = _namesake_rows()
    verdicts = {i: RANKED for i in rows}
    cue = cue_mod.Cue(
        words="what about Alex Gray and Denis?",
        entity_ids=["e-alex-gray", "e-denis"],
        name_scores={"e-alex-gray": 1.0, "e-denis": 1.0},
    )
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01")
    assert verdicts["f-trip"].allowed and verdicts["f-denis"].allowed


def _tied_raws() -> dict[str, dict]:
    return {
        "r-gray": {"src": ReadSrc.RAW, "author_entity_id": "e-alex-gray",
                   "text": "Out Nov 3-7, flying to Lisbon."},
        "r-stone": {"src": ReadSrc.RAW, "author_entity_id": "e-alex-stone",
                    "text": "Moved to Porto last month!"},
    }  # fmt: skip


def _learned_from(monkeypatch) -> None:
    facts = {
        "r-gray": {"facts": [{"kind": "fact", "relation": "travels_to", "value_type": "place",
                              "valid_from": "2026-11-03", "subject_id": "e-alex-gray"}]},
        "r-stone": {"facts": [{"kind": "fact", "relation": "moves_to", "value_type": "place",
                               "valid_from": "2026-09-01", "subject_id": "e-alex-stone"}]},
    }  # fmt: skip
    monkeypatch.setattr(basket, "_raw_learned", lambda conn, ids: {i: facts[i] for i in ids})


def test_a_message_by_a_tied_namesake_answers_only_when_its_fact_asks_the_relation(monkeypatch):
    """Al fits both Alexes: Gray's message about a November trip does not answer where an
    Alex lives now; Stone's message about the move made does."""
    _learned_from(monkeypatch)
    rows = _tied_raws()
    verdicts = {i: RANKED for i in rows}
    cue = _namesake_cue("which city does Al live in now?", relations=["lives_in"])
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01", conn=object())
    assert verdicts["r-stone"].allowed
    assert verdicts["r-gray"].reason == "namesake_unasked"


def test_a_tied_namesake_message_answers_the_question_that_asks_its_relation(monkeypatch):
    _learned_from(monkeypatch)
    rows = _tied_raws()
    verdicts = {i: RANKED for i in rows}
    cue = _namesake_cue("where is Al flying in November?", relations=["travels_to"])
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01", conn=object())
    assert verdicts["r-gray"].allowed and not verdicts["r-stone"].allowed


def test_a_tied_namesake_message_with_nothing_learned_stays_out(monkeypatch):
    monkeypatch.setattr(basket, "_raw_learned", lambda conn, ids: {})
    rows = _tied_raws()
    verdicts = {i: RANKED for i in rows}
    cue = _namesake_cue("where is Al flying in November?", relations=["travels_to"])
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01", conn=object())
    assert {v.reason for v in verdicts.values()} == {"namesake_unasked"}


def test_an_episode_about_someone_else_stays_out_of_a_question_naming_a_person(monkeypatch):
    """An episode has no subject of its own: whom it is about are its participants."""
    monkeypatch.setattr(
        basket,
        "_episode_people",
        lambda conn, ids: {"ep-anna": {"e-anna"}, "ep-denis": {"e-denis"}},
    )
    rows = {
        "ep-anna": {"src": ReadSrc.RECORD, "kind": "episode"},
        "ep-denis": {"src": ReadSrc.RECORD, "kind": "episode"},
        "ep-none": {"src": ReadSrc.RECORD, "kind": "episode"},
    }
    verdicts = {i: RANKED for i in rows}
    cue = cue_mod.Cue(words="is Denis moving?", entity_ids=["e-denis"])
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01", conn=object())
    assert verdicts["ep-denis"].allowed and verdicts["ep-none"].allowed
    assert verdicts["ep-anna"].reason == "other_subject"


def test_a_message_by_the_one_person_named_keeps_the_named_rule():
    rows = _tied_raws()
    verdicts = {i: RANKED for i in rows}
    cue = cue_mod.Cue(
        words="what about Alex Gray?", entity_ids=["e-alex-gray"], name_scores={"e-alex-gray": 1.0}
    )
    basket._off_target(rows, verdicts, cue, set(), "2026-10-01")
    assert verdicts["r-gray"].allowed and verdicts["r-stone"].reason == "other_subject"
