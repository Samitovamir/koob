"""Ordering by fit: a question that asks about something of a person's gets the fused order
weighed by how close each candidate is to the question, while a question about the newest
state («сейчас», "now") keeps the fused order and a single-valued slot always leads with
its newest value. Another person's message that names
the person asked fuses like a record about them."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from koob.memory import api, forget
from koob.memory.model import ReadSrc, TurnCtx, TurnOrigin
from koob.memory.recall import basket as B
from koob.memory.recall import cue as cue_mod
from koob.memory.recall import lexical, vectors
from koob.memory.recall.gate import Cand

from .._ledger import add, entity, fact, open_seeded

CLOCK = "2026-10-12T23:00:00.000000Z"


def t(words: str, *, n: int = 1) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}", thread_id=f"thread-{n}", session_id=f"session-{n}",
        origin=TurnOrigin.OWNER, owner_words=words, clock=CLOCK,
    )  # fmt: skip


def _cue(words: str) -> cue_mod.Cue:
    return cue_mod.parse(t(words), now=CLOCK)


@pytest.mark.parametrize(
    "words",
    ["Where does Denis live now?", "Где сейчас живёт Денис?", "В каком городе теперь Денис?",
     "What is Denis's latest job?", "Is Denis still in Berlin?"],
)  # fmt: skip
def test_a_question_about_the_newest_state_is_marked_current(words):
    assert _cue(words).current


@pytest.mark.parametrize(
    "words",
    ["What did Denis say about the trip?", "Что Денис рассказывал про поездку?",
     "When did Denis buy the boat?"],
)  # fmt: skip
def test_a_question_about_a_past_thing_is_not_marked_current(words):
    assert not _cue(words).current


def test_the_topic_is_what_the_question_asks_beyond_the_names():
    words = "What did Denis Smith say about the sailing trip?"
    cue = cue_mod.Cue(entity_ids=["e-denis"])
    topic = B._asked_topic(cue, lexical.query_stems(words), lexical.name_stems(words))
    assert {lexical.stem("sailing"), lexical.stem("trip")} <= topic
    assert lexical.stem("Denis") not in topic


@pytest.mark.parametrize(
    "words", ["What do you remember about Denis?", "Что ты помнишь про Дениса?",
              "Расскажи всё, что знаешь о Денисе."],
)  # fmt: skip
def test_a_question_that_only_asks_to_recall_a_person_has_no_topic(words):
    cue = cue_mod.Cue(entity_ids=["e-denis"])
    assert not B._asked_topic(cue, lexical.query_stems(words), lexical.name_stems(words))


def test_no_topic_without_a_person_or_for_the_newest_state():
    stems = lexical.query_stems("where does Denis keep the sailing boat now")
    assert not B._asked_topic(cue_mod.Cue(), stems, set())
    assert not B._asked_topic(cue_mod.Cue(entity_ids=["e-denis"], current=True), stems, set())


def _cand(i: str, slot: str | None = None) -> Cand:
    return Cand(id=i, kind="fact", version=1, text=i, slot=slot)


def _rows(*ids: str, raws: tuple[str, ...] = ()) -> dict:
    """Rows in recording order: the last id is the newest."""
    return {
        i: {"src": ReadSrc.RAW if i in raws else ReadSrc.RECORD,
            "recorded_at": f"2026-10-0{k + 1}T10:00:00Z"}
        for k, i in enumerate(ids)
    }  # fmt: skip


def _cos(monkeypatch, cos: dict[str, float]) -> None:
    monkeypatch.setattr(vectors, "cosines", lambda q, ids: {i: cos[i] for i in ids if i in cos})


def test_the_fused_order_weighs_closeness_and_keeps_its_scores(monkeypatch):
    _cos(monkeypatch, {"new": 0.1, "mid": 0.3, "old": 0.9})
    ranked = [("new", 0.050), ("mid", 0.045), ("old", 0.030)]
    rows = _rows("old", "mid", "new")
    out = B._by_fit(ranked, rows, {i: _cand(i) for i in rows}, [1.0])
    assert out == [("old", 0.050), ("mid", 0.045), ("new", 0.030)]


def test_a_raw_and_a_record_are_each_measured_from_their_own_best(monkeypatch):
    _cos(monkeypatch, {"rec2": 0.5, "raw": 0.45, "rec": 0.8})
    ranked = [("rec2", 0.050), ("raw", 0.040), ("rec", 0.030)]
    rows = _rows("rec2", "raw", "rec", raws=("raw",))
    out = B._by_fit(ranked, rows, {i: _cand(i) for i in rows}, [1.0])
    assert [i for i, _ in out] == ["raw", "rec", "rec2"]


def test_a_single_valued_slot_still_leads_with_its_newest_value(monkeypatch):
    _cos(monkeypatch, {"hobby": 0.2, "city-new": 0.2, "city-old": 0.9})
    ranked = [("hobby", 0.050), ("city-new", 0.040), ("city-old", 0.030)]
    rows = _rows("city-old", "hobby", "city-new")
    cands = {
        "city-new": _cand("city-new", "e-denis|lives_in"),
        "hobby": _cand("hobby"),
        "city-old": _cand("city-old", "e-denis|lives_in"),
    }
    out = B._by_fit(ranked, rows, cands, [1.0])
    assert out == [("city-new", 0.050), ("hobby", 0.040), ("city-old", 0.030)]


def test_a_candidate_without_a_vector_yet_counts_as_its_kinds_best(monkeypatch):
    _cos(monkeypatch, {"a": 0.4, "b": 0.6})
    ranked = [("a", 0.050), ("b", 0.045), ("fresh", 0.030)]
    rows = _rows("a", "b", "fresh")
    out = B._by_fit(ranked, rows, {i: _cand(i) for i in rows}, [1.0])
    assert [i for i, _ in out] == ["b", "fresh", "a"]


def test_without_a_query_vector_the_fused_order_stays(monkeypatch):
    _cos(monkeypatch, {"new": 0.1, "old": 0.9})
    ranked = [("new", 0.05), ("old", 0.03)]
    assert B._by_fit(ranked, _rows("old", "new"), {}, []) == ranked


def test_the_fused_order_re_deals_a_slot_newest_first_and_keeps_its_scores():
    order = [("city-old", 0.05), ("hobby", 0.04), ("city-new", 0.03)]
    cands = {
        "city-old": _cand("city-old", "e-denis|lives_in"),
        "hobby": _cand("hobby"),
        "city-new": _cand("city-new", "e-denis|lives_in"),
    }
    out = B._newest_in_slot(order, cands, {"city-new": 0, "hobby": 1, "city-old": 2})
    assert out == [("city-new", 0.05), ("hobby", 0.04), ("city-old", 0.03)]


def _raw_row(text: str, tier: str, author: str = "Mark") -> dict:
    return {"id": "r1", "src": ReadSrc.RAW, "text": text, "author_tier": tier,
            "author_name": author}  # fmt: skip


def test_another_persons_message_naming_the_person_asked_is_about_them():
    names = lexical.name_stems("Что у тебя есть про Дениса?")
    assert B._names_asked(_raw_row("Слышал от Дениса, что он едет в Тверь.", "other"), names)
    assert not B._names_asked(_raw_row("Слышал от Дениса, что он едет.", "user"), names)
    assert not B._names_asked(_raw_row("Едет в Тверь.", "other", author="Денис"), names)
    record = {**_raw_row("Денис едет в Тверь.", "other"), "src": ReadSrc.RECORD}
    assert not B._names_asked(record, names)


# ---------------------------------------------------------------- end to end

_CONCEPTS = {"sail": "boat", "boat": "boat", "yacht": "boat"}


class ConceptEmbedder:
    """A tiny concept space: only the words above carry meaning."""

    id = "concept@rel1"
    dim = 32

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"\w+", (text or "").casefold()):
            for stem, c in _CONCEPTS.items():
                if tok.startswith(stem):
                    h = int.from_bytes(hashlib.blake2b(c.encode(), digest_size=8).digest(), "little")
                    v[h % self.dim] += 1.0
        v[-1] += 0.01  # no zero vector
        return v

    def embed_docs(self, texts):
        return np.stack([self._one(x) for x in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text):
        return self._one(text)


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(ConceptEmbedder())
    forget.reset_wiring()
    forget.use_key(None)
    forget.wire(stop_hooks=[vectors.stop_hook])
    c = open_seeded(tmp_path)
    path = tmp_path / "vectors.db"
    vectors.attach(path)
    yield c, path
    c.close()
    forget.reset_wiring()
    forget.use_key(None)
    vectors.reset()
    vectors.set_embedder(None)


_NEWER = ["owns a red bicycle", "works at a bakery", "plays the cello", "collects stamps",
          "grows tomatoes", "reads crime novels", "drinks green tea", "knits scarves",
          "keeps two cats", "bakes rye bread", "runs on Sundays", "paints watercolours"]  # fmt: skip


def test_an_older_item_close_to_the_question_comes_before_the_persons_newer_ones(conn):
    c, path = conn
    add(c, entity("e-denis", "Denis Smith"))
    add(c, fact("f-boat", "e-denis", "likes", "yacht",
                text="Denis Smith took his yacht out on the lake"))  # fmt: skip
    for n, what in enumerate(_NEWER):
        add(c, fact(f"f-new{n}", "e-denis", "likes", what, text=f"Denis Smith {what}"))
    vdb = vectors.open_db(path)
    try:
        vectors.embed_pending(c, vdb)
    finally:
        vdb.close()
    ids = [i.id for i in api.recall(t("Has Denis ever tried sailing?"), store=c).items]
    assert ids and ids.index("f-boat") <= 2  # by recency alone it would come last


def test_the_top_hit_of_a_text_lane_fits_by_its_words_despite_a_low_cosine(monkeypatch):
    _cos(monkeypatch, {"close": 0.70, "reply": 0.55})
    ranked = [("close", 0.030), ("reply", 0.025)]
    rows = _rows("close", "reply", raws=("close", "reply"))
    cands = {i: _cand(i) for i in rows}
    lanes = {"raw": [("reply", 12.0), ("close", 3.0)]}
    assert [i for i, _ in B._by_fit(ranked, rows, cands, [1.0])] == ["close", "reply"]
    assert [i for i, _ in B._by_fit(ranked, rows, cands, [1.0], lanes)] == ["reply", "close"]


def test_a_text_hit_well_below_its_lane_best_gains_nothing():
    lanes = {"lexical": [("a", 10.0), ("b", 7.9)], "raw": [("c", 4.0), ("a", 3.9)]}
    fit = B._text_fit(lanes)
    assert fit["a"] == pytest.approx(1.0)
    assert "b" not in fit
    assert fit["c"] == pytest.approx(1.0)
