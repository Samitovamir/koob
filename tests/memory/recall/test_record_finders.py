"""Record finders: a stored record the question means reaches the basket.

- the meaning lane counts the top of its ranking as evidence above a lower cosine floor, so a
  paraphrase in the other language (cosines run lower across a language pair) is not lost;
  a hit far below the lane's best, or under the lower floor, stays weak;
- a person named by a short or affectionate form of the name, or by a word the user's own
  facts call them, resolves to the stored person, so their records are found;
- the meaning lane gets enough time to answer on a loaded machine and the basket stays
  within its latency budget.

Synthetic data only (an invented persona and contacts)."""

from __future__ import annotations

import numpy as np
import pytest

from koob.memory import api
from koob.memory.model import TurnCtx
from koob.memory.recall import basket, vectors

from .._ledger import add, entity, fact, open_seeded

CLOCK = "2026-10-12T09:00:00.000000Z"
DIM = 8


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()
    vectors.set_embedder(None)
    vectors.reset()


def _axis(k: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[k] = 1.0
    return v


def _at(cos: float, k: int) -> np.ndarray:
    """A unit vector at cosine `cos` to axis 0, leaning towards axis `k`."""
    return cos * _axis(0) + float(np.sqrt(1 - cos * cos)) * _axis(k)


class _FixedEmbedder:
    """Every question lands on axis 0; the records' vectors are set by the test."""

    id = "fixed@1"
    dim = DIM

    def embed_docs(self, texts):
        return np.zeros((len(texts), DIM), np.float32)

    def embed_query(self, text):
        return _axis(0)


def _index(vecs: dict[str, np.ndarray]) -> None:
    ids = list(vecs)
    vectors._index[0] = vectors._Index(
        _FixedEmbedder.id,
        DIM,
        ids,
        np.stack([vecs[i] for i in ids]).astype(np.float16),
        np.zeros(len(ids), np.int64),
        np.ones(len(ids), bool),
    )
    vectors.set_embedder(_FixedEmbedder())


def _ask(conn, words: str, n: int = 1) -> list[str]:
    turn = TurnCtx(
        turn_id=f"t-{n}", thread_id=f"th-{n}", session_id=f"s-{n}", owner_words=words, clock=CLOCK
    )
    return [i.id for i in api.recall(turn, store=conn).items]


# ---------------------------------------------------------------- the meaning lane


def test_vector_evidence_counts_the_top_of_the_lane_above_a_lower_floor():
    hits = [("a", 0.47), ("b", 0.44), ("c", 0.30)]
    assert basket._vector_evidence(hits) == {"a", "b"}


def test_vector_evidence_keeps_the_absolute_floor_anywhere_in_the_lane():
    hits = [("a", 0.62), ("b", 0.50), ("c", 0.58)]
    assert basket._vector_evidence(hits) == {"a", "c"}


def test_vector_evidence_leaves_weak_a_hit_far_below_the_best_or_under_the_lower_floor():
    assert basket._vector_evidence([("a", 0.52), ("b", 0.43)]) == {"a"}
    assert basket._vector_evidence([("a", 0.36), ("b", 0.20)]) == set()
    assert basket._vector_evidence([]) == set()


def test_cross_language_paraphrase_below_the_old_floor_is_found_when_it_ranks_first(
    conn, monkeypatch
):
    """A Russian question, an English record with no shared word: the meaning lane ranks the
    record first at a cosine under the absolute floor; it is served. With the top-of-lane rule
    off and the lane's evidence floor above its cosine, it is no evidence and the basket drops it."""
    assert add(conn, entity("e-marta", "Marta Lind")).ok
    assert add(
        conn,
        fact(
            "f-pool",
            "self",
            "attends",
            "Aquadrome",
            text="Denis Smith swims laps at the Aquadrome before work",
        ),
    ).ok
    assert add(
        conn,
        fact("f-marta", "e-marta", "works_at", "Nordwind", text="Marta Lind works at Nordwind"),
    ).ok
    _index({"f-pool": _at(0.46, 1), "f-marta": _at(0.21, 2)})
    question = "Куда я хожу плавать перед работой?"
    assert "f-pool" in _ask(conn, question, 1)
    assert "f-marta" not in _ask(conn, question, 2)
    monkeypatch.setattr(basket, "VEC_TOP", 0)
    monkeypatch.setattr(basket, "VEC_EVIDENCE", 0.5)
    assert "f-pool" not in _ask(conn, question, 3)


def test_vector_deadline_leaves_room_for_a_loaded_machine():
    """The query embedding takes ~18 ms on an idle machine and two to four times that under
    load; the lane runs beside the other finders, so its deadline covers the loaded case."""
    assert basket.FINDER_DEADLINES_MS["vector"] >= 70


# ---------------------------------------------------------------- people by another form


def test_possessive_form_of_a_short_name_finds_the_persons_record(conn):
    """«Тонина поездка», «у Тониной сестры»: a possessive adjective of a short name resolves to
    the stored person, so their record is found by the names lane."""
    assert add(conn, entity("e-tonya", "Тоня Белкина")).ok
    assert add(
        conn,
        fact(
            "f-trip",
            "e-tonya",
            "plans_trip_to",
            "Tver",
            text="Tonya Belkina is going to Tver in March",
        ),
    ).ok
    from koob.memory.recall import cue

    c = cue.parse(
        TurnCtx(turn_id="t-c", thread_id="th-c", owner_words="Когда Тонина поездка?"),
        now=CLOCK,
        conn=conn,
    )
    assert "e-tonya" in c.entity_ids
    assert "f-trip" in _ask(conn, "Когда Тонина поездка?", 4)
    assert "f-trip" in _ask(conn, "что слышно о Тониной поездке", 5)


def test_a_word_ending_like_a_possessive_names_nobody_without_a_stored_name(conn):
    assert add(conn, entity("e-tonya", "Тоня Белкина")).ok
    from koob.memory.recall import cue

    c = cue.parse(
        TurnCtx(turn_id="t-d", thread_id="th-d", owner_words="во сколько закрывается магазин?"),
        now=CLOCK,
        conn=conn,
    )
    assert c.entity_ids == []
