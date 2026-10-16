"""Basket speed: the memory spec §10.2, G9.

Spec marker: `[T perf/test_basket::p95_with_embedding]`: the whole pre-turn recall (cue
parse, finders, use-now, fusion, gate, served rows, pre-send check) ≤ 150 ms p95 on a 10^5
record store, query embedding included. The embedder is simulated at the measured cost of
embeddinggemma-300m (§10.3: 17.6 ms p95 per query) over a 10^5 × 768 fp16 matrix."""

from __future__ import annotations

import random
import time

import numpy as np
import pytest

from koob.memory import api
from koob.memory.model import TurnCtx
from koob.memory.recall import lexical, vectors

from .._ledger import open_seeded

N_RECORDS = 100_000
N_ENTITIES = 2_000
N_EPISODES = 3_000
DIM = 768
EMBED_S = 0.0176
T = "2026-10-01T10:00:00.000000Z"
NAMES = ["Саша", "Дима", "Илья", "Марк", "Лена", "Оля", "Nina", "Paul", "Катя", "Артём"]
TOPIC_EN = [
    "trip",
    "berlin",
    "exhibition",
    "gym",
    "coffee",
    "flat",
    "move",
    "exam",
    "course",
    "invoice",
    "hotel",
    "flight",
]
TOPIC_RU = [
    "поездка",
    "берлин",
    "выставка",
    "зал",
    "кофе",
    "квартира",
    "переезд",
    "экзамен",
    "курс",
    "счёт",
    "отель",
    "рейс",
]


class _SimEmbedder:
    id = "perf@1"
    dim = DIM

    def __init__(self) -> None:
        self.rng = np.random.default_rng(1)

    def embed_docs(self, texts):
        return self.rng.standard_normal((len(texts), DIM)).astype(np.float32)

    def embed_query(self, text):
        time.sleep(EMBED_S)  # ONNX releases the GIL like this does
        return self.rng.standard_normal(DIM).astype(np.float32)


def _seed(conn) -> None:
    rnd = random.Random(11)
    syll = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
    vocab = list({"".join(rnd.choice(syll) for _ in range(3)) for _ in range(9000)})[:8000]
    conn.execute("BEGIN")
    rowid = conn.execute("SELECT coalesce(max(fts_rowid), 0) + 1 FROM records").fetchone()[0]
    abase = conn.execute("SELECT coalesce(max(fts_rowid), 0) + 1 FROM aliases").fetchone()[0]
    ents = []
    for i in range(N_ENTITIES):
        name = NAMES[i] if i < len(NAMES) else "".join(rnd.choice(syll) for _ in range(2)).title()
        eid = f"e{i:05d}"
        ents.append(eid)
        conn.execute(
            "INSERT INTO records (id, fts_rowid, kind, version, author_tier, recorded_at,"
            " created_at, updated_at) VALUES (?, ?, 'entity', 1, 'user', ?, ?, ?)",
            (eid, rowid, T, T, T),
        )
        conn.execute(
            "INSERT INTO entities (id, etype, display_name) VALUES (?, 'person', ?)", (eid, name)
        )
        conn.execute(
            "INSERT INTO aliases (id, fts_rowid, entity_id, alias, alias_norm, script, kind, tier,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'cyrl', 'name', 'user', ?, ?)",
            (f"a{i}", abase + i, eid, name, name.casefold(), T, T),
        )
        conn.execute("INSERT INTO fts_names (rowid, name) VALUES (?, ?)", (abase + i, name))
        rowid += 1
    rec, facts, fts = [], [], []
    for i in range(N_RECORDS):
        rid = f"r{i:06d}"
        k = rnd.randrange(len(TOPIC_EN))
        text = " ".join(
            [rnd.choice(vocab) for _ in range(10)] + [TOPIC_EN[k], rnd.choice(TOPIC_EN)]
        )
        kw = " ".join([rnd.choice(vocab) for _ in range(3)] + [TOPIC_RU[k]])
        day = f"2026-{rnd.randrange(1, 11):02d}-{rnd.randrange(1, 28):02d}T10:00:00.000000Z"
        rec.append(
            (rid, rowid, "fact", 1 + i % 3, rnd.randrange(4), rnd.choice(("user", "other")), day)
        )
        facts.append((rid, rnd.choice(ents[:200] if i % 3 else ents), f"rel{i % 50}", text, kw))
        fts.append((rowid, lexical.stem(text), lexical.stem(kw)))
        rowid += 1
    conn.executemany(
        "INSERT INTO records (id, fts_rowid, kind, version, labels, author_tier, recorded_at,"
        " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 't', 't')",
        rec,
    )
    conn.executemany(
        "INSERT INTO facts (id, subject_id, relation, text_en, keywords_ru, fkind)"
        " VALUES (?, ?, ?, ?, ?, 'state')",
        facts,
    )
    conn.executemany(
        "INSERT INTO fts_records (rowid, text, keywords, aliases) VALUES (?, ?, ?, '')", fts
    )
    for i in range(N_EPISODES):
        eid = f"ep{i:05d}"
        conn.execute(
            "INSERT INTO records (id, fts_rowid, kind, version, author_tier, recorded_at,"
            " created_at, updated_at) VALUES (?, ?, 'episode', 1, 'system', ?, 't', 't')",
            (eid, rowid, T),
        )
        rowid += 1
        conn.execute(
            "INSERT INTO episodes (id, event_type, span_start, outcome, summary_en)"
            " VALUES (?, ?, ?, ?, ?)",
            (eid, rnd.choice(("meeting", "trip", "dinner", "call")), f"2026-0{rnd.randrange(1, 10)}-1{rnd.randrange(10)}",
             rnd.choice(("done", "cancelled", "postponed", None)), f"episode {i} {rnd.choice(TOPIC_EN)}"),
        )  # fmt: skip
        conn.execute(
            "INSERT INTO episode_links VALUES (?, 'participant', ?)", (eid, rnd.choice(ents[:50]))
        )
    conn.executemany(
        "INSERT INTO served (thread_id, session_id, item_id, item_kind, version, via, created_at)"
        " VALUES (?, ?, ?, 'fact', 1, 'basket', ?)",
        ((f"th{i % 40}", f"s{i % 40}", f"r{i * 3:06d}", T) for i in range(20_000)),
    )
    conn.execute("COMMIT")
    conn.execute("ANALYZE")


def _index() -> None:
    rng = np.random.default_rng(3)
    ids = [f"r{i:06d}" for i in range(N_RECORDS)]
    mat = rng.standard_normal((N_RECORDS, DIM)).astype(np.float16)
    vectors._index[0] = vectors._Index(
        "perf@1",
        DIM,
        ids,
        mat,
        np.zeros(N_RECORDS, np.int64),
        np.ones(N_RECORDS, bool),
    )


@pytest.fixture
def big(tmp_path):
    conn = open_seeded(tmp_path)
    _seed(conn)
    vectors.set_embedder(_SimEmbedder())
    _index()
    yield conn
    vectors.reset()
    vectors.set_embedder(None)
    conn.close()


QUESTIONS = [
    "что там у Саши с переездом?",
    "помнишь, мы хотели встретиться с Ильёй, но сорвалось из-за поездки?",
    "когда у меня экзамен по курсу",
    "where is the hotel invoice for Berlin",
    "что Лена говорила про выставку в мае",
    "напомни про кофе с Димой на прошлой неделе",
    "what did Paul say about the gym",
    "переезд в новую квартиру и счёт за отель",
]


def test_p95_with_embedding(big):
    """§10.2: basket p95 ≤ 150 ms at 10^5 records, query embedding included."""
    assert vectors.embed_query("warm") != []
    times, blocks, vec_hits = [], 0, 0
    for k in range(48):
        turn = TurnCtx(
            turn_id=f"t{k}",
            thread_id=f"perf-{k}",
            session_id=f"ps-{k}",
            owner_words=QUESTIONS[k % len(QUESTIONS)],
            clock="2026-10-14T09:00:00.000000Z",
        )
        t0 = time.perf_counter()
        b = api.recall(turn, store=big)
        times.append((time.perf_counter() - t0) * 1000)
        blocks += b.block is not None
        vec_hits += b.timings_ms.get("vector_wait", 99) < 40
        assert b.tokens <= 1500
    times.sort()
    p95 = times[int(len(times) * 0.95) - 1]
    print(f"basket p95 {p95:.1f} ms, p50 {times[len(times) // 2]:.1f} ms, blocks {blocks}/48")
    assert blocks >= 40  # the questions name people and topics the store holds
    assert p95 <= 150.0, f"p95 {p95:.1f} ms"
