"""Evidence recall of the LoCoMo contexts and the `ours_search` context (offline, synthetic)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC

import pytest

from koob.memory.eval.public import evidence as ev
from koob.memory.eval.public import locomo as lm
from koob.memory.eval.public import run as runmod

RAW = [
    {
        "sample_id": "conv-y",
        "conversation": {
            "speaker_a": "Denis Smith",
            "speaker_b": "Mia Lane",
            "session_1_date_time": "6:15 pm on 9 March, 2024",
            "session_1": [
                {"speaker": "Mia Lane", "dia_id": "D1:1", "text": "The pottery class starts Monday."},
                {"speaker": "Denis Smith", "dia_id": "D1:2", "text": "I adopted a cat named Pixel."},
            ],
        },
        "qa": [
            {"question": "What is the cat's name?", "answer": "Pixel", "evidence": ["D1:2"],
             "category": 4},
            {"question": "When does the class start?", "answer": "Monday", "evidence": ["D:1:1"],
             "category": 2},
        ],
    }
]  # fmt: skip


@pytest.fixture
def conv() -> lm.Conversation:
    return lm.parse(RAW, zone=UTC)[0]


def test_evidence_ids_are_normalized_from_the_release_spellings():
    got = ev.norm_evidence(["D1:3", "D:11:26", "D8:6; D9:17", "D1:3", "nonsense"])
    assert got == ("D1:3", "D11:26", "D8:6", "D9:17")


def test_handles_are_read_in_order_without_repeats():
    text = "[m:b2] one\n[m:a1] two · see [m:b2]\nno handle here"
    assert ev.handles(text) == ["b2", "a1"]
    assert ev.handles(None) == []


def test_provenance_maps_raw_chunks_and_the_records_derived_from_them():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE raw_chunks (id TEXT, source_ref TEXT)")
    conn.execute("CREATE TABLE edges (from_id TEXT, to_id TEXT, etype TEXT)")
    conn.executemany(
        "INSERT INTO raw_chunks VALUES (?, ?)",
        [("r1", "msg:conv-y:D1:1"), ("r2", "msg:conv-y:D1:2"), ("r9", "msg:other:D1:1")],
    )
    conn.executemany(
        "INSERT INTO edges VALUES (?, ?, ?)",
        [("f1", "r2", "derived_from"), ("f1", "r1", "derived_from"), ("f2", "r2", "about")],
    )
    prov = ev.provenance(conn, "conv-y")
    assert prov == {"r1": {"D1:1"}, "r2": {"D1:2"}, "f1": {"D1:1", "D1:2"}}
    assert ev.covered(["f1", "zz"], prov) == {"D1:1", "D1:2"}


def test_recall_summary_counts_items_questions_with_all_and_with_any():
    rows = [
        {"recall": {"x": ev.recall_row(("D1:1", "D1:2"), {"D1:1"})}, "tokens": {"x": 100}},
        {"recall": {"x": ev.recall_row(("D2:1",), {"D2:1"})}, "tokens": {"x": 300}},
        {"recall": {"x": ev.recall_row((), set())}, "tokens": {"x": 999}},  # no evidence: skipped
    ]
    s = ev.summarize(rows, ["x"])["x"]
    assert s["questions"] == 2 and s["evidence"] == 3
    assert s["item_recall"] == pytest.approx(66.7) and s["all_held"] == 50.0
    assert s["any_held"] == 100.0 and s["tokens_mean"] == 200
    assert rows[0]["recall"]["x"]["missing"] == ["D1:2"]


def test_tool_text_drops_items_already_shown_and_stops_at_the_budget():
    lines = ["<memory-tool>", "note line", "[m:a] " + "x" * 40, "[m:b] " + "y" * 40,
             "[m:c] " + "z" * 40, "</memory-tool>"]  # fmt: skip
    text = "\n".join(lines)
    frame = lm.tokens("\n".join([lines[0], lines[1], lines[5]]))
    one = lm.tokens(lines[2]) + 1
    got = lm.budgeted_tool_text(text, {"a"}, frame + one)
    assert got.splitlines() == [lines[0], lines[1], lines[3], lines[5]]
    assert lm.budgeted_tool_text(text, set(), frame) == ""  # no item fits: nothing at all
    assert lm.budgeted_tool_text("По «x» в памяти ничего нет.", set(), 500) == ""


class _BagEmbedder:
    vocab = ["cat", "pixel", "pottery", "class"]

    def _v(self, text):
        low = text.lower()
        return [float(low.count(w)) + 0.01 for w in self.vocab]

    def embed_docs(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)


def test_rag_top_names_the_dialog_ids_of_the_messages_it_picked(conv):
    rag = lm.RagIndex(conv, _BagEmbedder(), k=1)
    top = rag.top("the cat called pixel")
    assert [c.dia_id for c in top] == ["D1:2"]
    assert rag.context("the cat called pixel") == top[0].text


class _Neutral:
    def __call__(self, text):
        return {}

    def ready(self):
        return True


class _NoFacts:
    route, model_id = "cloud", "fake"

    def available(self):
        return True

    def __call__(self, snap):
        return {"entities": [], "facts": [], "impressions": []}


def test_contexts_phase_rereads_an_ingested_store_without_extraction(conv, tmp_path, monkeypatch):
    from koob.memory.recall import vectors

    root = tmp_path / "locomo"
    root.mkdir()
    (root / "locomo10.json").write_text(json.dumps(RAW), "utf-8")
    store = root / "runs" / conv.sample_id / "store"
    arr = lm.Arrival(conv, store, _NoFacts(), labeler=_Neutral())
    arr.start()
    try:
        arr.ingest()
    finally:
        arr.stop()
    monkeypatch.setattr(lm, "parse", lambda raw, zone=None: [conv])
    monkeypatch.setattr(vectors, "current_embedder", lambda **k: _BagEmbedder())
    info = runmod.contexts(root, conv.sample_id)
    assert info["extract"]["refused"] == 0 and info["extract"]["calls"] == 0
    rows = [json.loads(x) for x in (store.parent / "contexts.jsonl").read_text().splitlines()]
    assert [r["evidence"] for r in rows] == [["D1:2"], ["D1:1"]]
    for r in rows:
        assert set(r["recall"]) == {"ours", "ours_search", "rag"}
        assert r["recall"]["rag"]["hit"] == 1  # two messages, k=20: both always picked
        assert r["ours_search"].startswith(r["ours"])  # the basket first, the search after it
        assert r["tokens"]["ours_search"] <= lm.SEARCH_TOKENS
    assert info["recall"]["rag"]["item_recall"] == 100.0


def test_resume_needs_the_store_an_ingest_left(conv, tmp_path):
    arr = lm.Arrival(conv, tmp_path / "nothing", _NoFacts(), labeler=_Neutral())
    with pytest.raises(FileNotFoundError):
        arr.resume()


def test_summary_table_keeps_its_columns_for_long_system_names():
    from koob.memory.eval.public import scoring as sc

    one = {"n": 3, "J": 66.7, "J_strict": 33.3, "by_cat": {1: (1, 1), 4: (1, 2)}, "adv": (0, 0)}
    lines = sc.format_summary({"rag": one, "ours_search": one}).splitlines()
    starts = {ln.index("not run") for ln in lines[1:]}
    assert len(starts) == 1 and lines[0].index("adversarial") in starts
    assert [ln.split()[0] for ln in lines] == ["system", "rag", "ours_search"]
