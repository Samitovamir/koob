"""LoCoMo harness, offline: a synthetic two-person conversation, fake extractor and runners."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from koob.memory.eval.public import locomo as lm
from koob.memory.eval.public import scoring as sc

RAW = [
    {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Denis Smith",
            "speaker_b": "Mia Lane",
            # the keys run 1, 2 but the dates run 2, 1: arrival follows the dates
            "session_1_date_time": "6:15 pm on 9 March, 2024",
            "session_1": [
                {"speaker": "Mia Lane", "dia_id": "D1:1", "text": "The pottery class starts Monday."},
                {"speaker": "Denis Smith", "dia_id": "D1:2", "text": "I'll drive you there.",
                 "img_url": ["http://example.invalid/a.jpg"], "blip_caption": "a photo of a red car",
                 "query": "red car"},
            ],
            "session_2_date_time": "12:48 am on 2 March, 2024",
            "session_2": [
                {"speaker": "Denis Smith", "dia_id": "D2:1", "text": "I adopted a cat yesterday."},
                {"speaker": "Mia Lane", "dia_id": "D2:2", "text": "What is its name?"},
                {"speaker": "Denis Smith", "dia_id": "D2:3", "text": "Pixel."},
            ],
        },
        "qa": [
            {"question": "When did Denis adopt a cat?", "answer": "1 March 2024",
             "evidence": ["D2:1"], "category": 2},
            {"question": "What is the cat's name?", "answer": "Pixel", "evidence": ["D2:3"],
             "category": 4},
            {"question": "What class did Denis start?", "adversarial_answer": "pottery",
             "evidence": ["D1:1"], "category": 5},
        ],
    }
]  # fmt: skip


@pytest.fixture
def conv() -> lm.Conversation:
    return lm.parse(RAW, zone=UTC)[0]


def test_session_dates_are_read_as_wall_clock_in_the_given_zone():
    assert lm.parse_when("1:56 pm on 8 May, 2023", UTC) == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
    assert lm.parse_when("12:48 am on 1 February, 2023", UTC) == datetime(
        2023, 2, 1, 0, 48, tzinfo=UTC
    )
    assert lm.parse_when("12:05 pm on 3 June, 2023", UTC).hour == 12
    with pytest.raises(ValueError):
        lm.parse_when("yesterday", UTC)


def test_sessions_arrive_in_timestamp_order_not_key_order(conv):
    assert [s.n for s in conv.sessions] == [2, 1]
    assert conv.sessions[0].when < conv.sessions[1].when
    assert conv.asked_at > conv.sessions[-1].end


def test_both_speakers_are_contacts_of_one_chat_with_their_own_ids(conv):
    rows = [r for s in conv.sessions for r in lm.session_rows(conv, s)]
    assert {r["role"] for r in rows} == {"other"}
    assert {r["chat_id"] for r in rows} == {lm.chat_id(conv)}
    ids = {r["sender"]: r["sender_tg_id"] for r in rows}
    assert set(ids) == {"Denis Smith", "Mia Lane"} and len(set(ids.values())) == 2
    assert lm.sender_ids(conv) == ids  # stable: the same ids on every run


def test_timestamps_ride_in_the_row_never_in_the_text(conv):
    s = conv.sessions[0]
    rows = lm.session_rows(conv, s)
    assert [r["ts"] for r in rows] == sorted(r["ts"] for r in rows)
    assert rows[0]["ts"] == "2024-03-02T00:48:00.000Z"
    assert rows[1]["ts"] == "2024-03-02T00:48:30.000Z"  # PACE between turns
    for r in rows:
        assert "2024" not in r["text"] and "March" not in r["text"]


def test_an_image_turn_keeps_its_caption_and_drops_url_and_query(conv):
    rows = lm.session_rows(conv, conv.sessions[1])
    text = rows[1]["text"]
    assert text == "I'll drive you there. [shares a photo: a photo of a red car]"
    assert "example.invalid" not in text and "red car]" in text


def test_controls_see_the_same_message_text(conv):
    full = lm.transcript(conv)
    assert full.index("Session 2") < full.index("Session 1")
    assert "[shares a photo: a photo of a red car]" in full
    assert all("(" in c.text and ": " in c.text for c in lm.chunks(conv))


class _BagEmbedder:
    vocab = ["cat", "pixel", "pottery", "car", "drive", "class"]

    def _v(self, text):
        low = text.lower()
        return [float(low.count(w)) + 0.01 for w in self.vocab]

    def embed_docs(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)


def test_plain_rag_returns_the_nearest_messages_in_conversation_order(conv):
    rag = lm.RagIndex(conv, _BagEmbedder(), k=2)
    ctx = rag.context("the pottery class")
    lines = ctx.splitlines()
    assert len(lines) == 2 and "pottery class" in ctx
    orders = [next(c.order for c in rag.chunks if c.text == line) for line in lines]
    assert orders == sorted(orders)


# ---------------------------------------------------------------------- arrival


class _Neutral:  # a warm classifier that finds nothing: labels come from the lexicons only
    def __call__(self, text):
        return {}

    def ready(self):
        return True


class _Recorder:
    route, model_id = "cloud", "fake"

    def __init__(self):
        self.batches: list[list[tuple[str, str]]] = []

    def available(self):
        return True

    def __call__(self, snap):
        self.batches.append([(r["ts"], r["text"]) for r in snap["raws"]])
        return {"entities": [], "facts": [], "impressions": []}


def test_arrival_stores_each_session_in_order_with_its_timestamps(conv, tmp_path):
    rec = _Recorder()
    arr = lm.Arrival(conv, tmp_path / "store", rec, labeler=_Neutral())
    arr.start()
    try:
        arr.ingest()
        got = arr.sut.store().execute("SELECT ts, text FROM raw_chunks ORDER BY ts").fetchall()
        want = [(r["ts"], r["text"]) for s in conv.sessions for r in lm.session_rows(conv, s)]
        assert [tuple(x) for x in got] == want
        firsts = [b[0][1] for b in rec.batches if b]
        assert firsts.index("I adopted a cat yesterday.") < firsts.index(
            "The pottery class starts Monday."
        )
        assert [e["session"] for e in arr.log] == [2, 1]
        before = arr.sut.store().execute("SELECT COUNT(*) FROM raw_chunks").fetchone()[0]
        text, meta = arr.context(conv.questions[1], 0)
        assert isinstance(text, str) and "basket_items" in meta
        after = arr.sut.store().execute("SELECT COUNT(*) FROM raw_chunks").fetchone()[0]
        assert after == before  # a question leaves the store as it was
    finally:
        arr.stop()


# ---------------------------------------------------------------------- judge and cache


def test_mem0_judge_prompt_keeps_its_fields_and_trailing_spaces():
    text = sc.judge_input("lenient", "Q?", "gold", "gen")
    assert "Question: Q?\nGold answer: gold\nGenerated answer: gen\n" in text
    assert "(posed by one user to another user), \n" in text
    assert "be generous with your grading" in text
    strict = sc.judge_input("strict", "Q?", "gold", "gen")
    assert "Grade strictly" in strict and "Generated answer: gen" in strict


@pytest.mark.parametrize(
    ("out", "label"),
    [
        ({"label": "CORRECT"}, "CORRECT"),
        ({"label": "wrong", "explanation": "different date"}, "WRONG"),
        ({"label": "’CORRECT’"}, "CORRECT"),
        ('{"label": "WRONG"}', "WRONG"),
        ({"label": "CORRECT or WRONG"}, None),
        ({"label": ""}, None),
        ({}, None),
    ],
)
def test_judge_parser_reads_one_label_or_none(out, label):
    assert sc.parse_label(out) == label


def test_abstention_follows_locomo_rule():
    assert sc.abstains("Not mentioned in the conversation")
    assert sc.abstains("There is no information available about that.")
    assert not sc.abstains("Pottery")


def _runner(log):
    def run(role, out_model, text, *, model, lenient):
        log.append((role, model, text))
        return out_model(answer=f"ans:{text}"), {"cost_usd": 0.0}

    return run


def test_cache_key_covers_role_model_prompt_and_input(tmp_path):
    base = sc.cache_key("r", "m", "p", "t")
    assert base == sc.cache_key("r", "m", "p", "t")
    assert len({base, *(sc.cache_key(*k) for k in [("r2", "m", "p", "t"), ("r", "m2", "p", "t"),
                ("r", "m", "p2", "t"), ("r", "m", "p", "t2")])}) == 5  # fmt: skip


def test_a_cached_output_makes_no_second_call(tmp_path):
    log: list = []
    call = sc.CachedCall(
        "role-a", sc.AnswerOut, "model-a", tmp_path, runner=_runner(log), prompt="p1"
    )
    assert call("hello") == {"answer": "ans:hello"}
    assert call("hello") == {"answer": "ans:hello"}
    assert len(log) == 1 and call.hits == 1 and call.calls == 1
    other = sc.CachedCall(
        "role-a", sc.AnswerOut, "model-b", tmp_path, runner=_runner(log), prompt="p1"
    )
    other("hello")
    assert len(log) == 2  # another model is another key


def test_the_budget_refuses_the_call_past_its_cap_across_processes(tmp_path):
    log: list = []
    path = tmp_path / "budget.json"
    call = sc.CachedCall("r", sc.AnswerOut, "m", tmp_path, sc.Budget(path, 2), _runner(log), "p")
    call("a")
    call("b")
    call("a")  # a cache hit costs nothing
    again = sc.CachedCall("r", sc.AnswerOut, "m", tmp_path, sc.Budget(path, 2), _runner(log), "p")
    with pytest.raises(sc.BudgetExceeded):
        again("c")
    assert len(log) == 2 and sc.Budget(path, 2).counts == {"r": 2}


def test_scores_count_categories_one_to_four_and_report_adversarial_apart():
    rows = [
        {"system": "ours", "category": 1, "lenient": "CORRECT", "strict": "CORRECT"},
        {"system": "ours", "category": 2, "lenient": "CORRECT", "strict": "WRONG"},
        {"system": "ours", "category": 4, "lenient": "WRONG", "strict": "WRONG"},
        {"system": "ours", "category": 4, "lenient": None, "strict": None},
        {"system": "ours", "category": 5, "abstained": True},
        {"system": "ours", "category": 5, "abstained": False},
    ]
    s = sc.summarize(rows)["ours"]
    assert s["n"] == 4 and s["J"] == 50.0 and s["J_strict"] == 25.0
    assert s["unparsed"] == 1 and s["adv"] == [1, 2]
    assert s["by_cat"] == {"1": [1, 1], "2": [1, 1], "4": [0, 2]}


# ---------------------------------------------------------------------- answer phase


def test_answer_phase_judges_strict_only_where_lenient_passed(tmp_path, monkeypatch):
    import contextlib
    import json

    from koob.llm import runner as llm_runner
    from koob.memory.eval import cli_env
    from koob.memory.eval.public import run

    (tmp_path / "locomo10.json").write_text(json.dumps(RAW), "utf-8")
    conv = lm.parse(RAW)[0]
    ctx_rows = [
        {"qid": q.qid, "category": q.category, "question": q.question, "gold": q.answer,
         "ours": "Denis Smith adopted a cat named Pixel.", "rag": "(x) Denis Smith: Pixel."}
        for q in conv.questions
    ]  # fmt: skip
    run._jsonl(tmp_path / "runs" / "conv-x" / "contexts.jsonl", ctx_rows)
    seen: list[tuple[str, str]] = []

    def fake(role, out_model, text, *, model, lenient):
        seen.append((role, model))
        if role == sc.ANSWER_ROLE:
            return out_model(answer="Pixel"), {}
        ok = "Gold answer: Pixel" in text
        return out_model(explanation="x", label="CORRECT" if ok else "WRONG"), {}

    monkeypatch.setattr(llm_runner, "run_role", fake)
    monkeypatch.setattr(cli_env, "no_account", contextlib.nullcontext)
    out = run.answer(tmp_path, "conv-x", systems=("ours", "full"), answerer="sonnet", cap=50,
                     adversarial=True, strict_all=False)  # fmt: skip
    s = out["scores"]["ours"]
    assert s["n"] == 2 and s["J"] == 50.0 and s["J_strict"] == 50.0 and s["adv"] == [0, 1]
    answers = [m for r, m in seen if r == sc.ANSWER_ROLE]
    judges = [m for r, m in seen if r == sc.JUDGE_ROLE]
    assert set(answers) == {"claude-sonnet-5-5"} and set(judges) == {sc.JUDGE_MODEL}
    # 3 questions x 2 systems answered; both systems answer alike, so the lenient judge runs
    # once per scored question (the cache) and the strict one only where lenient passed
    assert len(answers) == 6 and len(judges) == 2 + 1
    rows = run._read_jsonl(tmp_path / "runs" / "conv-x" / "results-sonnet-ours.jsonl")
    assert {r["strict"] for r in rows if r.get("lenient") == "WRONG"} == {"WRONG"}
    assert all("lenient" not in r for r in rows if r["category"] == lm.ADVERSARIAL)
