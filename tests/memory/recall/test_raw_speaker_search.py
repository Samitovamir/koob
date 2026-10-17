"""Raw messages carry who sent them and when: the meaning lane embeds `<date> <sender>: <text>`,
the stemmed raw lane ranks the named sender's messages first, and the `search` tool reads raw
messages and meaning hits beside the records under the basket's truth and policy checks."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from koob.memory import api, forget
from koob.memory.model import ForgetScope, Label, RecordKind, TurnCtx, TurnOrigin
from koob.memory.recall import lexical, vectors

from .._ledger import open_seeded, raw
from ..forget._seed import entity, impression

_VPATH: list = [None]
CLOCK = "2026-10-12T12:00:00.000000Z"

# A tiny concept space standing in for a multilingual embedder; a sender's name is a concept
# too, so a vector meets a person only when the embedded text names them.
_CONCEPTS = {
    "ключ": "key", "key": "key",
    "гараж": "garage", "garage": "garage",
    "соседк": "neighbour", "neighbo": "neighbour",
    "коврик": "mat", "mat": "mat",
    "пароль": "code", "код": "code", "code": "code",
    "домофон": "intercom", "intercom": "intercom",
    "сахар": "sugar", "sugar": "sugar", "диабет": "sugar", "diabet": "sugar",
    "дач": "cottage", "cottage": "cottage",
    "лодк": "boat", "boat": "boat",
    "denis": "denis", "денис": "denis",
    "anna": "anna",
}  # fmt: skip


class ConceptEmbedder:
    id = "concept@1"
    dim = 64

    def __init__(self) -> None:
        self.docs: list[str] = []

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"\w+", (text or "").casefold()):
            c = next((c for s, c in _CONCEPTS.items() if tok.startswith(s)), None)
            if c is not None:
                h = int.from_bytes(hashlib.blake2b(c.encode(), digest_size=8).digest(), "little")
                v[h % self.dim] += 1.0
        return v

    def embed_docs(self, texts):
        self.docs += list(texts)
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text):
        return self._one(text)


@pytest.fixture
def emb():
    return ConceptEmbedder()


@pytest.fixture
def conn(tmp_path, emb):
    vectors.reset()
    vectors.set_embedder(emb)
    forget.reset_wiring()
    forget.use_key(None)
    forget.wire(stop_hooks=[vectors.stop_hook])
    c = open_seeded(tmp_path)
    _VPATH[0] = tmp_path / "vectors.db"
    vectors.attach(_VPATH[0])
    yield c
    c.close()
    forget.reset_wiring()
    forget.use_key(None)
    vectors.reset()
    vectors.set_embedder(None)


def t(words: str, *, n: int = 1) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}", thread_id=f"thread-{n}", session_id=f"session-{n}",
        origin=TurnOrigin.OWNER, owner_words=words, clock=CLOCK,
    )  # fmt: skip


def _raw(conn, rid: str, text: str, ts: str, *, tier: str = "user", labels: int = 0, **kw):
    raw(conn, rid, text, tier=tier, ts_=ts, labels=labels)
    for col, val in kw.items():  # author_entity_id
        conn.execute(f"UPDATE raw_chunks SET {col} = ? WHERE id = ?", (val, rid))
    lexical.index_raw(conn, rid)


def _embed(conn) -> int:
    vdb = vectors.open_db(_VPATH[0])
    try:
        return vectors.embed_pending(conn, vdb)
    finally:
        vdb.close()


def _people(conn) -> None:
    entity(conn, "e-denis", "Denis Smith", aliases=[("Денис", "name")])
    entity(conn, "e-anna", "Anna Brook")


def _search_text(items) -> str:
    return "\n".join(i.text for i in items)


def test_the_embedded_raw_text_carries_its_sender_and_date(conn, emb):
    _people(conn)
    _raw(conn, "r-denis", "ключи от гаража у соседки", "2026-10-06T10:00:00Z",
         tier="other", author_entity_id="e-denis")  # fmt: skip
    _raw(conn, "r-own", "ключи забрал", "2026-10-07T09:00:00Z")
    _embed(conn)
    assert "2026-10-06 Denis Smith: ключи от гаража у соседки" in emb.docs
    assert f"2026-10-07 {vectors.OWN_SPEAKER}: ключи забрал" in emb.docs


def test_a_question_naming_a_person_meets_that_persons_message_by_meaning(conn):
    _people(conn)
    _raw(conn, "r-anna", "ключи от гаража у соседки", "2026-10-05T10:00:00Z",
         tier="other", author_entity_id="e-anna")  # fmt: skip
    _raw(conn, "r-denis", "ключи от гаража у соседки", "2026-10-06T10:00:00Z",
         tier="other", author_entity_id="e-denis")  # fmt: skip
    _embed(conn)
    q = vectors.embed_query("what did Denis say about the garage keys?")
    raws = vectors.search_split(q, label_mask=-1, limit=0, raw_limit=2)[1]
    assert raws[0][0] == "r-denis"
    assert raws[0][1] > raws[1][1]


def test_the_stemmed_raw_lane_ranks_the_named_senders_message_first(conn):
    _people(conn)
    _raw(conn, "r-anna", "лодку продали в субботу", "2026-10-05T10:00:00Z",
         tier="other", author_entity_id="e-anna")  # fmt: skip
    _raw(conn, "r-denis", "лодку продали в субботу", "2026-10-06T10:00:00Z",
         tier="other", author_entity_id="e-denis")  # fmt: skip
    plain = [i for i, _ in lexical.search_raw(conn, "лодку продали")]
    named = [i for i, _ in lexical.search_raw(conn, "лодку продали", authors=["e-denis"])]
    assert set(plain) == set(named) == {"r-anna", "r-denis"}
    assert named[0] == "r-denis"
    b = api.recall(t("что Денис писал про лодку?"), store=conn)
    assert "r-denis" in [i.id for i in b.items]


def test_search_returns_a_fact_said_only_in_a_message(conn):
    _people(conn)
    _raw(conn, "r-keys", "ключи от гаража я оставил у соседки под ковриком", "2026-10-03T10:00:00Z")
    _raw(conn, "r-noise", "купил новые кроссовки и шнурки к ним", "2026-10-04T10:00:00Z")
    _embed(conn)
    found = api.search(
        "where are the garage keys?", turn=t("where are the garage keys?"), store=conn
    )
    raw_items = [i for i in found if i.kind == "raw"]
    assert [i.id for i in raw_items] == ["r-keys"]
    assert "ковриком" in raw_items[0].text
    assert raw_items[0].date == "2026-10-03"
    served = conn.execute("SELECT item_kind FROM served WHERE item_id = 'r-keys'").fetchall()
    assert [r[0] for r in served] == ["raw"]


def test_search_names_the_sender_of_another_persons_message(conn):
    _people(conn)
    _raw(conn, "r-denis", "лодку продали в субботу", "2026-10-06T10:00:00Z",
         tier="other", author_entity_id="e-denis")  # fmt: skip
    _embed(conn)
    found = api.search("лодку продали", turn=t("кто продал лодку?"), store=conn)
    item = next(i for i in found if i.id == "r-denis")
    assert item.said_by == "Denis Smith"


def test_search_never_serves_an_overtaken_value_from_a_message(conn):
    _raw(conn, "r-code", "код домофона на даче 4417", "2026-10-02T10:00:00Z")
    _raw(conn, "r-fix", "на даче домофон сменили: не 4417, а 9035", "2026-10-05T10:00:00Z")
    _embed(conn)
    q = "intercom code at the cottage"
    text = _search_text(api.search(q, turn=t(q), store=conn))
    assert "4417" not in text
    assert "9035" in text


def test_search_leaves_out_a_forgotten_message(conn):
    _raw(conn, "r-keys", "ключи от гаража я оставил у соседки под ковриком", "2026-10-03T10:00:00Z")
    _embed(conn)
    forget.stop_use(conn, ["r-keys"], scope=ForgetScope.ITEM, forever=True)
    q = "ключи от гаража"
    assert api.search(q, turn=t(q), store=conn) == []


def test_search_keeps_a_sensitive_message_for_a_direct_question(conn):
    _people(conn)
    _raw(conn, "r-sugar", "у меня диабет, сахар меряю по утрам, на даче тоже",
         "2026-10-06T10:00:00Z", tier="other", labels=int(Label.HEALTH),
         author_entity_id="e-denis")  # fmt: skip
    _embed(conn)
    # the brain's query names the illness; the user's own words did not ask about it
    side = api.search("Denis diabetes sugar", turn=t("what's new at the cottage?"), store=conn)
    assert "диабет" not in _search_text(side)
    ask = "что у Дениса со здоровьем, какой у него диабет?"
    direct = api.search("Denis diabetes sugar", turn=t(ask, n=2), store=conn)
    assert "диабет" in _search_text(direct)


def test_search_with_a_record_filter_returns_records_only(conn):
    _raw(conn, "r-keys", "ключи от гаража я оставил у соседки под ковриком", "2026-10-03T10:00:00Z")
    _embed(conn)
    q = "garage keys"
    found = api.search(q, turn=t(q), kinds=[RecordKind.FACT], store=conn)
    assert [i for i in found if i.kind == "raw"] == []


def test_search_leaves_out_an_older_impression_of_the_same_person(conn):
    _people(conn)
    impression(conn, "i-old", "e-denis", "Denis promises a lot and delivers little.")
    impression(conn, "i-new", "e-denis", "Denis delivered the whole boat repair on time.")
    conn.execute("UPDATE impressions SET observed_at = '2026-10-01T10:00:00Z' WHERE id = 'i-new'")
    _embed(conn)
    q = "can Denis deliver on a promise?"
    ids = [i.id for i in api.search(q, turn=t(q), store=conn)]
    assert "i-new" in ids
    assert "i-old" not in ids


def test_raw_vectors_of_an_older_recipe_are_embedded_again_once(conn):
    _raw(conn, "r-keys", "ключи от гаража у соседки", "2026-10-03T10:00:00Z")
    assert _embed(conn) == 1
    assert _embed(conn) == 0
    vdb = vectors.open_db(_VPATH[0])
    try:  # a row written by the text-only recipe: its version is the bare rev
        vdb.execute("UPDATE vectors SET version = 1 WHERE record_id = 'r-keys'")
        vdb.commit()
    finally:
        vdb.close()
    assert _embed(conn) == 1
    assert _embed(conn) == 0
    vdb = vectors.open_db(_VPATH[0])
    try:
        ver = vdb.execute("SELECT version FROM vectors WHERE record_id = 'r-keys'").fetchone()[0]
    finally:
        vdb.close()
    assert ver == vectors.raw_version(1)


def _learned_from(conn, rec_id: str, raw_id: str) -> None:
    conn.execute(
        "INSERT INTO edges (from_id, to_id, etype, created_at) VALUES (?, ?, 'derived_from', ?)",
        (rec_id, raw_id, CLOCK),
    )


def test_search_leaves_out_an_old_message_a_later_take_back_overtook(conn):
    _people(conn)
    _raw(conn, "r-old", "Денис много обещает и мало делает", "2026-09-01T10:00:00Z")
    impression(conn, "i-old", "e-denis", "Denis promises a lot and delivers little.")
    _learned_from(conn, "i-old", "r-old")
    _raw(conn, "r-new", "Беру назад про Дениса. Он один починил лодку.", "2026-10-01T10:00:00Z")
    impression(conn, "i-new", "e-denis", "Denis repaired the boat alone.", "Он один починил лодку.")
    conn.execute("UPDATE impressions SET observed_at = '2026-10-01T10:00:00Z' WHERE id = 'i-new'")
    _learned_from(conn, "i-new", "r-new")
    _embed(conn)
    q = "Денис много обещает?"
    ids = [i.id for i in api.search(q, turn=t(q), store=conn)]
    assert "r-old" not in ids
    assert "i-old" not in ids


def test_search_runs_the_basket_off_target_recheck(conn, monkeypatch):
    from koob.memory.recall import basket

    seen: list[set[str]] = []
    real = basket._off_target

    def spy(rows, verdicts, *a, **kw):
        seen.append(set(verdicts))
        return real(rows, verdicts, *a, **kw)

    monkeypatch.setattr(basket, "_off_target", spy)
    _people(conn)
    _raw(conn, "r-denis", "ключи от гаража у соседки", "2026-10-06T10:00:00Z",
         tier="other", author_entity_id="e-denis")  # fmt: skip
    _embed(conn)
    q = "где ключи от гаража?"
    api.search(q, turn=t(q), store=conn)
    assert any("r-denis" in s for s in seen)
