"""The raw-message meaning lane: usable raw chunks are embedded beside the records, so a value
said only in a message is found by a paraphrased question in the other language. The basket
renders such a raw with the truth overlay (retired values masked, the sender named), keeps
the label policy, and a forget or a chat ban removes the raw's vector with the raw."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from koob.memory import api, forget
from koob.memory.model import ChatMode, ForgetScope, Label, TurnCtx, TurnOrigin
from koob.memory.recall import lexical, vectors

from .._ledger import add, fact, open_seeded, raw
from ..forget._seed import entity

_VPATH: list = [None]
CLOCK = "2026-10-12T12:00:00.000000Z"

# A tiny bilingual concept space: the test's stand-in for a multilingual embedder. Words
# outside it carry no meaning, so only the concepts below decide the cosine.
_CONCEPTS = {
    "ключ": "key", "key": "key",
    "гараж": "garage", "garage": "garage",
    "соседк": "neighbour", "neighbo": "neighbour",
    "коврик": "mat", "mat": "mat",
    "оставил": "leave", "leave": "leave", "left": "leave",
    "пароль": "code", "код": "code", "code": "code",
    "домофон": "intercom", "intercom": "intercom",
    "сахар": "sugar", "sugar": "sugar", "диабет": "sugar", "diabet": "sugar",
    "здоров": "health", "health": "health",
    "дач": "cottage", "cottage": "cottage",
    "адрес": "address", "address": "address",
}  # fmt: skip


def _concept(token: str) -> str | None:
    for stem, concept in _CONCEPTS.items():
        if token.startswith(stem):
            return concept
    return None


class ConceptEmbedder:
    id = "concept@1"
    dim = 64

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"\w+", (text or "").casefold()):
            c = _concept(tok)
            if c is not None:
                h = int.from_bytes(hashlib.blake2b(c.encode(), digest_size=8).digest(), "little")
                v[h % self.dim] += 1.0
        return v

    def embed_docs(self, texts):
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim))

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
    for col, val in kw.items():  # author_entity_id, chat_id
        conn.execute(f"UPDATE raw_chunks SET {col} = ? WHERE id = ?", (val, rid))
    lexical.index_raw(conn, rid)


def _embed(conn) -> None:
    vdb = vectors.open_db(_VPATH[0])
    try:
        vectors.embed_pending(conn, vdb)
    finally:
        vdb.close()


def _vector_rows(conn, rid: str) -> int:
    vdb = vectors.open_db(_VPATH[0])
    try:
        return vdb.execute("SELECT count(*) FROM vectors WHERE record_id = ?", (rid,)).fetchone()[0]
    finally:
        vdb.close()


def _raw_lane(words: str, mask: int = -1) -> list[str]:
    q = vectors.embed_query(words)
    return [i for i, _ in vectors.search_split(q, label_mask=mask, limit=5, raw_limit=5)[1]]


def _texts(b) -> str:
    return "\n".join([b.block or "", *(i.text for i in b.items)])


def test_a_value_said_only_in_a_message_is_found_by_a_question_in_the_other_language(conn):
    _raw(conn, "r-keys", "ключи от гаража я оставил у соседки под ковриком", "2026-10-03T10:00:00Z")
    _raw(conn, "r-noise", "купил новые кроссовки и шнурки к ним", "2026-10-04T10:00:00Z")
    _embed(conn)
    assert lexical.search_raw(conn, "where did I leave the garage keys?") == []  # no shared word
    assert _raw_lane("where did I leave the garage keys?")[0] == "r-keys"
    b = api.recall(t("where did I leave the garage keys?"), store=conn)
    assert "r-keys" in [i.id for i in b.items]
    assert "ковриком" in _texts(b)
    assert "r-noise" not in [i.id for i in b.items]


def test_records_and_raws_keep_separate_meaning_lanes(conn):
    _raw(conn, "r-keys", "ключи от гаража я оставил у соседки под ковриком", "2026-10-03T10:00:00Z")
    _embed(conn)
    q = vectors.embed_query("garage keys")
    assert vectors.search(q, label_mask=-1) == []  # the record lane never returns raws
    assert vectors._index[0] is not None and bool(vectors._index[0].raw.all())


def test_a_corrected_value_in_a_message_is_never_served_as_current(conn):
    entity(conn, "e-denis", "Denis Smith")
    _raw(conn, "r-code", "код домофона на даче 4417", "2026-10-02T10:00:00Z")
    _raw(conn, "r-fix", "на даче домофон сменили: не 4417, а 9035", "2026-10-05T10:00:00Z")
    _embed(conn)
    b = api.recall(t("what's the intercom code at the cottage?"), store=conn)
    text = _texts(b)
    assert "4417" not in text
    assert "9035" in text


def test_a_message_from_another_person_names_its_sender_and_date(conn):
    entity(conn, "e-denis", "Denis Smith")
    _raw(conn, "r-denis", "ключи от гаража у соседки под ковриком", "2026-10-06T10:00:00Z",
         tier="other", author_entity_id="e-denis")  # fmt: skip
    _embed(conn)
    b = api.recall(t("where are the garage keys?"), store=conn)
    item = next(i for i in b.items if i.id == "r-denis")
    assert item.said_by == "Denis Smith"
    assert item.date == "2026-10-06"
    assert "со слов: Denis Smith" in (b.block or "")


def test_a_forgotten_message_leaves_both_raw_lanes_and_its_vector(conn):
    _raw(conn, "r-keys", "ключи от гаража я оставил у соседки под ковриком", "2026-10-03T10:00:00Z")
    _embed(conn)
    assert _vector_rows(conn, "r-keys") == 1
    forget.stop_use(conn, ["r-keys"], scope=ForgetScope.ITEM, forever=True)
    assert _vector_rows(conn, "r-keys") == 0  # the stop hook deleted the row at once
    assert _raw_lane("garage keys") == []
    assert lexical.search_raw(conn, "ключи от гаража") == []
    _embed(conn)  # the pull job never brings it back
    vdb = vectors.open_db(_VPATH[0])
    try:
        vectors.sweep_orphans(conn, vdb)
    finally:
        vdb.close()
    assert _vector_rows(conn, "r-keys") == 0
    b = api.recall(t("where did I leave the garage keys?", n=2), store=conn)
    assert "ковриком" not in _texts(b)


def test_a_redacted_message_is_embedded_again_from_its_new_text(conn):
    _raw(conn, "r-keys", "ключи от гаража я оставил у соседки под ковриком", "2026-10-03T10:00:00Z")
    _embed(conn)
    conn.execute("UPDATE raw_chunks SET text = 'купил шнурки', rev = rev + 1 WHERE id = 'r-keys'")
    vdb = vectors.open_db(_VPATH[0])
    try:
        assert vectors.sweep_orphans(conn, vdb) == 1  # the vector of the old text goes
        vectors.embed_pending(conn, vdb)
        ver = vdb.execute("SELECT version FROM vectors WHERE record_id = 'r-keys'").fetchone()[0]
    finally:
        vdb.close()
    assert ver == 2
    q = vectors.embed_query("garage keys")
    assert vectors.search_split(q, label_mask=-1, limit=0, raw_limit=1)[1][0][1] < 0.5


def test_a_sensitive_message_from_another_person_needs_a_direct_question(conn):
    entity(conn, "e-denis", "Denis Smith", aliases=[("Денис", "name")])
    _raw(conn, "r-sugar", "Денис: у меня диабет, сахар меряю по утрам, на даче тоже",
         "2026-10-06T10:00:00Z", tier="other", labels=int(Label.HEALTH),
         author_entity_id="e-denis")  # fmt: skip
    _embed(conn)
    side = api.recall(t("what's happening at the cottage with Denis Smith?"), store=conn)
    assert "диабет" not in _texts(side)
    direct = api.recall(t("что у Дениса со здоровьем, какой у него диабет?", n=2), store=conn)
    assert "диабет" in _texts(direct)


def test_a_message_from_a_banned_chat_is_absent(conn):
    _raw(conn, "r-group", "адрес дачи: Larch Lane 12, ключи под ковриком", "2026-10-06T10:00:00Z",
         tier="other", chat_id="tg:denis-group")  # fmt: skip
    _embed(conn)
    assert _raw_lane("cottage address") == ["r-group"]
    owner = t("полный запрет на чат tg:denis-group, ничего не запоминай", n=3)
    api.set_chat_mode("tg:denis-group", ChatMode.BANNED, turn=owner, store=conn)
    assert _raw_lane("cottage address") == []
    assert _vector_rows(conn, "r-group") == 0
    b = api.recall(t("what's the cottage address?", n=4), store=conn)
    assert "Larch" not in _texts(b)


def test_a_message_its_fact_quotes_whole_is_not_shown_twice(conn):
    _raw(conn, "r-city", "Я живу в Порту", "2026-10-03T10:00:00Z")
    _raw(conn, "r-more", "работаю в порту Лиссабона, смена с семи, пропуск у охраны",
         "2026-10-04T10:00:00Z")  # fmt: skip
    assert add(conn, fact("f-city", "self", "lives_in", "Porto", text="I live in Porto",
                          source_raw_id="r-city", quote_orig="я живу в Порту")).ok  # fmt: skip
    assert add(conn, fact("f-job", "self", "works_at", "Lisbon port", text="I work at the port",
                          source_raw_id="r-more", quote_orig="работаю в порту")).ok  # fmt: skip
    ids = [i.id for i in api.recall(t("где я живу и работаю, в каком порту?"), store=conn).items]
    assert "f-city" in ids and "r-city" not in ids  # its fact's line carries the same words
    assert "r-more" in ids  # says more than its fact quotes
