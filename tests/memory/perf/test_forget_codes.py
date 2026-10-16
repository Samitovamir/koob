"""Topic forget's code links (a code that appears in another message ties it to the same matter)
under G9: a code lookup is an `fts_raw` probe, not a scan of every message. Measured on
10^5 raws (~160 chars): the `instr` scan took ~50 ms per code (~155 ms for three codes, over
the 150 ms G9 budget); the probe takes well under a millisecond."""

from __future__ import annotations

import random
import statistics
import time

import pytest

from koob.memory import commands
from koob.memory.recall import lexical

from .. import _ledger as L

N_RAWS = 100_000
T = "2026-10-01T10:00:00.000000Z"
WORDS = [
    "привет",
    "как",
    "дела",
    "встреча",
    "завтра",
    "договор",
    "вклад",
    "банк",
    "ипотека",
    "проект",
    "отпуск",
    "Дима",
    "Саша",
    "Лера",
    "поездка",
    "Тверь",
    "Тверь",
    "работа",
    "созвон",
    "документы",
    "письмо",
    "сегодня",
    "потом",
    "номер",
]


def _insert(conn, rows: list[tuple[str, str]], start: int = 1) -> None:
    conn.executemany(
        "INSERT INTO raw_chunks (id, fts_rowid, source_kind, chat_id, turn_id, author_tier, ts,"
        " text, content_hmac, policy_version, created_at, updated_at)"
        " VALUES (?, ?, 'chat', 'c1', NULL, 'user', ?, ?, ?, 0, ?, ?)",
        [(rid, start + i, T, text, f"h-{rid}", T, T) for i, (rid, text) in enumerate(rows)],
    )
    conn.executemany(
        "INSERT INTO fts_raw (rowid, text) VALUES (?, ?)",
        [(start + i, lexical.stem(text)) for i, (_rid, text) in enumerate(rows)],
    )


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    conn = L.open_seeded(tmp_path_factory.mktemp("codes"))
    rnd = random.Random(7)
    stems = {w: lexical.stem(w) for w in WORDS}
    texts, fts = [], []
    for i in range(N_RAWS):
        words = [rnd.choice(WORDS) for _ in range(max(3, int(rnd.expovariate(1 / 20))))]
        if i % 37 == 0:
            words.append(str(rnd.randrange(10**5, 10**7)))
        if i % 500 == 0:
            words.append(f"ЮУАЦФМШГ-{rnd.randrange(1000, 9999)}")
        text = " ".join(words)
        texts.append((f"raw{i:06d}", N_RAWS + 10 + i, T, text, f"h{i}", T, T))
        fts.append((N_RAWS + 10 + i, " ".join(stems.get(w) or lexical.stem(w) for w in words)))
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO raw_chunks (id, fts_rowid, source_kind, chat_id, author_tier, ts, text,"
        " content_hmac, policy_version, created_at, updated_at)"
        " VALUES (?, ?, 'chat', 'c2', 'user', ?, ?, ?, 0, ?, ?)",
        texts,
    )
    conn.executemany("INSERT INTO fts_raw (rowid, text) VALUES (?, ?)", fts)
    _insert(conn, [("r-deposit", "Открыла вклад ЖЖЖЖЖ-4860, договор 4860-4216")])
    conn.execute("COMMIT")
    yield conn
    conn.close()


def test_code_lookup_is_an_index_probe(big):
    seen: list[str] = []
    big.set_trace_callback(seen.append)
    try:
        got = commands._raws_with_codes(big, {"жжжжж-4860", "4860-4216"})
    finally:
        big.set_trace_callback(None)
    assert [r[0] for r in got] == ["r-deposit"]
    assert not any("instr(" in s for s in seen), seen


def test_code_lookup_p95_on_1e5_raws(big):
    codes = {"жжжжж-4860", "4860-4216", "юуацфмшг-1234"}
    times = []
    for _ in range(30):
        t = time.perf_counter()
        commands._raws_with_codes(big, codes)  # a fresh cache: one probe per code
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    p95 = times[int(0.95 * (len(times) - 1))]
    assert p95 <= 10, (p95, statistics.median(times))


def test_same_code_matches_only_the_code(tmp_path):
    """Token-exact: «4860-4216» links «договор 4860-4216», not «14860-42169»; a code split by
    another word is not the code."""
    conn = L.open_seeded(tmp_path)
    try:
        _insert(
            conn,
            [
                ("r-a", "Вклад ЖЖЖЖЖ-4860, договор 4860-4216"),
                ("r-b", "Напомни про договор 4860-4216 завтра"),
                ("r-c", "Счёт 14860-42169 оплачен"),
                ("r-d", "ЖЖЖЖЖ и 4860 отдельно"),
            ],
        )
        conn.commit()
        assert {r[0] for r in commands._raws_with_codes(conn, {"4860-4216"})} == {"r-a", "r-b"}
        assert {r[0] for r in commands._raws_with_codes(conn, {"жжжжж-4860"})} == {"r-a"}
        conn.execute("UPDATE raw_chunks SET status = 'forgotten' WHERE id = 'r-b'")
        assert {r[0] for r in commands._raws_with_codes(conn, {"4860-4216"})} == {"r-a"}
    finally:
        conn.close()
