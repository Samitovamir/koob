"""Pre-send check speed: the memory spec §7.4, G9 (store).

Spec marker: `[T perf/test_presend::p95]`: ≤ 5 ms at 10^5 records."""

from __future__ import annotations

import random
import statistics
import time

from koob.memory import store
from koob.memory.model import ReadRef, ReadSrc

N_RECORDS = 100_000
N_RAW = 20_000
N_DERIVED = 200
T = "2026-10-12T09:00:00.000000Z"


def _seed(conn) -> None:
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO records (id, fts_rowid, kind, version, status, author_tier, recorded_at,"
        " created_at, updated_at) VALUES (?, ?, 'fact', ?, ?, 'user', ?, ?, ?)",
        (
            (f"r{i:06d}", i + 10, i % 7 + 1, "active" if i % 11 else "superseded", T, T, T)
            for i in range(N_RECORDS)
        ),
    )
    conn.executemany(
        "INSERT INTO raw_chunks (id, fts_rowid, source_kind, author_tier, ts, text, content_hmac,"
        " policy_version, created_at, updated_at) VALUES (?, ?, 'chat', 'user', ?, ?, ?, 0, ?, ?)",
        ((f"w{i:06d}", i + 1, T, f"message {i}", f"h{i}", T, T) for i in range(N_RAW)),
    )
    conn.executemany(
        "INSERT INTO derived (id, dkind, content, read_set, built_at, created_at, updated_at)"
        " VALUES (?, 'core_section', 'c', '[]', ?, ?, ?)",
        ((f"d{i:04d}", T, T, T) for i in range(N_DERIVED)),
    )
    conn.executemany(
        "INSERT INTO pending_updates (thread_id, item_id, old_version, new_version, created_at)"
        " VALUES (?, ?, 1, 2, ?)",
        ((f"thread-{i}", f"r{i:06d}", T) for i in range(500)),
    )
    conn.execute("COMMIT")
    conn.execute("ANALYZE")


def _read_set(rng: random.Random) -> list[ReadRef]:
    """A realistic prompt: core view + basket (~60 records), a few raw quotes, core sections."""
    refs = []
    for i in rng.sample(range(N_RECORDS), 60):
        refs.append(ReadRef(id=f"r{i:06d}", version=i % 7 + 1))
    for i in rng.sample(range(N_RAW), 10):
        refs.append(ReadRef(id=f"w{i:06d}", version=1, src=ReadSrc.RAW))
    for i in rng.sample(range(N_DERIVED), 8):
        refs.append(ReadRef(id=f"d{i:04d}", version=0, src=ReadSrc.DERIVED))
    return refs


def test_p95(tmp_path):
    """§7.4: pre-send check ≤ 5 ms at 10^5 records."""
    conn = store.open(tmp_path / "memory" / "memory.db", home=tmp_path / "home")
    _seed(conn)
    n = conn.execute("SELECT count(*) FROM records WHERE id <> 'self'").fetchone()[0]
    assert n == N_RECORDS  # plus the user entity (§4.5)
    rng = random.Random(7)
    sets = [_read_set(rng) for _ in range(300)]
    for refs in sets[:20]:  # warm the page cache, as a live process is
        store.pre_send_check(conn, thread_id="thread-9999", read_set=refs)
    times = []
    stale_seen = 0
    for refs in sets:
        t0 = time.perf_counter()
        result = store.pre_send_check(conn, thread_id="thread-9999", read_set=refs)
        times.append((time.perf_counter() - t0) * 1000)
        stale_seen += len(result.stale)
    p95 = statistics.quantiles(times, n=20)[18]
    assert stale_seen > 0  # the superseded tenth is caught, so the query really ran
    assert p95 <= 5.0, f"pre-send p95 {p95:.2f} ms at 10^5 records"
    conn.close()


def test_watermark_read_under_1ms(tmp_path):
    """§7.7: the turn-start version and event read is < 1 ms."""
    conn = store.open(tmp_path / "memory" / "memory.db", home=tmp_path / "home")
    conn.executemany(
        "INSERT INTO events (ts, kind, memory_version, item_ids) VALUES (?, ?, ?, '[]')",
        ((T, "forget" if i % 50 == 0 else "write", i) for i in range(1, 20_001)),
    )
    conn.execute("UPDATE meta SET memory_version = 20000")
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        store.changes_since(conn, memory_version=20_000, last_event_id=20_000)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.median(times) < 1.0
    conn.close()
