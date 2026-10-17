"""Record and raw-message embeddings in a separate, rebuildable `vectors.db` keyed
(record_id, embedder_id); in-process embedder, fp16 matrix, brute force under a 40 ms
deadline; orphan sweep at open.

Spec: §3.1, §3.2 invariants, §9.1a, §10.2, §10.3, §18 (G12).

- **File.** `vectors.db` beside memory.db holds `vectors(record_id, embedder_id, version,
  labels, own, dim, vec, raw)` (fp16, L2-normalised) and `vmeta` (`schema`, `serving`). It is
  never backed up and may be deleted at any time: the next `embed` job rebuilds it.
- **Raw messages.** Usable raw chunks (active, not quarantined, not junk) are embedded too,
  in their own language, as rows with `raw = 1` keyed by the chunk id and versioned by its
  `rev` (a redaction bumps it: the redacted text is re-embedded, the old vector goes).
  `search_split` scores both kinds in one scan: the records' meaning lane and the raw one.
- **Pull model.** `embed_pending` embeds usable records (active/archived) that have no vector
  for the current embedder (or one of an older record version). It runs as the background job
  kind `embed` (`store.register_job_runner`, not at startup); `enqueue` files that job.
- **Versioned re-embed (G12).** `embedder_id` is `<name>@<recipe rev>`. When it changes, the
  old vectors keep serving until the new embedder covers every embeddable record; then
  `serving` switches, the old rows go and `meta.embedder_id` in memory.db follows.
- **Hot path.** `search` scans the in-RAM fp16 matrix in chunks and stops at its deadline
  (partial or empty, never blocking the turn). The matrix belongs to one attached file, named
  by its resolved path: SQLite reports a connection's file with symlinks resolved (`/var` is
  `/private/var` on macOS), so an unresolved name would never match the file a job wrote and
  RAM would keep the first load forever. Every `embed_pending` re-reads RAM when it wrote the
  attached file; a background load never replaces a newer index (a generation count).
  `embed_query` never loads a model on the turn: the first call starts a local-only
  background load and returns []. Without fastembed both return [] and log once; nothing
  here raises into recall.
- **Forget (§9.1a, §9.1d).** forget.py deletes rows of forgotten record ids; `stop_hook`
  drops the event's record and raw ids from RAM and deletes their rows from the attached
  file; `sweep_orphans` (startup and purge) deletes rows and RAM entries of records and raws
  that are not usable (and raw rows of an older `rev`), which heals a crash between the
  ledger commit and the vector delete.
- **Labels.** Rows carry the record's labels and own-ness, refreshed by every `embed_pending`;
  the mask here is the finders' fence (a superset), use-now (`policy.decide`) still decides.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .. import store

log = logging.getLogger(__name__)

SCHEMA = 1
JOB_KIND = "embed"
CHUNK_ROWS = 16_384  # rows converted fp16 -> fp32 per step; ~1.3 ms each at 768-d
BATCH = 32  # documents per embedding call and per vectors.db transaction
_USABLE_SQL = "('active', 'archived')"
_OWN_TIERS = ("user", "user_confirmed")  # policy._OWN_SQL: user's own items pass the label fence
_SQL_CHUNK = 400
RAW_CHARS = 1_000  # a raw message is embedded from its first characters (chunks are short)
_RAW_USABLE_SQL = "status = 'active' AND quarantined = 0 AND junk = 0 AND text IS NOT NULL"

# The text each record kind is embedded from (records are stored in English, §10.3).
_TEXT_SQL = (
    "SELECT r.id, r.version, r.labels, r.author_tier,"
    " coalesce(f.text_en, i.text_en, d.chosen_text, lp.text_en, ls.rule_en, c.text_en,"
    " e.summary_en) AS text"
    " FROM records r"
    " LEFT JOIN facts f ON f.id = r.id LEFT JOIN impressions i ON i.id = r.id"
    " LEFT JOIN decisions d ON d.id = r.id LEFT JOIN loops lp ON lp.id = r.id"
    " LEFT JOIN lessons ls ON ls.id = r.id LEFT JOIN criteria c ON c.id = r.id"
    " LEFT JOIN episodes e ON e.id = r.id"
    f" WHERE r.status IN {_USABLE_SQL}"
)
# Raw messages are embedded from their own text, in the language they were written in.
_RAW_TEXT_SQL = (
    "SELECT c.id, c.rev, c.labels, c.author_tier, substr(c.text, 1, ?) AS text"
    " FROM raw_chunks c WHERE c.status = 'active' AND c.quarantined = 0 AND c.junk = 0"
    " AND c.text IS NOT NULL"
)

_DDL = """
CREATE TABLE IF NOT EXISTS vmeta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vectors (
    record_id    TEXT NOT NULL,
    embedder_id  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    labels       INTEGER NOT NULL DEFAULT 0,
    own          INTEGER NOT NULL DEFAULT 0,
    dim          INTEGER NOT NULL,
    vec          BLOB NOT NULL,
    embedded_at  TEXT NOT NULL,
    raw          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (record_id, embedder_id)
);
CREATE INDEX IF NOT EXISTS vectors_embedder ON vectors (embedder_id);
"""

# The clock the deadline reads; tests replace it.
_now: Callable[[], float] = time.perf_counter


# ---------------------------------------------------------------- embedders


class Embedder(Protocol):
    id: str  # versioned: "<name>@<rev>"
    dim: int

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class EmbedderSpec:
    name: str
    fastembed_id: str
    dim: int
    query_prefix: str = ""
    doc_prefix: str = ""
    rev: int = 1  # bump when the text recipe or prefixes change: forces a re-embed
    custom: dict[str, Any] | None = None  # add_custom_model kwargs when fastembed lacks it

    @property
    def id(self) -> str:
        return f"{self.name}@{self.rev}"


# Primary embedder is embeddinggemma-300m; multilingual-e5-small for emergencies only.
PRIMARY = EmbedderSpec(
    "embeddinggemma-300m",
    "google/embeddinggemma-300m",
    768,
    query_prefix="task: search result | query: ",
    doc_prefix="title: none | text: ",
)
FALLBACK = EmbedderSpec(
    "multilingual-e5-small",
    "intfloat/multilingual-e5-small",
    384,
    query_prefix="query: ",
    doc_prefix="passage: ",
    custom={"pooling": "MEAN", "normalization": True, "model_file": "onnx/model.onnx"},
)
SPECS = {s.name: s for s in (PRIMARY, FALLBACK)}


def cache_dir() -> Path:
    """Where fastembed keeps models: `KOOB_EMBED_CACHE`, `FASTEMBED_CACHE_PATH`, else
    the user's Caches folder (re-downloadable, never in backups)."""
    env = os.environ.get("KOOB_EMBED_CACHE") or os.environ.get("FASTEMBED_CACHE_PATH")
    return Path(env) if env else Path.home() / "Library" / "Caches" / "koob" / "fastembed"


class FastEmbedder:
    """An in-process fastembed ONNX model (`uv sync --extra memory`)."""

    def __init__(self, spec: EmbedderSpec, *, local_only: bool = True) -> None:
        from fastembed import TextEmbedding

        if spec.custom:
            from fastembed.common.model_description import ModelSource, PoolingType

            known = {m["model"] for m in TextEmbedding.list_supported_models()}
            if spec.fastembed_id not in known:
                c = spec.custom
                TextEmbedding.add_custom_model(
                    model=spec.fastembed_id,
                    pooling=getattr(PoolingType, c["pooling"]),
                    normalization=c["normalization"],
                    sources=ModelSource(hf=spec.fastembed_id),
                    dim=spec.dim,
                    model_file=c["model_file"],
                )
        self.spec = spec
        self.id = spec.id
        self.dim = spec.dim
        self.model = TextEmbedding(
            model_name=spec.fastembed_id, cache_dir=str(cache_dir()), local_files_only=local_only
        )

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray:
        docs = [self.spec.doc_prefix + t for t in texts]
        return np.array(list(self.model.embed(docs, batch_size=max(1, len(docs)))), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(next(iter(self.model.embed([self.spec.query_prefix + text]))), np.float32)


class HashEmbedder:
    """Deterministic offline embedder (hashed bag of words): tests and smoke runs only."""

    def __init__(self, id: str = "hash@1", dim: int = 64) -> None:  # noqa: A002
        self.id = id
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"\w+", (text or "").casefold()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 40) & 1 else -1.0
        return v

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)


_lock = threading.RLock()
_injected: list[Embedder | None] = [None]
_models: dict[str, Embedder] = {}
_failed: set[str] = set()  # local-only loads that found no model files
_loading: set[str] = set()
_logged: set[str] = set()


def _log_once(key: str, msg: str, *args: object) -> None:
    if key not in _logged:
        _logged.add(key)
        log.warning(msg, *args)


def _fastembed_present() -> bool:
    return importlib.util.find_spec("fastembed") is not None


def set_embedder(embedder: Embedder | None) -> None:
    """Inject the embedder (tests, eval); None goes back to the configured fastembed model."""
    _injected[0] = embedder


def configured_spec() -> EmbedderSpec:
    """`KOOB_EMBEDDER` names an emergency override; otherwise the primary."""
    return SPECS.get(os.environ.get("KOOB_EMBEDDER", ""), PRIMARY)


def _load(spec: EmbedderSpec, *, allow_download: bool) -> Embedder | None:
    try:
        emb = FastEmbedder(spec, local_only=not allow_download)
    except Exception as exc:  # no model files, broken cache, onnx error: never into recall
        with _lock:
            _failed.add(spec.id)
        _log_once(
            f"load:{spec.id}:{allow_download}",
            "memory vectors: %s not loaded (%s)",
            spec.id,
            type(exc).__name__,
        )
        return None
    with _lock:
        _models[spec.id] = emb
        _failed.discard(spec.id)
    return emb


def _bg_load(spec: EmbedderSpec) -> None:
    try:
        _load(spec, allow_download=False)
    finally:
        with _lock:
            _loading.discard(spec.id)


def _kick_model(spec: EmbedderSpec) -> None:
    """Start a local-only background load of one model (once; not after a miss)."""
    with _lock:
        if spec.id in _models or spec.id in _loading or spec.id in _failed:
            return
        _loading.add(spec.id)
    threading.Thread(target=_bg_load, args=(spec,), daemon=True).start()


def current_embedder(*, wait: bool = False, allow_download: bool = False) -> Embedder | None:
    """The embedder new vectors are made with. `wait=False` (the turn): a model not yet in
    RAM starts loading from local files in the background and None comes back at once."""
    if _injected[0] is not None:
        return _injected[0]
    if not _fastembed_present():
        _log_once("fastembed", "memory vectors: fastembed is not installed; vector search off")
        return None
    spec = configured_spec()
    with _lock:
        if spec.id in _models:
            return _models[spec.id]
        if not wait:
            _kick_model(spec)
            return None
    if spec.id in _failed and not allow_download:
        return None
    return _load(spec, allow_download=allow_download)


def embedder_ready() -> bool:
    """Whether an embedder is in RAM now (injected or loaded); never starts a load."""
    if _injected[0] is not None:
        return True
    with _lock:
        return configured_spec().id in _models


def _embedder_for(embedder_id: str | None) -> Embedder | None:
    """The embedder whose vectors are serving (queries must match them), else the current."""
    inj = _injected[0]
    if inj is not None:
        return inj
    if embedder_id and embedder_id in _models:
        return _models[embedder_id]
    old = next((sp for sp in SPECS.values() if sp.id == embedder_id), None)
    if old is not None and old.id != configured_spec().id and _fastembed_present():
        _kick_model(old)  # G12: the old vectors keep serving while the new ones are built
        return None
    return current_embedder(wait=False)


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float32)
    if m.ndim == 1:
        n = float(np.linalg.norm(m))
        return m / n if n > 0 else m
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def embed_query(text: str) -> list[float]:
    """The query vector for the serving index; [] when no embedder is ready (never blocks,
    never raises)."""
    try:
        idx = _index[0]
        emb = _embedder_for(idx.embedder_id if idx is not None else None)
        if emb is None or not (text or "").strip():
            return []
        if idx is not None and emb.id != idx.embedder_id:
            return []  # another embedder's space: no meaning search until vectors match
        return _unit(emb.embed_query(text)).tolist()
    except Exception as exc:
        _log_once("query", "memory vectors: query embedding failed (%s)", type(exc).__name__)
        return []


# ---------------------------------------------------------------- vectors.db


def open_db(path: Path | str) -> sqlite3.Connection:
    """vectors.db with the §3.1 pragmas and this module's schema (a stale schema is dropped:
    the file is rebuildable)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = store.connect(p)
    _ensure_schema(conn)
    return conn


def _ensure_schema(vdb: sqlite3.Connection) -> None:
    have = vdb.execute("SELECT name FROM sqlite_schema WHERE name = 'vmeta'").fetchone()
    if have is not None:
        row = vdb.execute("SELECT value FROM vmeta WHERE key = 'schema'").fetchone()
        if row is not None and int(row[0]) == SCHEMA:
            cols = {r[1] for r in vdb.execute("PRAGMA table_info(vectors)")}
            if cols and "raw" not in cols:  # a file from before raw rows: record rows stay
                vdb.execute("ALTER TABLE vectors ADD COLUMN raw INTEGER NOT NULL DEFAULT 0")
            return
        vdb.executescript("DROP TABLE IF EXISTS vectors; DROP TABLE IF EXISTS vmeta;")
    vdb.executescript(_DDL)
    vdb.execute("INSERT OR REPLACE INTO vmeta (key, value) VALUES ('schema', ?)", (str(SCHEMA),))


def _has_schema(vdb: sqlite3.Connection) -> bool:
    return vdb.execute("SELECT 1 FROM sqlite_schema WHERE name = 'vectors'").fetchone() is not None


def serving_id(vdb: sqlite3.Connection) -> str | None:
    if not _has_schema(vdb):
        return None
    row = vdb.execute("SELECT value FROM vmeta WHERE key = 'serving'").fetchone()
    return str(row[0]) if row else None


def _set_serving(vdb: sqlite3.Connection, embedder_id: str) -> None:
    vdb.execute("INSERT OR REPLACE INTO vmeta (key, value) VALUES ('serving', ?)", (embedder_id,))


def _usable(memory: sqlite3.Connection, ids: Iterable[str]) -> set[str]:
    """The ids that may hold a vector: usable records and usable raw chunks."""
    return set(_usable_versions(memory, ids))


def _usable_versions(memory: sqlite3.Connection, ids: Iterable[str]) -> dict[str, int | None]:
    """Usable ids -> the raw chunk's current `rev` (None for a record, whose version
    `embed_pending` checks)."""
    ids = list(ids)
    out: dict[str, int | None] = {}
    for i in range(0, len(ids), _SQL_CHUNK):
        chunk = ids[i : i + _SQL_CHUNK]
        qs = ",".join("?" * len(chunk))
        for r in memory.execute(
            f"SELECT id FROM records WHERE id IN ({qs}) AND status IN {_USABLE_SQL}", chunk
        ):
            out[r[0]] = None
        for r in memory.execute(
            f"SELECT id, rev FROM raw_chunks WHERE id IN ({qs}) AND {_RAW_USABLE_SQL}", chunk
        ):
            out[r[0]] = int(r[1])
    return out


def _delete_ids(
    vdb: sqlite3.Connection, ids: Sequence[str], embedder_id: str | None = None
) -> None:
    for i in range(0, len(ids), _SQL_CHUNK):
        chunk = list(ids[i : i + _SQL_CHUNK])
        qs = ",".join("?" * len(chunk))
        if embedder_id is None:
            vdb.execute(f"DELETE FROM vectors WHERE record_id IN ({qs})", chunk)
        else:
            vdb.execute(
                f"DELETE FROM vectors WHERE embedder_id = ? AND record_id IN ({qs})",
                [embedder_id, *chunk],
            )


def _tx(vdb: sqlite3.Connection) -> Any:
    class _T:
        def __enter__(self) -> sqlite3.Connection:
            vdb.execute("BEGIN IMMEDIATE")
            return vdb

        def __exit__(self, et: object, *_: object) -> None:
            vdb.execute("ROLLBACK" if et is not None else "COMMIT")

    return _T()


def embed_pending(
    memory: sqlite3.Connection,
    vectors: sqlite3.Connection,
    *,
    batch_size: int = BATCH,
    on_batch: Callable[[int], None] | None = None,
) -> int:
    """Embed usable records and raw chunks lacking a vector for the current embedder (or
    holding one of an older version); refresh labels of the rest; switch `serving` once a new
    embedder covers everything (G12). Returns the vectors written. Background only."""
    emb = current_embedder(wait=True, allow_download=True)
    if emb is None:
        return 0
    _ensure_schema(vectors)
    serving = serving_id(vectors)
    if serving is None:
        with _tx(vectors):
            _set_serving(vectors, emb.id)
        serving = emb.id
    have = {
        r[0]: (int(r[1]), int(r[2]), int(r[3]))
        for r in vectors.execute(
            "SELECT record_id, version, labels, own FROM vectors WHERE embedder_id = ?", (emb.id,)
        )
    }
    todo: list[tuple[str, int, int, int, str, int]] = []
    relabel: list[tuple[int, int, str]] = []
    sources = (
        (memory.execute(_TEXT_SQL), 0),
        (memory.execute(_RAW_TEXT_SQL, (RAW_CHARS,)), 1),
    )
    for rows, is_raw in sources:
        for rid, ver, labels, tier, text in rows:
            if not (text or "").strip():
                continue
            own = int(tier in _OWN_TIERS)
            old = have.get(rid)
            if old is None or old[0] != int(ver):
                todo.append((rid, int(ver), int(labels), own, text, is_raw))
            elif old[1:] != (int(labels), own):
                relabel.append((int(labels), own, rid))
    if relabel:
        with _tx(vectors):
            vectors.executemany(
                "UPDATE vectors SET labels = ?, own = ? WHERE record_id = ? AND embedder_id = ?",
                [(*r, emb.id) for r in relabel],
            )
    written = 0
    for i in range(0, len(todo), max(1, batch_size)):
        batch = todo[i : i + max(1, batch_size)]
        mat = _unit(emb.embed_docs([b[4] for b in batch])).astype("<f2")
        stamp = store.now()
        with _tx(vectors):
            vectors.executemany(
                "INSERT OR REPLACE INTO vectors (record_id, embedder_id, version, labels, own,"
                " dim, vec, embedded_at, raw) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (b[0], emb.id, b[1], b[2], b[3], emb.dim, mat[j].tobytes(), stamp, b[5])
                    for j, b in enumerate(batch)
                ],
            )
        # a record or raw forgotten (or redacted) while we embedded: its row goes now
        # (forget deletes first, so one of the two always sees the other)
        live = _usable_versions(memory, [b[0] for b in batch])
        gone = sorted(b[0] for b in batch if b[0] not in live or (b[5] and live[b[0]] != b[1]))
        if gone:
            with _tx(vectors):
                _delete_ids(vectors, gone, emb.id)
        written += len(batch) - len(gone)
        if on_batch is not None:
            on_batch(written)
    if serving != emb.id:
        with _tx(vectors):
            _set_serving(vectors, emb.id)
            vectors.execute("DELETE FROM vectors WHERE embedder_id != ?", (emb.id,))
        serving = emb.id
    _note_embedder(memory, serving)
    _refresh_ram(vectors)
    return written


def _note_embedder(memory: sqlite3.Connection, embedder_id: str) -> None:
    """`meta.embedder_id` in memory.db names the serving embedder (doctor, §18)."""
    row = memory.execute("SELECT embedder_id FROM meta WHERE id = 1").fetchone()
    if row is not None and row[0] != embedder_id:
        with store.write_tx(memory, priority="background"):
            memory.execute(
                "UPDATE meta SET embedder_id = ?, updated_at = ? WHERE id = 1",
                (embedder_id, store.now()),
            )


def sweep_orphans(memory: sqlite3.Connection, vectors: sqlite3.Connection) -> int:
    """Delete vectors of records that are not usable (crash between commit and delete),
    in vectors.db and in RAM; attaches the hot path to this file. Returns rows deleted."""
    if memory is None or vectors is None:
        return 0
    path = _resolved(store.db_file(vectors))
    if path is not None and _attached[0] is None:
        attach(path)
    ram = _index[0]
    if ram is not None and path is not None and _attached[0] == path:
        ram.drop(set(ram.ids) - _usable(memory, ram.ids))
    if not _has_schema(vectors):
        return 0
    rows = vectors.execute("SELECT record_id, version, raw FROM vectors").fetchall()
    live = _usable_versions(memory, {r[0] for r in rows})
    bad = sorted({r[0] for r in rows if r[0] not in live or (r[2] and live[r[0]] != int(r[1]))})
    if bad:
        with _tx(vectors):
            _delete_ids(vectors, bad)
    return len(bad)


# ---------------------------------------------------------------- jobs (G12)


def enqueue(memory: sqlite3.Connection) -> bool:
    """File one `embed` pull job unless one is already queued or running. True when filed.
    (A pull covers everything pending, so one waiting job is enough; after vectors.db is
    deleted the next one rebuilds it.)"""
    stamp = store.now()
    key = f"{JOB_KIND}:{store.memory_version(memory)}:{secrets.token_hex(6)}"
    with store.write_tx(memory, priority="background"):
        waiting = memory.execute(
            "SELECT 1 FROM jobs WHERE jkind = ? AND state IN ('queued', 'running') LIMIT 1",
            (JOB_KIND,),
        ).fetchone()
        if waiting is not None:
            return False
        memory.execute(
            "INSERT INTO jobs (id, jkind, payload, idempotency_key, created_at, updated_at)"
            " VALUES (?, ?, '{}', ?, ?, ?)",
            (f"job-{key.replace(':', '-')}", JOB_KIND, key, stamp, stamp),
        )
    return True


def _run_job(memory: sqlite3.Connection, job: dict[str, object]) -> object:
    db = store.db_file(memory)
    if db is None:
        return 0
    vdb = open_db(store.MemoryPaths.of(db).vectors)
    job_id = str(job.get("id") or "")
    try:
        return embed_pending(
            memory, vdb, on_batch=lambda _n: store.renew_lease(memory, job_id) if job_id else None
        )
    finally:
        vdb.close()


store.register_job_runner(JOB_KIND, _run_job)


# ---------------------------------------------------------------- the in-RAM index


class _Index:
    """fp16 matrix + ids + labels + own + raw + alive, for one embedder. Copy-on-grow, so a
    search holding the old arrays never sees a half-built one."""

    def __init__(
        self,
        embedder_id: str,
        dim: int,
        ids: list[str],
        mat: np.ndarray,
        labels: np.ndarray,
        own: np.ndarray,
        raw: np.ndarray | None = None,
    ) -> None:
        self.embedder_id = embedder_id
        self.dim = dim
        self.ids = ids
        self.pos = {rid: i for i, rid in enumerate(ids)}
        self.mat = mat
        self.labels = labels
        self.own = own
        self.raw = raw if raw is not None else np.zeros(len(ids), bool)
        self.alive = np.ones(len(ids), bool)

    def drop(self, ids: Iterable[str]) -> None:
        for rid in ids:
            i = self.pos.get(rid)
            if i is not None:
                self.alive[i] = False


_index: list[_Index | None] = [None]
_attached: list[Path | None] = [None]  # resolved (`_resolved`), so it compares with db_file
_load_started: list[bool] = [False]
_generation: list[int] = [0]  # bumped by every attach and refresh: an older load is dropped


def _resolved(path: Path | str | None) -> Path | None:
    """`path` with symlinks resolved (the file need not exist yet), as SQLite names it."""
    return Path(path).resolve() if path is not None else None


def _read_index(vdb: sqlite3.Connection) -> _Index | None:
    sid = serving_id(vdb)
    if sid is None:
        return None
    rows = vdb.execute(
        "SELECT record_id, labels, own, dim, vec, raw FROM vectors WHERE embedder_id = ?"
        " ORDER BY record_id",
        (sid,),
    ).fetchall()
    dims = {int(r[3]) for r in rows}
    if len(dims) > 1:
        return None
    dim = dims.pop() if dims else 0
    mat = (
        np.frombuffer(b"".join(r[4] for r in rows), dtype="<f2").reshape(len(rows), dim)
        if rows
        else np.zeros((0, 0), "<f2")
    )
    return _Index(
        sid,
        dim,
        [r[0] for r in rows],
        mat,
        np.array([int(r[1]) for r in rows], np.int64),
        np.array([bool(r[2]) for r in rows], bool),
        np.array([bool(r[5]) for r in rows], bool),
    )


def attach(path: Path | str | None, *, load: bool = False) -> None:
    """Point the hot path at a vectors.db (None detaches). `load=True` reads it now;
    otherwise the first search starts a background load."""
    with _lock:
        _attached[0] = _resolved(path)
        _index[0] = None
        _load_started[0] = False
        _generation[0] += 1
    if load and path is not None:
        load_index()


def load_index() -> bool:
    """Read the attached vectors.db into RAM (background or warm-up, never the turn). A load
    that finds no file or no index lets the next search try again; one that an attach or a
    refresh overtook while it read keeps its result out of RAM."""
    with _lock:
        path, gen = _attached[0], _generation[0]
    idx: _Index | None = None
    try:
        if path is not None and path.exists():
            vdb = store.connect(path)
            try:
                idx = _read_index(vdb) if _has_schema(vdb) else None
            finally:
                vdb.close()
    except Exception as exc:
        _log_once("load_index", "memory vectors: index not loaded (%s)", type(exc).__name__)
    with _lock:
        if _generation[0] == gen and _attached[0] == path:
            _index[0] = idx
            _load_started[0] = idx is not None
    return idx is not None


def _refresh_ram(vdb: sqlite3.Connection) -> None:
    """After `embed_pending`: re-read the index when the hot path uses this file (or uses
    none yet), so the next turn's search sees every row this pass wrote."""
    path = _resolved(store.db_file(vdb))
    if path is None:
        return
    if _attached[0] is None:
        attach(path)
    if _attached[0] == path:
        idx = _read_index(vdb)
        with _lock:
            if _attached[0] == path:
                _index[0] = idx
                _load_started[0] = True
                _generation[0] += 1


def stop_hook(event: Any) -> None:
    """§9.1d: the forget cache hook (`api.wire(stop_hooks=[vectors.stop_hook])`). Forgotten
    and redacted raw chunks leave RAM and the attached vectors.db at once (forget.py deletes
    the record rows itself; the purge's orphan sweep re-checks both)."""
    raw_ids = list(getattr(event, "raw_ids", None) or [])
    drop([*(getattr(event, "record_ids", None) or []), *raw_ids])
    path = _attached[0]
    if not raw_ids or path is None or not path.exists():
        return
    try:
        vdb = store.connect(path)
        try:
            if _has_schema(vdb):
                with _tx(vdb):
                    _delete_ids(vdb, raw_ids)
        finally:
            vdb.close()
    except Exception as exc:  # the purge's sweep deletes them; a forget never fails here
        _log_once("stop_hook", "memory vectors: raw rows not deleted at stop (%s)",
                  type(exc).__name__)  # fmt: skip


def drop(record_ids: Iterable[str]) -> None:
    idx = _index[0]
    if idx is not None:
        idx.drop(record_ids)


def reset() -> None:
    """Forget all module state (tests)."""
    with _lock:
        _injected[0] = None
        _index[0] = None
        _attached[0] = None
        _load_started[0] = False
        _generation[0] += 1
        _models.clear()
        _failed.clear()
        _logged.clear()


# ---------------------------------------------------------------- search (§10.2)


def search(
    query_vec: list[float], *, label_mask: int, deadline_ms: float = 40.0, limit: int = 50
) -> list[tuple[str, float]]:
    """Brute-force cosine over the in-RAM fp16 matrix: `(record_id, score)` best first, only
    record rows whose labels fit `label_mask` (or the user's own items). Past the deadline the
    scan stops: what was scored so far comes back (possibly nothing). Never raises, never blocks."""
    return search_split(query_vec, label_mask=label_mask, deadline_ms=deadline_ms,
                        limit=limit, raw_limit=0)[0]  # fmt: skip


def search_split(
    query_vec: list[float],
    *,
    label_mask: int,
    deadline_ms: float = 40.0,
    limit: int = 50,
    raw_limit: int = 50,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """One scan, two lanes: the best `limit` record rows and the best `raw_limit` raw-message
    rows, each `(id, score)` best first, under the same label fence and deadline as
    `search`. Never raises, never blocks."""
    t0 = _now()
    try:
        idx = _index[0]
        if idx is None:
            _kick_load()
            return [], []
        if not query_vec or len(query_vec) != idx.dim or not idx.ids:
            return [], []
        if limit <= 0 and raw_limit <= 0:
            return [], []
        q = _unit(np.asarray(query_vec, np.float32))
        n = len(idx.ids)
        scores = np.full(n, -np.inf, np.float32)
        budget = deadline_ms / 1000.0
        done = 0
        for start in range(0, n, CHUNK_ROWS):
            if _now() - t0 >= budget:
                log.debug("memory vectors: deadline after %d of %d rows", done, n)
                break
            end = min(n, start + CHUNK_ROWS)
            scores[start:end] = idx.mat[start:end].astype(np.float32) @ q
            done = end
        if done == 0:
            return [], []
        ok = idx.alive[:done] & (((idx.labels[:done] & ~int(label_mask)) == 0) | idx.own[:done])
        is_raw = idx.raw[:done]
        return (
            _best(idx, scores, ok & ~is_raw, limit),
            _best(idx, scores, ok & is_raw, raw_limit),
        )
    except Exception as exc:
        _log_once("search", "memory vectors: search failed (%s)", type(exc).__name__)
        return [], []


def _best(idx: _Index, scores: np.ndarray, ok: np.ndarray, limit: int) -> list[tuple[str, float]]:
    cand = np.flatnonzero(ok)
    if cand.size == 0 or limit <= 0:
        return []
    s = scores[cand]
    k = min(limit, cand.size)
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top], kind="stable")]
    return [(idx.ids[int(cand[t])], float(s[t])) for t in top]


def _kick_load() -> None:
    with _lock:
        if _attached[0] is None or _load_started[0]:
            return
        _load_started[0] = True
    threading.Thread(target=load_index, daemon=True).start()
