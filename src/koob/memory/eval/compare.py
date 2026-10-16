"""COMPARE-MEM: ready-made memories against the Cube memory, on the same dev scenarios.

Two more bench SUTs next to `plain`, `legacy` and `api` (bench.py), fed by the same replay:

- `FileMemoryAgentSUT` ("filemem"): a notes file kept by a model. At each schedule point
  (where the api SUT's extraction runs) and right after a memory command of the user
  (remember / fix / forget / undo / a chat setting), the model gets the notes and the new messages
  and returns the whole updated notes. The brain sees the notes plus a grep over them.
- `GraphitiSUT` ("graphiti"): graphiti-core on an embedded Kuzu file, LLM and embedder on local
  Ollama under the memory guard. The same batches become Graphiti episodes; the brain sees
  Zep-style facts with their date ranges plus entity summaries for the question.

Both apply the same glue at admission: rows from a chat set to `banned` are dropped, and in a
`no_dig` chat only the user's words are kept. Everything else is the engine's own work.

Then, on a fixed probe subset, the answer level: one brain (a model through `run_role`) answers
from each SUT's captured context, and a judge grades it against the gold (code checks first, a
model only when they cannot decide). Every model output is cached on disk, keyed by (role, model,
prompt hash, input), so a rerun makes no call.

Run one SUT per process (its own data dirs), e.g.
`python -m koob.memory.eval.compare run --sut filemem --out run.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .bench import ApiBenchSUT, BenchRunner, _Api, _BenchSUT, _blocks, _stale_values, judge_view
from .bench import summarize as bench_summarize
from .graders import SKIP
from .loader import LoadedSet, load_set
from .matchers import describe
from .report import WordingResult
from .schema import Probe, Scenario
from .sut import Guarded

ROLE_FILEMEM = "memory_filemem"
FILEMEM_MODEL = "claude-opus-5"  # the same model as our extractor (workers.extract.CLOUD_MODEL)
ROLE_BRAIN = "memory_compare_brain"
BRAIN_MODEL = "claude-sonnet-5"
ROLE_JUDGE = "memory_compare_judge"
JUDGE_MODEL = "claude-sonnet-5"  # no local Qwen fits under the memory guard; a cloud model judges

GRAPHITI_LLM = "gemma4:e4b"
GRAPHITI_EMBEDDER = "bge-m3"
OLLAMA = "http://127.0.0.1:11434"
BIG_MODEL_BYTES = 3_000_000_000  # an embedder under this is not a "big model" for the guard

# guarantee -> plain words (the panel)
GUARANTEES = {
    "G1": "after «забудь X», X is nowhere in what the brain sees",
    "G2": "pasted foreign text is not presented as their words",
    "G3": "a password or code is never shown",
    "G4": "a sensitive fact stays out of a send turn / unrelated advice",
    "G5": "a corrected value is never shown as current",
    "G6": "one correction updates a dependent conclusion",
}
PANEL: dict[tuple[str, str], str] = {
    ("inspect-d06", "p2"): "G1",
    ("inspect-d06", "p5"): "G1",
    ("keeps_true-d03", "p1"): "G1",
    ("keeps_true-d03", "p3"): "G1",
    ("poisoning-d01", "p2"): "G2",
    ("poisoning-d02", "p3"): "G2",
    ("understands-d03", "p2"): "G2",
    ("codes-c01", "p1"): "G3",
    ("codes-c01", "p2"): "G3",
    ("policy_leaks-d01", "p1"): "G3",
    ("policy_leaks-d02", "p2"): "G4",
    ("policy_leaks-d02", "p3"): "G4",
    ("keeps_true-d01", "p1"): "G5",
    ("safety_net-d02", "p3"): "G5",
    ("safety_net-d04", "p3"): "G5",
    ("safety_net-d05", "p3"): "G5",
    ("scout-d01", "r1"): "G5",
    ("scout-d02", "r1"): "G5",
    ("keeps_true-d02", "p4"): "G6",
    ("inspect-d04", "p1"): "G6",
    ("e2e-d01", "p4"): "G6",
    ("e2e-d02", "p2"): "G6",
    ("work-d02", "p1"): "G6",
}
ACCURACY: list[tuple[str, str]] = [
    ("knows-d01", "p1"),
    ("knows-d01", "p3"),
    ("knows-d02", "p1"),
    ("safety_net-d01", "p1"),
    ("safety_net-d07", "p2"),
    ("work-d01", "p2"),
    ("work-d03", "p2"),
    ("understands-d02", "p4"),
]
ANSWER_SET: set[tuple[str, str]] = set(PANEL) | set(ACCURACY)


class NotesOut(BaseModel):
    notes: str


class AnswerOut(BaseModel):
    answer: str


class VerdictOut(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str = ""


# ---------------------------------------------------------------------- cached model calls


def _prompt_hash(role: str) -> str:
    try:
        from koob.llm.runner import prompt_for

        text = prompt_for(role)
    except Exception:  # noqa: BLE001 - no prompt file (tests): the role name stands in
        text = role
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RoleCall:
    """`run_role` with an on-disk cache keyed by (role, model, prompt hash, input text)."""

    def __init__(
        self,
        role: str,
        out_model: type[BaseModel],
        model: str,
        root: Path,
        runner: Callable[..., tuple[BaseModel, dict]] | None = None,
    ) -> None:
        self.role, self.out_model, self.model = role, out_model, model
        self.root = Path(root)
        self._runner = runner
        self.prompt = _prompt_hash(role) if runner is None else role
        self.calls = 0
        self.hits = 0
        self.cost_usd = 0.0
        self.seconds = 0.0

    def path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.role}|{self.model}|{self.prompt}|{text}".encode()).hexdigest()
        return self.root / self.role / f"{key[:40]}.json"

    def __call__(self, text: str) -> dict[str, Any]:
        path = self.path(text)
        if path.exists():
            try:
                out = json.loads(path.read_text(encoding="utf-8"))
                self.hits += 1
                return out
            except (OSError, ValueError):
                pass
        runner = self._runner
        if runner is None:
            from koob.llm.runner import run_role

            runner = run_role
        t0 = time.perf_counter()
        obj, meta = runner(self.role, self.out_model, text, model=self.model, lenient=True)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.cost_usd += float((meta or {}).get("cost_usd") or 0.0)
        out = obj.model_dump(mode="json")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "hits": self.hits,
            "cost_usd": round(self.cost_usd, 4),
            "seconds": round(self.seconds, 1),
        }


# ---------------------------------------------------------------------- batching SUTs


class _BatchApi(_Api):
    def ingest_turn(self, **kw: Any) -> None:
        super().ingest_turn(**kw)
        self._s.after_ingest()  # type: ignore[attr-defined]

    def set_chat_mode(self, chat: str, mode: Any, **_: Any) -> None:
        self._s.on_chat_mode(chat, getattr(mode, "value", str(mode)))


class _BatchSUT(_BenchSUT):
    """Messages wait in a batch; the batch is folded into the memory at every schedule point
    (the api SUT's extraction points) and right after a user memory command."""

    def __init__(self) -> None:
        super().__init__()
        self.api = Guarded(_BatchApi(self))
        self.pending: list[str] = []
        self.urgent = False
        self.modes: dict[str, str] = {}
        self.flushes = 0
        self.dropped = 0
        self.proc_s = 0.0
        self.errors: list[str] = []
        self.clock: Any = None

    def start(self, scn: Scenario, clock: Any, scheduler: Any, workdir: Path) -> None:
        super().start(scn, clock, scheduler, workdir)
        self.clock = clock

    def now(self) -> str:
        return self.clock.now().isoformat() if self.clock is not None else ""

    def seed(self, scn: Scenario, personas: dict[str, Any]) -> None:
        for chat, mode in scn.setup.chat_policy.items():
            self.on_chat_mode(chat, getattr(mode, "value", str(mode)))
        for item in scn.setup.known:
            text = item.get("text") if isinstance(item, dict) else str(item)
            self.pending.append(f"[{self.now()}] owner (asked to remember): {text}")
            self.urgent = True
        if self.urgent:
            self.flush()

    def admit(self, row: dict[str, Any], turn_id: str | None = None) -> None:
        text = str(row.get("text") or row.get("caption") or "")
        role = row.get("role")
        if not text or role not in ("owner", "other", "tool", "cube"):
            return
        chat = str(row.get("chat_id") or row.get("source") or "")
        mode = self.modes.get(chat, "normal")
        if role != "owner" and (mode == "banned" or (mode == "no_dig" and role == "other")):
            self.dropped += 1
            return
        if role == "owner":
            who = "owner"
        elif role == "cube":
            who = "assistant (its reply)"
        elif role == "tool":
            who = f"{row.get('tool') or 'tool'} result from {row.get('source') or chat}"
        else:
            who = f"{row.get('sender') or '?'} in {chat}"
        self.pending.append(f"[{row.get('ts', '')}] {who}: {text}")
        self.admitted += 1

    def on_remember(self, text: str) -> None:
        self.urgent = True

    def on_fix(self, correction: str) -> None:
        self.urgent = True

    def on_forget(self, target: str) -> None:
        self.urgent = True

    def on_undo(self) -> None:
        self.urgent = True

    def on_chat_mode(self, chat: str, mode: str) -> None:
        self.modes[chat] = mode
        self.pending.append(f"[{self.now()}] settings: the owner set chat {chat} to mode '{mode}'")
        self.urgent = True

    def after_ingest(self) -> None:
        if self.urgent:
            self.flush()

    def run_worker(self, job: str, phase: str = "run", token: str | None = None) -> None:
        if phase != "snapshot":
            self.flush()

    def flush(self) -> None:
        self.urgent = False
        if not self.pending:
            return
        batch, self.pending = self.pending, []
        t0 = time.perf_counter()
        try:
            self.ingest(batch)
        finally:
            self.proc_s += time.perf_counter() - t0
            self.flushes += 1

    def ingest(self, batch: list[str]) -> None:
        raise NotImplementedError

    def before_view(self) -> None:
        if self.urgent and self.pending:
            self.flush()  # the command turn ends before the next question


_WORD = re.compile(r"\w{4,}", re.UNICODE)


def grep_lines(notes: str, query: str, limit: int = 12) -> list[str]:
    """Lines of `notes` sharing a 5-letter stem with a 4+ letter word of `query`."""
    stems = {w.lower().replace("ё", "е")[:5] for w in _WORD.findall(query)}
    out = []
    for line in notes.splitlines():
        low = line.lower().replace("ё", "е")
        if line.strip() and any(s in low for s in stems):
            out.append(line)
            if len(out) >= limit:
                break
    return out


class FileMemoryAgentSUT(_BatchSUT):
    """A model keeps one notes file; the brain sees the file and a grep over it."""

    name = "filemem"

    def __init__(self, updater: Callable[[str], dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.notes = ""
        if updater is None:
            from ...paths import WORKSPACE_DIR

            updater = RoleCall(
                ROLE_FILEMEM, NotesOut, FILEMEM_MODEL, WORKSPACE_DIR / "cache" / "memory-compare"
            )
        self.updater = updater

    def ingest(self, batch: list[str]) -> None:
        text = (
            f"Current time: {self.now()}\n\n<notes>\n{self.notes}\n</notes>\n\n"
            "<new_messages>\n" + "\n".join(batch) + "\n</new_messages>"
        )
        try:
            out = self.updater(text)
        except Exception as exc:  # noqa: BLE001 - a failed update keeps the old notes
            self.errors.append(f"update: {exc}"[:200])
            return
        self.notes = str(out.get("notes") or "")

    def view(self, wording: str | None) -> str:
        self.before_view()
        if wording is None:
            return self.notes
        hits = grep_lines(self.notes, wording)
        return self.notes + ("\n\n# grep\n" + "\n".join(hits) if hits else "")


# ---------------------------------------------------------------------- Graphiti


class GuardAbort(RuntimeError):
    pass


def ollama_json(path: str, body: dict | None = None, host: str = OLLAMA) -> dict:
    import urllib.request

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        host + path, data=data, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 - local Ollama
        return json.loads(r.read())


def ollama_guard(llm: str, http: Callable[[str], dict] | None = None) -> None:
    """Memory guard: one big model at a time, >= 24 GB free to load one, abort at 12 GB swap.
    Small embedders (< 3 GB) may sit next to the LLM."""
    from .judge import MAX_SWAP_GB, MIN_FREE_GB, free_memory_gb, swap_used_gb

    get = http or (lambda p: ollama_json(p))
    swap = swap_used_gb()
    if swap >= MAX_SWAP_GB:
        raise GuardAbort(f"swap {swap:.1f} GB >= {MAX_SWAP_GB:.0f} GB")
    sizes = {m["name"]: int(m.get("size") or 0) for m in get("/api/tags").get("models", [])}
    loaded = [m["name"] for m in get("/api/ps").get("models", [])]
    big = [m for m in loaded if m != llm and sizes.get(m, 0) >= BIG_MODEL_BYTES]
    if big:
        raise GuardAbort(f"another big model is loaded: {', '.join(big)}")
    if llm not in loaded and free_memory_gb() < MIN_FREE_GB:
        raise GuardAbort(f"{free_memory_gb():.1f} GB free < {MIN_FREE_GB:.0f} GB to load {llm}")


def chunk_lines(batch: list[str], max_lines: int = 30, max_chars: int = 2500) -> list[list[str]]:
    """Episodes small enough for the local model's 8k context (Graphiti also sends earlier
    episodes and its instructions with each one)."""
    out: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for line in batch:
        line = line[:max_chars]
        if cur and (len(cur) >= max_lines or size + len(line) > max_chars):
            out.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        out.append(cur)
    return out


class GraphitiSUT(_BatchSUT):
    """graphiti-core on an embedded Kuzu file; LLM and embedder on local Ollama."""

    name = "graphiti"

    def __init__(self, llm: str = GRAPHITI_LLM, embedder: str = GRAPHITI_EMBEDDER) -> None:
        super().__init__()
        self.llm, self.embedder_model = llm, embedder
        self.g: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.episodes = 0
        self.llm_calls = 0
        self.aborted: str | None = None

    def start(self, scn: Scenario, clock: Any, scheduler: Any, workdir: Path) -> None:
        super().start(scn, clock, scheduler, workdir)
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.graph_queries import get_fulltext_indices
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        cfg = LLMConfig(
            api_key="ollama",  # Ollama ignores it; no cloud key anywhere
            base_url=OLLAMA + "/v1",
            model=self.llm,
            small_model=self.llm,
            temperature=0,
        )
        llm = OpenAIGenericClient(config=cfg, max_tokens=4096)
        self._count_llm(llm)
        self.loop = asyncio.new_event_loop()
        self.g = Graphiti(
            graph_driver=KuzuDriver(db=str(workdir / "graphiti.kuzu")),
            llm_client=llm,
            embedder=OpenAIEmbedder(
                OpenAIEmbedderConfig(
                    api_key="ollama",
                    base_url=OLLAMA + "/v1",
                    embedding_model=self.embedder_model,
                    embedding_dim=1024,
                )
            ),
            cross_encoder=OpenAIRerankerClient(config=cfg),
        )
        # Kuzu's `build_indices_and_constraints` is a no-op: the fulltext indices are ours to make
        for q in ["INSTALL fts", "LOAD EXTENSION fts", *get_fulltext_indices(GraphProvider.KUZU)]:
            self._run(self.g.driver.execute_query(q))

    def _count_llm(self, llm: Any) -> None:
        inner = llm._generate_response  # noqa: SLF001 - count every LLM request

        async def counted(*a: Any, **kw: Any) -> Any:
            self.llm_calls += 1
            return await inner(*a, **kw)

        llm._generate_response = counted  # noqa: SLF001

    def _run(self, coro: Any) -> Any:
        assert self.loop is not None
        return self.loop.run_until_complete(coro)

    def stop(self) -> None:
        if self.g is not None:
            with contextlib.suppress(Exception):
                self._run(self.g.close())
            self.g = None
        if self.loop is not None:
            self.loop.close()
            self.loop = None

    def ingest(self, batch: list[str]) -> None:
        for chunk in chunk_lines(batch):
            self._episode(chunk)

    def _episode(self, chunk: list[str]) -> None:
        if self.aborted:
            return
        try:
            ollama_guard(self.llm)
        except GuardAbort as exc:
            self.aborted = str(exc)
            raise
        try:
            self._run(
                self.g.add_episode(
                    name=f"episode-{self.episodes + 1}",
                    episode_body="\n".join(chunk),
                    source_description=(
                        "chat log of the owner's assistant: the owner's own messages, his contacts' "
                        "messages, tool reads he asked for, the assistant's replies, settings lines"
                    ),
                    reference_time=self.clock.now(),
                )
            )
            self.episodes += 1
        except Exception as exc:  # noqa: BLE001 - a failed episode is lost, like in the app
            self.errors.append(f"add_episode: {type(exc).__name__}: {exc}"[:240])

    def view(self, wording: str | None) -> str:
        self.before_view()
        if not wording or self.g is None:
            return ""
        return self._run(self._context(wording))

    async def _context(self, wording: str) -> str:
        from graphiti_core.search.search_filters import SearchFilters
        from graphiti_core.search.search_utils import (
            edge_similarity_search,
            node_similarity_search,
        )

        vec = await self.g.embedder.create(input_data=[wording])
        edges = await edge_similarity_search(
            self.g.driver, vec, None, None, SearchFilters(), None, 10, 0.0
        )
        facts = [(e.uuid, e.fact, e.valid_at, e.invalid_at) for e in edges]
        try:  # Graphiti's own Kuzu fulltext path returns nothing; query the index directly
            rows, _, _ = await self.g.driver.execute_query(
                "CALL QUERY_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', $q) "
                "RETURN node.uuid AS uuid, node.fact AS fact, node.valid_at AS valid_at, "
                "node.invalid_at AS invalid_at, score ORDER BY score DESC LIMIT 5",
                q=wording,
            )
            seen = {f[0] for f in facts}
            facts += [
                (r["uuid"], r["fact"], r["valid_at"], r["invalid_at"])
                for r in rows
                if r["uuid"] not in seen
            ]
        except Exception:  # noqa: BLE001
            pass
        nodes = await node_similarity_search(self.g.driver, vec, SearchFilters(), None, 5, 0.0)

        def day(d: datetime | None) -> str:
            return d.date().isoformat() if d else "present"

        out = ["FACTS (each with its date range: valid from - invalid since):"]
        out += [f"- {f} ({day(v) if v else '?'} - {day(i)})" for _, f, v, i in facts]
        out += ["ENTITIES:"]
        out += [f"- {n.name}: {n.summary}" for n in nodes]
        return "\n".join(out)


# ---------------------------------------------------------------------- api with counters


class ApiCompareSUT(ApiBenchSUT):
    """The engine as in the exam: cloud extraction (cached), the label classifier on."""

    name = "api"

    def __init__(self, extract: str = "cloud", labeler: Any = True) -> None:
        super().__init__(extract=extract, labeler=labeler)
        self.proc_s = 0.0
        self.calls = 0
        self.hits = 0

    def run_worker(self, job: str, phase: str = "run", token: str | None = None) -> None:
        t0 = time.perf_counter()
        try:
            super().run_worker(job, phase, token)
        finally:
            self.proc_s += time.perf_counter() - t0

    def stop(self) -> None:
        if self._proc is not None:
            self.calls += int(getattr(self._proc, "calls", 0))
            self.hits += int(getattr(self._proc, "hits", 0))
        super().stop()


# ---------------------------------------------------------------------- runner


class CompareRunner(BenchRunner):
    """BenchRunner that also keeps, for the answer set, what the brain saw and when."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.captured: list[dict[str, Any]] = []
        self._first: set[str] = set()

    def _run_wording(self, p: Probe, wording: str | None, workdir: Path) -> WordingResult:
        sut: Any = self.sut
        seen: dict[str, str] = {}
        inner = sut.view

        def view(w: str | None) -> str:
            seen["text"] = inner(w)
            return seen["text"]

        sut.view = view
        try:
            res = super()._run_wording(p, wording, workdir)
        finally:
            sut.view = inner
        key = (self.scn.id, p.id)
        if key in ANSWER_SET and wording and p.id not in self._first:
            self._first.add(p.id)
            self.captured.append(
                {
                    "scenario": self.scn.id,
                    "probe": p.id,
                    "guarantee": PANEL.get(key),
                    "wording": wording,
                    "now": self.clock.now().isoformat(),
                    "view": seen.get("text", ""),
                    "view_verdict": res.status if res.status != SKIP else None,
                }
            )
        return res


def load_compare_set(root: Path) -> LoadedSet:
    """The dev set plus the compare-only scenarios (evals/memory/compare, dev personas)."""
    dev = load_set(root / "dev")
    extra = root / "compare"
    if extra.exists():
        more = load_set(extra, base_personas=dev.personas)
        dev.scenarios = [*dev.scenarios, *more.scenarios]
    return dev


def make_sut(name: str) -> _BenchSUT:
    from .bench import LegacySUT, PlainFileSUT

    if name == "plain":
        return PlainFileSUT()
    if name == "legacy":
        return LegacySUT(cache=True)
    if name == "api":
        return ApiCompareSUT()  # type: ignore[return-value]
    if name == "filemem":
        return FileMemoryAgentSUT()
    if name == "graphiti":
        return GraphitiSUT()
    raise ValueError(name)


def run_compare(
    loaded: LoadedSet,
    factory: Callable[[], Any],
    *,
    scenario_ids: list[str] | None = None,
    seed: int = 20261014,
    on_scenario: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    captured: list[dict[str, Any]] = []
    errors: list[str] = []
    stats = {
        "scenarios": 0,
        "admitted": 0,
        "flushes": 0,
        "dropped": 0,
        "proc_s": 0.0,
        "model_calls": 0,
        "cache_hits": 0,
        "cost_usd": 0.0,
        "local_llm_calls": 0,
        "episodes": 0,
    }
    t0 = time.perf_counter()
    for scn in loaded.scenarios:
        if scenario_ids and scn.id not in scenario_ids:
            continue
        sut = factory()
        before = _role_stats(sut)
        r = CompareRunner(scn, loaded.personas, sut=sut, scale="1x", seed=seed)
        res = r.run()
        after = _role_stats(sut)
        records += r.records
        captured += r.captured
        stats["scenarios"] += 1
        stats["admitted"] += sut.admitted
        stats["flushes"] += getattr(sut, "flushes", 0)
        stats["dropped"] += getattr(sut, "dropped", 0)
        stats["proc_s"] += getattr(sut, "proc_s", 0.0)
        stats["model_calls"] += after["calls"] - before["calls"] + getattr(sut, "calls", 0)
        stats["cache_hits"] += after["hits"] - before["hits"] + getattr(sut, "hits", 0)
        stats["cost_usd"] += after["cost_usd"] - before["cost_usd"]
        stats["local_llm_calls"] += getattr(sut, "llm_calls", 0)
        stats["episodes"] += getattr(sut, "episodes", 0)
        errors += [f"{scn.id}: {e}" for e in getattr(sut, "errors", [])]
        if res.error:
            errors.append(f"{scn.id}: {res.error}")
        if on_scenario:
            on_scenario(scn.id, stats)
        if getattr(sut, "aborted", None):
            errors.append(f"memory guard abort: {sut.aborted}")
            break
    stats["wall_s"] = round(time.perf_counter() - t0, 1)
    stats["proc_s"] = round(stats["proc_s"], 1)
    stats["cost_usd"] = round(stats["cost_usd"], 4)
    return {
        "records": records,
        "captured": captured,
        "errors": errors,
        "stats": stats,
        "summary": bench_summarize(records),
    }


def merge_runs(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """One run from runs over disjoint scenario lists (parallel processes): sums, wall = max."""
    out: dict[str, Any] = {"sut": parts[0]["sut"], "records": [], "captured": [], "errors": []}
    stats: dict[str, Any] = {}
    for p in parts:
        out["records"] += p["records"]
        out["captured"] += p["captured"]
        out["errors"] += p["errors"]
        for k, v in p["stats"].items():
            stats[k] = max(stats.get(k, 0), v) if k == "wall_s" else stats.get(k, 0) + v
    out["stats"] = stats
    out["summary"] = bench_summarize(out["records"])
    return out


def _role_stats(sut: Any) -> dict[str, Any]:
    up = getattr(sut, "updater", None)
    if isinstance(up, RoleCall):
        return {"calls": up.calls, "hits": up.hits, "cost_usd": up.cost_usd}
    return {"calls": 0, "hits": 0, "cost_usd": 0.0}


# ---------------------------------------------------------------------- answer level


def probe_index(loaded: LoadedSet) -> dict[tuple[str, str], Probe]:
    out: dict[tuple[str, str], Probe] = {}
    for s in loaded.scenarios:
        for p in s.probes:
            out[(s.id, p.id)] = p
    return out


def gold_of(p: Probe) -> dict[str, Any]:
    req: list[str] = []
    forb: list[str] = []
    rub: list[str] = []
    for e in _blocks(p):
        req += [describe(v) for v in e.must_include]
        forb += [describe(v) for v in _stale_values(e)]
        if e.judge:
            rub.append(e.judge)
    return {"required": req, "forbidden": forb, "rubric": " ".join(rub)}


def brain_text(item: dict[str, Any]) -> str:
    return (
        f"Today: {item['now']}\n\n<memory>\n{item['view']}\n</memory>\n\nOwner: {item['wording']}"
    )


def judge_text(item: dict[str, Any], answer: str, gold: dict[str, Any]) -> str:
    return (
        f"Owner's message: {item['wording']}\n\nAssistant's reply:\n<<<\n{answer}\n>>>\n\n"
        f"Gold:\n- required: {json.dumps(gold['required'], ensure_ascii=False)}\n"
        f"- forbidden: {json.dumps(gold['forbidden'], ensure_ascii=False)}\n"
        f"- rubric: {gold['rubric'] or '(none)'}"
    )


def grade_answers(
    captured: list[dict[str, Any]],
    probes: dict[tuple[str, str], Probe],
    brain: Callable[[str], dict[str, Any]],
    judge: Callable[[str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Brain answers each captured context; code checks first, the judge when they cannot decide."""
    out = []
    for item in captured:
        p = probes[(item["scenario"], item["probe"])]
        answer = str(brain(brain_text(item)).get("answer") or "")
        hit, stale, reasons = judge_view(p, answer)
        gold = gold_of(p)
        if not gold["rubric"] and hit is not False and stale is not True:
            passed, by, why = True, "code", ""
        else:
            v = judge(judge_text(item, answer, gold))
            passed, by, why = v.get("verdict") == "pass", "judge", str(v.get("reason") or "")
        out.append(
            {
                **{k: item[k] for k in ("scenario", "probe", "guarantee", "wording")},
                "answer": answer,
                "code_hit": hit,
                "code_stale": stale,
                "code_reasons": reasons,
                "passed": passed,
                "by": by,
                "reason": why,
            }
        )
    return out


def panel_view(captured: list[dict[str, Any]], probes: dict[tuple[str, str], Probe]) -> list[dict]:
    """View-level panel: G6 needs the dependent values visible; the rest need the bad ones absent."""
    out = []
    for item in captured:
        g = item.get("guarantee")
        if not g:
            continue
        hit, stale, _ = judge_view(probes[(item["scenario"], item["probe"])], item["view"])
        ok = (hit is not False and stale is not True) if g == "G6" else stale is not True
        out.append({"scenario": item["scenario"], "probe": item["probe"], "guarantee": g, "ok": ok})
    return out


# ---------------------------------------------------------------------- CLI


def _repo() -> Path:
    return Path(__file__).resolve().parents[4]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="compare")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sut", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--scenario", action="append")
    m = sub.add_parser("merge")
    m.add_argument("parts", nargs="+")
    m.add_argument("--out", required=True)
    a = sub.add_parser("answer")
    a.add_argument("runs", nargs="+")
    a.add_argument("--cache", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    loaded = load_compare_set(_repo() / "evals" / "memory")
    if args.cmd == "run":

        def progress(sid: str, st: dict[str, Any]) -> None:
            print(
                f"{sid}: flushes={st['flushes']} calls={st['model_calls']} "
                f"hits={st['cache_hits']} local={st['local_llm_calls']}",
                flush=True,
            )

        res = run_compare(
            loaded, lambda: make_sut(args.sut), scenario_ids=args.scenario, on_scenario=progress
        )
        res["sut"] = args.sut
        Path(args.out).write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
        print(json.dumps({"stats": res["stats"], "ALL": res["summary"].get("ALL")}))
        if res["errors"]:
            print("errors:", *res["errors"][:20], sep="\n  ")
        return
    if args.cmd == "merge":
        parts = [json.loads(Path(x).read_text(encoding="utf-8")) for x in args.parts]
        Path(args.out).write_text(
            json.dumps(merge_runs(parts), ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
        return
    probes = probe_index(loaded)
    root = Path(args.cache)
    brain = RoleCall(ROLE_BRAIN, AnswerOut, BRAIN_MODEL, root)
    judge = RoleCall(ROLE_JUDGE, VerdictOut, JUDGE_MODEL, root)
    report: dict[str, Any] = {}
    for path in args.runs:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = grade_answers(run["captured"], probes, brain, judge)
        report[run["sut"]] = {
            "answers": rows,
            "panel_view": panel_view(run["captured"], probes),
        }
        n = sum(r["passed"] for r in rows)
        print(
            f"{run['sut']}: {n}/{len(rows)} brain={brain.stats()} judge={judge.stats()}", flush=True
        )
    report["calls"] = {"brain": brain.stats(), "judge": judge.stats()}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
