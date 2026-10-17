"""`python -m koob.memory.eval.public locomo <phase>`: the LoCoMo harness for our memory.

Phases (each resumable: every model output is cached, every call counted against one cap):
- `ingest`: one conversation into a fresh store through the arrival path (cloud extraction),
  then every question's contexts (see `contexts`);
- `contexts`: every question's context for `ours`, `ours_search` and `rag` from the store an
  earlier `ingest` left (no arrival, no extraction), with each context's evidence recall
  (`evidence.py`), written to `contexts.jsonl` (the previous file is kept beside it);
- `recall`: the evidence-recall table of `contexts.jsonl`;
- `answer`: answers for the chosen systems with one answerer, then both judges, written to
  `results-<answerer>-<system>.jsonl` and `summary-<answerer>.json` (every system run so far);
- `report`: the table of a finished `answer` phase;
- `estimate`: calls for the whole release, from the data and a finished run's rates.

Layout under `--root` (default `<repo>/.devdata/benchmarks/locomo`, ignored by git):
`locomo10.json`, `budget.json` (calls by role), `cache/`, `runs/<conversation>/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from . import evidence as ev
from . import locomo as lm
from . import scoring as sc

DEFAULT_CAP = 700
WORKERS = 4
SYSTEMS = ("ours", "ours_search", "full", "rag")


def repo_root() -> Path:
    import koob

    return Path(koob.__file__).resolve().parents[2]


def default_root() -> Path:
    return repo_root() / ".devdata" / "benchmarks" / "locomo"


def _now_text(dt: datetime) -> str:
    local = dt.astimezone()
    return f"{local:%I:%M %p} on {local.day} {local:%B}, {local.year}".lstrip("0")


def _pick(root: Path, conv_id: str) -> lm.Conversation:
    for c in lm.load(root / "locomo10.json"):
        if c.sample_id == conv_id:
            return c
    raise SystemExit(f"no conversation {conv_id!r} in {root / 'locomo10.json'}")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


class BudgetedExtractor:
    """The cloud extractor with each real call counted against the run's cap."""

    route = "cloud"

    def __init__(self, inner: Any, budget: sc.Budget) -> None:
        self.inner, self.budget = inner, budget
        self.model_id = getattr(inner, "model_id", "worker-extract")
        self.stopped = False

    def available(self) -> bool:
        return True

    def __call__(self, snap: dict[str, Any]) -> dict[str, Any]:
        try:
            self.budget.take("memory_extract")
        except sc.BudgetExceeded:
            self.stopped = True
            raise
        return self.inner(snap)


# ---------------------------------------------------------------------- ingest


def _table_counts(conn: Any) -> dict[str, int]:
    names = [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if r[0] in ("raw_chunks", "records", "facts", "entities", "jobs")
    ]
    return {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}  # noqa: S608


def ingest(root: Path, conv_id: str, cap: int) -> dict[str, Any]:
    from ...recall import vectors
    from ...workers import extract as ex
    from ..extract_cache import CachedProcessor

    conv = _pick(root, conv_id)
    run = root / "runs" / conv_id
    budget = sc.Budget(root / "budget.json", cap)
    if budget.left() < len(conv.sessions):
        raise SystemExit(f"budget: {budget.left()} calls left, {len(conv.sessions)} sessions")
    store = run / "store"
    if store.exists():  # a fresh store every time; the old one is kept beside it
        store.rename(run / f"store-{datetime.now():%Y%m%d-%H%M%S}")
    inner = BudgetedExtractor(ex.CloudExtractor(), budget)
    proc = CachedProcessor(inner, root / "cache" / "extract")
    arr = lm.Arrival(conv, store, proc)
    arr.start()
    try:
        arr.ingest()
        counts = _table_counts(arr.sut.store())
        emb = vectors.current_embedder(wait=True)
        rows = _context_rows(arr, conv, emb)
    finally:
        arr.stop()
    _write_contexts(run, rows)
    info = {
        "conversation": conv_id,
        "data_sha256": lm.file_sha256(root / "locomo10.json"),
        "speaker_mapping": lm.SPEAKER_MAPPING,
        "sessions": arr.log,
        "extract": {"calls": proc.calls, "hits": proc.hits, "model": inner.model_id},
        "budget_stop": inner.stopped,
        "store": counts,
        "asked_at": conv.asked_at.isoformat(),
        "ctx_tokens_mean": {
            k: round(sum(len(r.get(k) or "") for r in rows) / 4 / max(len(rows), 1))
            for k in ("ours", "ours_search", "rag")
        },
        "embedder": getattr(emb, "id", None) or type(emb).__name__,
        "notes": arr.sut.notes,
    }
    (run / "ingest.json").write_text(json.dumps(info, indent=1, default=str), "utf-8")
    return info


def _context_rows(arr: lm.Arrival, conv: lm.Conversation, emb: Any) -> list[dict[str, Any]]:
    """Every question's contexts with their evidence recall and token counts."""
    rag = lm.RagIndex(conv, emb) if emb is not None else None
    prov = ev.provenance(arr.sut.store(), conv.sample_id)
    rows = []
    for n, q in enumerate(conv.questions):
        v = arr.views(q, n)
        top = rag.top(q.question) if rag is not None else []
        evid = ev.norm_evidence(q.evidence)
        held = {s: ev.covered(ev.handles(v[s]), prov) for s in ("ours", "ours_search")}
        held["rag"] = {c.dia_id for c in top}
        ctx = {
            "ours": v["ours"],
            "ours_search": v["ours_search"],
            "rag": "\n".join(c.text for c in top) if rag is not None else None,
        }
        rows.append(
            {
                "qid": q.qid,
                "category": q.category,
                "question": q.question,
                "gold": q.answer,
                **ctx,
                "ours_meta": v["ours_meta"],
                "search_meta": v["search_meta"],
                "evidence": list(evid),
                "recall": {s: ev.recall_row(evid, held[s]) for s in held if ctx[s] is not None},
                "tokens": {s: lm.tokens(t) for s, t in ctx.items() if t is not None},
            }
        )
    return rows


def _write_contexts(run: Path, rows: list[dict[str, Any]]) -> None:
    path = run / "contexts.jsonl"
    if path.exists():  # an earlier phase's contexts stay beside the new ones
        path.rename(run / f"contexts-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    _jsonl(path, rows)


class _NoExtraction:
    """An extractor that is never meant to run: re-reading a store needs no extraction. A call
    is counted and refused, so a run that needed one says so instead of spending it."""

    route, model_id = "cloud", "none"

    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> bool:
        return True

    def __call__(self, snap: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        raise RuntimeError("no extraction while contexts are read from an existing store")


def contexts(root: Path, conv_id: str) -> dict[str, Any]:
    """Every question's contexts from the store an earlier `ingest` left, with recall."""
    from ...recall import vectors
    from ..extract_cache import CachedProcessor

    conv = _pick(root, conv_id)
    run = root / "runs" / conv_id
    inner = _NoExtraction()
    proc = CachedProcessor(inner, root / "cache" / "extract")
    arr = lm.Arrival(conv, run / "store", proc)
    try:
        arr.resume()
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}: run the ingest phase first") from None
    try:
        emb = vectors.current_embedder(wait=True)
        rows = _context_rows(arr, conv, emb)
        counts = _table_counts(arr.sut.store())
    finally:
        arr.stop()
    _write_contexts(run, rows)
    scored = [r for r in rows if r["category"] in lm.CATEGORIES]
    info = {
        "conversation": conv_id,
        "store": counts,
        "extract": {"calls": proc.calls, "hits": proc.hits, "refused": inner.calls},
        "recall": ev.summarize(scored, ("ours", "ours_search", "rag")),
        "notes": arr.sut.notes,
    }
    (run / "contexts.json").write_text(json.dumps(info, indent=1, default=str), "utf-8")
    return info


def recall(root: Path, conv_id: str) -> dict[str, Any]:
    rows = _read_jsonl(root / "runs" / conv_id / "contexts.jsonl")
    scored = [r for r in rows if r["category"] in lm.CATEGORIES]
    return {"recall": ev.summarize(scored, ("ours", "ours_search", "rag"))}


# ---------------------------------------------------------------------- answer and judge


def _pool(fn: Any, items: list[Any]) -> tuple[list[Any], str | None]:
    """Results in order; on a budget stop, the done ones and the reason."""
    out: list[Any] = [None] * len(items)
    stop: str | None = None
    with ThreadPoolExecutor(WORKERS) as pool:
        futs = [pool.submit(fn, it) for it in items]
        for i, f in enumerate(futs):
            try:
                out[i] = f.result()
            except sc.BudgetExceeded as exc:
                stop = str(exc)
            except Exception as exc:  # noqa: BLE001 - one failed call is a row, not a crash
                out[i] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return out, stop


def answer(
    root: Path,
    conv_id: str,
    *,
    systems: tuple[str, ...],
    answerer: str,
    cap: int,
    adversarial: bool,
    strict_all: bool,
) -> dict[str, Any]:
    from ..cli_env import no_account

    conv = _pick(root, conv_id)
    run = root / "runs" / conv_id
    if (run / "contexts.jsonl").exists():
        ctxs = _read_jsonl(run / "contexts.jsonl")
    elif set(systems) <= {"full"}:  # the full-context control needs no ingested store
        ctxs = [
            {"qid": q.qid, "category": q.category, "question": q.question, "gold": q.answer}
            for q in conv.questions
        ]
    else:
        raise SystemExit(f"no contexts for {conv_id}: run the ingest phase first")
    full = lm.transcript(conv)
    now = _now_text(conv.asked_at)
    cats = (*lm.CATEGORIES, lm.ADVERSARIAL) if adversarial else lm.CATEGORIES
    budget = sc.Budget(root / "budget.json", cap)
    cache = root / "cache"
    ans = sc.CachedCall(sc.ANSWER_ROLE, sc.AnswerOut, sc.ANSWERERS[answerer], cache, budget)
    judge = sc.CachedCall(sc.JUDGE_ROLE, sc.JudgeOut, sc.JUDGE_MODEL, cache, budget)
    rows = []
    for system in systems:
        for c in ctxs:
            if c["category"] not in cats:
                continue
            ctx = full if system == "full" else (c.get(system) or "")
            rows.append({"system": system, **{k: c[k] for k in ("qid", "category", "question")},
                         "gold": c["gold"], "ctx_tokens": len(ctx) // 4,
                         "_in": sc.answer_input(now, ctx, c["question"])})  # fmt: skip
    stops: list[str] = []
    with no_account():
        outs, stop = _pool(lambda r: ans(r["_in"]), rows)
        stops += [stop] if stop else []
        for r, o in zip(rows, outs, strict=True):
            r["answer"] = (o or {}).get("answer") if isinstance(o, dict) else None
            r["answer_error"] = (o or {}).get("error") if isinstance(o, dict) else "budget"
            r["abstained"] = sc.abstains(r["answer"] or "")
        scored = [r for r in rows if r["category"] in lm.CATEGORIES and r["answer"] is not None]
        for kind in ("lenient", "strict"):
            todo = scored if kind == "lenient" or strict_all else [
                r for r in scored if r.get("lenient") == "CORRECT"
            ]  # fmt: skip
            if stops:
                break
            outs, stop = _pool(
                lambda r, k=kind: judge(sc.judge_input(k, r["question"], r["gold"], r["answer"])),
                todo,
            )
            stops += [stop] if stop else []
            for r, o in zip(todo, outs, strict=True):
                r[kind] = sc.parse_label(o) if isinstance(o, dict) and "error" not in o else None
                r[f"{kind}_why"] = (o or {}).get("explanation") if isinstance(o, dict) else None
        if not strict_all:  # strict is a subset of lenient: a lenient WRONG is a strict WRONG
            for r in scored:
                r.setdefault("strict", "WRONG" if r.get("lenient") == "WRONG" else None)
    for r in rows:
        r.pop("_in", None)
    for system in systems:
        _jsonl(
            run / f"results-{answerer}-{system}.jsonl", [r for r in rows if r["system"] == system]
        )
    every = [r for f in sorted(run.glob(f"results-{answerer}-*.jsonl")) for r in _read_jsonl(f)]
    summary = {
        "conversation": conv_id,
        "answerer": sc.ANSWERERS[answerer],
        "judge": sc.JUDGE_MODEL,
        "strict_scope": "all" if strict_all else "lenient-correct",
        "scores": sc.summarize(every),
        "calls": {"answer": ans.calls, "judge": judge.calls},
        "hits": {"answer": ans.hits, "judge": judge.hits},
        "budget": budget.counts,
        "stopped": stops,
    }
    (run / f"summary-{answerer}.json").write_text(json.dumps(summary, indent=1), "utf-8")
    return summary


# ---------------------------------------------------------------------- estimate


def estimate(root: Path, conv_id: str, answerer: str) -> dict[str, Any]:
    """Calls for all ten conversations at the rates one finished run showed."""
    run = root / "runs" / conv_id
    ing = json.loads((run / "ingest.json").read_text("utf-8"))
    summ = json.loads((run / f"summary-{answerer}.json").read_text("utf-8"))
    conv = _pick(root, conv_id)
    per_session = ing["extract"]["calls"] / max(len(conv.sessions), 1)
    systems = list(summ["scores"])
    lenient_ok = sum(s["correct"] for s in summ["scores"].values()) / max(
        sum(s["n"] for s in summ["scores"].values()), 1
    )
    total = {"extract": 0.0, "answer": 0, "judge_lenient": 0, "judge_strict": 0.0, "adv": 0}
    for c in lm.load(root / "locomo10.json"):
        n14 = len(c.scored())
        total["extract"] += per_session * len(c.sessions)
        total["answer"] += n14 * len(systems)
        total["judge_lenient"] += n14 * len(systems)
        total["judge_strict"] += n14 * len(systems) * lenient_ok
        total["adv"] += len(c.scored((lm.ADVERSARIAL,))) * len(systems)
    total = {k: round(v) for k, v in total.items()}
    total["sum_without_adv"] = sum(v for k, v in total.items() if k != "adv")
    return {"per_session_extract": round(per_session, 2), "lenient_rate": lenient_ok, **total}


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m koob.memory.eval.public")
    p.add_argument("bench", choices=["locomo"])
    p.add_argument(
        "phase", choices=["ingest", "contexts", "recall", "answer", "report", "estimate"]
    )
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--conv", default="conv-30")
    p.add_argument("--cap", type=int, default=DEFAULT_CAP)
    p.add_argument("--systems", default=",".join(SYSTEMS))
    p.add_argument("--answerer", choices=sorted(sc.ANSWERERS), default="sonnet")
    p.add_argument("--adversarial", action="store_true", help="also answer category 5")
    p.add_argument("--strict-all", action="store_true", help="strict judge on every answer")
    a = p.parse_args(argv)
    root = a.root or default_root()
    if a.phase == "ingest":
        out: Any = ingest(root, a.conv, a.cap)
    elif a.phase in ("contexts", "recall"):
        out = contexts(root, a.conv) if a.phase == "contexts" else recall(root, a.conv)
        print(ev.format_recall(out["recall"]))
    elif a.phase == "answer":
        systems = tuple(s for s in a.systems.split(",") if s)
        bad = set(systems) - set(SYSTEMS)
        if bad:
            raise SystemExit(f"unknown systems: {sorted(bad)}")
        out = answer(root, a.conv, systems=systems, answerer=a.answerer, cap=a.cap,
                     adversarial=a.adversarial, strict_all=a.strict_all)  # fmt: skip
        print(sc.format_summary(out["scores"]))
    elif a.phase == "report":
        out = json.loads((root / "runs" / a.conv / f"summary-{a.answerer}.json").read_text())
        print(sc.format_summary(out["scores"]))
    else:
        out = estimate(root, a.conv, a.answerer)
    print(json.dumps({k: v for k, v in out.items() if k != "scores"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
