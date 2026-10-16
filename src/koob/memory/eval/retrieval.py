"""Cross-lingual retrieval bench (spec 10.3, evals/memory/README.md: Cross-lingual retrieval set).

English canonical facts, Russian questions. Computes recall@1 and recall@3 and per-query
latency for in-process fastembed embedders. Two modes: `embedder` (question vs `text_en`;
`alias_only` questions are excluded and reported apart) and `full` (an approximation of the
section 10.3 lanes: the embedder plus an alias lane that boosts facts of any entity whose alias
appears in the question; the lexical and trigram lanes are not part of this bench).

The dependency is optional (`uv sync --extra memory`): fastembed is imported lazily.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .matchers import normalize


@dataclass(frozen=True)
class EmbedderSpec:
    name: str
    fastembed_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    custom: dict[str, Any] | None = None  # add_custom_model kwargs when fastembed does not list it


EMBEDDERS: dict[str, EmbedderSpec] = {
    "embeddinggemma-300m": EmbedderSpec(
        "embeddinggemma-300m",
        "google/embeddinggemma-300m",
        query_prefix="task: search result | query: ",
        doc_prefix="title: none | text: ",
    ),
    "multilingual-e5-small": EmbedderSpec(
        "multilingual-e5-small",
        "intfloat/multilingual-e5-small",
        query_prefix="query: ",
        doc_prefix="passage: ",
        custom={
            "dim": 384,
            "pooling": "MEAN",
            "normalization": True,
            "hf": "intfloat/multilingual-e5-small",
            "model_file": "onnx/model.onnx",
        },
    ),
    "multilingual-e5-large": EmbedderSpec(
        "multilingual-e5-large", "intfloat/multilingual-e5-large", "query: ", "passage: "
    ),
    "paraphrase-multilingual-mpnet": EmbedderSpec(
        "paraphrase-multilingual-mpnet",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ),
}


class FastEmbedder:
    def __init__(self, spec: EmbedderSpec, cache_dir: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised by the CLI only
            raise RuntimeError("fastembed is not installed: `uv sync --extra memory`") from exc
        self.spec = spec
        if spec.custom:
            from fastembed.common.model_description import ModelSource, PoolingType

            c = spec.custom
            known = {m["model"] for m in TextEmbedding.list_supported_models()}
            if spec.fastembed_id not in known:
                TextEmbedding.add_custom_model(
                    model=spec.fastembed_id,
                    pooling=getattr(PoolingType, c["pooling"]),
                    normalization=c["normalization"],
                    sources=ModelSource(hf=c["hf"]),
                    dim=c["dim"],
                    model_file=c["model_file"],
                )
        kw: dict[str, Any] = {"cache_dir": cache_dir} if cache_dir else {}
        self.model = TextEmbedding(model_name=spec.fastembed_id, **kw)

    def embed_docs(self, texts: list[str]) -> Any:
        import numpy as np

        return np.array(
            list(self.model.embed([self.spec.doc_prefix + t for t in texts])), dtype="float32"
        )

    def embed_query(self, text: str) -> Any:
        import numpy as np

        return np.array(
            next(iter(self.model.embed([self.spec.query_prefix + text]))), dtype="float32"
        )


@dataclass
class BenchResult:
    embedder: str
    mode: str
    n_questions: int
    recall_at_1: float
    recall_at_3: float
    latency_ms_p50: float
    latency_ms_p95: float
    floor: float | None
    passes_floor: bool | None
    by_tag: dict[str, dict[str, float]] = field(default_factory=dict)
    excluded_alias_only: dict[str, float] | None = None
    misses: list[str] = field(default_factory=list)


def load_set(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for key in ("entities", "facts", "questions"):
        if key not in data:
            raise ValueError(f"{path}: missing {key!r}")
    return data


def _alias_lane(question: str, entities: list[dict[str, Any]]) -> set[str]:
    nq = normalize(question)
    hit: set[str] = set()
    for e in entities:
        forms = [e.get("display_name", ""), *(a["form"] for a in e.get("aliases", []))]
        for f in forms:
            nf = normalize(f)
            if len(nf) >= 3 and nf in nq:
                hit.add(e["id"])
                break
    return hit


def rank(scores: Any, k: int) -> list[int]:
    import numpy as np

    return list(np.argsort(-scores)[:k])


def run_bench(
    data: dict[str, Any],
    embedder: Any,
    *,
    mode: str = "embedder",
    name: str = "",
    alias_boost: float = 0.12,
    warmup: int = 3,
) -> BenchResult:
    import numpy as np

    facts = data["facts"]
    fact_ids = [f["id"] for f in facts]
    fact_entity = [f.get("entity") for f in facts]
    doc = embedder.embed_docs([f["text_en"] for f in facts])
    doc = doc / np.linalg.norm(doc, axis=1, keepdims=True)
    qs = data["questions"]
    if mode == "embedder":
        scored = [q for q in qs if "alias_only" not in q.get("tags", [])]
    else:
        scored = list(qs)
    for q in scored[:warmup]:
        embedder.embed_query(q["ru"])
    lat: list[float] = []
    hits1: list[bool] = []
    hits3: list[bool] = []
    misses: list[str] = []
    tags: dict[str, list[tuple[bool, bool]]] = {}

    def evaluate(q: dict[str, Any]) -> tuple[bool, bool, float]:
        t0 = time.perf_counter()
        v = embedder.embed_query(q["ru"])
        v = v / np.linalg.norm(v)
        s = doc @ v
        if mode == "full":
            ents = _alias_lane(q["ru"], data["entities"])
            if ents:
                s = s + alias_boost * np.array([1.0 if e in ents else 0.0 for e in fact_entity])
        top = rank(s, 3)
        dt = (time.perf_counter() - t0) * 1000
        ans = set(q["answers"])
        return fact_ids[top[0]] in ans, any(fact_ids[i] in ans for i in top), dt

    for q in scored:
        h1, h3, dt = evaluate(q)
        hits1.append(h1)
        hits3.append(h3)
        lat.append(dt)
        if not h3:
            misses.append(q["id"])
        for t in q.get("tags", []):
            tags.setdefault(t, []).append((h1, h3))
    n = len(scored)
    res = BenchResult(
        embedder=name,
        mode=mode,
        n_questions=n,
        recall_at_1=sum(hits1) / n,
        recall_at_3=sum(hits3) / n,
        latency_ms_p50=statistics.median(lat),
        latency_ms_p95=sorted(lat)[min(n - 1, math.ceil(0.95 * n) - 1)],
        floor=(data.get("floor") or {}).get("recall_at_3"),
        passes_floor=None,
        by_tag={
            t: {
                "n": len(v),
                "r1": sum(a for a, _ in v) / len(v),
                "r3": sum(b for _, b in v) / len(v),
            }
            for t, v in sorted(tags.items())
        },
        misses=misses,
    )
    if res.floor is not None:
        res.passes_floor = res.recall_at_3 >= res.floor
    if mode == "embedder":
        apart = [q for q in qs if "alias_only" in q.get("tags", [])]
        if apart:
            r = [evaluate(q) for q in apart]
            res.excluded_alias_only = {
                "n": len(apart),
                "r1": sum(x[0] for x in r) / len(apart),
                "r3": sum(x[1] for x in r) / len(apart),
            }
    return res


def format_result(r: BenchResult) -> str:
    out = [
        f"{r.embedder} [{r.mode}] n={r.n_questions}  recall@1={r.recall_at_1:.3f}  recall@3={r.recall_at_3:.3f}  "
        f"query p50={r.latency_ms_p50:.1f} ms p95={r.latency_ms_p95:.1f} ms"
        + (
            f"  floor {r.floor}: {'PASS' if r.passes_floor else 'FAIL'}"
            if r.floor is not None
            else ""
        )
    ]
    if r.excluded_alias_only:
        a = r.excluded_alias_only
        out.append(
            f"  alias_only (excluded, reported apart) n={a['n']:.0f}  r@1={a['r1']:.3f}  r@3={a['r3']:.3f}"
        )
    for t, v in r.by_tag.items():
        out.append(f"  {t:<14} n={v['n']:<4.0f} r@1={v['r1']:.3f}  r@3={v['r3']:.3f}")
    if r.misses:
        out.append(
            f"  missed@3: {len(r.misses)} ({', '.join(r.misses[:12])}{'...' if len(r.misses) > 12 else ''})"
        )
    return "\n".join(out)
