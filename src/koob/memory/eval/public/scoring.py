"""Answerer and judges for the public benchmarks: fixed prompts, cached model calls, a call budget,
the judge-label parser and the score table.

Every model output is cached on disk, keyed by (role, model, prompt hash, account mode, input
text): a rerun of the same configuration makes no model call, and an edited prompt file or a new
model id misses the cache. A `Budget` counts the real calls of one run directory (across its
phases and processes) and refuses the call that would pass its cap.

Judges for LoCoMo categories 1-4:
- `lenient`: Mem0's original LoCoMo judge prompt (`ACCURACY_PROMPT`, mem0ai/mem0 at commit
  aae5989, `evaluation/metrics/llm_judge.py`, Apache-2.0), verbatim;
- `strict`: the same frame, but CORRECT only when the answer states the gold value itself.
Category 5 (adversarial) is scored by LoCoMo's own rule: the answer says the conversation does not
mention it (`not mentioned` / `no information available`).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

ANSWER_ROLE = "memory_public_answer"
JUDGE_ROLE = "memory_public_judge"
ANSWERERS = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
JUDGE_MODEL = "claude-sonnet-5-5"

# Verbatim from Mem0 (see the module doc), typographic quotes and trailing spaces included.
ACCURACY_PROMPT = (
    "\n"
    "Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:\n"
    "    (1) a question (posed by one user to another user), \n"
    "    (2) a ’gold’ (ground truth) answer, \n"
    "    (3) a generated answer\n"
    "which you will score as CORRECT/WRONG.\n"
    "\n"
    "The point of the question is to ask about something one user should know about the other user based on their prior conversations.\n"
    "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. \n"
    "\n"
    'For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it\'s the same date.\n'
    "\n"
    "Now it's time for the real question:\n"
    "Question: {question}\n"
    "Gold answer: {gold_answer}\n"
    "Generated answer: {generated_answer}\n"
    "\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. \n"
    "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n"
    "\n"
    'Just return the label CORRECT or WRONG in a json format with the key as "label".\n'
)  # noqa: E501

STRICT_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

Grade strictly. The generated answer is CORRECT only if it states the specific value the gold answer gives: the same person, object, place, activity, reason or number, and, for time related questions, the same date, month or year at the precision of the gold answer (a relative reference such as "last Tuesday" counts only if the answer also resolves it to that date). If the gold answer lists several items, the generated answer must give all of them. An answer that only touches on the same topic, stays vague, hedges between alternatives, contradicts the gold answer anywhere, or says the information is not available is WRONG. Wording and format may differ (e.g., "May 7th" vs "7 May" is the same date).

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""  # noqa: E501

JUDGES = {"lenient": ACCURACY_PROMPT, "strict": STRICT_PROMPT}
ABSTAIN_MARKS = ("not mentioned", "no information available")


class AnswerOut(BaseModel):
    answer: str


class JudgeOut(BaseModel):
    explanation: str = ""
    label: str


def answer_input(now_text: str, context: str, question: str) -> str:
    """The answerer's message: the same frame for every system, only the context differs."""
    return (
        f"Current time: {now_text}\n\n# CONTEXT\n{context.strip() or '(empty)'}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def judge_input(kind: str, question: str, gold: str, answer: str) -> str:
    return JUDGES[kind].format(question=question, gold_answer=gold, generated_answer=answer)


def parse_label(out: Any) -> Literal["CORRECT", "WRONG"] | None:
    """The judge's label from its JSON object (or plain text): CORRECT, WRONG, or None when it
    names neither or both (Mem0's script counts anything but CORRECT as wrong; we keep the
    None so a broken reply is visible)."""
    if isinstance(out, dict):
        out = out.get("label", "")
    text = str(out or "").upper()
    has_c, has_w = "CORRECT" in text, "WRONG" in text
    if has_c == has_w:
        return None
    return "CORRECT" if has_c else "WRONG"


def abstains(answer: str) -> bool:
    low = (answer or "").lower()
    return any(m in low for m in ABSTAIN_MARKS)


# ---------------------------------------------------------------------- budget and cache


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Real model calls of a run directory, by kind, persisted in `path`, capped at `cap`."""

    def __init__(self, path: Path, cap: int) -> None:
        self.path, self.cap = Path(path), cap
        self._lock = threading.Lock()
        try:
            self.counts: dict[str, int] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.counts = {}

    @property
    def used(self) -> int:
        return sum(self.counts.values())

    def left(self) -> int:
        return max(self.cap - self.used, 0)

    def take(self, kind: str) -> None:
        with self._lock:
            if self.used >= self.cap:
                raise BudgetExceeded(f"call cap {self.cap} reached ({self.counts})")
            self.counts[kind] = self.counts.get(kind, 0) + 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.counts, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)


def prompt_hash(role: str) -> str:
    try:
        from koob.llm.runner import prompt_for

        text = prompt_for(role)
    except Exception:  # noqa: BLE001 - no prompt file: the role name stands in
        text = role
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_key(role: str, model: str, prompt: str, text: str) -> str:
    from ..cli_env import MODE

    return hashlib.sha256(f"{role}|{model}|{prompt}|{MODE}|{text}".encode()).hexdigest()


class CachedCall:
    """`run_role` behind the disk cache and the budget. The caller holds `cli_env.no_account()`
    around a batch of calls (it sets process-wide variables, so threads must not each enter it)."""

    def __init__(
        self,
        role: str,
        out_model: type[BaseModel],
        model: str,
        root: Path,
        budget: Budget | None = None,
        runner: Callable[..., tuple[BaseModel, dict]] | None = None,
        prompt: str | None = None,
    ) -> None:
        self.role, self.out_model, self.model = role, out_model, model
        self.root, self.budget, self._runner = Path(root), budget, runner
        self.prompt = prompt if prompt is not None else prompt_hash(role)
        self.calls = self.hits = 0
        self.seconds = 0.0
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Lock] = {}

    def path(self, text: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.model)
        key = cache_key(self.role, self.model, self.prompt, text)
        return self.root / self.role / safe / f"{key[:40]}.json"

    def cached(self, text: str) -> dict[str, Any] | None:
        path = self.path(text)
        try:
            return json.loads(path.read_text(encoding="utf-8"))["out"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def __call__(self, text: str) -> dict[str, Any]:
        with self._lock:  # one call per input: a second thread with the same input waits
            key_lock = self._inflight.setdefault(self.path(text).name, threading.Lock())
        with key_lock:
            return self._call(text)

    def _call(self, text: str) -> dict[str, Any]:
        out = self.cached(text)
        if out is not None:
            with self._lock:
                self.hits += 1
            return out
        if self.budget is not None:
            self.budget.take(self.role)
        runner = self._runner
        if runner is None:
            from koob.llm.runner import run_role

            runner = run_role
        t0 = time.perf_counter()
        obj, meta = runner(self.role, self.out_model, text, model=self.model, lenient=True)
        secs = time.perf_counter() - t0
        out = obj.model_dump(mode="json")
        path = self.path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        blob = {"out": out, "secs": round(secs, 2), "cost_usd": (meta or {}).get("cost_usd")}
        tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        with self._lock:
            self.calls += 1
            self.seconds += secs
        return out


# ---------------------------------------------------------------------- scores


def summarize(rows: list[dict[str, Any]], cats: tuple[int, ...] = (1, 2, 3, 4)) -> dict[str, Any]:
    """{system: {n, J (lenient), J_strict, by_cat, adversarial}} from result rows."""
    out: dict[str, Any] = {}
    for r in rows:
        s = out.setdefault(
            r["system"],
            {"n": 0, "correct": 0, "strict_correct": 0, "unparsed": 0, "by_cat": {}, "adv": [0, 0]},
        )
        if r["category"] in cats:
            s["n"] += 1
            ok = r.get("lenient") == "CORRECT"
            s["correct"] += int(ok)
            s["strict_correct"] += int(ok and r.get("strict") == "CORRECT")
            s["unparsed"] += int(r.get("lenient") is None)
            c = s["by_cat"].setdefault(str(r["category"]), [0, 0])
            c[0] += int(ok)
            c[1] += 1
        elif r["category"] == 5:
            s["adv"][0] += int(bool(r.get("abstained")))
            s["adv"][1] += 1
    for s in out.values():
        n = max(s["n"], 1)
        s["J"] = round(100 * s["correct"] / n, 1)
        s["J_strict"] = round(100 * s["strict_correct"] / n, 1)
    return out


def format_summary(summary: dict[str, Any]) -> str:
    w = max([8, *(len(n) + 1 for n in summary)])
    rows = []
    for name, s in summary.items():
        cats = " ".join(f"c{k}:{v[0]}/{v[1]}" for k, v in sorted(s["by_cat"].items()))
        adv = f"{s['adv'][0]}/{s['adv'][1]}" if s["adv"][1] else "not run"
        rows.append((name, s, cats, adv))
    cw = max([24, *(len(r[2]) for r in rows)]) + 2
    lines = [
        f"{'system':<{w}}{'n':>5}{'J':>8}{'J strict':>10}  {'by category (correct/n)':<{cw}}adversarial"
    ]
    for name, s, cats, adv in rows:
        lines.append(f"{name:<{w}}{s['n']:>5}{s['J']:>8}{s['J_strict']:>10}  {cats:<{cw}}{adv}")
    return "\n".join(lines)


__all__ = [
    "ACCURACY_PROMPT",
    "ANSWERERS",
    "STRICT_PROMPT",
    "Budget",
    "BudgetExceeded",
    "CachedCall",
    "abstains",
    "answer_input",
    "cache_key",
    "judge_input",
    "parse_label",
    "summarize",
]
