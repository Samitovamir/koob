"""Pluggable judge for free text (spec 19.4): code checks first; a judge only when they cannot decide.

Interface plus a local-Ollama implementation behind a flag. It obeys the Ollama memory guard:
one big model at a time, at least 24 GB free before loading, unload after, abort at 12 GB swap.
Never used in pytest (tests inject fakes and exercise only the guard arithmetic).
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

MIN_FREE_GB = 24.0
MAX_SWAP_GB = 12.0


@dataclass
class JudgeVerdict:
    passed: bool
    reason: str = ""


class Judge(Protocol):
    def judge(self, rubric: str, answer: str, exact_values: list[str]) -> JudgeVerdict: ...

    def close(self) -> None: ...


class NoJudge:
    """Default: free-text rubrics are reported as skipped, never as passed."""

    def judge(self, rubric: str, answer: str, exact_values: list[str]) -> JudgeVerdict:
        raise JudgeUnavailable("no judge configured")

    def close(self) -> None:
        return None


class JudgeUnavailable(RuntimeError):
    pass


class MemoryGuardError(JudgeUnavailable):
    pass


def free_memory_gb() -> float:
    """Free + inactive + speculative pages (macOS `vm_stat`)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))  # type: ignore[union-attr]
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{label}:\s+(\d+)", out)
        pages += int(m.group(1)) if m else 0
    return pages * page / 1024**3


def swap_used_gb() -> float:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout
    m = re.search(r"used = ([\d.]+)([MG])", out)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v if m.group(2) == "G" else v / 1024


def guard_ok(free_gb: float, swap_gb: float, loaded_models: list[str], wanted: str) -> str | None:
    """None when it is safe to load `wanted`, else the reason (pure; unit-tested)."""
    if swap_gb >= MAX_SWAP_GB:
        return f"swap {swap_gb:.1f} GB >= {MAX_SWAP_GB:.0f} GB: abort"
    others = [m for m in loaded_models if m != wanted]
    if others:
        return f"another model is loaded ({', '.join(others)}): one big model at a time"
    if wanted not in loaded_models and free_gb < MIN_FREE_GB:
        return f"{free_gb:.1f} GB free < {MIN_FREE_GB:.0f} GB"
    return None


PROMPT = """You grade an assistant's answer. Be strict: a vague or approximate answer is a FAIL.
Rubric: {rubric}
Exact values that must be right: {values}
Answer under test:
<<<
{answer}
>>>
Reply with one JSON object: {{"pass": true|false, "reason": "<one short sentence>"}}"""


class OllamaJudge:
    """A local model from another family than the brain (default a Qwen). One call, then unload."""

    def __init__(
        self,
        model: str = "qwen3:14b",
        host: str = "http://127.0.0.1:11434",
        *,
        free_gb: Callable[[], float] = free_memory_gb,
        swap_gb: Callable[[], float] = swap_used_gb,
        http: Callable[[str, dict | None], dict] | None = None,
    ) -> None:
        self.model, self.host = model, host.rstrip("/")
        self._free, self._swap = free_gb, swap_gb
        self._http = http or self._request
        self.loaded_by_us = False

    def _request(self, path: str, body: dict | None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.host + path, data=data, headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 - local host from config
            return json.loads(r.read())

    def _loaded(self) -> list[str]:
        return [m["name"] for m in self._http("/api/ps", None).get("models", [])]

    def judge(self, rubric: str, answer: str, exact_values: list[str]) -> JudgeVerdict:
        why = guard_ok(self._free(), self._swap(), self._loaded(), self.model)
        if why:
            raise MemoryGuardError(why)
        body = {
            "model": self.model,
            "prompt": PROMPT.format(
                rubric=rubric, values=json.dumps(exact_values, ensure_ascii=False), answer=answer
            ),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "keep_alive": "30s",
        }
        self.loaded_by_us = True
        resp = self._http("/api/generate", body)
        try:
            data = json.loads(resp.get("response", "{}"))
            return JudgeVerdict(bool(data.get("pass")), str(data.get("reason", ""))[:300])
        except (json.JSONDecodeError, AttributeError):
            return JudgeVerdict(False, "judge returned unparsable output")

    def close(self) -> None:
        """Unload the model (keep_alive 0)."""
        if self.loaded_by_us:
            try:
                self._http("/api/generate", {"model": self.model, "keep_alive": 0})
            finally:
                self.loaded_by_us = False
