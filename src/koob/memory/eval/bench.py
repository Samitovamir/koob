"""Memory-level benchmark baselines: a plain growing file and the Cube's legacy memory.

No model is called. The same scenarios are replayed through the harness `Runner`; the two SUTs
here take every admitted message, and for each probe wording we ask one thing: what would the
brain see before answering? Then, on probes that ask for a value:

- hit   = every `must_include` value is present in what the brain sees;
- stale = a `must_not_include` / memory-visible `forbidden` value is shown as current
  (superseded, forgotten, banned-chat values): a mention its own clause negates does not
  count, nor, under `allow: history`, one marked as history (the graders' rule,
  `matchers.shown_as_current`, for every SUT alike);
- tokens = chars/4 of what the brain sees (a fixed estimate, same for every SUT);
- ms = time of the memory step (building what the brain sees), p50 / p95.

A probe with nothing to look up (state checks, hands, actions, behaviours only) is "n/a".
"""

from __future__ import annotations

import json
import math
import random
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .canary import StoreMap
from .graders import FAIL, PASS, SKIP
from .loader import LoadedSet
from .matchers import compile_value, describe, normalize, shown_as_current
from .observe import Inspector
from .report import ScenarioResult, WordingResult
from .routes import probe_route, route_tools
from .runner import Runner
from .schema import Expect, Probe, Scenario
from .sut import ApiSUT, Guarded, NotBuilt

SEEN_CHANNELS = {"core", "basket", "cli_prompt", "store_tools", "update_block", "all"}
TOKEN_COUNTER = "chars/4"
DEFAULT_SEED = 20261014
_REFS = ("contact:", "tg:", "mail:", "phone:", "@")


def tokens_of(text: str) -> int:
    return math.ceil(len(text) / 4)


# ---------------------------------------------------------------------- SUTs


class _Api:
    """The facade subset a replay uses; anything else is missing -> NotBuilt in `Guarded`."""

    def __init__(self, sut: _BenchSUT) -> None:
        self._s = sut

    def sweep(self, *, rows: list[dict[str, Any]] | None = None, **_: Any) -> int:
        for r in rows or []:
            self._s.admit(r)
        return len(rows or [])

    def ingest_turn(
        self, *, turn_id: str, rows: list[dict[str, Any]] | None = None, **_: Any
    ) -> None:
        for r in rows or []:
            self._s.admit(r, turn_id=turn_id)

    def remember(self, text: str, *, turn: Any, **_: Any) -> None:
        self._s.handled.add(turn.turn_id)
        self._s.on_remember(text)

    def fix(self, target: str, correction: str, *, turn: Any, **_: Any) -> None:
        self._s.handled.add(turn.turn_id)
        self._s.on_fix(correction)

    def forget(self, target: str, *, turn: Any, **_: Any) -> None:
        self._s.handled.add(turn.turn_id)
        self._s.on_forget(target)

    def undo_forget(self, forget_id: str, *, turn: Any, **_: Any) -> None:
        self._s.handled.add(turn.turn_id)
        self._s.on_undo()

    def set_chat_mode(self, chat: str, mode: Any, **_: Any) -> None:
        self._s.on_chat_mode(chat, getattr(mode, "value", str(mode)))

    def record_turn_source(self, *a: Any, **kw: Any) -> None:
        return None

    def actions_at_startup(self, **_: Any) -> None:
        return None


class _BenchSUT:
    """Shared plumbing: the SUT protocol with neutral answers, plus `view()` for the bench."""

    name = "bench"

    def __init__(self) -> None:
        self.api = Guarded(_Api(self))
        self.handled: set[str] = set()  # turns whose text a memory call already took
        self.misses: dict[str, int] = {}
        self.admitted = 0

    # --- bench surface
    def view(self, wording: str | None) -> str:
        raise NotImplementedError

    def admit(self, row: dict[str, Any], turn_id: str | None = None) -> None:
        raise NotImplementedError

    def tool_view(self, route: str, wording: str, turn: Any = None) -> str | None:
        """What this system's memory tools return for a tool-routed probe; None = it has no
        such tool (reported n/a; the probe is judged on the view alone)."""
        return None

    def on_remember(self, text: str) -> None: ...

    def on_fix(self, correction: str) -> None:
        self.miss("correction")

    def on_forget(self, target: str) -> None:
        self.miss("forget")

    def on_undo(self) -> None:
        self.miss("undo")

    def on_chat_mode(self, chat: str, mode: str) -> None:
        if mode != "normal":
            self.miss("chat_mode")

    def miss(self, what: str) -> None:
        self.misses[what] = self.misses.get(what, 0) + 1

    # --- SUT protocol
    def start(self, scn: Scenario, clock: Any, scheduler: Any, workdir: Path) -> None:
        self.workdir = workdir

    def stop(self) -> None: ...

    def store(self) -> Any:
        return None

    def seed(self, scn: Scenario, personas: dict[str, Any]) -> None:
        for item in scn.setup.known:
            self.on_remember(item.get("text") if isinstance(item, dict) else str(item))

    def run_worker(self, job: str, phase: str = "run", token: str | None = None) -> None:
        return None

    def snapshot(self) -> Any:
        return None

    def rollback(self, token: Any) -> None: ...

    def inspector(self) -> Inspector | None:
        return None

    def channel_texts(self) -> dict[str, str]:
        return {}

    def hands_log(self) -> list[dict[str, Any]]:
        return []

    def denials(self) -> list[dict[str, Any]]:
        return []

    def notifications(self) -> list[dict[str, Any]]:
        raise NotBuilt("bench: no notifications")

    def forget_journal_entries(self) -> int | None:
        return None

    def chat_policy(self) -> dict[str, str] | None:
        return None

    def inject_crash(self, point: str) -> None:
        raise NotBuilt(f"bench: crash point {point}")

    def corrupt(self, store: str) -> None:
        raise NotBuilt("bench: corruption")

    def store_map(self) -> StoreMap | None:
        return None


def _who(row: dict[str, Any]) -> str:
    role = row.get("role")
    if role == "owner":
        return "owner"
    return str(row.get("sender") or row.get("tool") or role or "?")


class PlainFileSUT(_BenchSUT):
    """One growing text file: every admitted message appended; the brain sees all of it."""

    name = "plain"

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._joined: tuple[int, str] | None = None

    def admit(self, row: dict[str, Any], turn_id: str | None = None) -> None:
        text = str(row.get("text") or row.get("caption") or "")
        if row.get("role") not in ("owner", "other", "tool") or not text:
            return  # the Cube's own scripted replies are not memory input
        chat = row.get("chat_id") or row.get("source") or ""
        self.lines.append(f"[{row.get('ts', '')}] {_who(row)} ({chat}): {text}")
        self.admitted += 1

    def on_remember(self, text: str) -> None:
        # the message itself was already appended by `ingest_turn`; a file has no "remember"
        return None

    def view(self, wording: str | None) -> str:
        if self._joined is None or self._joined[0] != len(self.lines):
            self._joined = (len(self.lines), "\n".join(self.lines))
        return self._joined[1]

    def snapshot(self) -> Any:
        return len(self.lines)

    def rollback(self, token: Any) -> None:
        if token is not None:
            del self.lines[token:]


class LegacySUT(_BenchSUT):
    """The Cube's old memory on a tmp vault (see `membench.legacy_adapter`).

    Upper bound: every user message, every non-noise message of other people and every tool
    text becomes a note (the real Cube saves only what the model chose to save). A correction
    overwrites the best recall hit (what a model's memory_search + memory_write would do); a
    forget deletes the best recall hit. No hit, or no legacy equivalent (chat modes) -> a miss.
    """

    name = "legacy"

    def __init__(
        self, memory_factory: Callable[[Path], Any] | None = None, cache: bool = False
    ) -> None:
        super().__init__()
        self._factory = memory_factory
        self._cache = cache
        self.mem: Any = None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._stamp: dict[str, str] = {}  # slug -> scenario time it was written
        self._shown: list[str] = []  # notes the latest user turn's recall showed

    def start(self, scn: Scenario, clock: Any, scheduler: Any, workdir: Path) -> None:
        super().start(scn, clock, scheduler, workdir)
        self._clock, self._stamp, self._shown = clock, {}, []
        factory = self._factory
        if factory is None:
            import importlib

            mod = "koob." + "membench.legacy_adapter"  # outside memory/, loaded on demand
            factory = importlib.import_module(mod).build
        self._tmp = tempfile.TemporaryDirectory(prefix="cb-legacy-vault-")
        vault = (
            Path(self._tmp.name).resolve() / "Koob"
        )  # /var vs /private/var: read() relates paths
        self.mem = (
            factory(vault, cache=True) if self._cache and self._factory is None else factory(vault)
        )

    def stop(self) -> None:
        if self.mem is not None:
            self.mem.close()
            self.mem = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def admit(self, row: dict[str, Any], turn_id: str | None = None) -> None:
        text = str(row.get("text") or row.get("caption") or "")
        role = row.get("role")
        if not text or role not in ("owner", "other", "tool"):
            return
        if role == "owner":
            self._shown = self._hits(text)  # what this turn's pre-turn recall showed
            if turn_id in self.handled:
                return  # a correction / forget / remember turn: the memory call took it
            source = "owner"
        elif role == "tool":
            source = "web"
        else:
            chat = str(row.get("chat_id") or "chat")[:100].replace("/", "_")
            source = f"telegram:{chat}"
        self._stamp[self.mem.add(text, source)] = str(row.get("ts") or self._now())
        self.admitted += 1

    def on_remember(self, text: str) -> None:
        if self.mem is not None:
            self._stamp[self.mem.add(text, "owner")] = self._now()
            self.admitted += 1

    def on_fix(self, correction: str) -> None:
        slug = self.mem.top_note(correction)
        if slug is None:
            self.miss("correction")
        else:
            self.mem.overwrite(slug, correction)
            self._stamp[slug] = self._now()

    def _now(self) -> str:
        clock = getattr(self, "_clock", None)
        return clock.now().isoformat(timespec="minutes") if clock is not None else ""

    def _hits(self, query: str, k: int = 3) -> list[str]:
        index = getattr(self.mem, "index", None)  # a stand-in memory may have no index
        if index is None or not query.strip():
            return []
        getattr(self.mem, "_flush", lambda: None)()
        return [Path(n.path).stem for n in index.find(query).notes[:k]]

    def tool_view(self, route: str, wording: str, turn: Any = None) -> str | None:
        """The old Cube's equivalents: memory_search + memory_read of the notes it would open
        (the ones the last turn showed, for «это», and the top hits for the question); a read
        shows the note whole with its `source` and `created`, stamped in scenario time as the
        note would be in life. memory_resolve has no legacy equivalent: n/a."""
        from .routes import route_tools

        tools = set(route_tools(route)) - {"memory_resolve"}
        if not tools or self.mem is None:
            return None
        legacy = self.mem._memory
        out = []
        for slug in dict.fromkeys([*self._shown, *self._hits(wording)]):
            try:
                text = legacy.read(f"notes/{slug}.md")["text"]  # the layout of note_path
            except Exception:  # noqa: BLE001 - a deleted or backstage note is not read
                continue
            stamp = self._stamp.get(slug, "")[:10]
            if stamp:
                text = re.sub(r"(?m)^(created|updated): .*$", rf"\1: {stamp}", text)
            out.append(text)
        return "\n".join(out)

    def on_forget(self, target: str) -> None:
        slug = self.mem.top_note(target)
        if slug is None or not self.mem.delete(slug):
            self.miss("forget")

    def on_undo(self) -> None:
        if not self.mem.restore_last_delete():
            self.miss("undo")

    def view(self, wording: str | None) -> str:
        return self.mem.view(wording or "")

    def snapshot(self) -> Any:
        return self.mem.mark() if self.mem is not None else None

    def rollback(self, token: Any) -> None:
        if token is not None and self.mem is not None:
            self.mem.rollback(token)


# ---------------------------------------------------------------------- runner


class ApiBenchSUT(ApiSUT):
    """The real engine behind the facade, for the bench: `view()` is what the brain sees before
    it answers (core text + the recall basket + the store-tool results for the wording)."""

    name = "api"

    def __init__(self, **kw: Any) -> None:
        # The labeler is on in every bench run outside pytest: with it off no label became final,
        # the policy dropped the cloud route and every extract job parked, so the published api
        # numbers measured a memory that never extracted.
        kw.setdefault("labeler", "auto")
        super().__init__(**kw)
        self.misses: dict[str, int] = {}
        self.admitted = 0

    def view(self, wording: str | None) -> str:
        if wording is None:
            return ""
        from ..model import TurnCtx

        turn = TurnCtx(
            turn_id="bench", thread_id="th-main", session_id="sess-bench", owner_words=wording
        )
        st, parts = self.store(), []

        def grab(fn: Callable[[], Any]) -> Any:
            try:
                return fn()
            except NotBuilt:
                return None

        core = grab(
            lambda: self.api.core(thread_id=turn.thread_id, session_id=turn.session_id, store=st)
        )
        if core is not None:
            parts.append(core.text)
        basket = grab(lambda: self.api.recall(turn, store=st))
        if basket is not None:
            parts += [basket.block or "", *(i.text for i in basket.items)]
        found = grab(lambda: self.api.search(wording, turn=turn, store=st))
        if found:
            parts += [i.text for i in found]
        return "\n".join(x for x in parts if x)


def _blocks(p: Probe) -> list[Expect]:
    return [p.expect] if p.expect is not None else list(p.expect_any)


def _present(nt: str, value: Any) -> bool:
    return any(pat.search(nt) for pat in compile_value(value))


def _stale_checks(e: Expect) -> list[tuple[Any, bool]]:
    """(value, history allowed) for every value the brain must not see as current."""
    out = [(v, False) for v in e.must_not_include]
    for fb in e.forbidden:
        if set(fb.channels) & SEEN_CHANNELS:
            out += [
                (v, fb.allow == "history")
                for v in fb.values
                if not (isinstance(v, str) and v.startswith(_REFS))
            ]
    return out


def _stale_values(e: Expect) -> list[Any]:
    return [v for v, _ in _stale_checks(e)]


def _is_stale(text: str, nt: str, value: Any, history_ok: bool) -> bool:
    """As the graders read it: a value its own clause negates («без кинзы», "not Boulder
    anymore") is not stale, nor, with `allow: history`, one marked as history; any other
    occurrence is."""
    return shown_as_current(text, value, history=history_ok)


def judge_view(p: Probe, text: str) -> tuple[bool | None, bool | None, list[str]]:
    """(hit, stale, reasons) of one wording against what the brain sees; None = n/a."""
    nt = normalize(text)
    hits: list[bool] = []
    stales: list[bool] = []
    reasons: list[str] = []
    for e in _blocks(p):
        if e.must_include:
            miss = [v for v in e.must_include if not _present(nt, v)]
            hits.append(not miss)
            reasons += [f"missing: {describe(v)}" for v in miss]
        sv = _stale_checks(e)
        if sv:
            bad = [v for v, h in sv if _is_stale(text, nt, v, h)]
            stales.append(bool(bad))
            reasons += [f"stale: {describe(v)}" for v in bad]
    hit = any(hits) if hits else None
    stale = all(stales) if stales else None
    return hit, stale, reasons


class BenchRunner(Runner):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.records: list[dict[str, Any]] = []

    def _run_wording(self, p: Probe, wording: str | None, workdir: Path) -> WordingResult:
        sut: _BenchSUT = self.sut  # type: ignore[assignment]
        t0 = time.perf_counter()
        text = sut.view(wording)
        route = probe_route(p, self.scn.group)
        tool = sut.tool_view(route, wording) if wording is not None and route_tools(route) else None
        if tool:
            text = text + "\n" + tool
        ms = (time.perf_counter() - t0) * 1000
        hit, stale, reasons = judge_view(p, text)
        self.records.append(
            {
                "group": self.scn.group,
                "scenario": self.scn.id,
                "probe": p.id,
                "hit": hit,
                "stale": stale,
                "tokens": tokens_of(text),
                "ms": ms,
                "route": route,
                "tool": (
                    None
                    if wording is None or not route_tools(route)
                    else "n/a"
                    if tool is None
                    else len(tool)
                ),
            }
        )
        if hit is None and stale is None:
            return WordingResult(wording, SKIP, skipped=["n/a at memory level"], basket_ms=ms)
        ok = hit is not False and stale is not True
        return WordingResult(wording, PASS if ok else FAIL, reasons=reasons, basket_ms=ms)


def pick_scenarios(scns: list[Scenario], per_group: int | None, seed: int) -> list[Scenario]:
    """All scenarios, or a seeded sample of `per_group` per group (for the slow scales)."""
    if not per_group:
        return list(scns)
    by: dict[str, list[Scenario]] = {}
    for s in scns:
        by.setdefault(s.group, []).append(s)
    keep: set[str] = set()
    for g, items in by.items():
        ids = sorted(s.id for s in items)
        keep |= set(random.Random(f"{seed}:{g}").sample(ids, min(per_group, len(ids))))
    return [s for s in scns if s.id in keep]


def run_bench(
    loaded: LoadedSet,
    sut_factory: Callable[[], _BenchSUT],
    *,
    groups: list[str] | None = None,
    scenario_ids: list[str] | None = None,
    scale: str = "1x",
    seed: int = DEFAULT_SEED,
    per_group: int | None = None,
    on_result: Callable[[ScenarioResult, int], None] | None = None,
) -> dict[str, Any]:
    """Replay the set; returns {"records", "misses", "errors", "admitted", "scenarios"}."""
    records: list[dict[str, Any]] = []
    misses: dict[str, int] = {}
    errors: list[str] = []
    admitted = 0
    n_scn = 0
    labeler: set[str] = set()
    notes: list[str] = []
    for scn in pick_scenarios(loaded.scenarios, per_group, seed):
        if groups and scn.group not in groups:
            continue
        if scenario_ids and scn.id not in scenario_ids:
            continue
        sut = sut_factory()
        r = BenchRunner(scn, loaded.personas, sut=sut, scale=scale, seed=seed)  # type: ignore[arg-type]
        res = r.run()
        n_scn += 1
        records += r.records
        admitted += sut.admitted
        on = getattr(sut, "labeler_on", None)
        labeler.add("n/a" if on is None else ("on" if on() else "off"))
        notes += [f"{scn.id}: {n}" for n in getattr(sut, "notes", [])]
        for k, v in sut.misses.items():
            misses[k] = misses.get(k, 0) + v
        if res.error:
            errors.append(f"{scn.id}: {res.error}")
        if on_result:
            on_result(res, len(r.records))
    return {
        "scenarios": n_scn,
        "records": records,
        "misses": misses,
        "errors": errors,
        "admitted": admitted,
        "labeler": "/".join(sorted(labeler)) or "n/a",
        "notes": notes,
    }


# ---------------------------------------------------------------------- numbers


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per group and pooled ("ALL"): rates over wordings that asked for a value."""
    out: dict[str, Any] = {}
    groups = sorted({r["group"] for r in records})
    for g in [*groups, "ALL"]:
        rs = [r for r in records if g == "ALL" or r["group"] == g]
        h = [r["hit"] for r in rs if r["hit"] is not None]
        s = [r["stale"] for r in rs if r["stale"] is not None]
        val = [r for r in rs if r["hit"] is not None or r["stale"] is not None]
        out[g] = {
            "wordings": len(rs),
            "na": len(rs) - len(val),
            "hit_n": len(h),
            "hit": round(sum(h) / len(h), 4) if h else None,
            "stale_n": len(s),
            "stale": round(sum(s) / len(s), 4) if s else None,
            "tokens_mean": round(sum(r["tokens"] for r in val) / len(val), 1) if val else None,
            "ms_p50": round(_pct([r["ms"] for r in val], 0.5) or 0.0, 3) if val else None,
            "ms_p95": round(_pct([r["ms"] for r in val], 0.95) or 0.0, 3) if val else None,
        }
    return out


# ---------------------------------------------------------------------- report

METRICS = [
    ("hit", "Hit rate (all expected values visible to the brain)", "{:.0%}"),
    ("stale", "Stale rate (a forbidden / superseded / forgotten value visible)", "{:.0%}"),
    ("tokens_mean", "Tokens carried per question (chars/4)", "{:,.0f}"),
    ("ms_p50", "Memory-step time p50 (ms)", "{:,.1f}"),
    ("ms_p95", "Memory-step time p95 (ms)", "{:,.1f}"),
    ("na", "n/a wordings (nothing to look up at memory level)", "{:d}"),
]


def render_markdown(data: dict[str, Any]) -> str:
    runs = data["runs"]  # {sut: {scale: {"summary", "misses", ...}}}
    suts = list(runs)
    scales = sorted(
        {s for r in runs.values() for s in r},
        key=lambda x: (x.split()[1:], int(x.split()[0].rstrip("x"))),
    )
    groups = sorted(
        {g for r in runs.values() for sc in r.values() for g in sc["summary"] if g != "ALL"}
    )
    cols = [(s, sc) for s in suts for sc in scales if sc in runs[s]]
    meta = data["meta"]
    out = [f"# Memory baseline, {meta['date']}", "", meta["method"], ""]
    for key, title, fmt in METRICS:
        out += [f"## {title}", ""]
        out.append("| group | " + " | ".join(f"{s} {sc}" for s, sc in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for g in [*groups, "ALL"]:
            cells = []
            for s, sc in cols:
                v = runs[s][sc]["summary"].get(g, {}).get(key)
                cells.append("n/a" if v is None else fmt.format(v))
            out.append(f"| {g} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## Misses (no legacy equivalent, or no recall hit)", ""]
    for s, sc in cols:
        m = runs[s][sc].get("misses") or {}
        text = ", ".join(f"{k} x{v}" for k, v in sorted(m.items())) or "none"
        out.append(f"- {s} {sc}: {text}")
    out += ["", "## Reproduce", "", "```bash", *meta["commands"], "```", ""]
    return "\n".join(out)


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
