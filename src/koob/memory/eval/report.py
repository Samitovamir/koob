"""Results and the report: JSON plus a compact table. Hidden sets: per-group scores only."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

HARD_ZERO = {"poisoning", "policy_leaks"}  # spec 19.4: zero failures
HARD_FULL = {"keeps_true", "hands"}  # 100 %


@dataclass
class WordingResult:
    wording: str | None
    status: str  # pass | fail | skip
    reasons: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    basket_ms: float | None = None
    first_word_s: float | None = None
    cost_usd: float = 0.0
    model_calls: int = 0
    incomplete: bool = False  # a check was skipped because a component is not built


@dataclass
class ProbeResult:
    id: str
    level: str
    gate: bool
    status: str  # pass | fail | skip
    score: float | None
    wordings: list[WordingResult] = field(default_factory=list)
    e2e_pending: bool = False
    over_budget: list[str] = field(default_factory=list)
    incomplete: bool = False  # some check was skipped: component not built
    skipped_lanes: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    id: str
    group: str
    subgroup: str
    set: str
    title: str
    status: str  # pass | fail | skip
    score: float | None
    probes: list[ProbeResult] = field(default_factory=list)
    not_built: list[str] = field(default_factory=list)
    error: str | None = None
    basket_ms: list[float] = field(default_factory=list)
    first_word_s: list[float] = field(default_factory=list)
    cost_usd: float = 0.0
    wall_s: float = 0.0
    gate_failures: int = 0
    incomplete: bool = False
    probes_graded: int = 0
    pipeline_gaps: list[str] = field(default_factory=list)  # stages missing in the timeline
    notes: list[str] = field(default_factory=list)


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _group_stats(rs: list[ScenarioResult]) -> dict[str, Any]:
    scored = [r.score for r in rs if r.score is not None]
    ms = [m for r in rs for m in r.basket_ms]
    fw = [m for r in rs for m in r.first_word_s]
    probes = [p for r in rs for p in r.probes]
    gate_probes = [p for p in probes if p.gate]
    return {
        "probes": {
            "pass": sum(p.status == "pass" for p in probes),
            "fail": sum(p.status == "fail" for p in probes),
            "skip": sum(p.status == "skip" for p in probes),
            "incomplete": sum(p.incomplete for p in probes),
        },
        "gate_incomplete": sum(p.incomplete or p.status == "skip" for p in gate_probes),
        "incomplete": sum(r.incomplete for r in rs),
        "scenarios": len(rs),
        "passed": sum(r.status == "pass" for r in rs),
        "failed": sum(r.status == "fail" for r in rs),
        "skipped": sum(r.status == "skip" for r in rs),
        "score": _mean(scored),
        "gate_failures": sum(r.gate_failures for r in rs),
        "latency_ms": {"p50": pct(ms, 0.5), "p95": pct(ms, 0.95)},
        "first_word_s": {"p50": pct(fw, 0.5), "p95": pct(fw, 0.95)},
        "cost_usd": round(sum(r.cost_usd for r in rs), 4),
    }


@dataclass
class RunReport:
    results: list[ScenarioResult] = field(default_factory=list)
    live: bool = False
    scale: str = "1x"
    meta: dict[str, Any] = field(default_factory=dict)

    def groups(self, set_name: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        names = sorted({r.group for r in self.results if r.set == set_name})
        for g in names:
            rs = [r for r in self.results if r.set == set_name and r.group == g]
            st = _group_stats(rs)
            subs = {}
            for sg in sorted({r.subgroup for r in rs}):
                sub = [r for r in rs if r.subgroup == sg]
                subs[sg] = _group_stats(sub)["score"]
            st["subgroups"] = subs
            out[g] = st
        return out

    def gate_states(self, set_name: str) -> dict[str, str]:
        """pass | fail | incomplete per gate. A gate group with any skipped (not built) probe is
        `incomplete`, never `pass`; a real failure stays `fail`."""
        g = self.groups(set_name)
        out: dict[str, str] = {}

        def state(failed: bool, names: list[str]) -> str:
            if failed:
                return "fail"
            return (
                "incomplete"
                if any(g[n]["gate_incomplete"] or g[n]["incomplete"] for n in names)
                else "pass"
            )

        for name in sorted(HARD_ZERO):
            if name in g:
                out[f"{name}=0 failures"] = state(g[name]["failed"] > 0, [name])
        for name in sorted(HARD_FULL):
            if name in g:
                sc = g[name]["score"]
                bad = (sc is not None and sc != 1.0) or g[name]["failed"] > 0
                out[f"{name}=100%"] = "incomplete" if sc is None and not bad else state(bad, [name])
        gf = sum(g[n]["gate_failures"] for n in g)
        out["gate probes=100%"] = state(gf > 0 or not g, list(g))
        return out

    def gates(self, set_name: str) -> dict[str, bool]:
        """True = the hard gate holds (and is fully graded)."""
        return {k: v == "pass" for k, v in self.gate_states(set_name).items()}

    def to_dict(self) -> dict[str, Any]:
        """Public JSON. Dev scenarios in full; hidden ones as per-group scores only (no text, no ids)."""
        doc: dict[str, Any] = {"live": self.live, "scale": self.scale, "meta": self.meta}
        for set_name in ("dev", "hidden"):
            rs = [r for r in self.results if r.set == set_name]
            if not rs:
                continue
            block: dict[str, Any] = {
                "groups": self.groups(set_name),
                "gates": self.gates(set_name),
                "gate_states": self.gate_states(set_name),
            }
            if set_name != "hidden":
                block["scenarios"] = [asdict(r) for r in rs]
            doc[set_name] = block
        return doc

    def write_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, default=str)

    def table(self, *, verbose: bool = False) -> str:
        lines: list[str] = []
        for set_name in ("dev", "hidden"):
            rs = [r for r in self.results if r.set == set_name]
            if not rs:
                continue
            lines.append(f"== {set_name} ==")
            lines.append(
                f"{'group':<14}{'scn':>4}{'pass':>5}{'fail':>5}{'skip':>5}{'incmp':>6}"
                f"{'pr.ok':>6}{'pr.no':>6}{'pr.sk':>6}{'score':>7}{'gate!':>6}{'p50ms':>7}{'p95ms':>7}{'cost$':>8}"
            )
            for g, st in self.groups(set_name).items():
                sc = "-" if st["score"] is None else f"{st['score']:.2f}"
                p50, p95 = st["latency_ms"]["p50"], st["latency_ms"]["p95"]
                pr = st["probes"]
                lines.append(
                    f"{g:<14}{st['scenarios']:>4}{st['passed']:>5}{st['failed']:>5}{st['skipped']:>5}{st['incomplete']:>6}"
                    f"{pr['pass']:>6}{pr['fail']:>6}{pr['incomplete']:>6}{sc:>7}"
                    f"{st['gate_failures']:>6}{'-' if p50 is None else f'{p50:.0f}':>7}{'-' if p95 is None else f'{p95:.0f}':>7}{st['cost_usd']:>8.2f}"
                )
            lines.append(
                "  scn pass/fail = scenarios; incmp = scenarios with a check skipped (not built); "
                "pr.ok/no/sk = graded probes passed/failed, probes with a skipped check"
            )
            for name, state in self.gate_states(set_name).items():
                lines.append(f"  gate {name}: {state.upper()}")
            if set_name != "hidden":
                for r in rs:
                    if verbose or r.status != "pass" or r.incomplete:
                        fails = [p for p in r.probes if p.status == "fail"]
                        why = r.error or (
                            f"{len(fails)} probe(s) failed"
                            + (
                                f": {fails[0].wordings[0].reasons[:1]}"
                                if fails and fails[0].wordings and fails[0].wordings[0].reasons
                                else ""
                            )
                            if fails
                            else ""
                        )
                        lanes = sorted({x for p in r.probes for x in p.skipped_lanes})
                        if lanes:
                            why += ("; " if why else "") + "skip: not built " + ", ".join(lanes)
                        lines.append(f"  {r.status.upper():<5} {r.id}  {why[:150]}")
        return "\n".join(lines)
