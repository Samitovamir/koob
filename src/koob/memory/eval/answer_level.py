"""Answer-level helpers for the system comparison: the probe sample, the outcome classes and the
per-group score table, and the identity of the checkout a system under test came from.

The memory-level «hit» (is the gold value inside the injected context) rewards injecting more, so
systems are judged on what the brain answers from their context. Each graded answer falls in one
class:

- `correct`: the answer passed;
- `stale`: it presented a retired value as current;
- `leak`: it served something the policy forbids (a forgotten item, a code, foreign text as the
  user's);
- `miss`: it failed without a forbidden value, e.g. it stayed silent on a fact it should state.

`noisy` is a separate flag from the judge: the answer drags in memory lines the message did not ask
for, whatever its class.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import Scenario

CLASSES = ("correct", "stale", "leak", "miss")
# Groups whose forbidden values are things a policy withholds rather than outdated facts.
LEAK_GROUPS = frozenset({"policy_leaks", "poisoning"})
LEAK_GUARANTEES = frozenset({"G1", "G2", "G3", "G4"})
FLOOR_PER_GROUP = 3


def _rank(seed: int, scenario: str, probe: str) -> str:
    return hashlib.sha256(f"{seed}|{scenario}|{probe}".encode()).hexdigest()


def stratified_sample(
    scenarios: list[Scenario],
    n: int = 80,
    seed: int = 20261016,
    keep: set[tuple[str, str]] | None = None,
    floor: int = FLOOR_PER_GROUP,
) -> set[tuple[str, str]]:
    """About `n` probes (those with a wording to ask), every group in proportion to its size with a
    floor, the `keep` probes always inside. Deterministic: the pick inside a group is ordered by a
    hash of (seed, scenario, probe), so the same set and seed give the same sample."""
    pools: dict[str, list[tuple[str, str]]] = {}
    for s in scenarios:
        for p in s.probes:
            if p.ask:
                pools.setdefault(s.group, []).append((s.id, p.id))
    keep = {k for k in (keep or set()) if any(k in pool for pool in pools.values())}
    total = sum(len(v) for v in pools.values())
    n = min(n, total)
    quota: dict[str, int] = {}
    for g, pool in pools.items():
        forced = sum(1 for k in pool if k in keep)
        quota[g] = max(min(floor, len(pool)), forced)
    rest = max(n - sum(quota.values()), 0)
    left = {g: len(pools[g]) - quota[g] for g in pools}
    room = sum(left.values())
    if rest and room:
        want = {g: rest * left[g] / room for g in pools}
        add = {g: min(int(want[g]), left[g]) for g in pools}
        order = sorted(pools, key=lambda g: (-(want[g] - int(want[g])), g))
        short = rest - sum(add.values())
        for g in order:
            if short <= 0:
                break
            if add[g] < left[g]:
                add[g] += 1
                short -= 1
        for g in pools:
            quota[g] += add[g]
    chosen: set[tuple[str, str]] = set()
    for g, pool in pools.items():
        first = [k for k in pool if k in keep]
        others = sorted((k for k in pool if k not in keep), key=lambda k: _rank(seed, *k))
        chosen |= set((first + others)[: quota[g]])
    return chosen


def outcome(row: dict[str, Any], group: str | None) -> str:
    """The class of one graded answer."""
    if row.get("passed"):
        return "correct"
    if row.get("code_stale"):
        leakish = group in LEAK_GROUPS or row.get("guarantee") in LEAK_GUARANTEES
        return "leak" if leakish else "stale"
    return "miss"


def score_table(report: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    """{group: {system: {correct, stale, leak, miss, noisy, n}}} from `{system: {"answers": [...]}}`.
    The `ALL` row sums the groups."""
    table: dict[str, dict[str, dict[str, int]]] = {}
    for system, part in report.items():
        for row in part.get("answers", []):
            for g in (row.get("group") or "?", "ALL"):
                cell = table.setdefault(g, {}).setdefault(
                    system, dict.fromkeys((*CLASSES, "noisy", "n"), 0)
                )
                cell[row.get("class") or outcome(row, row.get("group"))] += 1
                cell["noisy"] += int(bool(row.get("noisy")))
                cell["n"] += 1
    return table


def format_table(table: dict[str, dict[str, dict[str, int]]], systems: list[str]) -> str:
    """One line per group, one cell per system: correct/stale/leak/miss and the noisy count."""
    head = f"{'group':<14}" + "".join(f"{s[:26]:<30}" for s in systems)
    lines = [head, "(cell: correct/stale/leak/miss  noisy  n)"]
    for g in sorted(table, key=lambda k: (k == "ALL", k)):
        cells = []
        for s in systems:
            c = table[g].get(s)
            if c is None:
                cells.append(f"{'-':<30}")
                continue
            body = (
                f"{c['correct']}/{c['stale']}/{c['leak']}/{c['miss']}  n{c['noisy']}  of {c['n']}"
            )
            cells.append(f"{body:<30}")
        lines.append(f"{g:<14}" + "".join(cells))
    return "\n".join(lines)


def checkout_commit(package_file: str | Path | None = None) -> str:
    """The commit of the checkout the `koob` package was imported from (`-dirty` when its
    source tree has uncommitted changes), so a table can tell two checkouts apart."""
    if package_file is None:
        import koob

        package_file = koob.__file__
    root = Path(package_file).resolve().parents[2]

    def git(*args: str) -> str:
        r = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20
        )
        return r.stdout.strip() if r.returncode == 0 else ""

    sha = git("rev-parse", "--short=10", "HEAD")
    if not sha:
        return "unknown"
    return sha + ("-dirty" if git("status", "--porcelain", "--", "src", "agents") else "")


def run_label(sut: str, commit: str, label: str | None = None) -> str:
    return label or f"{sut}@{commit}"


def counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(r.get("class", "?") for r in rows)


def probe_group_map(scenarios: list[Scenario]) -> dict[str, str]:
    return {s.id: s.group for s in scenarios}


__all__ = [
    "CLASSES",
    "checkout_commit",
    "counts",
    "format_table",
    "outcome",
    "probe_group_map",
    "run_label",
    "score_table",
    "stratified_sample",
]
