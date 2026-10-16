"""Scenario loader: validates every file against the strict schema and fails loudly.

A hidden set is loaded from a path given at run time. Its errors never quote scenario text:
they say how many files failed and in which group directory (`redact=True`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import Scenario


class ScenarioError(Exception):
    """A scenario file does not match the format."""


@dataclass
class LoadedSet:
    root: Path
    personas: dict[str, Any]
    scenarios: list[Scenario] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    set_name: str = "dev"

    def by_group(self) -> dict[str, list[Scenario]]:
        out: dict[str, list[Scenario]] = {}
        for s in self.scenarios:
            out.setdefault(s.group, []).append(s)
        return out


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_scenario(
    path: Path, personas: dict[str, Any] | None = None, *, expect_set: str | None = None
) -> Scenario:
    """Load one file. Raises ScenarioError naming the file and every problem."""
    try:
        raw = _read(path)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: top level must be a mapping")
    try:
        scn = Scenario.model_validate(raw)
    except ValidationError as exc:
        lines = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise ScenarioError(f"{path}: " + "; ".join(lines)) from exc
    problems: list[str] = []
    if path.stem != scn.id:
        problems.append(f"file name {path.stem!r} != id {scn.id!r}")
    if path.parent.name != scn.group:
        problems.append(f"folder {path.parent.name!r} != group {scn.group!r}")
    if expect_set and scn.set != expect_set:
        problems.append(f"set {scn.set!r} does not match the tree ({expect_set})")
    if personas is not None:
        problems += _persona_problems(scn, personas)
    if problems:
        raise ScenarioError(f"{path}: " + "; ".join(problems))
    return scn


def _persona_problems(scn: Scenario, personas: dict[str, Any]) -> list[str]:
    if scn.persona not in personas:
        return [f"unknown persona {scn.persona!r}"]
    pers = personas[scn.persona] or {}
    contacts = set(pers.get("contacts") or {}) | set(scn.contacts)
    out: list[str] = []
    for e in scn.timeline:
        sender = getattr(e, "from_", None)
        if sender and not sender.startswith("unknown:") and sender not in contacts:
            out.append(f"event {e.id}: unknown contact {sender!r}")
    return out


def load_set(
    root: Path | str,
    *,
    redact: bool = False,
    strict: bool = True,
    expect_set: str | None = "auto",
    base_personas: dict[str, Any] | None = None,
) -> LoadedSet:
    """Load `<root>/<group>/*.yaml` (and `personas.yaml`). Raises ScenarioError listing all failures.

    `redact`: errors name only the count and the group directories (hidden sets).
    """
    root = Path(root)
    if not root.is_dir():
        raise ScenarioError(
            "scenario root is not a directory" if redact else f"{root}: not a directory"
        )
    pfile = root / "personas.yaml"
    personas = _read(pfile) if pfile.exists() else {}
    ls = LoadedSet(root=root, personas={**(base_personas or {}), **(personas or {})})
    # Hidden-only personas may sit in the hidden tree (any *personas*.yaml at the root).
    for extra in sorted(root.glob("personas*.yaml")):
        if extra != pfile:
            ls.personas.update(_read(extra) or {})
    if expect_set == "auto":
        if "hidden" in root.parts:
            expect_set = "hidden"
        else:
            expect_set = "dev" if (root / "personas.yaml").exists() else None
    failed_groups: dict[str, int] = {}
    for gdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if gdir.name == "crosslingual":
            continue
        for f in sorted(gdir.glob("*.yaml")):
            try:
                ls.scenarios.append(load_scenario(f, ls.personas, expect_set=expect_set))
            except ScenarioError as exc:
                ls.errors.append(str(exc))
                failed_groups[gdir.name] = failed_groups.get(gdir.name, 0) + 1
    if ls.scenarios:
        ls.set_name = ls.scenarios[0].set
    if ls.errors and strict:
        if redact:
            where = ", ".join(f"{g}: {n}" for g, n in sorted(failed_groups.items()))
            raise ScenarioError(f"{len(ls.errors)} scenario file(s) failed validation ({where})")
        raise ScenarioError("\n".join(ls.errors))
    return ls
