"""G13 import boundaries (the memory spec §2, §16.4).

`memory_imports` is an active gate: nothing in `koob/memory/` imports `koob.web` (absolute or
relative), `koob.assistant` or island code.
`attach_closure` is the cutover gate: the import closure of the runtime attach points must
not reach `koob.web`; it currently fails (`assistant/tools.py:46`, `assistant/context.py:24` ->
`web.nav`; `assistant/tools.py:924-1179` -> `web.tasks`) and stays xfail until those modules
move to a neutral package as part of the page removal.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PKG = SRC / "koob"
FORBIDDEN = ("koob.web", "koob.assistant")
ISLAND_WORDS = ("island", "isle")


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path) -> list[tuple[int, str]]:
    return _imports_of(
        path.read_text(encoding="utf-8"), _module_name(path), path.name == "__init__.py"
    )


def _imports_of(text: str, name: str, is_package: bool) -> list[tuple[int, str]]:
    """Every module name a file imports, relative imports resolved, nested ones included,
    plus string literals passed to import_module / __import__."""
    package = name if is_package else name.rpartition(".")[0]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            found.append((node.lineno, mod))
            found += [(node.lineno, f"{mod}.{a.name}") for a in node.names]
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.append((node.lineno, arg.value))
    return found


def _bad(mod: str) -> bool:
    if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN):
        return True
    return mod.startswith("koob.") and any(w in mod.split(".")[-1] for w in ISLAND_WORDS)


def test_memory_imports():
    """§2: [T test_boundaries::memory_imports]."""
    offenders = []
    files = sorted((PKG / "memory").rglob("*.py"))
    assert files, "memory package not found"
    for path in files:
        for line, mod in _imports(path):
            if _bad(mod):
                offenders.append(f"{path.relative_to(SRC)}:{line} imports {mod}")
    assert not offenders, "\n".join(offenders)


def test_checker_resolves_relative_and_dynamic_imports():
    """The checker itself catches `from ...web import nav` inside memory/adapters/."""
    probe = (
        "from ...web import nav\n"
        "from ...assistant import guard\n"
        "import importlib\n"
        "importlib.import_module('koob.web.tasks')\n"
        "from . import conversation\n"
    )
    mods = [m for _, m in _imports_of(probe, "koob.memory.adapters.turn", False)]
    assert "koob.web" in mods
    assert "koob.assistant" in mods
    assert "koob.web.tasks" in mods
    assert "koob.memory.adapters" in mods
    assert [m for m in mods if _bad(m)] == [
        "koob.web",
        "koob.web.nav",
        "koob.assistant",
        "koob.assistant.guard",
        "koob.web.tasks",
    ]


def _closure(roots: list[str]) -> set[str]:
    index = {_module_name(p): p for p in PKG.rglob("*.py")}
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        mod = todo.pop()
        if mod in seen or mod not in index:
            continue
        seen.add(mod)
        for _, dep in _imports(index[mod]):
            # `from pkg import name` may name a submodule or an attribute
            for cand in (dep, dep.rpartition(".")[0]):
                if cand in index and cand not in seen:
                    todo.append(cand)
    return seen


ATTACH_POINTS = [
    "koob.assistant.runtime",
    "koob.assistant.tools",
    "koob.assistant.context",
    "koob.memory.api",
]


@pytest.mark.xfail(
    strict=True, reason="cutover gate (§16.4): web.nav / web.tasks still imported by koob"
)
def test_attach_closure():
    """§16.4: the attach points' import closure never reaches koob.web."""
    reached = sorted(
        m for m in _closure(ATTACH_POINTS) if _bad(m) and m.startswith("koob.web")
    )
    assert not reached, reached


def test_memory_api_closure_is_clean():
    """The facade alone already stays clear of web and koob."""
    reached = sorted(m for m in _closure(["koob.memory.api"]) if _bad(m))
    assert not reached, reached
