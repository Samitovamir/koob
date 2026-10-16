"""Canary sweep (spec 19.5): plant unique tokens, run actions, scan every content-map store.

Tokens are swapped per run for fresh random ones of the same shape, so earlier runs never
pollute a scan. The scan reads bytes in UTF-8, JSON `\\uXXXX` escapes and UTF-16LE. Store
paths are config-driven (`StoreSpec`); `default_stores()` is the memory seams note Part B.
"""

from __future__ import annotations

import glob
import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

from .schema import Canary

LAT = string.ascii_uppercase
CYR = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЭЮЯ"


def fresh_token(c: Canary, rng: random.Random) -> Canary:
    """Same shape, new characters. A run of digits maps to one new run everywhere it occurs."""
    digit_map: dict[str, str] = {}

    def swap(s: str, alphabet: str) -> str:
        def letters(m: re.Match[str]) -> str:
            return "".join(rng.choice(alphabet) for _ in m.group())

        def digits(m: re.Match[str]) -> str:
            run = m.group()
            if run not in digit_map:
                digit_map[run] = "".join(rng.choice("0123456789") for _ in run)
            return digit_map[run]

        s = re.sub(r"\d+", digits, s)
        return re.sub(r"[^\W\d_]+", letters, s)

    return c.model_copy(
        update={"lat": swap(c.lat, LAT), "cyr": swap(c.cyr, CYR), "dig": swap(c.dig, "0123456789")}
    )


def spellings(c: Canary) -> dict[str, str]:
    return {"lat": c.lat, "cyr": c.cyr, "dig": c.dig}


def encodings(s: str) -> dict[str, bytes]:
    """The byte forms a stored copy can take. JSON escapes in both hex cases."""
    out = {"utf8": s.encode("utf-8"), "utf16le": s.encode("utf-16-le")}
    esc = json.dumps(s, ensure_ascii=True)[1:-1]
    if esc != s:
        out["json_escape"] = esc.encode("ascii")
        upper = re.sub(r"\\u([0-9a-f]{4})", lambda m: "\\u" + m.group(1).upper(), esc).encode()
        if upper != out["json_escape"]:
            out["json_escape_upper"] = upper
    return out


@dataclass(frozen=True)
class Hit:
    canary: str
    spelling: str
    encoding: str
    path: str
    store: str


@dataclass
class StoreSpec:
    """One content-map store: a name and the globs (files or directories) that make it up."""

    name: str
    globs: list[str]
    note: str = ""


@dataclass
class StoreMap:
    stores: list[StoreSpec]
    excluded: dict[str, str] = field(default_factory=dict)  # name -> reason (audio, WebKit)
    # allowance kind -> store names where a hit is fine ("turn_conversation:<ev>" matches by prefix)
    allow: dict[str, list[str]] = field(
        default_factory=lambda: {
            "none": [],
            "turn_conversation": ["career_db", "sdk_transcripts"],
            "trash_until_purge": ["memory_db"],
            "history": ["memory_db"],
        }
    )


def default_stores(
    *,
    data_dir: Path,
    workspace_dir: Path,
    claude_projects_dir: Path | None = None,
    vault_dir: Path | None = None,
) -> StoreMap:
    """Part B of the memory seams note as globs. The bot stub's sent log is added by the caller."""
    stores = [
        StoreSpec(
            "memory_db",
            [str(data_dir / "memory" / "memory.db*"), str(data_dir / "memory" / "vectors.db*")],
        ),
        StoreSpec(
            "memory_files",
            [
                str(data_dir / "memory" / "hotlog*"),
                str(data_dir / "memory" / "journal"),
                str(data_dir / "memory" / "backups"),
            ],
        ),
        StoreSpec(
            "career_db", [str(data_dir / "koob.db*"), str(data_dir / "backups" / "career-*.db")]
        ),
        StoreSpec("llm_logs", [str(data_dir / "logs" / "llm")]),
        StoreSpec("server_log", [str(data_dir / "logs" / "server.log*")]),
        StoreSpec("workspace", [str(workspace_dir)]),
    ]
    if claude_projects_dir:
        stores.append(StoreSpec("sdk_transcripts", [str(claude_projects_dir)]))
    if vault_dir:
        stores.append(StoreSpec("vault", [str(vault_dir), str(vault_dir / ".history")]))
    return StoreMap(
        stores=stores,
        excluded={
            "audio": "synthesised speech: no text content",
            "webkit_store": "page prefs only; message routes covered by the no-store test",
        },
    )


def _files(spec: StoreSpec) -> list[Path]:
    seen: dict[str, Path] = {}
    for g in spec.globs:
        for m in map(Path, glob.glob(g, recursive=True)):
            if m.is_dir():
                for f in m.rglob("*"):
                    if f.is_file():
                        seen[str(f)] = f
            elif m.is_file():
                seen[str(m)] = m
    return sorted(seen.values())


def scan(smap: StoreMap, canaries: list[Canary], *, turn_ids: set[str] | None = None) -> list[Hit]:
    """Every hit of every spelling in every encoding. Whether a hit is allowed is decided by `unlisted`."""
    needles: list[tuple[str, str, str, bytes]] = []
    for c in canaries:
        for sp, text in spellings(c).items():
            for enc, b in encodings(text).items():
                needles.append((c.id, sp, enc, b))
    hits: dict[tuple[str, str, str, str], Hit] = {}
    for spec in smap.stores:
        for f in _files(spec):
            try:
                data = f.read_bytes()
            except OSError:
                continue
            for cid, sp, enc, b in needles:
                if b in data:
                    hits.setdefault((cid, sp, enc, str(f)), Hit(cid, sp, enc, str(f), spec.name))
    return sorted(hits.values(), key=lambda h: (h.canary, h.path, h.spelling, h.encoding))


def unlisted(
    hits: list[Hit], canaries: list[Canary], smap: StoreMap, *, purged: bool = False
) -> dict[str, list[Hit]]:
    """Hits outside each token's allowed list. `purged`: trash_until_purge no longer allows anything."""
    allowed_stores: dict[str, set[str]] = {}
    for c in canaries:
        ok: set[str] = set()
        for a in c.allowed:
            kind = a.split(":", 1)[0]
            if kind == "trash_until_purge" and purged:
                continue
            ok |= set(smap.allow.get(kind, []))
        allowed_stores[c.id] = ok
    out: dict[str, list[Hit]] = {c.id: [] for c in canaries}
    for h in hits:
        if h.store not in allowed_stores.get(h.canary, set()):
            out[h.canary].append(h)
    return out
