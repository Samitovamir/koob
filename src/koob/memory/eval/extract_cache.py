"""Extraction processors for the eval, with an on-disk cache.

`--extract cloud|local` runs the extract worker over the scenario's chats. A model call costs
money or minutes, so every processor output is cached under the workspace cache, keyed by
(model id, prompt hash, input hash): a rerun of the same set makes no model call. The eval is not
a turn path, so a slow first run is fine; the second is free.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MODES = ("off", "cloud", "local")

# Row ids are minted from the wall clock (ULIDs) and derived ids hash them (`x` + 24 hex), so
# the same scenario gets new ids on every run. The cache keys on the snapshot with ids replaced
# by their order of appearance, and stores the answer the same way: a rerun hits.
_ID = re.compile(r"\b(?:[0-9A-HJKMNP-TV-Z]{26}|x[0-9a-f]{24})\b")
_PH = re.compile(r"@@ID(\d+)@@")


def canonical(snap: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """The snapshot as JSON with ids as `@@ID<n>@@`, and the map id -> placeholder."""
    blob = json.dumps(snap, ensure_ascii=False, sort_keys=True, default=str)
    ids: dict[str, str] = {}

    def sub(m: re.Match[str]) -> str:
        return ids.setdefault(m.group(0), f"@@ID{len(ids)}@@")

    return _ID.sub(sub, blob), ids


def cache_root() -> Path:
    from ...paths import WORKSPACE_DIR

    return WORKSPACE_DIR / "cache" / "memory-eval-extract"


def prompt_hash() -> str:
    """Hash of the extraction prompt text; a prompt edit invalidates the cache."""
    from ..workers import extract

    try:
        from koob.llm.runner import prompt_for

        text = prompt_for(extract.ROLE)
    except Exception:  # noqa: BLE001 - no prompt file: the prompt id stands in
        text = extract.PROMPT_ID
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# The key is what the extraction model reads of a snapshot: `workers.extract.user_text`,
# read back, so it follows the extractor's input as it changes. Everything else in a snapshot
# is bookkeeping: record versions and raw revisions (the read set), labels and their finality
# (the route already filtered on them), `text_en`, dedup-only sections. Those changed from run
# to run with the label-pass timing, and rows minted in one millisecond came back in a random id
# order, so a key over the whole snapshot missed for the same model input.
VIEW_VERSION = "view-2"
_FIRST = ("known_entities", "messages")  # ids are numbered in this order, then the rest


def _dump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)


def _model_input(snap: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """(the prompt text around the data block, the data block as JSON)."""
    from ..workers import extract

    text = extract.user_text(snap)
    a, b = text.find("<data>\n"), text.rfind("\n</data>")
    if a < 0 or b < a:
        return text, {}
    return text[:a] + text[b:], json.loads(text[a + len("<data>\n") : b])


def canonical_view(snap: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """The model's input as one string with ids as `@@ID<n>@@`, and the map id ->
    placeholder. Every list it reads is sorted by content (ids masked), so neither the id order
    nor a bookkeeping field moves the key; the messages keep their order, which the model reads
    as the conversation. Equal items keep the snapshot's order."""
    ids: dict[str, str] = {}

    def sub(text: str) -> str:
        return _ID.sub(lambda m: ids.setdefault(m.group(0), f"@@ID{len(ids)}@@"), text)

    def peek(text: str) -> str:
        return _ID.sub(lambda m: ids.get(m.group(0), "@@ID?@@"), text)

    try:
        wrapper, view = _model_input(snap)
    except (KeyError, TypeError):  # not a worker snapshot (a partial one in a test): all of it
        blob, whole = canonical(snap)
        return "snapshot=" + blob, whole
    parts = [wrapper]
    keys = [k for k in _FIRST if k in view] + sorted(k for k in view if k not in _FIRST)
    for key in keys:
        val = view[key]
        if isinstance(val, list) and key != "messages":
            val = sorted(val, key=lambda x: peek(_dump(x)))
        parts.append(f"{key}=" + sub(_dump(val)))
    return "\n".join(parts), ids


def view_hash(snap: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_view(snap)[0].encode("utf-8")).hexdigest()[:32]


def bookkeeping_variants(
    snap: dict[str, Any], *, radius: int = 6, limit: int = 200_000
) -> Iterator[dict[str, Any]]:
    """Snapshots with the same model input as `snap` that an earlier run may have keyed the
    old whole-snapshot cache on: the entities in another id order, other record versions
    (nearest first). For replaying a cache written before `canonical_view`; never changes
    what the model reads."""
    ents = list(snap.get("entities") or [])
    facts = list(snap.get("facts") or [])
    order = range(len(ents))
    perms = list(itertools.permutations(order)) if len(ents) <= 5 else [tuple(order)]
    n = 0
    for r in range(radius + 1):
        ring = [
            o
            for o in itertools.product(range(-r, r + 1), repeat=len(ents))
            if max(map(abs, o), default=0) == r
        ]
        for perm in perms:
            for offs in ring:
                vers = [
                    int(ents[i].get("version") or 1) + d for i, d in zip(perm, offs, strict=True)
                ]
                if any(v < 1 for v in vers):
                    continue
                es = [{**ents[i], "version": v} for i, v in zip(perm, vers, strict=True)]
                for fd in (0, 1, -1, 2, 3, -2) if facts else (0,):
                    fs = [{**f, "version": int(f.get("version") or 1) + fd} for f in facts]
                    if any(f["version"] < 1 for f in fs):
                        continue
                    n += 1
                    if n > limit:
                        return
                    yield {**snap, "entities": es, "facts": fs}


def input_hash(snap: dict[str, Any]) -> str:
    """Stable across runs: ids do not count, only their pattern."""
    return hashlib.sha256(canonical(snap)[0].encode("utf-8")).hexdigest()[:32]


def _to_placeholders(out: Any, ids: dict[str, str]) -> Any:
    blob = json.dumps(out, ensure_ascii=False)
    blob = _ID.sub(lambda m: ids.get(m.group(0), m.group(0)), blob)
    return json.loads(blob)


def _from_placeholders(out: Any, ids: dict[str, str]) -> Any:
    back = {ph: i for i, ph in ids.items()}
    blob = json.dumps(out, ensure_ascii=False)
    return json.loads(_PH.sub(lambda m: back.get(m.group(0), m.group(0)), blob))


class CachedProcessor:
    """Wrap a `workers.extract` processor: same call, answers from disk when it has them."""

    def __init__(self, inner: Any, root: Path | None = None, *, prompt: str | None = None) -> None:
        self.inner = inner
        self.route = getattr(inner, "route", "cloud")
        self.model_id = str(getattr(inner, "model_id", "worker-extract"))
        self.root = Path(root) if root is not None else cache_root()
        self.prompt = prompt if prompt is not None else prompt_hash()
        self.hits = 0
        self.calls = 0  # model calls this processor really made

    def _dir(self) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.model_id)
        return self.root / safe

    def _path(self, snap: dict[str, Any]) -> Path:
        """The old key: the whole snapshot (bookkeeping included). Read as a fallback."""
        key = hashlib.sha256(
            f"{self.model_id}|{self.prompt}|{input_hash(snap)}".encode()
        ).hexdigest()
        return self._dir() / f"{key[:40]}.json"

    def view_path(self, snap: dict[str, Any]) -> Path:
        """The key: what the model reads (`canonical_view`)."""
        key = hashlib.sha256(
            f"{self.model_id}|{self.prompt}|{VIEW_VERSION}|{view_hash(snap)}".encode()
        ).hexdigest()
        return self._dir() / f"v-{key[:40]}.json"

    def lookup(self, snap: dict[str, Any], *, fuzzy: bool = False) -> dict[str, Any] | None:
        """The cached answer for `snap`, ids restored, or None. Order: the view key, the old
        whole-snapshot key, then (`fuzzy`, for a cache written before the view key) the old key
        of each bookkeeping variant - the same model input under other versions or id order."""
        tries: list[tuple[Path, dict[str, str]]] = [
            (self.view_path(snap), canonical_view(snap)[1]),
            (self._path(snap), canonical(snap)[1]),
        ]
        for path, ids in tries:
            out = self._read(path, ids)
            if out is not None:
                return out
        if not fuzzy:
            return None
        have = {p.name for p in self._dir().glob("*.json")} if self._dir().is_dir() else set()
        if not have:
            return None
        for v in bookkeeping_variants(snap):
            path = self._path(v)
            if path.name in have:
                # the variant's own placeholder map: its ids are ours, in its order
                return self._read(path, canonical(v)[1])
        return None

    @staticmethod
    def _read(path: Path, ids: dict[str, str]) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return _from_placeholders(json.loads(path.read_text(encoding="utf-8")), ids)
        except (OSError, ValueError):
            return None  # a torn cache file: ask again

    def available(self) -> bool:
        if self.root.exists() and any(self.root.rglob("*.json")):
            return True  # a warm cache answers without the model
        fn = getattr(self.inner, "available", None)
        try:
            return bool(fn()) if fn else True
        except Exception:  # noqa: BLE001
            return False

    def __call__(self, snap: dict[str, Any]) -> dict[str, Any]:
        out = self.lookup(snap)
        if out is not None:
            self.hits += 1
            return out
        path, ids = self.view_path(snap), canonical_view(snap)[1]
        from .cli_env import no_account

        with no_account():  # the extractor never sees the user's account e-mail
            out = self.inner(snap)
        self.calls += 1
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(_to_placeholders(out, ids), ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(path)
        except OSError:
            pass  # a read-only cache: the answer is still used
        return out


def make_processor(
    mode: str, *, root: Path | None = None, model: str | None = None
) -> CachedProcessor | None:
    """The cached processor for `--extract`; None for `off`. Never called from pytest with
    cloud/local (tests pass a fake through `ApiSUT(processor=...)`)."""
    if mode == "off":
        return None
    from ..workers import extract

    if mode == "cloud":
        return CachedProcessor(extract.CloudExtractor(model), root)
    if mode == "local":
        return CachedProcessor(extract.OllamaExtractor(), root)
    raise ValueError(f"--extract: one of {', '.join(MODES)}")
