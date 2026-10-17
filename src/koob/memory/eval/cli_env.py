"""Eval-only: model calls through the claude CLI without the user's account metadata.

On a subscription the CLI adds the signed-in account's e-mail to every session's context (it
reads it from the `oauthAccount` entry of its global config; no flag turns it off). In an exam
that is contamination: a persona's question («какая у меня почта?») gets the user's real
address instead of what memory holds. Inside `no_account()` a call runs with an empty
throwaway config dir (its account entry has no e-mail) while the login stays the default one
(`CLAUDE_SECURESTORAGE_CONFIG_DIR=""` keeps the usual keychain item). Only the eval's own
call sites use it (compare brain and judge, the extraction cache, Graphiti's LLM); the
product's `llm` runner is untouched.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

KEYS = ("CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR")
# Part of the compare answer cache key: answers made with the e-mail in context never count.
MODE = "no-email"
PERSONAL = ("emailAddress", "displayName")
_DIR: list[Path] = []


def _owner_config() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(base) / ".claude.json" if base else Path.home() / ".claude.json"


def _strip(data: dict) -> bool:
    acct = data.get("oauthAccount")
    if not isinstance(acct, dict) or not any(k in acct for k in PERSONAL):
        return False
    data["oauthAccount"] = {k: v for k, v in acct.items() if k not in PERSONAL}
    return True


def config_dir() -> Path:
    """The throwaway config dir (one per process). It starts with the account entry minus the
    e-mail and name (the CLI fetches the whole profile again when the entry is missing), and
    an e-mail the CLI wrote back is dropped again before the next call."""
    if not _DIR:
        d = Path(tempfile.mkdtemp(prefix="cb-eval-cli-"))
        atexit.register(shutil.rmtree, d, True)  # the account ids never outlive the process
        try:
            acct = json.loads(_owner_config().read_text(encoding="utf-8")).get("oauthAccount")
        except (OSError, ValueError, AttributeError):
            acct = None
        if isinstance(acct, dict):
            seed = {"oauthAccount": acct}
            _strip(seed)
            (d / ".claude.json").write_text(json.dumps(seed), encoding="utf-8")
        _DIR.append(d)
    d = _DIR[0]
    cfg = d / ".claude.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else None
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict) and _strip(data):
        cfg.write_text(json.dumps(data), encoding="utf-8")
    return d


@contextlib.contextmanager
def no_account() -> Iterator[Path]:
    old = {k: os.environ.get(k) for k in KEYS}
    d = config_dir()
    os.environ["CLAUDE_CONFIG_DIR"] = str(d)
    os.environ["CLAUDE_SECURESTORAGE_CONFIG_DIR"] = ""
    try:
        yield d
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def has_email(d: Path) -> bool:
    """Whether the throwaway config carries the account e-mail (never prints it)."""
    try:
        acct = json.loads((d / ".claude.json").read_text(encoding="utf-8")).get("oauthAccount")
        return isinstance(acct, dict) and bool(acct.get("emailAddress"))
    except (OSError, ValueError, AttributeError):
        return False
