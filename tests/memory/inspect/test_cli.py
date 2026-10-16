"""`cb memory what|where-from|forget|undo|purge-status|run-purges` (§17: the commands also
work through `cb memory …`), over the api and the default store."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from koob import cli, paths
from koob.memory import forget, store

from ._world import DIMA_WORK, MOVE, status, world


@pytest.fixture
def mem_default(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    # the koob.db side is the runtime's wiring; this test has no koob.db
    monkeypatch.setattr(cli, "_memory_conversation", lambda: None)
    conn = store.open(store.MemoryPaths.default().db, home=tmp_path / "home")
    world(conn)
    yield conn
    conn.close()
    forget.reset_wiring()


def _cb(*args: str):
    res = CliRunner().invoke(cli.app, ["memory", *args])
    assert res.exit_code == 0, res.output
    return res.output


def test_what_and_where_from(mem_default):
    out = _cb("what", "Саша")
    assert "Про" in out and f"m:{MOVE}" in out
    out = _cb("where-from", f"m:{MOVE}")
    assert "«Саша переезжает в Тверь в ноябре»" in out


def test_forget_undo_status_and_purges(mem_default):
    out = _cb("forget", f"m:{DIMA_WORK}")
    assert "Перестал использовать" in out
    fid = re.search(r"\b(fg[0-9a-f]{10,})\b", out).group(1)
    assert status(mem_default, DIMA_WORK) == "forgotten"
    out = _cb("purge-status", fid)
    assert "в корзине" in out.casefold() and "записи" in out and "ledger" not in out
    assert "Нечего" in _cb("run-purges")
    assert "Вернул" in _cb("undo", fid)
    assert status(mem_default, DIMA_WORK) == "active"

    out = _cb("forget", "всё", "про", "Диму", "--forever")
    assert "удалил полностью" in out.casefold()  # the CLI runs the due purge after the call
    assert status(mem_default, DIMA_WORK) in ("forgotten", "purged")
