"""Fixtures for the race suite: a migrated memory.db on disk (threads and child processes
need a real file, WAL), a fake clock, and connections opened per thread."""

from __future__ import annotations

from pathlib import Path

import pytest

from koob.memory import store
from koob.memory.clock import ManualClock

from ._kit import T0


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "memory" / "memory.db"
    path.parent.mkdir(parents=True)
    (tmp_path / "data" / "home").mkdir()
    store.open(path, home=tmp_path / "data" / "home").close()
    return path


@pytest.fixture
def conn(db_path: Path):
    c = store.open(db_path, home=db_path.parent.parent / "home")
    yield c
    c.close()


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(T0)
