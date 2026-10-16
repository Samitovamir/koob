"""Opening the memory store (the memory spec §3.1)."""

from __future__ import annotations

import pytest

from koob.memory import store


def test_refuses_old_sqlite(tmp_path, monkeypatch):
    """§3.1: [T storage/test_open::refuses_old_sqlite]."""
    monkeypatch.setattr(store, "sqlite_version", lambda: (3, 50, 2))
    with pytest.raises(store.SQLiteTooOld, match="3.51.3"):
        store.open(tmp_path / "memory.db", home=tmp_path / "home")
    assert not (tmp_path / "memory.db").exists()
    monkeypatch.setattr(store, "sqlite_version", lambda: (3, 51, 3))
    store.open(tmp_path / "memory.db", home=tmp_path / "home").close()


def test_bundled_sqlite_is_new_enough():
    """The pinned interpreter (`.python-version`) links SQLite >= 3.51.3 (§3.1, §20)."""
    assert store.sqlite_version() >= store.MIN_SQLITE


def test_refuses_synced_path(tmp_path):
    """§3.1: [T storage/test_open::refuses_synced_path]."""
    home = tmp_path / "home"
    icloud = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    icloud.mkdir(parents=True)
    with pytest.raises(store.SyncedPathRefused):
        store.open(icloud / "memory" / "memory.db", home=home)
    # Desktop and Documents are fine while iCloud Desktop & Documents is off ...
    store.open(home / "Documents" / "m1.db", home=home).close()
    # ... and refused once it is on.
    (icloud / "Documents").mkdir()
    with pytest.raises(store.SyncedPathRefused):
        store.open(home / "Documents" / "m2.db", home=home)
    with pytest.raises(store.SyncedPathRefused):
        store.open(home / "Desktop" / "m3.db", home=home)
    store.open(home / "dev" / "data" / "memory.db", home=home).close()


def test_pragmas_on_every_connection(tmp_path):
    """§3.1: [T storage/test_open::pragmas_on_every_connection]."""
    path = tmp_path / "memory.db"
    store.open(path, home=tmp_path / "home").close()
    for conn in (store.connect(path), store.connect(path)):
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # INCREMENTAL
        conn.close()
    hot = store.connect(tmp_path / "hotlog.db", journal_mode="DELETE")
    assert hot.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert hot.execute("PRAGMA secure_delete").fetchone()[0] == 1
    hot.close()
