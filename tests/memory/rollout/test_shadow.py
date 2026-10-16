"""the memory spec §20 rollout step 1: in shadow mode the legacy
`memory_write` / `memory_forget` mirror into the ledger, so fixes and forgets made before the
cutover reach it. Spec marker: `[T rollout/test_shadow::<case>]`."""

from __future__ import annotations

import pytest

from koob import paths
from koob.assistant import tools
from koob.memory import api, store
from koob.memory.model import TurnCtx, TurnOrigin

from ..forget._seed import career  # noqa: F401 - fixture
from ..inspect._world import DIMA_WORK, MOVE, status, turn, world


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """The default memory.db (what shadow_mirror opens) in tmp_path, with the world."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    conn = store.open(store.MemoryPaths.default().db, home=tmp_path / "home")
    world(conn)
    yield conn
    conn.close()


def _texts(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT text_en FROM facts")]


def test_forget_reaches_ledger(ledger):
    """§20: in shadow mode a legacy forget still reaches the ledger."""
    api.shadow_mirror("forget", {"slug": "sasha-move"}, turn=turn("забудь про переезд Саши"))
    assert status(ledger, MOVE) == "forgotten"
    assert status(ledger, DIMA_WORK) == "active"


def test_a_forget_by_slug_forgets_only_that_notes_record(ledger):
    """I-8: «убери старую заметку про Сашу …» makes legacy forget one note (its slug); the
    ledger forgets only the record imported from that note, not everything about that person."""
    from ..inspect._world import LIVES_OWN, raw

    raw(ledger, "raw-import-move", "Саша переезжает в Тверь", source_kind="import:obsidian")
    ledger.execute(
        "UPDATE raw_chunks SET source_ref = 'obsidian:sasha-move' WHERE id = 'raw-import-move'"
    )
    ledger.execute("UPDATE facts SET source_raw_id = 'raw-import-move' WHERE id = ?", (MOVE,))
    ledger.commit()
    assert api._slug_records(ledger, "sasha-move") == [MOVE]
    assert api._slug_records(ledger, "") == [] and api._slug_records(ledger, "a/b") == []
    words = "убери всё про Сашу из старой заметки и запиши новую"
    api.shadow_mirror("forget", {"slug": "sasha-move"}, turn=turn(words))
    assert status(ledger, MOVE) == "forgotten"
    assert status(ledger, LIVES_OWN) == "active" and status(ledger, DIMA_WORK) == "active"


def test_a_forget_without_his_words_or_from_a_page_changes_nothing(ledger):
    duty = TurnCtx(turn_id="duty-1", thread_id="thread-duty", origin=TurnOrigin.DUTY)
    api.shadow_mirror("forget", {"slug": "dima-work"}, turn=duty)
    page = turn("открой эту страницу", tainted=True)
    api.shadow_mirror("forget", {"slug": "dima-work"}, turn=page)
    assert status(ledger, DIMA_WORK) == "active"


def test_a_legacy_write_is_remembered_and_a_correction_is_a_fix(ledger):
    api.shadow_mirror(
        "write",
        {"slug": "coffee", "title": "Coffee", "body": "Denis drinks coffee without sugar",
         "type": "preference", "source": "owner"},
        turn=turn("запомни: я пью кофе без сахара"),
    )  # fmt: skip
    assert any("coffee without sugar" in t for t in _texts(ledger))
    api.shadow_mirror(
        "write",
        {"slug": "dima", "title": "Дима", "body": "Дима работает в Сбере", "type": "user"},
        turn=turn("нет, Дима работает не в Яндексе, а в Сбере", n=2),
    )
    assert status(ledger, DIMA_WORK) == "superseded"


def test_a_note_from_a_page_is_the_pages_claim(ledger):
    api.shadow_mirror(
        "write",
        {"slug": "pavel", "title": "Pavel", "body": "Denis owes Pavel 500 dollars",
         "type": "user", "source": "web:example.org"},
        turn=turn("открой эту страницу"),
    )  # fmt: skip
    row = ledger.execute(
        "SELECT r.author_tier FROM records r JOIN facts f ON f.id = r.id"
        " WHERE f.text_en LIKE '%Pavel%'"
    ).fetchone()
    assert row is not None and row[0] == "tool"


def test_the_mirror_never_breaks_the_legacy_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "file-not-dir")
    (tmp_path / "file-not-dir").write_text("x")
    api.shadow_mirror("forget", {"slug": "x"}, turn=turn("забудь про Сашу"))  # no raise


def test_the_legacy_tools_mirror_only_in_shadow(ledger, career, monkeypatch):  # noqa: F811
    monkeypatch.setattr(tools.memory, "forget", lambda conn, slug, turn_id: {"ok": True})
    tools.watch_turn_words(4343, lambda: "забудь, где работает Дима")
    try:
        monkeypatch.setattr(tools, "memory_shadow", lambda: "off")
        tools.memory_forget(career, tools.SlugIn(slug="dima-work"), turn_id=4343)
        assert status(ledger, DIMA_WORK) == "active"
        monkeypatch.setattr(tools, "memory_shadow", lambda: "on")
        tools.memory_forget(career, tools.SlugIn(slug="dima-work"), turn_id=4343)
    finally:
        tools.watch_turn_words(4343, None)
    assert status(ledger, DIMA_WORK) == "forgotten"


def _turn_row(conn, tid: int, origin: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO koob_threads(id, created_at, updated_at) VALUES (9, 'x', 'x')"
    )
    conn.execute(
        "INSERT INTO koob_turns(id, thread_id, started_at, origin) VALUES (?, 9, 'x', ?)",
        (tid, origin),
    )
    conn.commit()


def test_a_duty_note_is_never_mirrored_as_his_word(ledger, career, monkeypatch):  # noqa: F811
    """I-7: the shadow mirror reads the turn's real origin; a duty's own summary note is the
    Cube's (system), never `user`, and the user's own turn's note stays the user's."""
    monkeypatch.setattr(tools, "memory_shadow", lambda: "on")
    _turn_row(career, 5151, "duty")
    _turn_row(career, 5152, "owner")
    assert tools._ledger_turn(career, 5151).origin is TurnOrigin.DUTY
    assert tools._ledger_turn(career, 5152).origin is TurnOrigin.OWNER
    assert tools._ledger_turn(career, 777).origin is TurnOrigin.TASK  # no row, no words
    assert tools._ledger_turn(career, None).origin is TurnOrigin.OWNER  # a direct call
    note = {"title": "Tea", "type": "preference", "source": "owner"}
    tools._shadow_mirror(
        "write", {**note, "slug": "tea", "body": "Denis drinks green tea"}, career, 5151, {}
    )
    tools.watch_turn_words(5152, lambda: "запомни: я пью кофе по утрам")
    try:
        tools._shadow_mirror(
            "write", {**note, "slug": "cof", "body": "Denis drinks coffee in the morning"},
            career, 5152, {},
        )  # fmt: skip
    finally:
        tools.watch_turn_words(5152, None)
    tiers = dict(
        ledger.execute(
            "SELECT f.text_en, r.author_tier FROM records r JOIN facts f ON f.id = r.id"
            " WHERE f.text_en LIKE '%tea%' OR f.text_en LIKE '%coffee%'"
        ).fetchall()
    )
    assert any("tea" in x for x in tiers), tiers
    assert all(t != "user" for x, t in tiers.items() if "tea" in x), tiers
    assert any(t == "user" for x, t in tiers.items() if "coffee" in x), tiers
