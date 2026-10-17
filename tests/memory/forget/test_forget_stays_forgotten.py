"""A forget always leaves its mark and always wins: a paraphrased order still finds the fact
and the other records holding its value, no recall path serves a suppressed item, and a
restore after corruption re-applies every journaled forget (the core view checks the forget
marks for every section it renders). Uses a synthetic persona."""

from __future__ import annotations

import pytest

from koob.memory import api, backup, forget, store
from koob.memory.model import ForgetScope
from koob.memory.recall.core_view import compute, render

from . import _seed as s
from .test_forget_orders import _status, _turn

clock = s.clock  # fixture

_LOCKER = ("f-locker", "ep-locker", "in-locker")


def _core(conn) -> str:
    return render(compute(conn))


def _locker_world(mem) -> None:
    """The user gave a gym locker code in one message: a fact, an episode and an intention
    each hold the value; an unrelated fact about the user's sister stays."""
    s.relation(mem, "locker_code", "code")
    s.relation(mem, "lives_in", "residence")
    msg = "My new gym locker code is 4417, I'll change it after the trip to Lisbon."
    s.raw(mem, "raw-locker", msg)
    s.fact(
        mem, "f-locker", "self", "locker_code", value="4417",
        text_en="Denis's gym locker code is 4417", quote="My new gym locker code is 4417",
        raw="raw-locker", keywords_en="gym locker code", keywords_ru="шкафчик код спортзал",
    )  # fmt: skip
    s.record(
        mem, "ep-locker", "episode", tier="system", origin_quote="My new gym locker code is 4417",
        fts=("denis got a new gym locker with code 4417", "", ""),
    )  # fmt: skip
    mem.execute(
        "INSERT INTO episodes (id, summary_en) VALUES ('ep-locker', "
        "'Denis got a new gym locker with code 4417')"
    )
    s.edge(mem, "ep-locker", "raw-locker", "derived_from")
    s.record(
        mem, "in-locker", "intention", origin_quote="I'll change it after the trip to Lisbon",
        fts=("change the gym locker code 4417 after the lisbon trip", "", ""),
    )  # fmt: skip
    mem.execute(
        "INSERT INTO intentions (id, trigger, action) VALUES ('in-locker',"
        ' \'{"kind":"date_reached","ts":"2026-11-02T09:00"}\','
        ' \'{"text":"Change the gym locker code 4417 after the Lisbon trip"}\')'
    )
    s.edge(mem, "in-locker", "raw-locker", "derived_from")
    s.raw(mem, "raw-sis", "My sister Anna lives in Porto.")
    s.fact(
        mem, "f-sis", "self", "lives_in", value="Porto", text_en="Denis's sister Anna lives in Porto",
        quote="My sister Anna lives in Porto", raw="raw-sis", keywords_en="sister anna porto",
    )  # fmt: skip
    mem.commit()


@pytest.mark.parametrize(
    "words",
    [
        "Forget the number of my locker, I don't want it stored",
        "forget the digits for my locker please",
        "Забудь номер моего шкафчика",
    ],
)
def test_paraphrased_forget_reaches_every_record_holding_the_value(mem, clock, words):
    _locker_world(mem)
    res = api.forget(words, turn=_turn(words), because_quote=words, store=mem)
    assert res.outcome == "stopped", res.line
    for rid in _LOCKER:
        assert _status(mem, rid) == "forgotten", rid
    assert _status(mem, "f-sis") == "active"
    assert mem.execute("SELECT count(*) FROM suppressions").fetchone()[0] >= 1


def test_suppressed_copy_is_served_by_no_recall_lane(mem, clock):
    """A copy of a forgotten fact that is still active (a write that slipped past the
    commit check) is never served: the basket's hydrate and the core view drop it."""
    _locker_world(mem)
    fid = forget.stop_use(mem, ["f-locker"], scope=ForgetScope.ITEM, forever=False)
    assert fid
    s.fact(
        mem, "f-copy", "self", "locker_code", value="4417",
        text_en="Denis keeps code 4417 for the gym locker", raw="raw-locker",
    )  # fmt: skip
    mem.commit()
    from koob.memory.recall import basket

    rows = basket._hydrate(mem, ["f-copy", "f-sis"], "2026-10-17")
    assert "f-copy" not in rows and "f-sis" in rows
    assert "4417" not in _core(mem)


@pytest.fixture
def mem_paths(tmp_path):
    return store.MemoryPaths(tmp_path / "data" / "memory")


def _corrupt(path) -> None:
    data = bytearray(path.read_bytes())
    for i in range(4096, min(len(data), 4096 * 12)):
        data[i] = (i * 31) & 0xFF
    path.write_bytes(bytes(data))


def _people_world(conn) -> None:
    with store.write_tx(conn):
        s.relation(conn, "works_at", "employer")
        s.entity(conn, "e-kim", "Kimberly Vance", [("Kimberly Vance", "name"), ("Kim", "name")])
        s.raw(conn, "raw-kim", "Kimberly Vance works at Northwind Labs.")
        s.fact(
            conn, "f-kim", "e-kim", "works_at", value="Northwind Labs",
            text_en="Kimberly Vance works at Northwind Labs",
            quote="Kimberly Vance works at Northwind Labs", raw="raw-kim",
        )  # fmt: skip
        # the user mentioned her today: she is in the core's «People» section
        now = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
        conn.execute(f"UPDATE records SET recorded_at = {now}")  # noqa: S608
        conn.execute(f"UPDATE raw_chunks SET ts = {now}")  # noqa: S608


@pytest.mark.parametrize("purged", [False, True])
def test_forever_forget_wins_over_a_restore_after_corruption(mem_paths, tmp_path, clock, purged):
    home = tmp_path / "home"
    conn = store.open(mem_paths.db, home=home)
    _people_world(conn)
    assert "Northwind" in _core(conn)
    backup.make_set(conn, clock=clock)  # the nightly set, before the forget
    clock.advance(3600)
    fid = forget.stop_use(conn, ["e-kim"], scope=ForgetScope.ABOUT_ENTITY, forever=True)
    if purged:
        forget.purge(conn, fid)
    conn.close()
    clock.advance(3600)
    _corrupt(mem_paths.db)
    assert not backup.check_live(mem_paths.db)

    backup.recover(mem_paths.db, home=home)

    conn = store.open(mem_paths.db, home=home)
    try:
        for rid in ("e-kim", "f-kim"):
            assert _status(conn, rid) in ("forgotten", "purged"), rid
        text = _core(conn)
        assert "Kimberly" not in text and "Northwind" not in text
    finally:
        conn.close()


def _revive(conn) -> int:
    """A later restore step (the hot log replay, the sweep) writes the person back."""
    with store.write_tx(conn):
        conn.execute("UPDATE records SET status = 'active' WHERE id IN ('e-kim', 'f-kim')")
        conn.execute("UPDATE entities SET display_name = 'Kimberly Vance' WHERE id = 'e-kim'")
        conn.execute(
            "UPDATE facts SET text_en = 'Kimberly Vance works at Northwind Labs',"
            " value_text = 'Northwind Labs', value_norm = 'northwind labs' WHERE id = 'f-kim'"
        )
    return 2


def test_a_forget_wins_over_what_a_restore_step_writes_back(mem_paths, tmp_path, clock):
    home = tmp_path / "home"
    conn = store.open(mem_paths.db, home=home)
    _people_world(conn)
    the_set = backup.make_set(conn, clock=clock)
    clock.advance(3600)
    fid = forget.stop_use(conn, ["e-kim"], scope=ForgetScope.ABOUT_ENTITY, forever=True)
    forget.purge(conn, fid)
    conn.close()

    backup.restore_report(the_set, mem_paths.db, home=home, sweep=_revive)

    conn = store.open(mem_paths.db, home=home)
    try:
        for rid in ("e-kim", "f-kim"):
            assert _status(conn, rid) in ("forgotten", "purged"), rid
        assert "Northwind" not in _core(conn)
    finally:
        conn.close()


def test_core_view_checks_the_journal_for_every_section(mem_paths, tmp_path, clock):
    """Rows the journal holds forgotten that are active again never reach the core."""
    home = tmp_path / "home"
    conn = store.open(mem_paths.db, home=home)
    try:
        _people_world(conn)
        forget.stop_use(conn, ["e-kim"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
        _revive(conn)
        text = _core(conn)
        assert "Kimberly" not in text and "Northwind" not in text
    finally:
        conn.close()
