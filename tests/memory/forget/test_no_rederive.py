"""Acceptance for the memory spec §9.6: raw follows its records; suppressions block
re-derivation. Spec marker: `[T forget/test_no_rederive::<case>]`."""

from __future__ import annotations

from koob.memory import forget
from koob.memory.model import ForgetScope

from . import _seed as s

clock = s.clock  # fixture

RAW = "Илья переезжает в Берлин, а ещё он купил собаку"


def _scenario_a(mem) -> None:
    s.relation(mem, "moves_to", "residence")
    s.relation(mem, "owns", "possession", "multi")
    s.entity(mem, "ent-ilya", "Илья", [("Илья", "name"), ("Ilya", "translit")])
    s.raw(mem, "raw-r", RAW, turn="3")
    s.fact(
        mem,
        "f-move",
        "ent-ilya",
        "moves_to",
        value="Berlin",
        text_en="Ilya is moving to Berlin",
        quote="Илья переезжает в Берлин",
        raw="raw-r",
    )
    s.fact(
        mem,
        "f-dog",
        "ent-ilya",
        "owns",
        value="dog",
        text_en="Ilya bought a dog",
        quote="он купил собаку",
        raw="raw-r",
    )
    s.episode(
        mem,
        "ep-1",
        "Ilya is moving to Berlin and bought a dog",
        links=[("about", "ent-ilya")],
        derived_from=["f-move", "f-dog", "raw-r"],
    )
    s.derived(mem, "d-day", "Oct 1: Ilya moving to Berlin", [("f-move", 1), ("raw-r", 1)])


def _usable_text(mem) -> str:
    """Everything a rebuild may read: usable kind rows, active raw, valid derived."""
    parts = []
    for table in ("facts", "episodes"):
        for row in mem.execute(
            f"SELECT k.* FROM {table} k JOIN records r ON r.id = k.id "
            "WHERE r.status IN ('active', 'archived')"
        ):
            parts += [str(v) for v in tuple(row) if isinstance(v, str)]
    parts += [
        r[0] or "" for r in mem.execute("SELECT text FROM raw_chunks WHERE status = 'active'")
    ]
    parts += [r[0] for r in mem.execute("SELECT content FROM derived WHERE valid = 1")]
    return " ".join(parts)


def test_episode_rebuild_keeps_forgotten_out(mem, clock):
    """§9.6 (red-team scenario A): a request to forget that a person is moving. The raw message stays
    for the dog fact with the span redacted; the episode and the summary built from the move are
    out of use; a rebuild from old raw or a lower tier cannot re-create the proposition; the
    user's new statement after the forget is admitted."""
    _scenario_a(mem)
    fid = forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)

    status = dict(mem.execute("SELECT id, status FROM records").fetchall())
    assert status["f-move"] == "forgotten" and status["ep-1"] == "forgotten"
    assert status["f-dog"] == "active" and status["ent-ilya"] == "active"
    text, rev, raw_status = mem.execute(
        "SELECT text, rev, status FROM raw_chunks WHERE id = 'raw-r'"
    ).fetchone()
    assert raw_status == "active" and rev == 2
    assert text == "[забыто], а ещё он купил собаку"
    assert mem.execute("SELECT valid FROM derived WHERE id = 'd-day'").fetchone()[0] == 0
    usable = _usable_text(mem).casefold()
    assert "berlin" not in usable and "берлин" not in usable and "переезжа" not in usable

    old = dict(subject_id="ent-ilya", relation="moves_to", value_norm="berlin")
    assert forget.blocked_by(mem, source_ts=s.OLD, author_tier="system", **old) == fid
    assert forget.blocked_by(mem, source_ts=s.OLD, author_tier="user", **old) == fid
    clock.advance(60)
    later = s.now_iso(clock)
    assert forget.blocked_by(mem, source_ts=later, author_tier="tool", **old) == fid
    assert forget.blocked_by(mem, source_ts=later, author_tier="user", **old) is None
    other = dict(subject_id="ent-ilya", relation="moves_to", value_norm="paris")
    assert forget.blocked_by(mem, source_ts=s.OLD, author_tier="system", **other) is None

    forget.undo(mem, fid)
    assert mem.execute("SELECT text FROM raw_chunks WHERE id = 'raw-r'").fetchone()[0] == RAW
    assert forget.blocked_by(mem, source_ts=s.OLD, author_tier="system", **old) is None


def test_raw_forgotten_when_nothing_else_uses_it(mem, clock):
    """§9.6: raw that supports no other usable record is forgotten whole."""
    s.relation(mem, "moves_to", "residence")
    s.entity(mem, "ent-ilya", "Илья", [("Илья", "name")])
    s.raw(mem, "raw-1", "Илья переезжает в Берлин", turn="3")
    s.fact(
        mem,
        "f-move",
        "ent-ilya",
        "moves_to",
        value="Berlin",
        text_en="Ilya is moving to Berlin",
        quote="Илья переезжает в Берлин",
        raw="raw-1",
    )
    forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)
    assert tuple(
        mem.execute("SELECT status, text FROM raw_chunks WHERE id = 'raw-1'").fetchone()
    ) == (
        "forgotten",
        None,
    )


def test_unlocatable_quote_forgets_the_raw(mem, clock):
    """§9.6: when the quote does not locate a span, the shared raw is forgotten whole; the
    other record keeps its own quote as evidence."""
    _scenario_a(mem)
    mem.execute("UPDATE facts SET quote_orig = 'Илья уезжает' WHERE id = 'f-move'")
    forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)
    assert mem.execute("SELECT status FROM raw_chunks WHERE id = 'raw-r'").fetchone()[0] == (
        "forgotten"
    )
    assert mem.execute("SELECT quote_orig FROM facts WHERE id = 'f-dog'").fetchone()[0] == (
        "он купил собаку"
    )
