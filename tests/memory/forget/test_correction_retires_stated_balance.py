"""A correction that contrasts an amount and states the new balance also retires the old balance
from the user's earlier message, even when the balance's other operand was never said. The test
goes through the same `api` calls a conversation makes."""

from __future__ import annotations

import pytest

from koob.memory import api, forget
from koob.memory.recall import basket, vectors

from .._ledger import open_seeded
from ..recall.test_raw_meaning_lane import ConceptEmbedder
from .test_forget_and_fix_leave_marks import _shown, said, t
from .test_forget_kin_and_slot_fix import _marks


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(ConceptEmbedder())
    basket._NO_CONTRAST.clear()
    basket._NO_REVISION.clear()
    forget.reset_wiring()
    forget.use_key(None)
    forget.wire(stop_hooks=[vectors.stop_hook])
    c = open_seeded(tmp_path)
    vectors.attach(tmp_path / "vectors.db")
    yield c
    c.close()
    forget.reset_wiring()
    vectors.set_embedder(None)


@pytest.mark.parametrize(
    ("first", "correction", "old", "new", "ask"),
    [
        ("Denis Smith paid the plumber 400 dollars for the boiler on Tuesday, so the house fund "
         "is down to 1600 dollars.",
         "Actually the plumber was 450 dollars, not 400, so the house fund is at 1550 dollars.",
         ("400", "1600"), ("450", "1550"), "how much is left in the house fund?"),
        ("Denis Smith paid the plumber $400 for the boiler on Tuesday, so the house fund is "
         "down to $1,600.",
         "Actually the plumber was $450, not $400, so the house fund is at $1,550.",
         ("$400", "1,600"), ("450", "1,550"), "how much is left in the house fund?"),
        ("Денис Смит заплатил сантехнику 400 долларов за котёл во вторник, в фонде на дом "
         "осталось 1600 долларов.",
         "Сантехник обошёлся не в 400, а в 450 долларов, в фонде на дом осталось 1550 долларов.",
         ("400", "1600"), ("450", "1550"), "сколько осталось в фонде на дом?"),
    ],
)  # fmt: skip
def test_a_contrasted_amount_with_a_new_balance_retires_the_old_balance(
    conn, first, correction, old, new, ask
):
    rid = said(conn, 1, first)
    api.fix(correction, correction, turn=t(correction, 2), because_quote=correction, store=conn)
    marks = " ".join(_marks(conn, rid))
    assert old[0].lstrip("$") in marks and old[1] in marks, marks
    assert "tuesday" not in marks.casefold() and "вторник" not in marks, marks
    said(conn, 2, correction, "2026-10-02")
    shown = _shown(conn, ask)
    assert new[1] in shown
    assert old[1] not in shown, shown


def test_a_balance_the_correction_repeats_unchanged_is_not_retired(conn):
    rid = said(conn, 1, "Denis Smith paid the plumber 400 dollars for the boiler, and the house "
                        "fund holds 1600 dollars.")  # fmt: skip
    correction = "Actually the plumber was 450 dollars, not 400; the house fund still holds 1600."
    api.fix(correction, correction, turn=t(correction, 2), because_quote=correction, store=conn)
    marks = _marks(conn, rid)
    assert any("400" in m for m in marks), marks
    assert not any("1600" in m for m in marks), marks


def test_an_unrelated_new_amount_beside_a_contrast_retires_nothing_else(conn):
    rid = said(conn, 1, "Denis Smith paid the plumber 400 dollars for the boiler and 300 for the "
                        "new pipes, so the house fund is down to 1600 dollars.")  # fmt: skip
    correction = "Actually the plumber was 450 dollars, not 400, and I tipped him 20."
    api.fix(correction, correction, turn=t(correction, 2), because_quote=correction, store=conn)
    marks = " ".join(_marks(conn, rid))
    assert "400" in marks, marks
    assert "300" not in marks and "1600" not in marks, marks
