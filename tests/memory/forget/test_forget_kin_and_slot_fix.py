"""A forget takes what was derived from the fact along with it, and a correction that only
names the new value retires the old one by slot, through the same `api` calls a conversation
makes. Uses a synthetic persona."""

from __future__ import annotations

import pytest

from koob.memory import api, episodes, forget
from koob.memory.model import AuthorTier, Episode, FKind
from koob.memory.recall import basket, core_view, vectors

from .._ledger import DayClock, add, entity, fact, open_seeded, status
from ..recall.test_raw_meaning_lane import ConceptEmbedder
from .test_forget_and_fix_leave_marks import _shown, said, t


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(ConceptEmbedder())
    basket._NO_CONTRAST.clear()
    basket._NO_REVISION.clear()
    forget.reset_wiring()
    forget.use_key(None)
    forget.wire(stop_hooks=[vectors.stop_hook], clock=DayClock())
    c = open_seeded(tmp_path)
    vectors.attach(tmp_path / "vectors.db")
    yield c
    c.close()
    forget.reset_wiring()
    vectors.set_embedder(None)


def _night_episodes(conn) -> list[str]:
    episodes.commit(conn, episodes.propose_episodes(conn, since="2026-01-01T00:00:00Z"))
    return [r[0] for r in conn.execute("SELECT id FROM records WHERE kind = 'episode'")]


@pytest.mark.parametrize(
    ("first", "words"),
    [
        ("Last weekend I took a trip to the coast, we stayed in a cabin by the dunes.",
         "Forget about my trip, I don't want it to come up again."),
        ("На выходных у меня была поездка на побережье, жили в домике у дюн.",
         "Забудь про мою поездку, не хочу о ней вспоминать."),
    ],
)  # fmt: skip
def test_a_described_forget_takes_the_fact_and_its_night_episode_without_asking(conn, first, words):
    rid = said(conn, 1, first)
    add(conn, fact("f-op", "self", "travels_to", "the coast", fkind=FKind.EVENT,
                   text="Denis took a trip to the coast and stayed in a cabin by the dunes",
                   source_raw_id=rid, quote_orig=first))  # fmt: skip
    eps = _night_episodes(conn)
    assert eps, "the night job built no episode"
    res = api.forget(words, turn=t(words), because_quote=words, store=conn)
    assert res.outcome == "stopped", res.line
    assert status(conn, "f-op") == "forgotten"
    assert all(status(conn, e) == "forgotten" for e in eps)


def test_a_forget_takes_a_written_fact_and_an_episode_built_from_its_message_by_another_fact(
    conn,
):
    rid = said(conn, 1, "Last weekend I took a trip to the coast, we stayed in a cabin.")
    add(conn, fact("f-hot", "self", "travels_to", "the coast", fkind=FKind.EVENT,
                   text="Denis took a trip to the coast"))  # fmt: skip
    add(conn, fact("f-seed", "self", "stayed_in", "a cabin", fkind=FKind.EVENT,
                   text="Denis stayed in a cabin", source_raw_id=rid, quote_orig="a cabin"))  # fmt: skip
    ep = Episode(id="ep-1", author_tier=AuthorTier.SYSTEM, recorded_at="2026-10-12T05:00:00Z",
                 event_type="trip", summary_en="Denis took a trip to the coast and a cabin")  # fmt: skip
    add(conn, ep, derived_from=["f-seed"])
    words = "Forget about my trip, I don't want it to come up again."
    res = api.forget(words, turn=t(words), because_quote=words, store=conn)
    assert res.outcome == "stopped", res.line
    assert status(conn, "f-hot") == "forgotten" and status(conn, "ep-1") == "forgotten"


def test_two_different_user_facts_tied_by_a_forget_still_ask(conn):
    r1 = said(conn, 1, "My knee surgery is on Monday at the clinic.")
    r2 = said(conn, 2, "My dental surgery is next Friday downtown.", "2026-10-02")
    add(conn, fact("f-a", "self", "surgery_plan", "Monday", text="Denis has surgery on Monday",
                   source_raw_id=r1, quote_orig="My knee surgery is on Monday"))  # fmt: skip
    add(conn, fact("f-b", "self", "surgery_plan", "next Friday",
                   text="Denis has surgery next Friday", source_raw_id=r2,
                   quote_orig="My dental surgery is next Friday"))  # fmt: skip
    words = "Forget about my surgery, it's private."
    res = api.forget(words, turn=t(words), because_quote=words, store=conn)
    assert res.outcome != "stopped", res.line
    assert status(conn, "f-a") == "active" and status(conn, "f-b") == "active"


def _marks(conn, rid: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT retracted_value_norm FROM raw_retractions WHERE raw_id = ?", (rid,)
        )
    ]


@pytest.mark.parametrize(
    ("first", "correction", "old", "new", "kept", "ask"),
    [
        ("Denis Smith's dentist appointment is on the 14th of November at 3 pm on Hill Road.",
         "The dentist got moved to the 21st, same time, 3 pm.", "14th", "21st", "3 pm",
         "when is the dentist appointment?"),
        ("Запись к стоматологу у Дениса Смита 14 ноября в 15:00 на Холмовой.",
         "Стоматолога перенесли на 21-е, в то же время, в 15:00.", "14 ноября", "21-е",
         "15:00", "когда стоматолог?"),
    ],
)  # fmt: skip
def test_a_correction_naming_only_the_new_day_retires_the_old_day_by_slot(
    conn, first, correction, old, new, kept, ask
):
    rid = said(conn, 1, first)
    api.fix(correction, correction, turn=t(correction, 2), because_quote=correction, store=conn)
    marks = _marks(conn, rid)
    assert marks, "no slot mark on his earlier message"
    assert not any(kept in m for m in marks), marks  # a value the correction repeats stays
    said(conn, 2, correction, "2026-10-02")
    shown = _shown(conn, ask)
    assert old.casefold() not in shown.casefold()
    assert new.casefold() in shown.casefold()


def test_a_correction_of_an_amount_retires_the_balance_computed_from_it(conn):
    rid = said(conn, 1, "I lent Max Rowe 500 dollars for the car repair, he paid back 200, "
                        "so he still owes me 300.")  # fmt: skip
    correction = "Make the car repair loan 600 dollars."
    api.fix(correction, correction, turn=t(correction, 2), because_quote=correction, store=conn)
    marks = " ".join(_marks(conn, rid))
    assert "500" in marks and "300" in marks and "200" not in marks, marks
    said(conn, 2, correction, "2026-10-02")
    shown = _shown(conn, "how much does Max owe me for the car repair?")
    assert "600" in shown
    assert "500" not in shown and "owes me 300" not in shown


def test_a_corrected_message_stays_served_once_the_correction_is_ingested(conn):
    rid = said(conn, 1, "The team offsite is in Leeds next month, I booked the hotel already.")
    correction = "Actually the team offsite is in York, not Leeds."
    api.fix(correction, correction, turn=t(correction, 2), because_quote=correction, store=conn)
    said(conn, 2, correction, "2026-10-02")
    rows = basket._hydrate(conn, [rid], "2026-10-12")
    assert rid in rows, "the corrected message was dropped by a second cut of the same value"
    assert "leeds" not in rows[rid]["text"].casefold()
    shown = _shown(conn, "where is the team offsite?")
    assert "york" in shown.casefold() and "leeds" not in shown.casefold()


def test_records_learned_later_from_a_corrected_message_lose_the_old_amount_and_balance(conn):
    rid = said(conn, 1, "I lent Max Rowe 500 dollars for the car repair, he paid back 200, "
                        "so he still owes me 300.")  # fmt: skip
    correction = "Make the car repair loan 600 dollars."
    api.fix(correction, correction, turn=t(correction, 2), because_quote=correction, store=conn)
    said(conn, 2, correction, "2026-10-02")
    add(conn, fact("f-loan", "self", "lent_amount", "500 dollars", source_raw_id=rid,
                   text="Denis lent Max Rowe 500 dollars for the car repair",
                   quote_orig="I lent Max Rowe 500 dollars"))  # fmt: skip
    add(conn, fact("f-owes", "self", "owed_balance", "300", source_raw_id=rid,
                   text="Max Rowe still owes Denis 300 for the car repair",
                   quote_orig="so he still owes me 300"))  # fmt: skip
    add(conn, fact("f-back", "self", "paid_back", "200", source_raw_id=rid,
                   text="Max Rowe paid back 200 for the car repair",
                   quote_orig="he paid back 200"))  # fmt: skip
    add(conn, fact("f-left", "self", "loan_balance", "300 dollars", tier=AuthorTier.SYSTEM,
                   text="Max Rowe has 300 dollars of the car repair loan left to repay"),
        derived_from=["f-loan", "f-back"])  # fmt: skip
    rows = basket._hydrate(conn, ["f-loan", "f-owes", "f-back", "f-left"], "2026-10-12")
    assert set(rows) == {"f-back"}


BRO = "ent-bro"


def _core(conn) -> str:
    return "\n".join(core_view.render([sec]) for sec in core_view.compute(conn))


def _debt(conn) -> None:
    rid = said(conn, 1, "Max Rowe's brother borrowed 40 thousand from me last spring and "
                        "still has not paid it back.")  # fmt: skip
    add(conn, entity("ent-max", "Max Rowe"))
    add(conn, entity(BRO, "Max Rowe's brother"))
    add(conn, fact("f-debt", BRO, "owes", "40 thousand since last spring", source_raw_id=rid,
                   text="Max Rowe's brother owes Denis a loan of 40 thousand since last spring",
                   quote_orig="borrowed 40 thousand from me last spring"))  # fmt: skip
    add(conn, fact("f-borrow", BRO, "borrowed_from", "Denis", source_raw_id=rid,
                   text="Max Rowe's brother borrowed money from Denis",
                   quote_orig="borrowed 40 thousand from me"))  # fmt: skip
    add(conn, fact("f-unpaid", BRO, "repaid", "not yet", source_raw_id=rid,
                   text="Max Rowe's brother has not paid Denis back",
                   quote_orig="still has not paid it back"))  # fmt: skip
    other = said(conn, 2, "Max Rowe moved to Porto in the summer.", "2026-10-02")
    add(conn, fact("f-max", "ent-max", "lives_in", "Porto", source_raw_id=other,
                   text="Max Rowe lives in Porto", quote_orig="moved to Porto"))  # fmt: skip


def test_a_forget_for_good_takes_the_same_messages_facts_and_hides_the_emptied_person(conn):
    _debt(conn)
    assert "brother" in _core(conn).casefold()
    words = "Forget the loan of 40 thousand for good"
    res = api.forget(words, turn=t(words), because_quote=words, forever=True, store=conn)
    assert res.outcome == "stopped", res.line
    assert {status(conn, f) for f in ("f-debt", "f-borrow", "f-unpaid")} == {"forgotten"}
    assert status(conn, "f-max") == "active"
    assert forget.hidden_by_forget(conn, [BRO, "ent-max"]) == {BRO}
    assert "brother" not in _core(conn).casefold()
    # the user can name the person anew: something new about them shows them again
    add(conn, fact("f-new", BRO, "works_at", "a bakery",
                   text="Max Rowe's brother works at a bakery"))  # fmt: skip
    assert forget.hidden_by_forget(conn, [BRO]) == set()


def test_a_plain_forget_keeps_the_same_messages_other_facts(conn):
    _debt(conn)
    words = "Forget the loan of 40 thousand"
    res = api.forget(words, turn=t(words), because_quote=words, store=conn)
    assert res.outcome == "stopped", res.line
    assert status(conn, "f-debt") == "forgotten"
    assert status(conn, "f-unpaid") == "active"
    assert forget.hidden_by_forget(conn, [BRO]) == set()
