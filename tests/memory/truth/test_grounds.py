"""G6: a conclusion travels with the facts it rests on. The core and the basket show
a decision with its premises; when a premise is corrected the decision shows as on hold with
the new value next to it (never as a current plan, never with the retired value); a newer
value on a premise's measure shows as «since then». Fictional people, own wordings."""

from __future__ import annotations

import pytest

from koob.memory import api, justify, ledger, store
from koob.memory.model import AuthorTier, Decision, Domain

from ..inspect._world import TS, add, fact, person, status, turn

GOSHA, LOFT = "ent-gosha", "ent-loft"
NUTS = "f-gosha-nuts"
CAFE = "d-romashka"
RENT, RENT2 = "f-loft-rent", "f-loft-rent2"
LEASE = "d-loft-lease"
LATER = "2026-10-14T09:00:00.000000Z"


@pytest.fixture
def db(mem):
    ledger.seed_relations(mem)
    ledger.seed_owner(mem)
    return mem


def _just(*members: str) -> list[dict]:
    return [{"group_no": 0, "member": m, "provenance": "stated"} for m in members]


def _decision(did: str, text: str, quote: str, at: str = TS) -> Decision:
    return Decision(
        id=did, author_tier=AuthorTier.USER, recorded_at=at, labels_final=True,
        domain=Domain.SOCIAL, chosen_text=text, decided_at=at, origin_quote=quote,
    )  # fmt: skip


def _cafe(conn) -> None:
    add(conn, person(GOSHA, "Гоша"))
    add(conn, fact(NUTS, GOSHA, "has_condition", "аллергия на арахис",
                   "Gosha has a peanut allergy.", "у Гоши аллергия на арахис"))  # fmt: skip
    add(conn, _decision(CAFE, "They will order the nut-free set at «Ромашка» for Gosha's party.",
                        "берём сет без орехов в «Ромашке» — из-за аллергии"),
        justifications=_just(NUTS))  # fmt: skip


def test_core_decision_names_its_premise(db):
    _cafe(db)
    core = api.core(thread_id="t-1", store=db).text
    line = next(ln for ln in core.splitlines() if "Ромашка" in ln)
    assert "decided:" in line and "because" in line and "peanut allergy" in line


def test_correction_puts_the_decision_on_hold_with_the_new_value(db):
    _cafe(db)
    words = "Исправь: у Гоши не аллергия на арахис, а на киви."
    res = api.fix(f"m:{NUTS}", words, turn=turn(words, n=2), because_quote=words, store=db)
    assert res.outcome == "committed", res
    assert status(db, CAFE) == "needs_review"
    core = api.core(thread_id="t-2", store=db).text
    lines = [ln for ln in core.splitlines() if "Ромашка" in ln]
    assert lines and all("on hold" in ln and "no longer holds" in ln for ln in lines), core
    assert not any("decided:" in ln and "on hold" not in ln for ln in lines)
    assert "peanut" not in core and "арахис" not in core


def test_corrected_premise_is_named_next_to_the_held_decision(db):
    """A premise outside the sensitive fence: the on-hold line names the corrected value, and
    the retired one is nowhere in the core."""
    _loft(db, newer=False)
    words = "Поправь: Лофт стоит не 8 тысяч, а 10 тысяч в месяц."
    res = api.fix(f"m:{RENT}", words, turn=turn(words, n=4), because_quote=words, store=db)
    assert res.outcome == "committed", res
    assert status(db, LEASE) == "needs_review"
    with store.write_tx(db):  # the label classifier finished: the new value may be named
        db.execute("UPDATE records SET labels_final = 1 WHERE id = ?", (res.record_ids[0],))
    core = api.core(thread_id="t-4", store=db).text
    line = next(ln for ln in core.splitlines() if "Лофт" in ln and "rents" in ln)
    assert "on hold" in line and "10 тысяч" in line, core
    assert "8 thousand" not in core and "8 тысяч" not in core


def _loft(conn, *, newer: bool = True) -> None:
    add(conn, person(LOFT, "студия «Лофт»"))
    add(conn, fact(RENT, LOFT, "other", "8 тысяч в месяц", "Loft is cheap, 8 thousand a month.",
                   "дёшево, 8 тысяч в месяц"))  # fmt: skip
    add(conn, _decision(LEASE, "She rents the studio «Лофт» for the year.",
                        "снимаю «Лофт» на год: дёшево"),
        justifications=_just(RENT))  # fmt: skip
    if newer:
        add(conn, fact(RENT2, LOFT, "other", "15 тысяч с марта",
                       "Loft will cost 15 thousand from March.", "с марта 15 тысяч",
                       recorded=LATER))  # fmt: skip


@pytest.mark.parametrize("words", ["Продлевать Лофт на следующий год?", "Keep renting Loft?"])
def test_basket_decision_shows_what_moved_under_it(db, words):
    """The user's own later price on the premise's measure replaces it (`ledger._same_measure`):
    the decision no longer holds, with the new value; the replaced one is never shown."""
    _loft(db)
    basket = api.recall(turn(words, n=3, thread="t-3"), store=db)
    item = next(i for i in basket.items if i.id == LEASE)
    assert "больше не держится" in item.text and "15 thousand" in item.text
    assert "8 thousand" not in item.text
    assert "проверить" in item.text


@pytest.mark.parametrize("words", ["Продлевать Лофт на следующий год?", "Keep renting Loft?"])
def test_basket_decision_shows_a_newer_value_someone_else_reports(db, words):
    """Someone else's newer price does not replace the user's: the premise holds, «с тех пор»
    names it."""
    _loft(db, newer=False)
    add(db, fact(RENT2, LOFT, "other", "15 тысяч с марта", "Loft will cost 15 thousand from March.",
                 "с марта 15 тысяч", tier=AuthorTier.OTHER, recorded=LATER),
        tier=AuthorTier.OTHER)  # fmt: skip
    basket = api.recall(turn(words, n=3, thread="t-3"), store=db)
    item = next(i for i in basket.items if i.id == LEASE)
    assert "основание" in item.text and "8 thousand" in item.text
    assert "с тех пор" in item.text and "15 thousand" in item.text
    assert "проверить" in item.text


def test_unmoved_premise_is_not_flagged(db):
    _loft(db, newer=False)
    g = justify.grounds(db, LEASE, lambda ids: set(ids))
    assert [t for _, t, _ in g.held] == ["Loft is cheap, 8 thousand a month."]
    assert not g.shaken


def test_forgotten_or_fenced_premise_is_never_named(db):
    _loft(db)
    g = justify.grounds(db, LEASE, lambda ids: set())  # the view's fence refuses all
    assert not g.held and not g.since
    with store.write_tx(db):
        db.execute("UPDATE records SET status = 'forgotten' WHERE id = ?", (RENT,))
    g = justify.grounds(db, LEASE, lambda ids: set(ids))
    assert g == justify.Grounds()
