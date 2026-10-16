"""§8.1 justifications: well-founded support and independence by root origins.

Spec marker: `[T truth/test_support::<case>]`. Fictional people only."""

from __future__ import annotations

import pytest

from koob.memory import justify, ledger, store
from koob.memory.model import OWNER_ENTITY_ID, AuthorTier, OpKind, Proposal, ProposalOp

from ..inspect._world import add, fact, person, raw

DIMA, SEREGA, PETYA, LENA = "ent-dima", "ent-serega", "ent-petya", "ent-lena"
OTHER = AuthorTier.OTHER


@pytest.fixture
def db(mem):
    ledger.seed_relations(mem)
    return mem


def _link(conn, target: str, member: str, group: int, prov: str = "inferred") -> None:
    j = {"target": target, "group_no": group, "member": member, "provenance": prov}
    op = ProposalOp(op=OpKind.LINK, target=target, group=target, payload={"justification": j})
    assert store.commit(conn, Proposal(ops=[op], author_tier=AuthorTier.USER), ledger.apply_op).ok


def _just(*groups: list[str], prov: str = "rule") -> list[dict]:
    return [
        {"group_no": g, "member": m, "provenance": prov} for g, ms in enumerate(groups) for m in ms
    ]


def _people(conn) -> None:
    for eid, name in ((DIMA, "Дима"), (SEREGA, "Серёга"), (PETYA, "Петя"), (LENA, "Лена")):
        add(conn, person(eid, name))
    raw(conn, "r-dima", "Дима: уезжаю в Тверь в пятницу", tier=OTHER, author=DIMA, chat_id="tg:1")
    raw(conn, "r-ser", "Серёга: Дима говорит, в пятницу едет в Тверь", tier=OTHER,
        author=SEREGA, chat_id="tg:2")  # fmt: skip
    raw(conn, "r-pet", "Петя: Дима сказал, что переезжает в Тверь", tier=OTHER, author=PETYA,
        chat_id="tg:3")  # fmt: skip
    raw(conn, "r-lena", "Лена: Дима точно едет в Тверь, я помогаю с вещами", tier=OTHER,
        author=LENA, chat_id="tg:4")  # fmt: skip
    claims = (
        ("f-dima", DIMA, "r-dima", "уезжаю в Тверь в пятницу"),
        ("f-ser", DIMA, "r-ser", "в пятницу едет в Тверь"),  # relayed: DIMA is the speaker
        ("f-pet", DIMA, "r-pet", "переезжает в Тверь"),
        ("f-lena", LENA, "r-lena", "Дима точно едет в Тверь"),
        ("f-rumor", None, "r-ser", "Дима говорит, в пятницу едет"),  # no original speaker
    )
    for fid, speaker, rid, quote in claims:
        add(conn, fact(fid, DIMA, "other", "moving to Tver", "Dima is moving to Tver", quote,
                       tier=OTHER, speaker=speaker, raw=rid))  # fmt: skip


def test_inferred_cycle_unsupported(db):
    """§8.1: inferred cycles never keep themselves alive."""
    add(db, fact("f-a", OWNER_ENTITY_ID, "other", "Lena likes jazz", "Lena likes jazz", "Лена любит джаз"))  # fmt: skip
    add(db, fact("f-b", OWNER_ENTITY_ID, "other", "jazz bar gift", "Give Lena a jazz bar voucher", "подарю Лене джаз-бар"),
        justifications=_just(["f-a"], prov="inferred"))  # fmt: skip
    assert justify.supported(db, "f-b")  # grounded through the root f-a

    _link(db, "f-a", "f-b", 0)  # now each rests only on the other
    assert not justify.supported(db, "f-a")
    assert not justify.supported(db, "f-b")

    raw(db, "r-a", "Лена любит джаз, она сама сказала")
    _link(db, "f-a", "r-a", 1, prov="stated")  # a grounded way into the cycle
    assert justify.supported(db, "f-a") and justify.supported(db, "f-b")


def test_two_retellings_one_root(db):
    """§8.1: two retellings of one root are not independent."""
    _people(db)
    add(db, fact("c-help", OWNER_ENTITY_ID, "other", "help Dima pack", "Help Dima pack on Friday",
                 "помогу Диме собраться"),
        justifications=_just(["f-ser"], ["f-pet"], ["f-lena"]))  # fmt: skip
    assert justify.roots(db, "c-help", 0) == {f"entity:{DIMA}"}
    assert justify.roots(db, "c-help", 1) == {f"entity:{DIMA}"}
    assert not justify.independent(db, "c-help", (0, 1))
    assert justify.independent(db, "c-help", (0, 2))  # Lena's own word is a second source


def test_relay_shares_root_with_original(db):
    """§8.1: a relayed claim's root is its named original speaker; with none named
    ("they say that…") the relayer is the root."""
    _people(db)
    add(db, fact("c-farewell", OWNER_ENTITY_ID, "other", "farewell on Thursday",
                 "A farewell dinner for Dima on Thursday", "проводим Диму в четверг"),
        justifications=_just(["r-dima"], ["f-ser"], ["f-rumor"]))  # fmt: skip
    assert justify.roots(db, "c-farewell", 0) == {f"entity:{DIMA}"}  # his own message
    assert not justify.independent(db, "c-farewell", (0, 1))
    assert justify.roots(db, "c-farewell", 2) == {f"entity:{SEREGA}"}
    assert justify.independent(db, "c-farewell", (0, 2))


def test_his_own_statements_need_no_independence_test(db):
    """§8.1: user-tier statements need no independence test; the latest wins."""
    add(db, fact("f-m1", OWNER_ENTITY_ID, "other", "Monday call", "Call on Monday", "созвон в понедельник"))  # fmt: skip
    add(db, fact("f-m2", OWNER_ENTITY_ID, "other", "Monday 10am", "The call is at 10", "созвон в 10"))  # fmt: skip
    add(db, fact("c-call", OWNER_ENTITY_ID, "other", "block Monday 10", "Block Monday 10am", "занять утро понедельника"),
        justifications=_just(["f-m1"], ["f-m2"], prov="stated"))  # fmt: skip
    assert justify.roots(db, "c-call", 0) == {justify.OWNER_ROOT}
    assert justify.independent(db, "c-call", (0, 1))
    assert not justify.independent(db, "c-call", (0, 7))  # no such group
