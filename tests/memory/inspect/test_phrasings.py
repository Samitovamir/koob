"""G8 on user-like phrasings (§1.2): the commands take the user's whole sentence, not a
cleaned-up name - inflections, diminutives, Latin spelling, «всё про», words around the
target. Fictional people only."""

from __future__ import annotations

import pytest

from koob.memory import api

from ._world import DIMA_WORK, MOVE, TIRED, status, turn, world


@pytest.mark.parametrize(
    "words",
    [
        "что ты помнишь про Сашу",
        "что знаешь о Саше",
        "расскажи, что помнишь про Сашу?",
        "what do you remember about Sasha",
        "Саша",
    ],
)
def test_what_phrasings(mem, words):
    world(mem)
    ids = {i.id for i in api.what(words, turn=turn(words), store=mem).items}
    assert {MOVE, TIRED} <= ids
    assert DIMA_WORK not in ids


@pytest.mark.parametrize(
    "words",
    [
        "забудь про переезд Саши",
        "забудь, что Саша переезжает",
        "удали из памяти переезд Саши",
        "забудь пожалуйста про её переезд",
    ],
)
def test_forget_one_item_phrasings(mem, words):
    world(mem)
    res = api.forget(words, turn=turn(words), because_quote=words, store=mem)
    assert res.outcome == "stopped", res
    assert res.record_ids == [MOVE]
    assert status(mem, TIRED) == "active"


@pytest.mark.parametrize(
    "words",
    [
        "забудь всё про Диму",
        "сотри всё о Дмитрии",
        "forget everything about Dima",
        "забудь Диму",
    ],
)
def test_forget_everything_about_phrasings(mem, words):
    world(mem)
    res = api.forget(words, turn=turn(words), because_quote=words, store=mem)
    assert res.outcome == "stopped", res
    assert status(mem, DIMA_WORK) == "forgotten"
    assert status(mem, MOVE) == "active"
