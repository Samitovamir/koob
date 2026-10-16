""""Where is this from" (§17, MCP `where_from` §10.4): lineage in plain words - quote, chat,
date, who said it - and nothing of an item that is forgotten or not for this turn."""

from __future__ import annotations

from koob.memory import api, commands

from ._world import HEALTH, LIVES_DIMA, MOVE, RAW_DIMA, RAW_MOVE, turn, world


def test_where_from_his_own_words(mem):
    world(mem)
    lin = api.where_from(f"m:{MOVE}", turn=turn("откуда ты это знаешь"), store=mem)
    assert lin.item_id == MOVE
    text = "\n".join(lin.lines)
    assert "«Саша переезжает в Тверь в ноябре»" in text
    assert "ты сам" in text
    assert "12 октября 2026" in text
    assert "разговор с Кубом" in text
    assert RAW_MOVE in {r.id for r in lin.sources}


def test_where_from_another_persons_claim(mem):
    world(mem)
    lines = commands.where_from(mem, LIVES_DIMA, turn("откуда это про Пермь"))
    text = "\n".join(lines)
    assert "Дима" in text
    assert "Telegram" in text
    assert "3 октября 2026" in text
    assert "«Саша вообще-то живёт в Перми»" in text


def test_where_from_forgotten_item_shows_no_content(mem):
    world(mem)
    api.forget(f"m:{MOVE}", turn=turn("забудь переезд"), because_quote="переезд", store=mem)
    text = "\n".join(commands.where_from(mem, f"m:{MOVE}", turn("откуда это", n=2)))
    assert "забыт" in text
    assert "Тверь" not in text and "Tver" not in text


def test_where_from_unknown_and_sensitive(mem):
    world(mem)
    assert commands.where_from(mem, "m:nope", turn("откуда это")) == ["Такой записи нет."]
    text = "\n".join(commands.where_from(mem, HEALTH, turn("откуда это", n=2)))
    assert "диабет" not in text and RAW_DIMA not in text
