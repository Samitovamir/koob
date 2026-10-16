"""«что ты помнишь про X» (§17): usable records by kind, with date and source; impressions
marked «впечатление»; competing claims attributed; sensitive items per §5.5; every item the
brain gets writes a `served` row (§10.4)."""

from __future__ import annotations

from koob.memory import api, commands

from ._world import (
    DIMA_WORK,
    HEALTH,
    LIVES_DIMA,
    LIVES_OWN,
    MOVE,
    SASHA,
    TIRED,
    second_sasha,
    turn,
    world,
)


def _ids(answer) -> set[str]:
    return {i.id for i in answer.items} | {i.id for i in answer.competing_claims}


def test_what_lists_usable_records_with_handles_dates_and_sources(mem):
    world(mem)
    answer = api.what("Саша", turn=turn("что ты помнишь про Сашу"), store=mem)
    assert {MOVE, TIRED} <= {i.id for i in answer.items}
    assert DIMA_WORK not in _ids(answer)
    move = next(i for i in answer.items if i.id == MOVE)
    assert move.date == "2026-10-12" and move.author_tier == "user"

    lines = commands.what(mem, "Сашу", turn("что ты помнишь про Сашу", n=2))
    text = "\n".join(lines)
    assert text.startswith("Про Сашу помню") or text.startswith("Про «Саша» помню")
    assert f"[m:{MOVE}]" in text
    assert "впечатление: «кажется, Саша устала от работы»" in text


def test_what_attributes_competing_claims(mem):
    world(mem)
    answer = api.what("Саша", turn=turn("что ты помнишь про Сашу"), store=mem)
    assert {i.id for i in answer.competing_claims} == {LIVES_OWN, LIVES_DIMA}
    dima = next(i for i in answer.competing_claims if i.id == LIVES_DIMA)
    assert dima.said_by == "Дима"
    text = "\n".join(commands.what(mem, "Саша", turn("что ты помнишь про Сашу", n=2)))
    assert "Расходится" in text
    assert "по словам Дима" in text or "по словам Димы" in text
    assert "по твоим словам" in text


def test_others_sensitive_item_only_on_a_direct_question(mem):
    world(mem)
    plain = api.what("Саша", turn=turn("что ты помнишь про Сашу"), store=mem)
    assert HEALTH not in _ids(plain)
    text = "\n".join(commands.what(mem, "Саша", turn("что ты помнишь про Сашу", n=2)))
    assert "диабет" not in text and "diabetes" not in text.casefold()
    assert "личн" in text  # an honest count, without the content

    direct = api.what("Саша", turn=turn("что ты знаешь про здоровье Саши", n=3), store=mem)
    assert HEALTH in _ids(direct)


def test_what_by_words_without_a_name(mem):
    world(mem)
    answer = api.what("переезд", turn=turn("что ты помнишь про переезд"), store=mem)
    assert [i.id for i in answer.items] == [MOVE]


def test_what_two_namesakes_lists_both(mem):
    world(mem)
    second_sasha(mem)
    text = "\n".join(commands.what(mem, "Саша", turn("что ты помнишь про Сашу")))
    assert "Ozon" in text or "Озоне" in text
    assert "Тверь" in text


def test_what_nothing_known_says_so(mem):
    world(mem)
    lines = commands.what(mem, "Аркадий", turn("что ты помнишь про Аркадия"))
    assert lines == ["Про «Аркадий» ничего не помню."]


def test_what_writes_served_rows_for_the_brain(mem):
    world(mem)
    api.what("Саша", turn=turn("что ты помнишь про Сашу"), store=mem)
    rows = mem.execute(
        "SELECT item_id, via, thread_id, session_id, turn_id FROM served WHERE item_id = ?",
        (MOVE,),
    ).fetchall()
    assert [tuple(r) for r in rows] == [(MOVE, "tool", "thread-owner", "session-owner", "turn-1")]
    # the entity record itself is not an item the brain was told
    assert mem.execute("SELECT count(*) FROM served WHERE item_id = ?", (SASHA,)).fetchone()[0] == 0


def test_get_returns_one_item_and_hides_others_sensitive(mem):
    world(mem)
    item = api.get(f"m:{MOVE}", turn=turn("что это"), store=mem)
    assert item is not None and item.id == MOVE and item.text
    assert api.get(HEALTH, turn=turn("что это", n=2), store=mem) is None
    assert api.get("m:nope", turn=turn("что это", n=3), store=mem) is None
