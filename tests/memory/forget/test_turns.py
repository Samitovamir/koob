"""Acceptance for the memory spec §9.5 and §9.1b-c: affected turns in koob.db.
Spec marker: `[T forget/test_turns::<case>]`."""

from __future__ import annotations

import json

from koob.memory import forget
from koob.memory.adapters.conversation import FORGOTTEN
from koob.memory.model import ForgetScope

from . import _seed as s

career = s.career  # fixture
clock = s.clock  # fixture
conv = s.conv  # fixture


def _ilya(mem, career, *, turn_id: str) -> None:
    s.relation(mem, "moves_to", "residence")
    s.entity(
        mem, "ent-ilya", "Илья", [("Илья", "name"), ("Илюша", "diminutive"), ("Ilya", "translit")]
    )
    s.raw(mem, "raw-1", "Запомни: Илья переезжает в Берлин", turn=turn_id, chat="1")
    s.fact(
        mem,
        "f-move",
        "ent-ilya",
        "moves_to",
        value="Berlin",
        text_en="Ilya is moving to Berlin",
        quote="Илья переезжает в Берлин",
        keywords_ru="переезд Берлин",
        keywords_en="move Berlin",
        raw="raw-1",
    )


def test_paraphrase_in_answer_found(mem, career, conv, clock):
    """§9.5: a later turn that paraphrases the forgotten item is found by match keys; the
    island shows «[забыто]»; an unrelated turn stays; «Вернуть» brings both back."""
    th = s.thread(career)
    t_said = s.turn(
        career, th, ("user", "Запомни: Илья переезжает в Берлин"), ("assistant", "Запомнил.")
    )
    t_para = s.turn(
        career,
        th,
        ("user", "что нового у друзей?"),
        ("assistant", "Илюша скоро будет жить в Берлине."),
        tool={"name": "memory_search", "result": "ilya moving berlin"},
    )
    t_other = s.turn(career, th, ("user", "какая погода?"), ("assistant", "Солнечно."))
    _ilya(mem, career, turn_id=t_said)

    fid = forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)

    assert s.texts(career, t_said) == [FORGOTTEN, FORGOTTEN]
    assert s.texts(career, t_para) == [FORGOTTEN, FORGOTTEN]
    tool = career.execute(
        "SELECT tool_json FROM koob_messages WHERE turn_id = ? AND role = 'tool'", (int(t_para),)
    ).fetchone()[0]
    assert json.loads(tool) == {
        "name": "memory_search",
        "label": FORGOTTEN,
        "detail": "",
        "result": "",
    }
    assert s.texts(career, t_other) == ["какая погода?", "Солнечно."]
    ctx = career.execute("SELECT context FROM koob_turns WHERE id = ?", (int(t_said),)).fetchone()
    assert ctx[0] is None
    line = forget.read_journal(forget.journal_path(mem))[1]  # [0] is the `stopping` line
    assert set(line.turn_ids) == {t_said, t_para}

    forget.undo(mem, fid)
    assert s.texts(career, t_said) == ["Запомни: Илья переезжает в Берлин", "Запомнил."]
    assert s.texts(career, t_para)[1] == "Илюша скоро будет жить в Берлине."
    ctx = career.execute("SELECT context FROM koob_turns WHERE id = ?", (int(t_said),)).fetchone()
    assert ctx[0] == "page: /koob"


def test_served_turn_blanked_and_sessions_restarted(mem, career, clock):
    """§9.5, §9.1c: a turn whose served set held the id is blanked even without a key hit;
    every thread that served it restarts without the affected turns."""
    restarted: list[tuple[list[str], list[str]]] = []
    store_ = s.CareerConversation(career, restarter=lambda th, ex: restarted.append((th, ex)) or th)
    forget.wire(conversation=store_)
    th = s.thread(career)
    t1 = s.turn(career, th, ("user", "что у меня на неделе?"), ("assistant", "Встреча в среду."))
    _ilya(mem, career, turn_id=None)
    s.served(mem, th, "f-move", t1)

    forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)

    assert s.texts(career, t1) == [FORGOTTEN, FORGOTTEN]
    assert restarted == [([th], [t1])]


def test_streaming_answer_rematched(mem, career, conv, clock):
    """§9.1c: the answer that streamed during the forget, written after it, is re-matched
    by the after-turn adapter and blanked under the same forget."""
    th = s.thread(career)
    t_said = s.turn(career, th, ("user", "Запомни: Илья переезжает в Берлин"))
    _ilya(mem, career, turn_id=t_said)
    fid = forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)
    clock.advance(5)
    t_late = s.turn(career, th, ("assistant", "Кстати, Илья переезжает в Берлин в мае."))
    t_calm = s.turn(career, th, ("assistant", "Готово, напоминание на завтра."))

    assert forget.rematch_turn(mem, t_late) == [fid]
    assert forget.rematch_turn(mem, t_calm) == []
    assert s.texts(career, t_late) == [FORGOTTEN]
    assert s.texts(career, t_calm) == ["Готово, напоминание на завтра."]
    assert t_late in forget.read_journal(forget.journal_path(mem))[-1].turn_ids

    forget.undo(mem, fid)
    assert s.texts(career, t_late) == ["Кстати, Илья переезжает в Берлин в мае."]


def test_overlapping_forgets_keep_turn_blank(mem, career, conv, clock):
    """Two forgets blank the same turn: undoing one keeps it blank for the other; undoing
    both brings the original back."""
    th = s.thread(career)
    t = s.turn(career, th, ("user", "Илья переезжает в Берлин, а Маша любит кофе"))
    _ilya(mem, career, turn_id=t)
    s.entity(mem, "ent-masha", "Маша", [("Маша", "name")])
    s.relation(mem, "prefers", "preference", "multi")
    s.raw(mem, "raw-2", "Илья переезжает в Берлин, а Маша любит кофе", turn=t, chat="1")
    s.fact(
        mem,
        "f-coffee",
        "ent-masha",
        "prefers",
        value="coffee",
        text_en="Masha likes coffee",
        quote="Маша любит кофе",
        raw="raw-2",
    )

    a = forget.stop_use(mem, ["f-move"], scope=ForgetScope.ITEM, forever=False)
    b = forget.stop_use(mem, ["f-coffee"], scope=ForgetScope.ITEM, forever=False)
    assert s.texts(career, t) == [FORGOTTEN]
    forget.undo(mem, a)
    assert s.texts(career, t) == [FORGOTTEN]
    forget.undo(mem, b)
    assert s.texts(career, t) == ["Илья переезжает в Берлин, а Маша любит кофе"]


def test_match_keys_cover_aliases_values_and_shingles(mem, career, clock):
    """§9.5: the subject's aliases, the value's stems in RU and EN, 4-word shingles."""
    _ilya(mem, career, turn_id=None)
    keys = forget.match_keys(mem, ["f-move"])
    for k in ("Илья", "Илюша", "Ilya", "Berlin", "илья переезжает в берлин"):
        assert k in keys
    assert any(k.startswith("берлин") for k in keys) and any(k.startswith("переезд") for k in keys)
