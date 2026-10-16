"""the memory spec §6.1 safety net.
Spec marker: `[T hotpath/test_safety_net::<case>]`."""

from __future__ import annotations

from koob.memory import api
from koob.memory.model import TurnOrigin
from koob.memory.writers import hot

from ._world import sasha_world, turn_of


def test_claim_without_commit_blocked(mem):
    """§6.1: the Stop hook blocks once a reply claiming a memory change with no commit."""
    words = "запомни: я люблю зелёный чай"
    turn = turn_of(words, n=2)
    assert api.claims_unbacked_change(turn, "Запомнил.", store=mem)
    assert api.claims_unbacked_change(turn, "Got it, noted.", store=mem)
    assert not api.claims_unbacked_change(turn, "Какой чай ты любишь?", store=mem)
    task = turn_of(words, n=3, origin=TurnOrigin.TASK)
    assert not api.claims_unbacked_change(task, "Запомнил.", store=mem)
    api.remember(words, turn=turn, because_quote="я люблю зелёный чай", store=mem)
    assert not api.claims_unbacked_change(turn, "Запомнил.", store=mem)


def test_failure_line_once(mem):
    """§6.1: on failure the Cube says the failure line once."""
    sasha_world(mem)
    turn = turn_of("нет, это не тот Саша, я про Сашу с работы", n=2)
    net = api.after_turn_check(turn, "Понял.", store=mem)
    assert net.fired and net.job_id
    assert hot.failure_line(mem, "thread-1") is None  # the job has not failed
    mem.execute("UPDATE jobs SET state = 'failed' WHERE id = ?", (net.job_id,))
    mem.commit()
    line = hot.failure_line(mem, "thread-1")
    assert line == "Не смог применить исправление про Саша — уточни."
    assert hot.failure_line(mem, "thread-1") is None
    # the same turn checked again reuses its job
    assert api.after_turn_check(turn, "Понял.", store=mem).job_id == net.job_id


def test_freeze_blocks_send(mem):
    """§6.1: frozen cue entities leave basket and update blocks and hands are denied on them.
    hot owns the freeze set (`hot.frozen_entities` / `frozen_records` / `is_frozen`);
    recall and the hands gate read it."""
    sasha, bday = sasha_world(mem)
    plain = api.after_turn_check(turn_of("какая погода завтра?", n=2), "Солнечно.", store=mem)
    assert not plain.fired
    turn = turn_of("я же говорил, у Саши день рождения не тогда", n=3)
    net = api.after_turn_check(turn, "Понял.", store=mem)
    assert net.fired and net.frozen_entities == [sasha.id]
    assert hot.is_frozen(mem, sasha.id)
    assert bday.id in hot.frozen_records(mem)
    mem.execute("UPDATE jobs SET state = 'done' WHERE id = ?", (net.job_id,))
    mem.commit()
    assert not hot.is_frozen(mem, sasha.id)
    hot.freeze(mem, [sasha.id], seconds=0.0)
    assert not hot.is_frozen(mem, sasha.id)  # the 60 s (here 0 s) passed: the freeze lifts
    # a correction committed in the turn: nothing to catch
    q = "у Саши день рождения 14 мая"
    fixed_turn = turn_of(f"нет, {q}", n=4)
    api.fix(f"m:{bday.id}", q, turn=fixed_turn, because_quote=q, store=mem)
    assert not api.after_turn_check(fixed_turn, "Исправил.", store=mem).fired
    promise = api.after_turn_check(turn_of("ок", n=5), "Хорошо, напомню завтра.", store=mem)
    assert promise.fired
