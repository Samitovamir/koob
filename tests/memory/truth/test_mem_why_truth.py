"""Tests for fixes #5 and #6 from the design notes: a correction retires every carrier of the
old value, a speaker correction moves the claim to the right person, and forget sticks (scope,
wording, codes). Synthetic data only; no model call."""

from __future__ import annotations

from koob.memory import api, commands, entities, forget
from koob.memory.model import Entity, Episode

from ..hotpath._world import turn_of
from ..races._kit import TS0, add_raw, fact, handle, person, seed, status_of, ulid


def _edge(conn, frm: str, to: str, etype: str = "derived_from") -> None:
    conn.execute(
        "INSERT INTO edges (from_id, to_id, etype, created_at) VALUES (?, ?, ?, ?)",
        (frm, to, etype, TS0),
    )


def _usable_values(conn) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT coalesce(f.value_text, '') || ' ' || f.text_en FROM facts f"
            " JOIN records r ON r.id = f.id WHERE r.status IN ('active', 'archived')"
        )
    ]


# ------------------------------------------------------------- #5: every carrier retired


def test_fix_retires_a_duplicate_copy_of_the_old_value(mem):
    """Two copies of one value (two sources): the fix of one retires both (understands-d02/p3,
    the duplicate «loves cilantro» survived its supersede)."""
    arman = person("Арман")
    a = fact("ret-a", arman, "returns_on", "9 мая", "Arman returns on May 9", "Арман вернётся 9 мая",
             value_norm="--05-09")  # fmt: skip
    b = fact("ret-b", arman, "returns_on", "9 мая", "Arman is back on 9 May", "вернётся 9-го",
             value_norm="--05-09")  # fmt: skip
    seed(mem, arman, a, b)
    q = "нет, Арман вернётся не 9 мая, а 10 мая"
    r = api.fix(handle(a.id), q, turn=turn_of(q, n=2), because_quote=q, store=mem)
    assert r.outcome == "committed"
    assert status_of(mem, a.id) == "superseded" and status_of(mem, b.id) == "superseded"
    assert not any("9 мая" in v or "May 9" in v for v in _usable_values(mem))


def test_fix_aimed_at_an_episode_corrects_its_seed_fact(mem):
    """The episode carries its seed's value: the fix lands on the fact, and the episode goes
    with it (safety_net-d02: the episode moved, the fact kept 05-09)."""
    arman = person("Арман")
    f = fact("ret-f", arman, "returns_on", "9 мая", "Arman returns on May 9", "Арман вернётся 9 мая",
             value_norm="--05-09")  # fmt: skip
    ep = Episode(id=ulid("episode:ret"), author_tier=f.author_tier, recorded_at=TS0,
                 labels_final=True, summary_en="Arman returns on May 9.")  # fmt: skip
    seed(mem, arman, f, ep, extra=lambda c: _edge(c, ep.id, f.id))
    q = "нет, Арман вернётся не 9 мая, а 10 мая"
    r = api.fix(handle(ep.id), q, turn=turn_of(q, n=2), because_quote=q, store=mem)
    assert r.outcome == "committed"
    assert status_of(mem, f.id) == "superseded"
    assert status_of(mem, ep.id) == "superseded"
    assert any("10" in v for v in _usable_values(mem))


def test_speaker_correction_moves_the_claim_to_the_right_person(mem):
    """«я перепутал: X говорил Воронин, а не Воронцова» (e2e-d01): the claim is Воронин's,
    and a person minted only from the mistaken claim is no longer current."""
    wrong = Entity(id=ulid("entity:vorontsova"), author_tier=person("x").author_tier,
                   recorded_at=TS0, labels_final=True, etype="person",
                   display_name="Женя Воронцова")  # fmt: skip
    claim = fact("load", wrong, "said", "нагрузочный тест", "Zhenya Vorontsova ran a load test.",
                 "гоняла нагрузочный тест")  # fmt: skip
    raw = add_raw(mem, "claim", "Женя Воронцова гоняла нагрузочный тест NATS", chat_id="owner")

    def links(c):
        _edge(c, claim.id, raw)
        _edge(c, wrong.id, raw)

    seed(mem, wrong, claim, extra=links)
    q = "Стоп, я перепутал: про нагрузочный тест NATS говорил Женя Воронин, коллега, а не Женя Воронцова, соседка."
    r = api.fix(q, q, turn=turn_of(q, n=2), because_quote=q, store=mem)
    assert r.outcome == "committed", r.line
    assert status_of(mem, claim.id) == "superseded"
    assert status_of(mem, wrong.id) == "retracted"
    usable = " ".join(_usable_values(mem))
    assert "Воронцов" not in usable and "Vorontsov" not in usable
    new = mem.execute(
        "SELECT e.display_name FROM facts f JOIN entities e ON e.id = f.subject_id WHERE f.id = ?",
        (r.record_ids[0],),
    ).fetchone()
    assert new[0] == "Женя Воронин"


# ------------------------------------------------------------------ #6: forget that sticks


def test_forget_wording_drops_his_reason_and_reads_what_he_said():
    """«Забудь, что я говорила про X, это было нечестно» and «Про X забудь, не хочу, чтобы…»
    name X (inspect-d01, safety_net-d04 were `not_found`)."""
    q, whole = commands._forget_query(
        "Забудь, что я говорила про Сашкину выгоренность, это было нечестно."
    )
    assert "нечестно" not in q and "говорила" not in q and "выгоренность" in q and whole
    q, _ = commands._forget_query("Про давление забудь, не хочу, чтобы это где-то висело.")
    assert commands._content_words(q) == ["давление"]
    assert commands._same_word("выгоренность", "выгоревшая")
    assert not commands._same_word("переговоры", "перегородка")


def test_code_like_tokens_are_never_entity_names():
    for code in ("ТГКЛСФ-5104 (aka PBUUUQX-5104)", "PBUUUQX-5104", "6034-5094", "№ 7731-0942"):
        assert entities.code_like(code), code
    for name in ("Костя", "Релиз 1 июля", "Женя Воронин", "NATS JetStream", "Сашка Белова"):
        assert not entities.code_like(name), name


def test_forget_takes_what_was_minted_only_from_the_forgotten_message(mem):
    """keeps_true-d03: the canary minted as an entity name from the forgotten message goes
    with it; a person with another source stays."""
    kid = person("Костя")
    canary = Entity(id=ulid("entity:canary"), author_tier=kid.author_tier, recorded_at=TS0,
                    labels_final=True, etype="person", display_name="Зефира")  # fmt: skip
    told = fact("fight", kid, "event", "драка", "Kostya got into a fight at school.",
                "Костя подрался в школе")  # fmt: skip
    raw = add_raw(mem, "fight", "Костя подрался в школе с мальчиком по кличке Зефира", chat_id="o")
    other = add_raw(mem, "maths", "Костя получил пятёрку по математике", chat_id="o")

    def links(c):
        for rid in (told.id, canary.id, kid.id):
            _edge(c, rid, raw)
        _edge(c, kid.id, other)

    seed(mem, kid, canary, told, extra=links)
    plan = forget._plan(mem, [told.id], forget.ForgetScope.ITEM, "fg-test")
    assert told.id in plan.records and canary.id in plan.records
    assert kid.id not in plan.records
    assert raw in plan.raw_forgotten
