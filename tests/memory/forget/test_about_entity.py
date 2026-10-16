"""Acceptance for the memory spec §9.8: "forget everything about X" (scope `about_entity`).
Spec marker: `[T forget/test_about_entity::<case>]`."""

from __future__ import annotations

from koob.memory import forget
from koob.memory.adapters.conversation import FORGOTTEN
from koob.memory.model import ForgetScope

from . import _seed as s

clock = s.clock  # fixture
career = s.career  # fixture
conv = s.conv  # fixture


def _world(mem) -> None:
    s.relation(mem, "moves_to", "residence")
    s.relation(mem, "friend_of", "social", "multi")
    s.relation(mem, "lives_in", "residence")
    s.entity(
        mem,
        "ent-x",
        "Илья",
        [("Илья", "name"), ("Илюша", "diminutive"), ("Ilya", "translit")],
        stable_ids=[("tg_user_id", "777001")],
    )
    s.entity(mem, "ent-me", "Денис", [("Денис", "name")])
    s.entity(mem, "ent-masha", "Маша", [("Маша", "name")])
    s.raw(mem, "raw-x", "Илья переезжает в Берлин", turn="1")
    s.fact(
        mem,
        "f-subject",
        "ent-x",
        "moves_to",
        value="Berlin",
        text_en="Ilya is moving to Berlin",
        quote="Илья переезжает в Берлин",
        raw="raw-x",
    )
    s.fact(
        mem,
        "f-value",
        "ent-me",
        "friend_of",
        value="Ilya",
        value_entity="ent-x",
        text_en="Denis is friends with Ilya",
    )
    s.impression(mem, "imp-x", "ent-x", "Ilya seemed tired", "Илюша выглядел уставшим")
    s.record(mem, "loop-x", "loop")
    mem.execute("INSERT INTO loops (id, owner, text_en) VALUES ('loop-x', 'user', 'call him')")
    s.edge(mem, "loop-x", "ent-x", "about")
    s.episode(mem, "ep-about", "Farewell dinner", links=[("about", "ent-x")])
    # Mentions elsewhere: Masha's move, said by Ilya.
    s.raw(mem, "raw-m", "Илья сказал, что Маша переехала в Париж", turn="2")
    s.fact(
        mem,
        "f-masha",
        "ent-masha",
        "lives_in",
        value="Paris",
        text_en="Ilya says Masha moved to Paris",
        quote="Илья сказал, что Маша переехала в Париж",
        raw="raw-m",
        speaker="ent-x",
    )
    s.episode(
        mem,
        "ep-party",
        "Party with Masha and Ilya",
        links=[("participant", "ent-x"), ("participant", "ent-masha")],
    )


def test_subject_records_forgotten(mem, clock):
    """X's entity with aliases and stable ids, and every record with X as subject, value,
    about-entity, about edge or episode `about` link, are forgotten whole."""
    _world(mem)
    fid = forget.stop_use(mem, ["ent-x"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    status = dict(mem.execute("SELECT id, status FROM records").fetchall())
    for rid in ("ent-x", "f-subject", "f-value", "imp-x", "loop-x", "ep-about"):
        assert status[rid] == "forgotten", rid
    for rid in ("ent-me", "ent-masha", "f-masha", "ep-party"):
        assert status[rid] == "active", rid
    assert mem.execute("SELECT count(*) FROM aliases WHERE entity_id = 'ent-x'").fetchone()[0] == 0
    assert mem.execute("SELECT count(*) FROM fts_names WHERE name MATCH 'Илю'").fetchone()[0] == 0
    assert tuple(
        mem.execute("SELECT display_name, stable_ids FROM entities WHERE id = 'ent-x'").fetchone()
    ) == (FORGOTTEN, "[]")
    forget.undo(mem, fid)
    assert mem.execute("SELECT count(*) FROM aliases WHERE entity_id = 'ent-x'").fetchone()[0] == 3
    assert mem.execute("SELECT count(*) FROM fts_names WHERE name MATCH 'Илю'").fetchone()[0] == 1


def test_mentions_masked_elsewhere(mem, clock):
    """A record about something else keeps its claim with X masked: names → «[забыто]»,
    speaker and participant links cleared, raw span-masked (rev+1); undo restores."""
    _world(mem)
    fid = forget.stop_use(mem, ["ent-x"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    text_en, quote, speaker = mem.execute(
        "SELECT text_en, quote_orig, speaker_entity_id FROM facts WHERE id = 'f-masha'"
    ).fetchone()
    assert text_en == f"{FORGOTTEN} says Masha moved to Paris"
    assert quote == f"{FORGOTTEN} сказал, что Маша переехала в Париж"
    assert speaker is None
    links = mem.execute("SELECT target_id FROM episode_links WHERE episode_id = 'ep-party'")
    assert [r[0] for r in links] == ["ent-masha"]
    text, rev = mem.execute("SELECT text, rev FROM raw_chunks WHERE id = 'raw-m'").fetchone()
    assert text.startswith(FORGOTTEN) and rev == 2
    fts = mem.execute(
        "SELECT text FROM fts_records WHERE rowid = "
        "(SELECT fts_rowid FROM records WHERE id = 'f-masha')"
    ).fetchone()[0]
    assert "ilya" not in fts.casefold()

    forget.undo(mem, fid)
    assert mem.execute("SELECT speaker_entity_id FROM facts WHERE id = 'f-masha'").fetchone()[
        0
    ] == ("ent-x")
    assert (
        mem.execute("SELECT text FROM raw_chunks WHERE id = 'raw-m'")
        .fetchone()[0]
        .startswith("Илья")
    )


def test_no_new_entity_from_old_raw(mem, clock):
    """X does not come back under a new entity id from raw older than the forget: its
    stable ids and (alias, proposition) pairs are suppressed; another Илья is not."""
    _world(mem)
    fid = forget.stop_use(mem, ["ent-x"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    old = dict(source_ts=s.OLD, author_tier="other")
    assert forget.blocked_by(mem, stable_ids=[("tg_user_id", "777001")], **old) == fid
    assert (
        forget.blocked_by(
            mem, alias_norms=["илюша"], relation="moves_to", value_norm="berlin", **old
        )
        == fid
    )
    assert (
        forget.blocked_by(
            mem, alias_norms=["илья"], relation="moves_to", value_norm="moscow", **old
        )
        is None
    )
    assert forget.blocked_by(mem, stable_ids=[("tg_user_id", "555")], **old) is None


def test_turn_with_only_the_name_is_masked(mem, career, conv, clock):
    """§9.8 turns: a turn hit by the forgotten records is blanked whole; a turn whose only
    hit is X's name is masked in place."""
    th = s.thread(career)
    t_move = s.turn(career, th, ("user", "Илья переезжает в Берлин"))
    t_name = s.turn(career, th, ("user", "Илюша звонил вчера, всё хорошо"))
    t_other = s.turn(career, th, ("user", "купи молоко"))
    _world(mem)
    mem.execute("UPDATE raw_chunks SET turn_id = ?, chat_id = ? WHERE id = 'raw-x'", (t_move, th))
    forget.stop_use(mem, ["ent-x"], scope=ForgetScope.ABOUT_ENTITY, forever=False)
    assert s.texts(career, t_move) == [FORGOTTEN]
    assert s.texts(career, t_name) == [f"{FORGOTTEN} звонил вчера, всё хорошо"]
    assert s.texts(career, t_other) == ["купи молоко"]
