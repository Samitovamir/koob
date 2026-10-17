"""What is proven wrong stays out, a direct question by target.

- a view the user took back («беру назад про неё…») overtakes the user's older view of that
  person even when the question's words reach only the old one, and the raw that carries only
  the old view goes with it; a newer view that does not retract only competes among candidates;
- a record that only takes back what the user asked to forget would name the forgotten claim;
- «direct» (§5.5) is decided by the question's resolved target, not by the spelling of the
  name in their words; the topic test is unchanged, so a name alone unlocks nothing sensitive;
- the basket's frame adds no bracket of its own (a «)» reads as part of a memory).

Synthetic data only; names and wordings differ from the dev probes on purpose."""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import AuthorTier, FKind, Impression, Label, TurnCtx, TurnOrigin
from koob.memory.recall import basket, cue, vectors
from koob.memory.writers import hot

from .._ledger import add, entity, fact, open_seeded, raw, ts

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str, *, n: int = 1) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id=f"thread-{n}",
        session_id=f"session-{n}",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        message_id=f"msg-{n}",
        clock=CLOCK,
    )


def _imp(iid: str, about: str, day: str, en: str, ru: str) -> Impression:
    return Impression(
        id=iid, about_entity_id=about, text_en=en, observed_at=f"{day}T10:00:00Z",
        author_tier=AuthorTier.USER, recorded_at=ts(), origin_quote=ru, quote_orig=ru,
    )  # fmt: skip


def _older(conn, ids: list[str], words: str) -> set[str]:
    rows = basket._hydrate(conn, ids, CLOCK[:10])
    c = cue.parse(t(words), now=CLOCK, conn=conn)
    return basket._older_impressions(rows, set(rows), c, conn)


# ---------------------------------------------------------------- retraction overtakes


@pytest.mark.parametrize(
    ("en", "ru"),
    [
        ("The owner takes back what she said about Tamara: Tamara ran the launch alone.",
         "Беру свои слова назад про Тамару — она одна вытянула запуск."),
        ("The owner was wrong about Tamara; she ran the launch alone.",
         "Я была неправа насчёт Тамары, запуск она вытянула одна."),
    ],
)  # fmt: skip
def test_retraction_overtakes_the_old_view_outside_the_candidates(conn, en, ru):
    assert add(conn, entity("e-tam", "Тамара")).ok
    assert add(conn, _imp("i-tam", "e-tam", "2026-02-01", "Tamara forgets every deadline.",
                          "Тамара вечно всё забывает и срывает сроки")).ok  # fmt: skip
    assert add(
        conn,
        fact("f-tam", "e-tam", "other", "вытянула запуск", fkind=FKind.OPINION, text=en,
             quote_orig=ru),
    ).ok  # fmt: skip
    # the question reaches only the old impression; the retraction overtakes it anyway
    assert _older(conn, ["i-tam"], "Тамара не сорвёт сроки?") == {"i-tam"}


def test_newer_view_that_does_not_retract_does_not_hide_from_outside(conn):
    """Only what is proven wrong is removed: a newer remark on another side of her is no proof."""
    assert add(conn, entity("e-zoya", "Зоя")).ok
    assert add(conn, _imp("i-zoya", "e-zoya", "2026-02-01", "Zoya is always late.",
                          "Зоя вечно опаздывает")).ok  # fmt: skip
    assert add(
        conn,
        fact("f-zoya", "e-zoya", "other", "хорошо рисует", fkind=FKind.OPINION,
             text="The owner thinks Zoya draws well.", quote_orig="Зоя классно рисует"),
    ).ok  # fmt: skip
    assert _older(conn, ["i-zoya"], "Зоя опоздает?") == set()


def test_raw_with_only_the_overtaken_view_is_not_quoted(conn):
    assert add(conn, entity("e-mar", "Марат")).ok
    raw(
        conn,
        "raw-mar",
        "Марат обаятельный, но ничего не доводит до конца.",
        ts_="2026-02-01T10:00:00Z",
    )
    imp = _imp("i-mar", "e-mar", "2026-02-01", "Marat never finishes anything.",
               "Марат обаятельный, но ничего не доводит до конца.")  # fmt: skip
    assert add(conn, imp).ok
    conn.execute(
        "INSERT INTO edges (from_id, to_id, etype, created_at) VALUES ('i-mar', 'raw-mar',"
        " 'derived_from', ?), ('e-mar', 'raw-mar', 'derived_from', ?)",  # the person, too
        (ts(), ts()),
    )
    assert add(
        conn,
        fact("f-mar", "e-mar", "other", "довёл проект", fkind=FKind.OPINION,
             text="The owner retracts his remark about Marat: Marat finished the project.",
             quote_orig="Забираю слова про Марата, проект он довёл"),
    ).ok  # fmt: skip
    assert _older(conn, ["i-mar", "raw-mar"], "Марат доведёт дело?") == {"i-mar", "raw-mar"}


def _learned(conn, rid: str, text: str, day: str, imp: Impression) -> None:
    raw(conn, rid, text, ts_=f"{day}T10:00:00Z")
    assert add(conn, imp).ok
    conn.execute(
        "INSERT INTO edges (from_id, to_id, etype, created_at) VALUES (?, ?, 'derived_from', ?)",
        (imp.id, rid, ts()),
    )


def test_retraction_in_the_source_message_overtakes_the_old_raw(conn):
    """The new view's quote keeps only «she ran the launch»; the take-back said before it in
    the same message still proves the old view wrong, so the old message is not quoted."""
    assert add(conn, entity("e-den", "Денис")).ok
    _learned(conn, "raw-old", "Денис милый, но всё время срывает сроки.", "2026-02-01",
             _imp("i-old", "e-den", "2026-02-01", "Denis keeps missing deadlines.",
                  "Денис милый, но всё время срывает сроки."))  # fmt: skip
    _learned(conn, "raw-new", "Беру назад про Дениса. Он один вытянул запуск.", "2026-06-01",
             _imp("i-new", "e-den", "2026-06-01", "Denis ran the launch alone.",
                  "Он один вытянул запуск."))  # fmt: skip
    # the question reaches only the old message
    rows = basket._hydrate(conn, ["raw-old"], CLOCK[:10])
    c = cue.parse(t("Денису можно поручить срочное?"), now=CLOCK, conn=conn)
    assert basket._older_impressions(rows, set(rows), c, conn, any_view=True) == {"raw-old"}


def test_newer_view_without_take_back_in_its_message_keeps_the_old_raw(conn):
    assert add(conn, entity("e-den", "Денис")).ok
    _learned(conn, "raw-old", "Денис милый, но всё время срывает сроки.", "2026-02-01",
             _imp("i-old", "e-den", "2026-02-01", "Denis keeps missing deadlines.",
                  "Денис милый, но всё время срывает сроки."))  # fmt: skip
    _learned(
        conn,
        "raw-new",
        "Денис классно рисует.",
        "2026-06-01",
        _imp("i-new", "e-den", "2026-06-01", "Denis draws well.", "Денис классно рисует."),
    )
    rows = basket._hydrate(conn, ["raw-old"], CLOCK[:10])
    c = cue.parse(t("Денису можно поручить срочное?"), now=CLOCK, conn=conn)
    assert basket._older_impressions(rows, set(rows), c, conn, any_view=True) == set()


def test_retraction_lexicon_reads_both_languages():
    for text in (
        "Беру назад про Олю, она молодец.",
        "Забудь, что я говорила про Олину депрессию.",
        "The owner takes it back: Olya is great.",
        "The owner retracts what he said about Olya.",
        "The owner asked to forget what she said about Olya's health.",
    ):
        assert hot.retracts(text), text
    for text in ("Оля берёт отпуск назад в июле?", "Olya took the train back home.", "Зоя рисует"):
        assert not hot.retracts(text), text


# ---------------------------------------------------------------- direct question by target


def _kim_world(conn):
    assert add(conn, entity("e-lev", "Лев Цой")).ok
    assert add(
        conn,
        fact("f-trip", "e-lev", "travels_to", "Осло", text="Lev Tsoi flies to Oslo in March.",
             keywords_ru="Осло, март, улетаю", quote_orig="в марте улетаю в Осло",
             tier=AuthorTier.OTHER, author_entity_id="e-lev",
             labels=Label.UNKNOWN_SENSITIVE),
    ).ok  # fmt: skip
    assert add(
        conn,
        fact("f-ill", "e-lev", "has_condition", "астма", text="Lev Tsoi has asthma.",
             quote_orig="у меня астма", tier=AuthorTier.OTHER, author_entity_id="e-lev",
             labels=Label.HEALTH),
    ).ok  # fmt: skip


def test_target_resolved_by_the_cue_names_the_subject(conn):
    _kim_world(conn)
    words = "Куда этот товарищ улетает в марте?"  # names no one: not a direct question
    assert not policy.decide(conn, "f-trip", t(words)).allowed
    v = policy.decide_targeted(conn, ["f-trip"], t(words), ["e-lev"])["f-trip"]
    assert v.allowed and v.reason == "direct_question"


def test_target_alone_unlocks_nothing_sensitive(conn):
    _kim_world(conn)
    for words in ("Что нового у него?", "Как он там?"):
        out = policy.decide_targeted(conn, ["f-trip", "f-ill"], t(words), ["e-lev"])
        assert not out["f-trip"].allowed  # open topic: the item's own words are its topic
        assert not out["f-ill"].allowed  # health: the topic lexicon, as before
    v = policy.decide_targeted(conn, ["f-ill"], t("Как у него со здоровьем?"), ["e-lev"])
    assert v["f-ill"].allowed
    # another person's target does not name this subject
    assert not policy.decide_targeted(conn, ["f-trip"], t("Куда улетает в марте?"), ["e-x"])[
        "f-trip"
    ].allowed


# ---------------------------------------------------------------- the frame


def test_basket_frame_has_no_bracket_of_its_own(conn):
    assert add(conn, fact("f-tea", "self", "likes", "улун", text="The owner likes oolong tea.",
                          quote_orig="люблю улун")).ok  # fmt: skip
    rows = basket._hydrate(conn, ["f-tea"], CLOCK[:10])
    block = basket.render([basket._item(rows["f-tea"], None, None)], "turn-1")
    assert block and "[m:f-tea]" in block
    frame = block.splitlines()[1]
    assert "(" not in frame and ")" not in frame


# ---------------------------------------------------------------- a spent retraction


def _olya(conn, *, forgotten: bool):
    assert add(conn, entity("e-olya", "Оля Гусева")).ok
    assert add(conn, _imp("i-olya", "e-olya", "2026-02-01", "Olya seems depressed lately.",
                          "Оля какая-то подавленная последнее время")).ok  # fmt: skip
    if forgotten:
        conn.execute("UPDATE records SET status = 'forgotten' WHERE id = 'i-olya'")
    assert add(
        conn,
        fact("f-olya", "e-olya", "other", "Олина подавленность",
             text="The owner retracts what he said about Olya's low mood, calling it unfair.",
             quote_orig="Забудь, что я говорил про Олину подавленность, это было нечестно."),
    ).ok  # fmt: skip
    return basket._hydrate(conn, ["f-olya"], CLOCK[:10])


def test_retraction_of_a_forgotten_claim_is_not_shown(conn):
    rows = _olya(conn, forgotten=True)
    assert basket._spent_retractions(conn, rows, {"f-olya"}) == {"f-olya"}


def test_spent_retraction_on_a_second_entity_of_the_asked_person(conn):
    """The forget sits on «Оля Гусева», the retraction on a second «Оля»; the question asks
    about both: the forget is the one this record names."""
    rows = _olya(conn, forgotten=True)
    assert add(conn, entity("e-olya-2", "Оля")).ok
    conn.execute("UPDATE facts SET subject_id = 'e-olya-2' WHERE id = 'f-olya'")
    rows = basket._hydrate(conn, ["f-olya"], CLOCK[:10])
    assert basket._spent_retractions(conn, rows, {"f-olya"}) == set()
    got = basket._spent_retractions(conn, rows, {"f-olya"}, ["e-olya", "e-olya-2"])
    assert got == {"f-olya"}


def test_retraction_stays_while_the_old_claim_is_not_forgotten(conn):
    rows = _olya(conn, forgotten=False)
    assert basket._spent_retractions(conn, rows, {"f-olya"}) == set()


# ---------------------------------------------------------------- one person minted twice


def _ida_view(conn, eid: str, tier=AuthorTier.USER):
    assert add(conn, entity("e-ida-1", "Ида")).ok
    assert add(conn, entity(eid, "Ида", tier=tier)).ok
    assert add(conn, _imp("i-ida", "e-ida-1", "2026-02-01", "Ida is unreliable.",
                          "Ида ненадёжная, всё бросает")).ok  # fmt: skip
    assert add(
        conn,
        fact("f-ida", eid, "other", "довела", fkind=FKind.OPINION,
             text="The owner takes back what he said about Ida: she saw it through.",
             quote_orig="Беру назад про Иду, она всё довела"),
    ).ok  # fmt: skip


def test_retraction_on_a_second_entity_of_the_same_person(conn):
    """The extractor minted her twice from their words: one person for the overtaking."""
    _ida_view(conn, "e-ida-2")
    assert _older(conn, ["i-ida"], "Иде можно доверить проект?") == {"i-ida"}


def test_namesake_from_other_peoples_words_is_not_the_same_person(conn):
    _ida_view(conn, "e-ida-x", tier=AuthorTier.OTHER)
    assert _older(conn, ["i-ida"], "Иде можно доверить проект?") == set()
