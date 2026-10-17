"""Recall: who said it, people by role or kin word, namesakes, the user's own labelled fact
asked exactly, and an end before its own start.

- a line names who said it: a record minted without its author takes its source message's
  sender, a raw line from another person names its sender; «кто сказал» / "who told me" keeps
  their words that name the relative behind a claim;
- a role or kin word the relation table has no slot for («тётя», "landlord", «научрук») finds
  the person the user's own facts call so, in the other language too; a kin link written from
  the other side (the person's fact pointing at the user) counts;
- two people under one first name are told apart before serving (a surname, a kin word, the
  question's other words); with nothing pointing at one, both stay; two people are never one
  on a first name alone;
- the user's own fact under a strong label (a minor, health) is rescued when the question asks
  exactly that record's relation about its subject, and not when it asks about someone else;
- a fact whose end falls before its own start is not dropped as expired; a real end is.

Synthetic data only; names and wordings differ from the dev probes on purpose."""

from __future__ import annotations

import pytest

from koob.memory import entities
from koob.memory.model import AuthorTier, Label, TurnCtx, TurnOrigin
from koob.memory.recall import basket, cue, vectors

from .._ledger import add, entity, fact, open_seeded, raw

CLOCK = "2026-10-12T09:00:00.000000Z"


@pytest.fixture
def conn(tmp_path):
    vectors.reset()
    vectors.set_embedder(None)
    c = open_seeded(tmp_path)
    yield c
    c.close()


def t(words: str) -> TurnCtx:
    return TurnCtx(
        turn_id="turn-1", thread_id="thread-1", session_id="session-1",
        origin=TurnOrigin.OWNER, owner_words=words, message_id="msg-1", clock=CLOCK,
    )  # fmt: skip


def ids(conn, words: str) -> list[str]:
    return cue.parse(t(words), now=CLOCK, conn=conn).entity_ids


# ---------------------------------------------------------------- who said it


def _sender(conn):
    assert add(conn, entity("e-mira", "Мира Лужина", tier=AuthorTier.OTHER)).ok
    raw(conn, "r-mira", "Слушай, взнос за секцию подняли до 4000", tier="other")
    conn.execute("UPDATE raw_chunks SET author_entity_id = 'e-mira' WHERE id = 'r-mira'")
    conn.commit()


def test_record_without_author_names_its_source_sender(conn):
    _sender(conn)
    assert add(
        conn,
        fact("f-fee", "self", "other", "4000", text="The club fee went up to 4000.",
             quote_orig="взнос за секцию подняли до 4000", source_raw_id="r-mira",
             tier=AuthorTier.OTHER),
    ).ok  # fmt: skip
    rows = basket._hydrate(conn, ["f-fee"], CLOCK[:10])
    item = basket._item(rows["f-fee"], None, None)
    assert item.said_by == "Мира Лужина"
    assert "со слов: Мира Лужина" in basket.line(item)


def test_raw_line_from_another_person_names_the_sender(conn):
    _sender(conn)
    rows = basket._hydrate(conn, ["r-mira"], CLOCK[:10])
    assert "со слов: Мира Лужина" in basket.line(basket._item(rows["r-mira"], None, None))


@pytest.mark.parametrize(
    ("words", "asks"),
    [("Кто сказал, что взнос подняли?", True), ("Who told me the fee went up?", True),
     ("От кого я узнал про взнос?", True), ("Сколько теперь взнос?", False),
     ("what's the club fee now?", False)],
)  # fmt: skip
def test_who_said_questions_are_told_apart(words, asks):
    assert cue.parse(t(words), now=CLOCK).asks_source is asks


def test_who_said_keeps_his_words_that_name_the_relative():
    quote = (
        "бабушка вчера по телефону долго рассказывала, что дядя Остап наконец продал гараж "
        "на Садовой и теперь ищет дачу поближе к городу"
    )
    row = {"src": basket.ReadSrc.RECORD, "author_tier": "user", "value_text": "продал гараж",
           "quote_orig": quote}  # fmt: skip
    shown = "Uncle Ostap sold his garage."
    assert "бабушка" not in basket._source_value(row, shown)
    assert "бабушка" in basket._source_value({**row, "_asks_source": True}, shown)


# ---------------------------------------------------------------- role and kin words


def _landlord(conn):
    assert add(conn, entity("e-gleb", "Глеб Сизов")).ok
    assert add(
        conn,
        fact("f-ll", "e-gleb", "knows", None, text="Gleb Sizov is the owner's landlord.",
             quote_orig="Глеб — хозяин нашей квартиры"),
    ).ok  # fmt: skip


@pytest.mark.parametrize(
    "words",
    ["Что писал арендодатель?", "Когда приедет хозяин квартиры?", "what did my landlord say?"],
)
def test_role_word_finds_the_person_in_either_language(conn, words):
    _landlord(conn)
    assert "e-gleb" in ids(conn, words)


def test_role_word_in_his_russian_words_answers_an_english_question(conn):
    assert add(conn, entity("e-ilya", "Илья Маркович")).ok
    assert add(
        conn,
        fact("f-adv", "e-ilya", "knows", None, text="Ilya Markovich reviews the owner's thesis.",
             quote_orig="мой научрук Илья Маркович"),
    ).ok  # fmt: skip
    assert "e-ilya" in ids(conn, "when does my advisor want the draft?")
    assert "e-ilya" not in ids(conn, "when does the shop open?")


def test_role_word_is_a_word_start_not_any_substring(conn):
    assert add(conn, entity("e-rita", "Рита Гусева")).ok
    assert add(
        conn,
        fact("f-rest", "e-rita", "works_at", "Bistro", text="Rita Guseva runs a restaurant."),
    ).ok  # fmt: skip
    assert "e-rita" not in ids(conn, "Что нового у тёти?")  # «aunt» is not in «restaurant»


def test_kin_link_written_from_the_other_side(conn):
    assert add(conn, entity("e-yana", "Яна")).ok
    assert add(
        conn,
        fact("f-sib", "e-yana", "sibling_of", None, value_entity_id="self",
             text="Yana is the owner's sister."),
    ).ok  # fmt: skip
    assert "e-yana" in ids(conn, "Когда у сестры экзамен?")
    assert "e-yana" in ids(conn, "is my sister coming on friday?")


# ---------------------------------------------------------------- namesakes


def _two_niks(conn):
    assert add(conn, entity("e-nika-o", "Ника Орлова")).ok
    assert add(conn, entity("e-nika-s", "Ника Сомова")).ok
    assert add(
        conn,
        fact("f-o-city", "e-nika-o", "lives_in", "Тула", text="Nika Orlova lives in Tula.",
             quote_orig="Ника Орлова живёт в Туле"),
    ).ok  # fmt: skip
    assert add(
        conn,
        fact("f-o-job", "e-nika-o", "colleague_of", None, value_entity_id="self",
             text="Nika Orlova works with the owner.", quote_orig="Ника с работы, коллега"),
    ).ok  # fmt: skip
    assert add(
        conn,
        fact("f-s-city", "e-nika-s", "lives_in", "Псков", text="Nika Somova lives in Pskov.",
             quote_orig="Ника Сомова живёт в Пскове"),
    ).ok  # fmt: skip
    assert add(
        conn,
        fact("f-s-sib", "e-nika-s", "sibling_of", None, value_entity_id="self",
             text="Nika Somova is the owner's cousin and like a sister.",
             quote_orig="Ника — моя сестра"),
    ).ok  # fmt: skip


@pytest.mark.parametrize("words", ["Где сейчас Ника Сомова?", "where does Nika Somova live now?"])
def test_namesake_told_apart_by_surname(conn, words):
    _two_niks(conn)
    got = ids(conn, words)
    assert "e-nika-s" in got and "e-nika-o" not in got, got
    b = basket.assemble(conn, t(words))
    assert "Tula" not in (b.block or "") and "Туле" not in (b.block or ""), b.block


@pytest.mark.parametrize("words", ["Где живёт моя сестра Ника?", "where is my sister Nika now?"])
def test_namesake_told_apart_by_kin_word(conn, words):
    _two_niks(conn)
    got = ids(conn, words)
    assert got.count("e-nika-s") == 1 and "e-nika-o" not in got, got


@pytest.mark.parametrize("words", ["Ника с работы ещё в Туле?", "is Nika from work still around?"])
def test_namesake_told_apart_by_what_is_known(conn, words):
    _two_niks(conn)
    got = ids(conn, words)
    assert "e-nika-o" in got and "e-nika-s" not in got, got


def test_namesakes_with_nothing_pointing_both_stay(conn):
    _two_niks(conn)
    got = ids(conn, "Ника звонила?")
    assert {"e-nika-o", "e-nika-s"} <= set(got), got


def test_namesake_groups_never_join_on_a_first_name():
    names = {"a": "Ника Орлова", "b": "Nika Orlova", "c": "Ника", "d": "Ника Сомова"}
    groups = entities.namesake_groups(names)
    flat = [set(g) for g in groups]
    assert any({"c", "d"} <= g for g in flat), groups  # one first name, two people
    assert not any({"a", "b"} <= g for g in flat), groups  # one full name: one person
    assert entities.namesake_groups({"x": "Ника Орлова", "y": "Тимур Орлов"}) == []


# ---------------------------------------------------------------- the user's own labelled fact


def _row(**kw):
    base = {"author_tier": AuthorTier.USER.value, "src": basket.ReadSrc.RECORD,
            "labels": int(Label.MINOR), "subject_id": "e-kid", "relation": "has_phone",
            "text": "Fedya's phone number is 555-0142.", "value_text": "555-0142",
            "quote_orig": "номер Феди 555-0142"}  # fmt: skip
    return {**base, **kw}


def _cue(words: str, entity_ids: list[str]) -> cue.Cue:
    c = cue.parse(t(words), now=CLOCK)
    return c.model_copy(update={"entity_ids": entity_ids})


def test_own_minor_fact_rescued_when_its_relation_is_asked():
    c = _cue("скинь номер телефона Феди", ["e-kid"])
    stems = basket.lexical.query_stems(c.words)
    assert basket._asked(_row(), {"lexical"}, c, stems)
    c_en = _cue("send me Fedya's phone number", ["e-kid"])
    assert basket._asked(_row(), {"lexical"}, c_en, basket.lexical.query_stems(c_en.words))


def test_own_minor_fact_not_rescued_for_someone_else_or_another_kind():
    c = _cue("скинь номер телефона Феди", ["e-other"])
    assert not basket._asked(_row(), {"lexical"}, c, basket.lexical.query_stems(c.words))
    c2 = _cue("Федя пойдёт в кино?", ["e-kid"])
    assert not basket._asked(_row(), {"lexical"}, c2, basket.lexical.query_stems(c2.words))


def test_own_health_fact_rescued_by_its_own_value_words():
    row = _row(labels=int(Label.HEALTH), subject_id="self", relation="other",
               text="The owner takes a pill every morning.", value_text="витамин",
               quote_orig="пью витамин по утрам")  # fmt: skip
    c = _cue("во сколько мне пить витамин?", [])
    assert basket._asked(row, {"lexical"}, c, basket.lexical.query_stems(c.words))


# ---------------------------------------------------------------- an end before its start


def test_end_before_its_own_start_is_not_an_end(conn):
    assert add(
        conn,
        fact("f-winter", "self", "travels_to", "Сочи", text="The owner is in Sochi for winter.",
             valid_from="2026-12-20", valid_to="2026-01-05"),
    ).ok  # fmt: skip
    assert add(
        conn,
        fact("f-spring", "self", "travels_to", "Ялта", text="The owner was in Yalta in spring.",
             valid_from="2026-03-01", valid_to="2026-04-01"),
    ).ok  # fmt: skip
    rows = basket._hydrate(conn, ["f-winter", "f-spring"], CLOCK[:10])
    assert "f-winter" in rows and rows["f-winter"]["valid_to"] is None
    assert "f-spring" not in rows


def test_standing_namesakes_are_named_on_their_lines(conn):
    _two_niks(conn)
    c = cue.parse(t("Ника звонила?"), now=CLOCK, conn=conn)
    assert c.namesakes == {"e-nika-o": "Ника Орлова", "e-nika-s": "Ника Сомова"}
    rows = basket._hydrate(conn, ["f-o-city"], CLOCK[:10])
    bare = {**rows["f-o-city"], "text": "Nika lives in Tula.", "quote_orig": "Ника в Туле"}
    item = basket._item({**bare, "_who": "Ника Орлова"}, None, None)
    assert item.text.startswith("о ком: Ника Орлова · "), item.text
    named = basket._item({**rows["f-o-city"], "_who": "Ника Орлова"}, None, None)
    assert "о ком" not in named.text  # the line already names her in full
    told = cue.parse(t("Где сейчас Ника Сомова?"), now=CLOCK, conn=conn)
    assert told.namesakes == {} and told.ruled_out == ["e-nika-o"]
