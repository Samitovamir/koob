"""The finder brings back the user's committed facts.

- natural questions reach the record through its entity and relation (the pet example): the
  first person, a relation word («где живу», «брат»), the question word (where, when, who);
- remove only what is proven wrong: a broad label (legal, unknown) on the user's own record is
  no proof it does not apply when the question asks for it; health keeps the policy's own test;
- name forms (#12): a declined diminutive meets its name by morphology and by its family; a
  name in the other script meets the record by transliteration;
- the value is in the line: the source-language keywords the value lacks.

Synthetic data only; names and wordings differ from the dev probes on purpose."""

from __future__ import annotations

import pytest

from koob.memory import api, policy
from koob.memory.model import AuthorTier, FKind, Impression, Label, TurnCtx, TurnOrigin
from koob.memory.recall import cue, lexical, vectors

from .._ledger import add, entity, fact, open_seeded, ts

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


def _basket(conn, words: str, n: int = 1) -> str:
    b = api.recall(t(words, n=n), store=conn)
    return "\n".join([b.block or "", *(i.text for i in b.items)])


def _home(conn) -> None:
    assert add(
        conn,
        fact(
            "f-home", "self", "lives_in", "Таганка",
            text="The owner rents a flat near Taganka.",
            keywords_ru="квартира, аренда, Таганка", quote_orig="снимаем квартиру на Таганке",
            labels=Label.LEGAL | Label.UNKNOWN_SENSITIVE,
        ),
    ).ok  # fmt: skip


# ---------------------------------------------------------------- the cue reads the question


@pytest.mark.parametrize(
    ("words", "rel", "vtype", "me"),
    [
        ("Подскажи, где я сейчас живу?", "lives_in", "place", True),
        ("Where do we live these days?", "lives_in", "place", True),
        ("Какой у нас был адрес до переезда?", "home_address", None, True),
        ("Когда у Тимура днюха?", "birthday", "date", False),
        ("Кем работает мой брат?", "sibling_of", "person", True),
        ("With whom did I plan to meet on Friday?", "wants_to_meet", "person", True),
    ],
)
def test_cue_names_relation_value_type_and_person(words, rel, vtype, me):
    ws = policy.words(words)
    assert rel in cue.relations_of(ws)
    assert (vtype in cue.value_types_of(ws)) if vtype else True
    assert policy._hits(ws, cue._SELF_LEX) is me


def test_cue_short_stems_do_not_misfire():
    for words in ("Который из вариантов лучше?", "Дай другой пример", "Котлеты или рыба?"):
        assert "has_pet" not in cue.relations_of(policy.words(words))
        assert "friend_of" not in cue.relations_of(policy.words(words))


# ---------------------------------------------------------------- relation lane and use-now


@pytest.mark.parametrize(
    "words", ["Подскажи, где я сейчас живу?", "Какой у нас адрес?", "Where do I live now?"]
)
def test_where_i_live_reaches_his_residence(conn, words):
    _home(conn)
    assert "Таганка" in _basket(conn, words)


def test_residence_stays_out_of_an_unrelated_turn(conn):
    _home(conn)
    assert "Таганка" not in _basket(conn, "Что приготовить на ужин из курицы?")


def test_health_label_keeps_the_policy_test(conn):
    """The override never covers health: a diagnosis does not ride on «где я живу»."""
    _home(conn)
    assert add(
        conn,
        fact("f-ill", "self", "home_address", "палата 4", text="The owner stays in ward 4.",
             labels=Label.HEALTH),
    ).ok  # fmt: skip
    text = _basket(conn, "Где я живу?")
    assert "Таганка" in text and "палата" not in text


def test_kin_word_reaches_the_relative(conn):
    assert add(conn, entity("e-bro", "Тимур")).ok
    assert add(
        conn,
        fact("f-bro", "self", "sibling_of", "Тимур", value_entity_id="e-bro",
             text="Timur is the owner's brother."),
    ).ok  # fmt: skip
    assert add(
        conn, fact("f-job", "e-bro", "works_at", "Ростелеком", text="Timur works at Rostelecom.")
    ).ok
    assert "Rostelecom" in _basket(conn, "А где сейчас трудится мой брат?")


# ---------------------------------------------------------------- name forms


def test_declined_diminutive_meets_the_name_by_morphology(conn):
    """«Егорушка» → «егорушк» → «егор» ~ «Егор» (trigrams alone miss it): no table holds this name."""
    assert add(conn, entity("e-egor", "Егор Лавров")).ok
    hits = dict(lexical.search_names(conn, "Как там Егорушка поживает после переезда?"))
    assert hits.get("e-egor", 0) >= cue.NAME_FLOOR


def test_diminutive_meets_the_full_name_by_family(conn):
    """«Женька» shares a root with «Женя», a diminutive of «Евгений»."""
    assert add(conn, entity("e-zh", "Евгений Котов")).ok
    hits = dict(lexical.search_names(conn, "Когда Женька вернётся из Твери?"))
    assert hits.get("e-zh", 0) >= cue.NAME_FLOOR


def test_name_in_latin_meets_his_russian_words(conn):
    note = "Запомни: у Барсика стрижка 5 мая в 14:00, салон «Хвост» на Ленинском."
    assert add(conn, fact("f-groom", "self", "other", None, text=note)).ok
    assert "Хвост" in _basket(conn, "When is Barsik's grooming?")


# ---------------------------------------------------------------- the value is in the line


def test_line_carries_the_keywords_the_value_lacks(conn):
    assert add(conn, entity("e-studio", "Свет", etype="org")).ok
    assert add(
        conn,
        fact("f-studio", "e-studio", "other", "Арбат",
             text="The studio Svet is on Arbat, a 5-minute walk from home.",
             keywords_ru="Арбат, 5 минут пешком, Пресня"),
    ).ok  # fmt: skip
    assert add(conn, fact("f-old", "e-studio", "other", "Пресня", text="Svet was on Presnya.")).ok
    conn.execute("UPDATE records SET status = 'superseded' WHERE id = 'f-old'")
    text = _basket(conn, "Почему мы выбрали студию Свет?")
    assert "5 минут пешком" in text
    assert "Пресня" not in text  # a retired value never rides along in the keywords


# ---------------------------------------------------------------- what the finder must not add


def test_cause_clause_is_context_not_the_asked_relation():
    c = cue.parse(t("С кем мы так и не встретились из-за командировки?"), now=CLOCK)
    assert "wants_to_meet" in c.relations and "travels_to" not in c.relations


def test_past_question_skips_a_plan(conn):
    _home(conn)
    assert add(
        conn,
        fact("f-dorm", "self", "lives_in", "общага", fkind=FKind.PLAN, valid_from="2026-12-01",
             text="From December the owner will live in the dorm."),
    ).ok  # fmt: skip
    text = _basket(conn, "Где мы жили раньше, до всех переездов?")
    assert "общага" not in text and "dorm" not in text


def test_newest_impression_of_a_person_speaks_for_now(conn):
    assert add(conn, entity("e-vera", "Вера")).ok
    for iid, day, en, ru in (
        (
            "i-old",
            "2026-02-01",
            "Vera is often late with tasks.",
            "Вера вечно опаздывает со сдачей",
        ),
        ("i-new", "2026-09-01", "Vera now delivers on time.", "Вера теперь всё сдаёт вовремя"),
    ):
        imp = Impression(
            id=iid, about_entity_id="e-vera", text_en=en, observed_at=f"{day}T10:00:00Z",
            author_tier=AuthorTier.USER, recorded_at=ts(), origin_quote=ru, quote_orig=ru,
        )  # fmt: skip
        assert add(conn, imp).ok
    text = _basket(conn, "Можно поручить Вере срочную задачу?")
    assert "вовремя" in text and "опаздывает" not in text


def test_newer_opinion_fact_overtakes_an_older_impression(conn):
    assert add(conn, entity("e-oleg", "Олег")).ok
    imp = Impression(
        id="i-oleg", about_entity_id="e-oleg", text_en="Oleg rarely answers on time.",
        observed_at="2026-02-01T10:00:00Z", author_tier=AuthorTier.USER, recorded_at=ts(),
        origin_quote="Олег вечно молчит неделями", quote_orig="Олег вечно молчит неделями",
    )  # fmt: skip
    assert add(conn, imp).ok
    assert add(
        conn,
        fact("f-oleg", "e-oleg", "other", "отвечает в тот же день", fkind=FKind.OPINION,
             text="The owner takes it back: Oleg now answers the same day."),
    ).ok  # fmt: skip
    text = _basket(conn, "Олегу можно написать по срочному вопросу?")
    assert "молчит" not in text
