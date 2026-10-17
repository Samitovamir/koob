"""the memory spec §5.5: use now. The user's own items are used wherever they apply, with a
note; the direct-question rule binds only other people's sensitive items; send turns hold
sensitive items for the user's word. Spec marker: `[T policy/test_use_now::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import Label, TurnCtx, TurnOrigin

from .conftest import owner


@pytest.fixture
def world(db):
    db.entity("e-self", "Денис")
    db.entity("e-sasha", "Саша", "Александр", "Sasha")
    db.entity("e-dima", "Дима", "Дмитрий")
    db.entity("e-gleb", "Глеб")
    db.fact(
        "f-allergy",
        subject="e-self",
        text_en="Allergic to nuts",
        quote="у меня аллергия на орехи",
        labels=Label.HEALTH,
        tier="user",
        applicability={"situation": "food", "general": True},
    )
    db.fact("f-jazz", subject="e-self", text_en="Likes jazz", quote="люблю джаз", tier="user")
    db.fact(
        "f-iron",
        subject="e-self",
        text_en="Takes iron for three months",
        quote="пью железо три месяца",
        labels=Label.HEALTH,
        tier="user_confirmed",
    )
    db.fact(
        "f-claim",
        subject="e-self",
        text_en="Dima says Denis has a nut allergy",
        quote="Дима говорит, что у тебя аллергия на орехи",
        labels=Label.HEALTH,
        tier="other",
        author="e-dima",
    )
    db.fact(
        "f-diabetes",
        subject="e-sasha",
        text_en="Sasha has diabetes",
        quote="у меня диабет",
        labels=Label.HEALTH,
        tier="other",
        author="e-sasha",
    )
    db.fact(
        "f-riga",
        subject="e-sasha",
        text_en="Sasha moved to Riga",
        quote="я переехал в Ригу",
        tier="other",
        author="e-sasha",
    )
    db.fact(
        "f-leaving",
        subject="e-sasha",
        text_en="Sasha plans to leave the company",
        quote="скоро уйду из компании",
        labels=Label.THIRD_PARTY_PRIVATE,  # a confident open-topic label
        tier="other",
        author="e-sasha",
    )
    db.fact(
        "f-addr",
        subject="e-gleb",
        text_en="Gleb lives at Lenina 5, apt 12",
        quote="живу на Ленина 5, кв. 12",
        labels=Label.LOCATION_PRECISE,
        tier="other",
        author="e-gleb",
    )
    db.fact(
        "f-home",
        subject="e-self",
        text_en="Denis lives at Pushkina 3",
        quote="я живу на Пушкина 3",
        labels=Label.LOCATION_PRECISE,
        tier="user",
    )
    db._record("l-doctor", "loop", tier="user", labels=Label.HEALTH, final=True, status="active")
    db.conn.execute(
        "INSERT INTO loops (id, owner, text_en, due) VALUES ('l-doctor', 'user',"
        " 'Book a doctor before Friday', '2026-10-16')"
    )
    return db


def _d(world, item, words, **kw):
    return policy.decide(world.conn, item, owner(words, **kw))


def test_own_sensitive_used_where_relevant_with_note(world):
    """§5.5: the user's own sensitive items are used where their applicability matches, with one
    short note («учёл: …»), and never volunteered out of context."""
    food = _d(world, "f-allergy", "Закажи мне пиццу на вечер")
    assert food.allowed and food.note == "учёл: аллергия на орехи" and not food.hold
    assert _d(world, "f-allergy", "Купи торт с орехами к чаю").allowed
    assert _d(world, "f-allergy", "Запиши меня к врачу на пятницу").allowed
    assert _d(world, "f-iron", "Что мне врач прописал по ферритину?").allowed
    # not where it does not apply
    for words in ("Какая завтра погода в Москве?", "Найди билеты в Ригу на субботу"):
        assert not _d(world, "f-allergy", words).allowed
    # the user's non-sensitive items: no fence
    assert _d(world, "f-jazz", "Какая завтра погода?").allowed
    # «Дима говорит, что у тебя…» is Дима's claim, not the user's own item
    assert not _d(world, "f-claim", "Закажи мне пиццу на вечер").allowed
    # neutral turn (core view): the user's own items may sit there; others' sensitive ones may not
    core = TurnCtx(turn_id="core", thread_id="thread-1", owner_words="")
    assert policy.use_now(world.conn, "f-allergy", core)
    assert not policy.use_now(world.conn, "f-diabetes", core)
    # the user's own reminder fires on its event
    fire = TurnCtx(turn_id="ev", thread_id="duty", origin=TurnOrigin.EVENT)
    assert policy.use_now(world.conn, "l-doctor", fire)


def test_because_quote_checks():
    """§5.5: because_quote must be the user's own words and hit the topic or name the target."""
    words = "Напиши Диме, что у Саши аллергия на орехи и сын болеет, пусть не берёт торт"
    turn = owner(words)
    ok = policy.because_quote_ok
    assert ok("у Саши аллергия", turn, labels=Label.HEALTH, target=None)
    assert ok("у САШИ  аллергия", turn, labels=Label.HEALTH, target=None)  # normalized
    assert not ok("у Саши диабет", turn, labels=Label.HEALTH, target=None)  # not the user's words
    assert not ok("пусть не берёт торт", turn, labels=Label.HEALTH, target=None)  # no topic
    assert ok("Напиши Диме", turn, labels=Label.NONE, target="Дима")  # names the target
    assert not ok("Напиши Диме", turn, labels=Label.NONE, target="Саша")
    assert not ok("у Саши аллергия", turn, labels=Label.HEALTH | Label.MINOR, target=None)
    assert ok(
        "аллергия на орехи и сын болеет", turn, labels=Label.HEALTH | Label.MINOR, target=None
    )
    assert not ok("", turn, labels=Label.HEALTH, target="Саша")
    # a weak quote never carries anything (§13.4)
    weak = owner("запомни главное из письма")
    assert not ok("запомни главное", weak, labels=Label.HEALTH, target=None)
    assert not ok("запомни главное", weak, labels=Label.NONE, target=None)
    # fenced foreign text is not the user's words: TurnCtx.owner_words excludes it
    fenced = owner("перешли Диме")
    assert not ok("у Саши аллергия", fenced, labels=Label.HEALTH, target=None)
    # a task turn's words are the task's order
    task = TurnCtx(
        turn_id="t",
        thread_id="task",
        origin=TurnOrigin.TASK,
        task_words="узнай у Саши про аллергию",
    )
    assert ok("про аллергию", task, labels=Label.HEALTH, target=None)


def test_entity_name_alone_unlocks_nothing_sensitive(world):
    """§5.5: for items from other people, chats or files, an entity's name alone unlocks
    only its non-sensitive items."""
    for words in ("Что нового у Саши?", "Расскажи про Александра", "what's up with Sasha"):
        assert _d(world, "f-riga", words).allowed
        v = _d(world, "f-diabetes", words)
        assert not v.allowed and v.reason == "name_only"
        assert not _d(world, "f-leaving", words).allowed
    # topic without the subject is not a direct question either
    assert not _d(world, "f-diabetes", "Как вообще со здоровьем?").allowed
    # asking about the speaker's health never surfaces the speaker's claim about someone else
    assert not _d(world, "f-claim", "Как у Димы со здоровьем?").allowed
    # a direct question: topic lexicon + subject, inflected
    assert _d(world, "f-diabetes", "Как у Саши со здоровьем?").allowed
    assert _d(world, "f-diabetes", "У Сашей был диабет?").allowed
    # an open-topic label: the item's own words are its topic
    assert _d(world, "f-leaving", "Саша уходит из компании?").allowed


def test_send_turn_sensitive_only_on_his_word(world):
    """§5.5: on a send turn sensitive items stay visible, marked «не отправлять без его
    слова», and enter an outward message only when the user's words in this turn ask for them."""
    conn = world.conn
    words = "Напиши Диме, чтобы заказал торт на мой день рождения"
    send = owner(words)
    assert policy.is_send_turn(send)
    v = policy.decide(conn, "f-allergy", send)
    assert v.allowed and v.hold and v.hold_line == "не отправлять без его слова"
    assert v.note == "учёл: аллергия на орехи"
    assert not policy.may_send(conn, "f-allergy", send, None)
    assert not policy.may_send(conn, "f-allergy", send, "заказал торт")
    assert not policy.may_send(conn, "f-allergy", send, "у меня аллергия")  # not the user's words here
    assert policy.may_send(conn, "f-jazz", send, None)  # nothing sensitive: nothing held

    asked = owner("Напиши Диме, чтобы заказал торт без орехов, у меня аллергия")
    v = policy.decide(conn, "f-allergy", asked)
    assert v.allowed and not v.hold
    assert policy.may_send(conn, "f-allergy", asked, "у меня аллергия")

    # others' sensitive items: a send naming the person is no direct question
    assert not policy.decide(conn, "f-diabetes", owner("Напиши Саше, что встреча в семь")).allowed
    told = owner("Напиши Диме, что у Саши диабет и сладкое не надо")
    v = policy.decide(conn, "f-diabetes", told)
    assert v.allowed and not v.hold
    assert policy.may_send(conn, "f-diabetes", told, "у Саши диабет")
    # «напиши мне» is not an outward send
    assert not policy.is_send_turn(owner("напиши мне список покупок"))
    assert policy.is_send_turn(owner("send to Dima the new address"))
    assert not policy.is_send_turn(owner("can I ask you a question about my visa?"))
    assert not policy.decide(conn, "f-allergy", owner("Закажи мне пиццу")).hold


def test_send_turn_others_address_held(world):
    """§5.5: other people's precise places are everyday data, but held on send turns."""
    conn = world.conn
    for words in ("Где живёт Глеб?", "Что нового у Глеба?"):
        v = policy.decide(conn, "f-addr", owner(words))
        assert v.allowed and not v.hold
    send = owner("Напиши Мире, что Глеб переехал")
    v = policy.decide(conn, "f-addr", send)
    assert v.allowed and v.hold
    assert not policy.may_send(conn, "f-addr", send, "Глеб переехал")
    asked = owner("Отправь Мире адрес Глеба")
    v = policy.decide(conn, "f-addr", asked)
    assert v.allowed and not v.hold
    assert policy.may_send(conn, "f-addr", asked, "адрес Глеба")
    # the user's own address is not held
    mine = owner("Напиши Мире, чтобы приехала ко мне домой")
    v = policy.decide(conn, "f-home", mine)
    assert v.allowed and not v.hold


def test_unusable_items_never_used(world):
    world.fact("f-old", subject="e-self", text_en="old", status="superseded", tier="user")
    world.fact("f-forgot", subject="e-self", text_en="x", status="forgotten", tier="user")
    for item in ("f-old", "f-forgot", "missing"):
        assert not _d(world, item, "что угодно").allowed
