"""the memory spec §5.3: labels at arrival, refinement and inheritance.
Spec marker: `[T policy/test_labels::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import Label, ReadRef, ReadSrc, Route

from .conftest import owner


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Обострился гастрит, ферритин 9. Врач прописал железо", Label.HEALTH),
        ("Я должен Диме 15 000 рублей", Label.MONEY),
        ("Паспорт продлить до марта, виза до 2027", Label.IDENTITY_DOCS),
        ("Новый код от домофона 5521, квартира 12", Label.LOCATION_PRECISE),
        ("Дима сказал по секрету, что уходит", Label.THIRD_PARTY_PRIVATE),
        ("адвокат по иску позвонит в пятницу", Label.LEGAL),
        ("в субботу везу сына в школу на олимпиаду", Label.MINOR),
        ("my doctor said the allergy is mild", Label.HEALTH),
    ],
)
def test_arrival_lexicons(text, expected):
    assert policy.label(text) & expected


def test_arrival_plain_text_unlabeled():
    assert policy.label("у Саши день рождения 12 мая") == Label.NONE
    assert policy.label("перевод статьи на английский готов") == Label.NONE


def test_arrival_entity_types_and_fail_closed():
    assert policy.label("встреча в среду", entity_types=["minor"]) & Label.MINOR
    # a weak cue alone is uncertain: fail-closed
    assert policy.label("сдал анализ во вторник") == Label.UNKNOWN_SENSITIVE
    # the in-process classifier: high sets the label, uncertain sets unknown_sensitive
    policy.set_classifier(lambda t: {Label.HEALTH: 0.9} if "сахар" in t else {Label.INTIMATE: 0.3})
    assert policy.label("опять скачет сахар") & Label.HEALTH
    assert policy.label("мы с ней снова разговариваем") == Label.UNKNOWN_SENSITIVE


def test_refine_only_adds():
    """§5.3: workers may add labels, never remove them."""
    arrival = policy.label("Обострился гастрит, везу сына к врачу")
    assert arrival & Label.HEALTH and arrival & Label.MINOR
    # a worker proposing fewer labels keeps every arrival bit and adds its own
    refined = policy.refine_labels(arrival, Label.MONEY)
    assert refined == arrival | Label.MONEY
    assert policy.refine_labels(refined, Label.NONE) == refined
    for lbl in policy.bits(arrival):
        assert policy.refine_labels(arrival, arrival & ~lbl) & lbl
    # only the user can remove one (via «исправь»)
    assert policy.refine_labels(refined, Label.MINOR, by_owner=True) == Label.MINOR


def test_summary_inherits_health(db):
    """§5.3: a derived item's labels are the OR of its read set's labels."""
    db.entity("e-self", "Тимур")
    db.entity("e-oksana", "Оксана")
    health = db.fact(
        "f-gastrit",
        subject="e-self",
        text_en="Gastritis flared; ferritin 9",
        quote="Обострился гастрит, ферритин 9",
        labels=Label.HEALTH,
        tier="user",
    )
    plain = db.fact(
        "f-swim",
        subject="e-arman",
        text_en="Takes Arman to swimming on Saturday",
        quote="в субботу везу Армана на плавание",
        tier="other",
        author="e-oksana",
    )
    read_set = [ReadRef(id=health, version=1), ReadRef(id=plain, version=1)]
    assert policy.inherit([Label.HEALTH, Label.NONE, Label.MINOR]) == Label.HEALTH | Label.MINOR
    labels = policy.labels_of(db.conn, read_set)
    assert labels & Label.HEALTH
    db.derived("d-day", read_set, labels=labels, content="Day: gastritis flare; swimming Saturday")
    ref = ReadRef(id="d-day", version=0, src=ReadSrc.DERIVED)
    assert policy.labels_of(db.conn, [ref]) & Label.HEALTH
    # the summary keeps its label downstream: no cloud, and no use on an unrelated turn
    assert Route.CLOUD not in policy.route(db.conn, [ref])
    assert not policy.use_now(db.conn, "d-day", owner("Что посоветуешь на выходные с детьми?"))
    # even a derived row written with too few labels inherits its sources'
    db.derived("d-bad", read_set, labels=Label.NONE)
    assert (
        policy.labels_of(db.conn, [ReadRef(id="d-bad", version=0, src=ReadSrc.DERIVED)])
        & Label.HEALTH
    )


def test_unknown_member_is_fail_closed(db):
    assert policy.labels_of(db.conn, [ReadRef(id="gone", version=1)]) == Label.UNKNOWN_SENSITIVE
