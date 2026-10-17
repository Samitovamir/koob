"""build(): the side a correction corrects away is never a new fact, and a later change the
model returned without `replaces` names the earlier record. Synthetic names and wordings."""

from __future__ import annotations

from koob.memory.model import Extraction
from koob.memory.workers import extract

from ._kit import fact

MELNIK = [
    {"handle": "m:ent_m", "id": "ent_m", "etype": "person", "display_name": "Гоша Мельник",
     "version": 1}
]  # fmt: skip
SEMINAR = [
    {"handle": "m:ent_s", "id": "ent_s", "etype": "thing", "display_name": "семинар по керамике",
     "version": 1}
]  # fmt: skip


def _ops(built, kind: str) -> list[dict]:
    return [o.payload for o in built.ops if o.payload.get("record", {}).get("kind") == kind]


def _person(ref: str, name: str, kind_words: list[str]) -> dict:
    return {"ref": ref, "etype": "person", "display_name": name, "aliases": [name],
            "kind_words": kind_words}  # fmt: skip


def test_the_corrected_away_side_is_no_fact_and_no_person():
    text = "Стоп, отчёт по продажам прислал Гоша Мельник, мой брат, а не Гоша Мельников, тренер."
    snap = extract.snapshot_of([{"id": "r1", "text": text}], relations=["sibling_of"],
                               entities=MELNIK)  # fmt: skip
    gold = {
        "entities": [_person("e1", "Гоша Мельников", ["тренер"])],
        "facts": [
            fact("f1", "m:ent_m", "other", "отчёт по продажам", text[6:-1], "r1", fkind="event"),
            fact(
                "f2",
                "m:ent_m",
                "sibling_of",
                None,
                "Гоша Мельник, мой брат",
                "r1",
                value_entity="self",
            ),  # fmt: skip
            fact("f3", "e1", "other", "тренер", "Гоша Мельников, тренер", "r1"),
        ],
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    kept = {p["record"]["quote_orig"] for p in _ops(built, "fact")}
    assert "Гоша Мельников, тренер" not in kept and len(kept) == 2  # the true side stays
    assert not _ops(built, "entity")  # no new «Мельников (aka тренер)»
    assert "corrected_away:f3" in built.dropped
    assert "corrected_away_entity:e1" in built.dropped


def test_the_old_side_first_form_and_its_kind_word():
    text = "Это была не тётя Лиза, а сестра Мила."
    snap = extract.snapshot_of([{"id": "r1", "text": text}], relations=["sibling_of"])
    gold = {
        "entities": [_person("e1", "Лиза", ["тётя"]), _person("e2", "Мила", ["сестра"])],
        "facts": [
            fact("f1", "e1", "other", "тётя", "тётя Лиза", "r1"),
            fact("f2", "e2", "sibling_of", None, "сестра Мила", "r1", value_entity="self"),
        ],
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    (kept,) = _ops(built, "fact")
    assert kept["record"]["quote_orig"] == "сестра Мила"
    (mila,) = _ops(built, "entity")
    assert mila["record"]["display_name"] == "Мила"
    assert [a["alias"] for a in mila["aliases"] if a.get("kind") == "nickname"] == ["сестра"]


def test_a_person_a_kept_record_names_is_still_created():
    text = "Отчёт прислал Гоша Мельник, а не Гоша Мельников."
    snap = extract.snapshot_of([{"id": "r1", "text": text}], entities=MELNIK)
    gold = {
        "entities": [_person("e1", "Гоша Мельников", [])],
        "facts": [
            fact(
                "f1",
                "e1",
                "other",
                "отчёт",
                "не Гоша Мельников",
                "r1",
                polarity="negated",
                fkind="event",
            ),  # fmt: skip
        ],
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    assert len(_ops(built, "fact")) == 1 and len(_ops(built, "entity")) == 1


def test_not_only_and_the_new_side_stand():
    snap = extract.snapshot_of(
        [
            {"id": "r1", "text": "Nadia speaks not only Greek but Turkish."},
            {"id": "r2", "text": "Нет, Надя живёт не в Омске, а в Томске."},
        ],
        relations=["speaks", "lives_in"],
    )
    gold = {
        "entities": [_person("e1", "Nadia", []), _person("e2", "Надя", [])],
        "facts": [
            fact("f1", "e1", "speaks", "Greek", "only Greek", "r1"),
            fact("f2", "e1", "speaks", "Turkish", "Turkish", "r1"),
            fact("f3", "e2", "lives_in", "Томск", "в Омске, а в Томске", "r2"),
        ],
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    assert {p["record"]["value_text"] for p in _ops(built, "fact")} == {"Greek", "Turkish", "Томск"}
    assert not [d for d in built.dropped if d.startswith("corrected_away")]


def _event(ref: str, raw: str, value: str, quote: str, **kw) -> dict:
    return fact(ref, "m:ent_s", "other", value, quote, raw, **kw)


def test_a_later_change_names_the_earlier_record_in_the_batch():
    snap = extract.snapshot_of(
        [
            {
                "id": "r1",
                "text": "Семинар по керамике в Твери, 4–5 марта.",
                "ts": "2026-02-01T10:00:00Z",
            },
            {
                "id": "r2",
                "text": "Семинар по керамике теперь в Твери, 11–12 марта.",
                "ts": "2026-02-09T10:00:00Z",
            },
        ],  # fmt: skip
        entities=SEMINAR,
    )
    gold = {
        "facts": [
            _event("f2", "r2", "Тверь, 11–12 марта", "в Твери, 11–12 марта"),
            _event("f1", "r1", "Тверь, 4–5 марта", "в Твери, 4–5 марта"),
        ]
    }
    built = extract.build(Extraction.model_validate(gold), snap)
    facts = {p["record"]["value_text"]: p for p in _ops(built, "fact")}
    assert facts["Тверь, 11–12 марта"]["model_matches"] == [
        facts["Тверь, 4–5 марта"]["record"]["id"]
    ]
    assert facts["Тверь, 4–5 марта"]["model_matches"] == []  # never backwards in time


def _known(value: str, *, speaker: str = "self", tier: str = "user") -> list[dict]:
    return [{"handle": "m:fk", "id": "fk", "subject": "m:ent_s", "relation": "other",
             "value_text": value, "version": 1, "tier": tier, "speaker": speaker}]  # fmt: skip


def test_a_later_change_names_his_known_record_only():
    raws = [{"id": "r2", "text": "Семинар по керамике в Твери, 11–12 марта."}]
    gold = {"facts": [_event("f2", "r2", "Тверь, 11–12 марта", "в Твери, 11–12 марта")]}
    x = Extraction.model_validate(gold)

    def matches(known: list[dict]) -> list[str]:
        snap = extract.snapshot_of(raws, entities=SEMINAR)
        snap["facts"] = known
        (only,) = _ops(extract.build(x, snap), "fact")
        return only["model_matches"]

    assert matches(_known("Тверь, 4–5 марта")) == ["fk"]
    # another person's claim is never named, a different slot («1 марта» alone) neither
    assert matches(_known("Тверь, 4–5 марта", speaker="ent_x", tier="other")) == []
    assert matches(_known("1 марта")) == []
    assert matches(_known("Тверь, 11–12 марта")) == []  # the same value is no change
