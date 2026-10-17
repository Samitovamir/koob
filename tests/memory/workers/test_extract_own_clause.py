"""build() settles a fact from the words of its own clause, never from the model's input.
Synthetic wordings in both languages."""

from __future__ import annotations

from koob.memory.model import OWNER_ENTITY_ID
from koob.memory.workers import extract as X


def _rec(quote: str, value: str, *, kind: str = "preference", **kw) -> dict:
    rec = {
        "kind": kind,
        "subject_id": OWNER_ENTITY_ID,
        "relation": "drinks",
        "value_text": value,
        "value_norm": X.norm(value),
        "quote_orig": quote,
        "fkind": "opinion" if kind == "preference" else "state",
        "text_en": f"The owner: {value}",
    }
    if kind == "preference":
        rec["applicability"] = {"unstated": True}
    rec.update(kw)
    return rec


def test_what_he_buys_for_someone_else_gets_that_scope():
    r = _rec("Сам пью эспрессо, а для тёщи беру ромашковый чай", "ромашковый чай")
    X._settle_words(r, r["quote_orig"], [], [])
    assert r["applicability"] == {"for_whom": "для тёщи"}
    own = _rec("Сам пью эспрессо, а для тёщи беру ромашковый чай", "эспрессо")
    X._settle_words(own, own["quote_orig"], [], [])
    assert own["applicability"] == {"unstated": True}  # the user's own clause stays the user's


def test_for_other_by_dative_name_and_english():
    assert X.for_other("Племяннику заказываю овсяное печенье", "овсяное печенье")
    assert X.for_other("I pick up oat milk for my roommate", "oat milk") == "for my roommate"
    assert X.for_other("Взял сыр для Гоши", "сыр", ["Гоша"]) == "для Гоши"
    assert X.for_other("Для поездки беру термос", "термос", ["Гоша"]) is None
    assert X.for_other("Bought a planner for March", "planner", ["Mira"]) is None
    assert X.for_other("Купил себе кроссовки", "кроссовки") is None


def test_someone_elses_subject_is_left_alone():
    r = _rec("для сестры беру соевое", "соевое", subject_id="e-sister")
    X._settle_words(r, r["quote_orig"], [], [])
    assert r["applicability"] == {"unstated": True}


def test_this_time_only_is_a_one_off():
    r = _rec("Сегодня без сахара, что-то не хочется", "без сахара")
    X._settle_words(r, r["quote_orig"], [], [])
    assert r["stability"] == "one_off"
    lasting = _rec("С сегодняшнего дня пью без сахара", "без сахара")
    X._settle_words(lasting, lasting["quote_orig"], [], [])
    assert "stability" not in lasting
    habit = _rec("Today I walked instead of taking the tram", "walked", kind="fact")
    X._settle_words(habit, habit["quote_orig"], [], [])
    assert habit["stability"] == "one_off"
    plain = _rec("Today I moved to a new flat", "new flat", kind="fact")
    X._settle_words(plain, plain["quote_orig"], [], [])
    assert "stability" not in plain  # a fact needs a contrast to read as an exception


def test_a_bare_amount_in_thousands_gets_its_number():
    r = _rec("Лимит на продукты был три тысячи, теперь полтора", "полтора", kind="fact")
    X._settle_words(r, r["quote_orig"], [], [])
    assert r["value_norm"] == "1500" and "1500" in r["text_en"]
    assert r["value_text"] == "полтора"  # the model's value (read by later batches) stays
    slot = [{"subject": f"m:{OWNER_ENTITY_ID}", "relation": "drinks", "value_text": "$2,200"}]
    d = _rec("it's 2.5 now", "2.5", kind="fact")
    X._settle_words(d, d["quote_orig"], [], slot)
    assert d["value_norm"] == "2500"
    hours = _rec("Стоит 4000, идёт полтора часа", "полтора часа", kind="fact")
    X._settle_words(hours, hours["quote_orig"], [], [])
    assert hours["value_norm"] == X.norm("полтора часа")
    alone = _rec("теперь полтора", "полтора", kind="fact")
    X._settle_words(alone, alone["quote_orig"], [], [])
    assert alone["value_norm"] == "полтора"
