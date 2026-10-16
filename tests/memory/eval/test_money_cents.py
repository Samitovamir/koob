"""Money with cents in every usual written form, and the currency check.

The same expansion serves must_include and must_not_include, so a tighter or looser
matcher moves every system under test alike."""

from __future__ import annotations

import pytest

from koob.memory.eval.matchers import find
from koob.memory.eval.schema import Expect

CASES = [
    # cents, both decimal marks, sign before or after, grouped thousands
    ("$3.50", {"money": 3.5, "cur": "USD"}, True),
    ("3,50 $", {"money": 3.5, "cur": "USD"}, True),
    ("3.5", {"money": 3.5}, True),
    ("1 200,50 ₽", {"money": "1200.50", "cur": "RUB"}, True),
    ("1,200.50 rub", {"money": 1200.5, "cur": "RUB"}, True),
    ("1.200,50", {"money": 1200.5}, True),
    ("€ 12,05", {"money": 12.05, "cur": "EUR"}, True),
    ("3 доллара 50 центов", {"money": 3.5, "cur": "USD"}, True),
    ("1 200 руб. 50 коп.", {"money": 1200.5, "cur": "RUB"}, True),
    ("3.55", {"money": 3.5}, False),
    ("13.50", {"money": 3.5}, False),
    # a whole amount is not the head of a decimal or of a bigger number
    ("3.00", {"money": 3}, True),
    ("$3.50", {"money": 3}, False),
    ("52,000", {"money": 52}, False),
    ("1 200 000", {"money": 1200}, False),
    ("1 200 000", {"money": 200}, False),
    ("1 200 000", {"money": 1200000}, True),
    ("152000", {"money": 52000}, False),
    # the README's forms keep working
    ("52 000 руб", {"money": 52000, "cur": "RUB"}, True),
    ("52k", {"money": 52000}, True),
    ("52 тыс", {"money": 52000, "cur": "RUB"}, True),
    ("3,5k", {"money": 3500}, True),
    ("1.5 млн", {"money": 1500000}, True),
    ("6000", {"money": 6000, "cur": "RUB"}, True),
    # another currency's mark is not this amount; an unmarked amount still is
    ("$3.50", {"money": 3.5, "cur": "RUB"}, False),
    ("3.50 €", {"money": 3.5, "cur": "USD"}, False),
    ("3 доллара 50 центов", {"money": 3.5, "cur": "RUB"}, False),
    ("$700", {"money": 700, "cur": "RUB"}, False),
    ("700 р", {"money": 700, "cur": "USD"}, False),
    ("700 раз", {"money": 700, "cur": "USD"}, True),
    ("52 000 ₽ / $700", {"money": 52000, "cur": "RUB"}, True),
]


@pytest.mark.parametrize(("text", "value", "want"), CASES)
def test_money_forms(text, value, want):
    assert find(text, value) is want


def test_decimal_money_loads_and_must_not_catches_it_in_any_form():
    e = Expect(must_include=[{"money": 3.5, "cur": "USD"}], must_not_include=[{"money": 4.2}])
    assert e.must_include and e.must_not_include
    assert find("it was 4,20 now", {"money": 4.2})


def test_bad_money_is_rejected():
    with pytest.raises(ValueError):
        find("x", {"money": "3.5.1"})
