from __future__ import annotations

from pathlib import Path

import pytest

from koob.memory.eval.loader import load_set
from koob.memory.eval.matchers import find
from koob.memory.eval.noise import expand
from koob.memory.eval.schema import Noise

DEV = Path(__file__).resolve().parents[3] / "evals" / "memory" / "dev"


@pytest.mark.parametrize(
    "text,val,ok",
    [
        ("Это 17 мая", {"date": "2026-05-17"}, True),
        ("May 17th", {"date": "2026-05-17"}, True),
        ("17.05", {"date": "--05-17"}, True),
        ("27 мая", {"date": "2026-05-17"}, False),
        ("2 ноября", {"date": "--11-02"}, True),
        ("52 000 руб", {"money": 52000, "cur": "RUB"}, True),
        ("52k", {"money": 52000}, True),
        ("152000", {"money": 52000}, False),
        ("в 16.00", {"time": "16:00"}, True),
        ("Ёлка", "елка", True),
        ("Кинза", ["cilantro", "кинза"], True),
        ("nothing", ["cilantro", "кинза"], False),
    ],
)
def test_find(text, val, ok):
    assert find(text, val) is ok


def _noise(**kw):
    base = {
        "ev": "noise", "seed": 4101, "count": 60,
        "span": {"from": "2026-01-05", "to": "2026-03-01"},
        "chats": {"tg:family": 0.5, "island": 0.5},
        "kinds": {"smalltalk": 0.3, "distractor_plan": 0.4, "namesake": 0.3},
        "cast": ["dima"],
    }  # fmt: skip
    base.update(kw)
    return Noise.model_validate(base)


CONTACTS = {"dima": {"ru": "Дима Орлов", "en": "Dmitry Orlov"}}


def test_noise_is_deterministic_and_seed_sensitive():
    a, b = expand(_noise(), contacts=CONTACTS), expand(_noise(), contacts=CONTACTS)
    assert a == b and len(a) == 60
    c = expand(_noise(seed=4102), contacts=CONTACTS)
    assert [m.text for m in c] != [m.text for m in a]


def test_noise_scale_and_span():
    base = expand(_noise(), contacts=CONTACTS)
    ten = expand(_noise(), scale="10x", contacts=CONTACTS)
    assert len(ten) == 600 and len(base) == 60
    assert all("2026-01-05" <= m.at.date().isoformat() <= "2026-03-01" for m in ten)


def test_noise_respects_protect():
    n = _noise(
        protect={"values": ["Орлов", "Дима"], "slots": []}, kinds={"smalltalk": 1, "namesake": 1}
    )
    assert all("Дима" not in m.text and "Орлов" not in m.text for m in expand(n, contacts=CONTACTS))
    slot = _noise(
        kinds={"distractor_plan": 1},
        protect={"values": [], "slots": [{"subject": "dima", "family": "travel"}]},
    )
    texts = [m.text for m in expand(slot, contacts=CONTACTS)]
    assert all("Дима" not in t for t in texts if "едет" in t or "билеты" in t)


def test_dev_noise_directives_expand():
    s = load_set(DEV)
    n = 0
    for sc in s.scenarios:
        for e in sc.timeline:
            if e.ev == "noise":
                out = expand(e, contacts=s.personas[sc.persona]["contacts"])
                assert len(out) == e.count
                n += 1
    assert n >= 10
