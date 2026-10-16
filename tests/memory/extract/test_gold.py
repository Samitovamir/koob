"""the memory spec §4.1, §4.3, §6.3: unknown stays NULL; preferences carry a scope key;
code drops what the sources do not back; the eval path scores against gold. Gold extractions
stand in for the model. Spec marker: `[T extract/test_gold::<case>]`."""

from __future__ import annotations

import json

from koob.memory.model import AuthorTier, Provenance
from koob.memory.workers import extract, scheduler

from ..workers._kit import SASHA, Gold, batch, fact, raw

DIMS = ("valid_from", "valid_to", "stability", "applicability", "value_entity_id", "keywords_ru")


def test_null_expected_fields(env):
    """§4.1: unknown stays NULL; the extractor invents no values."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе")
    extract.configure(cloud=Gold(SASHA))
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    row = conn.execute(f"SELECT {', '.join(DIMS)}, relation, value_text FROM facts").fetchone()
    assert tuple(row[-2:]) == ("works_at", "Yandex")
    assert all(v is None for v in tuple(row)[: len(DIMS)])  # nothing the message did not say


def test_unbacked_items_dropped(env):
    """A quote not verbatim in its raw, or an invented handle, never reaches the ledger."""
    conn, _ = env
    raw(conn, "r1", "Саша работает в Яндексе")
    snap, rs = extract.snapshot(conn, ["r1"])
    gold = {
        "entities": SASHA["entities"],
        "facts": [
            fact("f1", "e1", "works_at", "Google", "Саша работает в Google", "r1"),
            fact("f2", "m:nobody", "works_at", "Yandex", "Саша работает в Яндексе", "r1"),
            fact("f3", "e1", "works_at", "Yandex", "работает в  Яндексе", "r1"),
        ],
    }
    p = extract.propose(snap, rs, Gold(gold))
    recs = [op.payload["record"] for op in p.ops]
    assert [r["kind"] for r in recs] == ["entity", "fact"]
    assert recs[1]["value_text"] == "Yandex" and p.model_id == "gold-cloud"
    assert len({op.group for op in p.ops}) == 1  # the fact needs its new entity: one group


def test_relayed_claim_and_stated_reason(env):
    """§8.1: «Дима говорит, что …» in the user's message is the quoted person's claim (tier other);
    a `stated` reason on foreign text is `inferred` (§6.3); each op reads only what it used (§7.2)."""
    conn, _ = env
    raw(conn, "r1", "Дима говорит, что Саша переезжает в Тверь")
    raw(conn, "r2", "forwarded: Sasha loves Tver", tier="tool")
    gold = {
        "entities": [
            {"ref": "e1", "etype": "person", "display_name": "Дима"},
            {"ref": "e2", "etype": "person", "display_name": "Саша"},
        ],
        "facts": [
            fact(
                "f1",
                "e2",
                "moves_to",
                "Tver",
                "Саша переезжает в Тверь",
                "r1",
                fkind="plan",
                speaker="e1",
            ),
            fact(
                "f2",
                "e2",
                "likes",
                "Tver",
                "Sasha loves Tver",
                "r2",
                fkind="opinion",
                reasons=[{"member": "r2", "provenance": "stated"}],
            ),
        ],
    }
    snap, rs = extract.snapshot(conn, ["r1", "r2"])
    ops = {op.payload["record"]["id"]: op for op in extract.propose(snap, rs, Gold(gold)).ops}
    f1, f2 = (o for o in ops.values() if o.payload["record"]["kind"] == "fact")
    assert f1.payload["record"]["author_tier"] == AuthorTier.OTHER.value
    assert f2.payload["record"]["author_tier"] == AuthorTier.TOOL.value
    assert f2.payload["justifications"][0]["provenance"] == Provenance.INFERRED.value
    assert {r.id for r in f1.read_set} == {"r1"} and {r.id for r in f2.read_set} == {"r2"}
    again = extract.propose(snap, rs, Gold(gold))
    assert sorted(o.payload["record"]["id"] for o in again.ops) == sorted(ops)  # deterministic


def test_preference_scope_keys(env):
    """§4.3: a preference carries a scope, {general: true} or {unstated: true}."""
    conn, clk = env
    raw(conn, "r1", "Я люблю кинзу. Вообще обожаю грузинскую кухню")
    gold = {
        "facts": [
            fact(
                "p1",
                "self",
                "likes",
                "cilantro",
                "Я люблю кинзу",
                "r1",
                kind="preference",
                fkind="opinion",
            ),
            fact(
                "p2",
                "self",
                "likes",
                "Georgian food",
                "Вообще обожаю грузинскую кухню",
                "r1",
                kind="preference",
                fkind="opinion",
                applicability={"general": True},
            ),
        ]
    }
    extract.configure(cloud=Gold(gold))
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    scopes = {
        v: json.loads(a)
        for v, a in conn.execute(
            "SELECT f.value_text, f.applicability FROM facts f JOIN records r ON r.id = f.id"
            " WHERE r.kind = 'preference'"
        )
    }
    assert scopes == {"cilantro": {"unstated": True}, "Georgian food": {"general": True}}


def test_eval_scores_against_gold():
    """The checkpoint eval path: numbers per case and in total, no store, no model."""
    case = {
        "id": "c1",
        "raws": [{"id": "r1", "text": "Саша работает в Яндексе"}],
        "relations": ["works_at"],
        "gold": SASHA,
    }
    good = extract.evaluate([case], Gold(SASHA))["total"]
    assert (good["precision"], good["recall"], good["dims_accuracy"]) == (1.0, 1.0, 1.0)
    inventing = {**SASHA, "facts": [{**SASHA["facts"][0], "stability": "lasting"}]}
    bad = extract.evaluate([case], Gold(inventing))["total"]
    assert bad["invented"] == 1 and bad["dims_accuracy"] < 1.0
