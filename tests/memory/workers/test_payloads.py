"""the memory spec §6.5 (v0.4): the payload shape of every op kind and the extraction
output, as typed models the ledger accepts. Spec marker: `[T workers/test_payloads::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory.model import (
    OP_PAYLOADS,
    AddPayload,
    AliasIn,
    DerivedIn,
    Extraction,
    FlagPayload,
    ForgetPayload,
    JustificationIn,
    LinkPayload,
    MergePayload,
    OpKind,
    ProposalOp,
    RetractPayload,
    SplitPayload,
    SupersedePayload,
    UpdatePayload,
    payload_dict,
    validate_payload,
)

from .. import _ledger as L


def _op(kind: OpKind, payload, target: str | None = "x") -> ProposalOp:
    return ProposalOp(op=kind, target=target, group="g", payload=payload_dict(payload))


def test_every_op_kind_validates():
    examples = {
        OpKind.ADD: AddPayload(record={"kind": "fact"}, derived_from=["r1"]),
        OpKind.UPDATE: UpdatePayload(
            fields={"importance": 0.4}, add_aliases=[AliasIn(alias="Шурик")]
        ),
        OpKind.SUPERSEDE: SupersedePayload(by="f2", fix=True),
        OpKind.RETRACT: RetractPayload(),
        OpKind.FLAG: FlagPayload(status="archived"),
        OpKind.LINK: LinkPayload(from_id="a", to_id="b", etype="mentions"),
        OpKind.MERGE: MergePayload(drop="e2"),
        OpKind.SPLIT: SplitPayload(record_ids=["f1"], display_name="Alexander K."),
        OpKind.FORGET: ForgetPayload(),
    }
    assert set(examples) == set(OpKind)
    for kind, payload in examples.items():
        assert type(validate_payload(_op(kind, payload))) is OP_PAYLOADS[kind]
    assert "review" not in payload_dict(FlagPayload(status="archived"))
    derived = AddPayload(derived=DerivedIn(dkind="summary", content="Day summary"))
    assert validate_payload(_op(OpKind.ADD, derived, None))
    for bad in (
        lambda: AddPayload(),
        lambda: AddPayload(record={}, derived=DerivedIn(dkind="summary", content="x")),
        lambda: SupersedePayload(),
        lambda: FlagPayload(),
        lambda: LinkPayload(from_id="a"),
        lambda: LinkPayload(alias=AliasIn(alias="Саша"), from_id="a", to_id="b", etype="about"),
    ):
        with pytest.raises(ValueError):
            bad()


def test_typed_payloads_commit_through_the_ledger(tmp_path):
    conn = L.open_seeded(tmp_path)
    assert L.add(conn, L.entity("e-sasha", "Саша")).ok
    L.raw(conn, "r1", "у Саши день рождения 12 мая")
    old = L.fact(
        "f1", "e-sasha", "birthday", "05-12", source_raw_id="r1", quote_orig="день рождения 12 мая"
    )
    new = L.fact("f2", "e-sasha", "birthday", "05-14", origin_quote="нет, 14 мая")

    def commit(kind, payload, target=None):
        return L.commit(
            conn,
            ProposalOp(op=kind, target=target, group="e-sasha", payload=payload_dict(payload)),
        )

    assert commit(OpKind.ADD, AddPayload(record=old.model_dump(mode="json"))).ok
    fix = SupersedePayload(record=new.model_dump(mode="json"), fix=True)
    assert commit(OpKind.SUPERSEDE, fix, "f1").ok
    assert L.status(conn, "f1") == "superseded"
    member = JustificationIn(member="r1", provenance="rule")
    assert commit(OpKind.LINK, LinkPayload(justification=member), "f2").ok
    assert commit(OpKind.FLAG, FlagPayload(review=True), "f2").ok


def test_extraction_gold_shape():
    gold = {
        "entities": [
            {"ref": "e1", "etype": "person", "display_name": "Sasha", "aliases": ["Саша"]}
        ],
        "facts": [
            {
                "ref": "f1",
                "subject": "e1",
                "relation": "moves_to",
                "value_text": "Tver",
                "text_en": "Sasha plans to move to Tver",
                "quote_orig": "собирается переехать в Тверь",
                "fkind": "plan",
                "speaker": "e1",
                "source_raw_id": "r1",
                "stability": None,
            }
        ],
        "impressions": [],
    }
    x = Extraction.model_validate(gold)
    assert x.facts[0].valid_to is None and x.facts[0].polarity.value == "affirmed"
    with pytest.raises(ValueError):
        Extraction.model_validate({"facts": [{**gold["facts"][0], "quote_orig": None}]})
