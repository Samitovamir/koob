"""memory_extract@4: a retraction takes the author's own claim back and is never a fact; a pet
or thing the user names is an entity that the user's common noun for it reaches. A gold
processor stands in for the model; synthetic names."""

from __future__ import annotations

from koob.memory import entities
from koob.memory.model import OWNER_ENTITY_ID, AuthorTier, Extraction, OpKind
from koob.memory.workers import extract

from .. import _ledger as L
from ._kit import Gold, batch, fact

TAKE_BACK = "Про Мари беру свои слова назад, я всё перепутал."


def _his_known_fact(conn) -> None:
    L.add(conn, L.entity("e-mari", "Мари", labels_final=True))
    L.add(
        conn,
        L.fact(
            "f-mari-job", "e-mari", "works_at", "Сбер", text="Mari works at Sber", labels_final=True
        ),
    )


def _run(conn, clk, gold, raw_ids: list[str]) -> None:
    extract.configure(cloud=Gold(gold), local=Gold(gold, route="local"))
    batch(conn, raw_ids)
    from koob.memory.workers import scheduler

    scheduler.tick(conn, clk)


def test_his_retraction_puts_his_claim_on_hold_and_is_no_fact(env):
    conn, clk = env
    _his_known_fact(conn)
    L.raw(conn, "r1", TAKE_BACK)
    conn.execute("UPDATE raw_chunks SET turn_id = 't1' WHERE id = 'r1'")
    conn.commit()

    def gold(snap: dict) -> dict:
        (known,) = [f for f in snap["facts"] if f["id"] == "f-mari-job"]
        assert known["speaker"] == OWNER_ENTITY_ID  # code only: never in the model's input
        assert '"speaker"' not in extract.user_text(snap)
        return {
            "facts": [  # the model also wrote the retraction down as a statement
                fact(
                    "f1",
                    "self",
                    "other",
                    "retracts what he said about Mari",
                    "беру свои слова назад",
                    "r1",
                ),
            ],  # fmt: skip
            "retractions": [
                {
                    "ref": "x1",
                    "targets": [known["handle"]],
                    "quote_orig": "беру свои слова назад",
                    "source_raw_id": "r1",
                },
            ],  # fmt: skip
        }

    _run(conn, clk, gold, ["r1"])
    assert L.status(conn, "f-mari-job") == "needs_review"  # no longer usable, nothing shown
    rows = conn.execute(
        "SELECT f.text_en FROM facts f JOIN records r ON r.id = f.id WHERE r.status = 'active'"
    ).fetchall()
    assert [r[0] for r in rows] == []


def _known(fid: str, subject: str, speaker: str, tier: str) -> dict:
    return {"handle": f"m:{fid}", "id": fid, "subject": f"m:{subject}", "relation": "works_at",
            "value_text": "Сбер", "version": 1, "tier": tier, "speaker": speaker}  # fmt: skip


def test_a_foreign_retraction_reaches_only_the_authors_own_claim():
    raws = [{"id": "r1", "text": "забудь, что я писала про Мари", "tier": "other",
             "speaker": "e-pat"}]  # fmt: skip
    known = [
        _known("f-his", "e-mari", OWNER_ENTITY_ID, "user"),
        _known("f-hers", "e-mari", "e-pat", "other"),
    ]
    snap = extract.snapshot_of(raws, relations=["works_at"], facts=known)
    gold = {"retractions": [{"ref": "x1", "targets": ["m:f-his", "m:f-hers"],
                             "quote_orig": "забудь, что я писала", "source_raw_id": "r1"}]}  # fmt: skip
    built = extract.build(Extraction.model_validate(gold), snap)
    flags = [(o.target, o.payload) for o in built.ops if o.op is OpKind.FLAG]
    assert flags == [("f-hers", {"status": "needs_review"})]
    assert "retraction_not_theirs:x1" in built.dropped


def test_an_earlier_claim_of_the_batch_taken_back_is_not_written():
    raws = [
        {
            "id": "r1",
            "text": "Курс керамики у Мари стоит девять тысяч",
            "ts": "2026-10-01T10:00:00Z",
        },
        {"id": "r2", "text": "Нет, про курс беру свои слова назад", "ts": "2026-10-01T10:05:00Z"},
        {"id": "r3", "text": "Курс керамики у Мари по субботам", "ts": "2026-10-01T10:09:00Z"},
    ]
    snap = extract.snapshot_of(raws, relations=["other"])
    gold = {
        "facts": [
            fact("f1", "self", "other", "девять тысяч", "стоит девять тысяч", "r1"),
            fact("f3", "self", "other", "по субботам", "по субботам", "r3"),
        ],
        "retractions": [{"ref": "x1", "targets": ["f1", "f3"],
                         "quote_orig": "беру свои слова назад", "source_raw_id": "r2"}],
    }  # fmt: skip
    built = extract.build(Extraction.model_validate(gold), snap)
    written = [o.payload["record"]["value_text"] for o in built.ops if o.op is OpKind.ADD]
    assert written == ["по субботам"]  # a later message is not taken back by an earlier one
    assert "retracted_in_batch:f1" in built.dropped


def test_a_named_pet_is_an_entity_his_word_for_it_reaches(env):
    conn, clk = env
    L.raw(conn, "r1", "Моя кошка Пломбир третий день не ест корм.")
    conn.execute("UPDATE raw_chunks SET turn_id = 't1' WHERE id = 'r1'")
    conn.commit()
    gold = {
        "entities": [{"ref": "e1", "etype": "pet", "display_name": "Пломбир",
                      "kind_words": ["кошка", "cat", "ТХ-2041"]}],
        "facts": [fact("f1", "self", "has_pet", "кошка Пломбир", "Моя кошка Пломбир", "r1",
                       value_entity="e1")],
    }  # fmt: skip
    _run(conn, clk, gold, ["r1"])
    (eid, etype) = conn.execute(
        "SELECT id, etype FROM entities WHERE display_name = 'Пломбир'"
    ).fetchone()
    assert etype == "pet"
    names = {
        r[0] for r in conn.execute("SELECT alias_norm FROM aliases WHERE entity_id = ?", (eid,))
    }
    assert "кошка" in names and "cat" not in names and not any("2041" in n for n in names)
    # «отвезти кошку к ветеринару»: the user's common noun, inflected, reaches the pet
    assert eid in {r.entity_id for r in entities.resolve_rows(conn, "кошку", "any")}
    (tier,) = conn.execute("SELECT author_tier FROM records WHERE id = ?", (eid,)).fetchone()
    assert tier == AuthorTier.USER.value
