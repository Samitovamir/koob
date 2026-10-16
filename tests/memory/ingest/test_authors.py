"""A person's own message names its author by the sender's stable id (§13.3).
The sender is resolved to the entity holding that id, or minted once as a person;
extraction then reads «я» in that message as this person. Synthetic names and ids."""

from __future__ import annotations

import json

from koob.memory import api, entities, ledger
from koob.memory.model import AuthorTier, Entity, StableId
from koob.memory.workers import extract

from ._rows import TS, other, raws


def _row(text: str, *, tg: int, name: str | None, msg_id: int, **kw) -> dict:
    return other(text, chat=f"tg:{tg}", msg_id=msg_id, sender=name, sender_tg_id=tg, **kw)


def _people(conn, tg: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT entity_id FROM entity_stable_ids WHERE kind = 'tg_user_id' AND value = ?",
            (tg,),
        )
    ]


def test_a_persons_message_is_authored_by_the_sender_minted_once(mem):
    ledger.seed_owner(mem)
    api.sweep(rows=[_row("в пятницу улетаю в Тбилиси на неделю", tg=5550101,
                         name="Зарина Валиева", msg_id=1)], store=mem)  # fmt: skip
    api.sweep(rows=[_row("и возьму с собой этюдник, хочу порисовать горы", tg=5550101,
                         name="Зарина Валиева", msg_id=2)], store=mem)  # fmt: skip
    a, b = raws(mem)
    (eid,) = _people(mem, "5550101")  # one person for one id, however many messages
    assert a["author_entity_id"] == b["author_entity_id"] == eid
    assert a["author_tier"] == AuthorTier.OTHER.value  # the author is named; the tier stays hers
    e = mem.execute("SELECT etype, display_name, stable_ids FROM entities WHERE id = ?", (eid,))
    etype, name, sids = e.fetchone()
    assert (etype, name) == ("person", "Зарина Валиева")
    assert {"kind": "tg_user_id", "value": "5550101", "tier": "platform"}.items() <= json.loads(
        sids
    )[0].items()
    # her platform name is a confirmed alias with its other-script forms (§13.3)
    hit = {r.entity_id for r in entities.resolve_rows(mem, "Зарина", "person")}
    assert eid in hit


def test_a_known_contact_resolves_by_id_never_by_name(mem):
    ledger.seed_owner(mem)
    known = Entity(
        id="e-timur", etype="person", display_name="Тимур", author_tier=AuthorTier.USER,
        recorded_at=TS, stable_ids=[StableId(kind="tg_user_id", value="5550102", tier="platform")],
    )  # fmt: skip
    from .. import _ledger as L

    L.add(mem, known)
    api.sweep(rows=[_row("я переехал на Васильевский", tg=5550102, name="Тим К.", msg_id=3)],
              store=mem)  # fmt: skip
    # the same display name from another id is another person, never merged by the name
    api.sweep(rows=[_row("я завтра на смене до ночи", tg=5550199, name="Тимур", msg_id=4)],
              store=mem)  # fmt: skip
    first, second = raws(mem)
    assert first["author_entity_id"] == "e-timur"
    assert second["author_entity_id"] not in (None, "e-timur")
    assert _people(mem, "5550102") == ["e-timur"]


def test_a_username_reads_as_a_name_and_no_name_or_a_forward_mints_nobody(mem):
    ledger.seed_owner(mem)
    api.sweep(rows=[_row("я опоздаю минут на десять", tg=5550103, name="lev_marten", msg_id=5)],
              store=mem)  # fmt: skip
    api.sweep(rows=[_row("я на месте, жду у входа", tg=5550104, name=None, msg_id=6)],
              store=mem)  # fmt: skip
    api.sweep(rows=[_row("я продаю велосипед недорого", tg=5550105, name="Ольга Нестерова",
                         msg_id=7, fwd_from="channel:market")], store=mem)  # fmt: skip
    named, unnamed, forwarded = raws(mem)
    eid = named["author_entity_id"]
    name = mem.execute("SELECT display_name FROM entities WHERE id = ?", (eid,)).fetchone()[0]
    assert name == "Lev Marten"
    handles = {r[0] for r in mem.execute("SELECT alias FROM aliases WHERE entity_id = ?", (eid,))}
    assert "lev_marten" in handles
    assert unnamed["author_entity_id"] is None and not _people(mem, "5550104")  # no stand-in
    assert forwarded["author_entity_id"] is None and not _people(mem, "5550105")


def test_the_model_reads_the_authors_handle_and_name(mem):
    ledger.seed_owner(mem)
    api.sweep(rows=[_row("я наконец-то сдала на права!", tg=5550106, name="Вера Лис", msg_id=8)],
              store=mem)  # fmt: skip
    (r,) = raws(mem)
    snap, _ = extract.snapshot(mem, [r["id"]])
    (msg,) = snap["raws"]
    assert msg["speaker"] == r["author_entity_id"]
    text = extract.user_text(snap)
    assert f'"author": "m:{r["author_entity_id"]}"' in text and '"author_name": "Вера Лис"' in text


def test_sender_name_and_ids_are_platform_metadata_only():
    assert entities.sender_stable_ids("tg:42") == [("tg_user_id", "42")]
    assert entities.sender_stable_ids("Nika.Rowe@Example.org") == [
        ("email", "nika.rowe@example.org")
    ]
    assert entities.sender_stable_ids("tg:-1001234") == []  # a group or a channel peer
    assert entities.sender_stable_ids("@nika") == []  # a handle is not a stable id
    assert entities.sender_name("@nika_rowe") == "Nika Rowe"
    assert entities.sender_name("Ника") == "Ника"
    assert entities.sender_name("unknown:courier") is None
    assert entities.sender_name("ID-55501") is None
    assert entities.sender_name("  ") is None
