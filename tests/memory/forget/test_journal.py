"""Acceptance for the memory spec §9.3: the deletion journal.
Spec marker: `[T forget/test_journal::<case>]`."""

from __future__ import annotations

import json

import pytest

from koob.memory import forget, store
from koob.memory.model import ForgetScope, ForgetStage, JournalLine

from . import _seed as s

clock = s.clock  # fixture

CANARY = ("QZXJOURNAL7", "ЖУРНАЛКАНАРЕЙКА", "86753094")


def _seed_sasha(mem) -> None:
    s.relation(mem, "birthday", "birthday")
    s.entity(mem, "ent-sasha", "Саша", [("Саша", "name"), ("Sasha", "translit")])
    s.raw(mem, "raw-1", f"Запомни: у Саши день рождения 12 мая {CANARY[1]}", turn="1")
    s.fact(
        mem,
        "f-bday",
        "ent-sasha",
        "birthday",
        value=f"12 мая {CANARY[2]}",
        text_en=f"Sasha's birthday is May 12 {CANARY[0]}",
        quote=f"у Саши день рождения 12 мая {CANARY[1]}",
        raw="raw-1",
    )


def test_written_at_stop_use(mem, clock):
    """§9.3: the journal line is written at stop-use, not at purge."""
    _seed_sasha(mem)
    fid = forget.stop_use(mem, ["f-bday"], scope=ForgetScope.ITEM, forever=False)
    path = forget.journal_path(mem)
    assert path is not None and path.parent.name == "journal" and path.name == "forget.jsonl"
    lines = forget.read_journal(path)
    # The write-ahead `stopping` line, then the full `stopped` line
    assert [x.stage for x in lines] == [ForgetStage.STOPPING, ForgetStage.STOPPED]
    assert lines[0].target_ids == ["f-bday"] and lines[0].scope is ForgetScope.ITEM
    line = lines[1]
    assert line.stage is ForgetStage.STOPPED and line.forget_id == fid
    assert "f-bday" in line.record_ids
    raw_hmac = mem.execute("SELECT content_hmac FROM raw_chunks WHERE id = 'raw-1'").fetchone()[0]
    assert line.raw_hmacs == [raw_hmac]
    # The family is the vocabulary's (store.open seeds config/memory/relations.yaml).
    fam = mem.execute("SELECT family FROM relations WHERE relation = 'birthday'").fetchone()[0]
    sup = forget.suppression_hmac("ent-sasha", fam, f"12 мая {CANARY[2]}".casefold())
    assert sup in line.suppression_hmacs
    assert line.memory_version == store.memory_version(mem)
    assert forget.status(mem, fid) == "in_trash"  # the purge waits in the trash (24 h)


def test_no_content(mem, clock):
    """§9.3: journal lines hold ids and HMACs only, through every stage."""
    _seed_sasha(mem)
    fid = forget.stop_use(mem, ["f-bday"], scope=ForgetScope.ITEM, forever=False)
    forget.undo(mem, fid)
    fid2 = forget.stop_use(mem, ["f-bday"], scope=ForgetScope.ITEM, forever=True)
    assert forget.purge(mem, fid2) is True
    path = forget.journal_path(mem)
    stages = [x.stage for x in forget.read_journal(path)]
    assert stages == [
        ForgetStage.STOPPING,
        ForgetStage.STOPPED,
        ForgetStage.UNDONE,
        ForgetStage.STOPPING,
        ForgetStage.STOPPED,
        ForgetStage.PURGED,
    ]
    assert s.hits([path], [*CANARY, "Саши", "Sasha", "мая"]) == []
    for raw in path.read_text(encoding="utf-8").splitlines():
        assert set(json.loads(raw)) <= set(JournalLine.model_fields)


def test_text_refused_by_append():
    """A caller cannot slip content into the journal through an id field."""
    line = JournalLine(
        ts="2026-10-12T09:00:00.000000Z",
        stage=ForgetStage.STOPPED,
        forget_id="fg-1",
        record_ids=["у Саши день рождения"],
        memory_version=1,
    )
    with pytest.raises(ValueError):
        forget.append_journal(__import__("pathlib").Path("/nonexistent/forget.jsonl"), line)


def test_replays_without_the_key(mem, clock, tmp_path):
    """§9.3: on a new Mac (another key) ids and copied HMACs still replay the forget."""
    _seed_sasha(mem)
    backup = tmp_path / "old.db"
    mem.execute(f"VACUUM INTO '{backup}'")
    fid = forget.stop_use(mem, ["f-bday"], scope=ForgetScope.ITEM, forever=True)
    assert forget.purge(mem, fid)
    forget.use_key(b"a-different-key-on-a-new-mac-000")
    old = store.connect(backup)
    try:
        assert forget.replay_journal(old, forget.journal_path(mem)) == [fid]
        status_, text = old.execute(
            "SELECT r.status, f.text_en FROM records r JOIN facts f USING (id) WHERE id = 'f-bday'"
        ).fetchone()
        assert status_ == "purged" and CANARY[0] not in text
        assert old.execute("SELECT status FROM raw_chunks WHERE id = 'raw-1'").fetchone()[0] == (
            "purged"
        )
        live = {r[0] for r in mem.execute("SELECT hmac FROM suppressions")}
        assert live <= {r[0] for r in old.execute("SELECT hmac FROM suppressions")}
    finally:
        old.close()
