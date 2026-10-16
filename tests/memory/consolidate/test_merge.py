"""Acceptance for the memory spec §6.4: duplicate merges keep every name, date and
number of their inputs. Spec marker: `[T consolidate/test_merge::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import api
from koob.memory.model import OpKind, Proposal
from koob.memory.workers import nightly

from .. import _ledger as L


@pytest.fixture(autouse=True)
def _on_ac():
    nightly.use_power(lambda: True)
    yield
    nightly.use_power(None)


def _dupe(conn, fid: str, text: str, value: str = "German") -> None:
    L.add(conn, L.fact(fid, "sasha", "nationality", value, text=text), rule_match=False)


def _seed(conn) -> None:
    L.add(conn, L.entity("sasha", "Sasha"))


def test_survival(mem):
    """§6.4: a duplicate merge keeps every name, date and number of its inputs."""
    _seed(mem)
    _dupe(mem, "f-a", "Sasha is German")
    _dupe(mem, "f-b", "Sasha is German, born in Hamburg on 3 May 1990")
    counts = api.run_nightly(store=mem)
    # the keeper is the one that loses nothing: f-b holds every detail of f-a
    assert counts["merged"] == 1
    assert L.status(mem, "f-b") == "active"
    assert L.status(mem, "f-a") == "superseded"
    assert ("f-b", "f-a") in L.edges(mem, "supersedes")


def test_survival_keeps_apart_when_a_detail_would_be_lost(mem):
    _seed(mem)
    _dupe(mem, "f-a", "Sasha is German, says Dima")
    _dupe(mem, "f-b", "Sasha is German since 2019")
    counts = api.run_nightly(store=mem)
    assert counts["merged"] == 0 and counts["kept_apart"] == 1
    assert L.status(mem, "f-a") == L.status(mem, "f-b") == "active"


def test_survival_values_normalized():
    """«3 мая», May 3rd and 05-03 are one date; a missing number fails."""
    assert nightly.survival_ok(["Sasha's birthday is 3 May"], "Sasha birthday May 3rd", [])
    assert not nightly.survival_ok(["Room 214 in Berlin"], "Room in Berlin", [])
    assert not nightly.survival_ok(["Sasha and Dima"], "Sasha", [])


def test_supersession_not_rejected(mem):
    """§6.4: a value covered by a supersede op in the same proposal passes the survival check."""
    _seed(mem)
    _dupe(mem, "f-a", "Sasha is German")
    _dupe(mem, "f-b", "Sasha is German since 2019")
    L.add(
        mem, L.fact("f-c", "sasha", "lives_in", "Munich", text="Sasha lives in Munich since 2019")
    )
    merge = L.op(OpKind.MERGE, target="f-a", group="g1", drop="f-b")
    alone = Proposal(ops=[merge])
    assert nightly.check_merges(mem, alone) == ["g1"]  # 2019 would be lost
    covered = Proposal(ops=[merge, L.op(OpKind.SUPERSEDE, target="f-c", group="g1", by="f-a")])
    assert nightly.check_merges(mem, covered) == []


def test_merge_is_delta_only(mem):
    """A second night with nothing new reads no records and merges nothing."""
    _seed(mem)
    _dupe(mem, "f-a", "Sasha is German")
    _dupe(mem, "f-b", "Sasha is German")
    first = api.run_nightly(store=mem)
    assert first["delta"] >= 2 and first["merged"] == 1
    second = api.run_nightly(store=mem)
    assert second["delta"] == 0 and second["merged"] == 0
