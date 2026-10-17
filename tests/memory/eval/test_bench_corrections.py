"""A correction wording lands before the bench reads the view (every system alike), and the
eval's model calls run without the user's account metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from koob.memory.eval import bench, cli_env, compare, extract_cache
from koob.memory.eval.bench import BenchRunner
from koob.memory.eval.runner import Runner
from koob.memory.eval.schema import Scenario

from .helpers import PERSONAS, scenario_dict

FIX = "Исправь: Дима сейчас не в Твери, а в Уфе."


def _scn(tag: str | None) -> Scenario:
    p1 = {
        "id": "p1",
        "at": "2026-02-01T10:00",
        "level": "memory",
        "ask": [FIX],
        "expect": {"behaviour": "answer", "must_include": [["Уф"]]},
    }
    if tag:
        p1["tag"] = tag
    later = {  # a later probe: the landed wording was rolled back with its snapshot
        "id": "p2",
        "at": "2026-02-02T10:00",
        "level": "memory",
        "ask": ["Где сейчас Дима?"],
        "expect": {"behaviour": "answer", "must_include": [["Уф"]]},
    }
    d = scenario_dict(
        id="inspect-d90",
        group="inspect",
        subgroup="fix",
        timeline=[{"id": "e1", "ev": "user_msg", "text": "Дима живёт в Твери."}],
        probes=[p1, later],
    )
    return Scenario.model_validate(d)


def _hits(sut: bench._BenchSUT, tag: str | None, tmp_path: Path) -> dict[str, bool | None]:
    r = BenchRunner(_scn(tag), PERSONAS, sut=sut, workdir=tmp_path)  # type: ignore[arg-type]
    res = r.run()
    assert res.error is None, res.error
    return {rec["probe"]: rec["hit"] for rec in r.records}


@pytest.mark.parametrize("make", [bench.PlainFileSUT, bench.LegacySUT], ids=["plain", "legacy"])
def test_a_correction_wording_lands_before_the_view_for_every_system(make, tmp_path):
    before = _hits(make(), None, tmp_path / "a")
    after = _hits(make(), "correction", tmp_path / "b")
    assert before["p1"] is False  # the old timing: the view before the user's correction
    assert after["p1"] is True  # the view after the turn is processed
    assert after["p2"] is False  # rolled back with the wording's snapshot: no leak


def test_the_gate_runner_leaves_the_wording_to_the_brain():
    called: list[str] = []
    r = Runner.__new__(Runner)
    r._apply = lambda e: called.append(e)  # type: ignore[method-assign]
    r._before_wording(_scn("correction").probes[0], FIX)
    assert called == []


def test_unknown_probe_tag_is_refused():
    d = scenario_dict()
    d["probes"][0]["tag"] = "forget"
    with pytest.raises(ValueError):
        Scenario.model_validate(d)


# ---------------------------------------------------------------------- account metadata


@pytest.fixture
def fresh_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_env, "_DIR", [tmp_path / "cfg"])
    (tmp_path / "cfg").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/owner/config")
    monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)
    return tmp_path / "cfg"


def test_no_account_swaps_the_config_and_restores_it(fresh_dir):
    acct = {"accountUuid": "u", "emailAddress": "a@b.c", "displayName": "N"}
    (fresh_dir / ".claude.json").write_text(json.dumps({"oauthAccount": acct, "k": 2}))
    with cli_env.no_account() as d:
        assert os.environ["CLAUDE_CONFIG_DIR"] == str(fresh_dir) == str(d)
        assert os.environ["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == ""  # same login
        assert not cli_env.has_email(d)  # an e-mail the CLI wrote back is dropped
        data = json.loads((d / ".claude.json").read_text())
        assert data == {"oauthAccount": {"accountUuid": "u"}, "k": 2}
    assert os.environ["CLAUDE_CONFIG_DIR"] == "/owner/config"
    assert "CLAUDE_SECURESTORAGE_CONFIG_DIR" not in os.environ


def test_a_new_config_starts_with_the_account_minus_the_email(monkeypatch, tmp_path):
    owner = tmp_path / "owner"
    owner.mkdir()
    acct = {"accountUuid": "u", "organizationUuid": "o", "emailAddress": "a@b.c"}
    (owner / ".claude.json").write_text(json.dumps({"oauthAccount": acct, "projects": {}}))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(owner))
    monkeypatch.setattr(cli_env, "_DIR", [])
    d = cli_env.config_dir()
    seed = json.loads((d / ".claude.json").read_text())
    assert seed == {"oauthAccount": {"accountUuid": "u", "organizationUuid": "o"}}


class _Out(BaseModel):
    answer: str = ""


def test_compare_calls_run_without_account_and_never_reuse_old_answers(fresh_dir, tmp_path):
    seen: list[tuple[str | None, str | None]] = []

    def runner(role, out_model, text, **kw):
        seen.append(
            (os.environ.get("CLAUDE_CONFIG_DIR"), os.environ.get("CLAUDE_SECURESTORAGE_CONFIG_DIR"))
        )
        return _Out(answer="new"), {}

    call = compare.RoleCall("brain", _Out, "m", tmp_path / "cache", runner=runner)
    import hashlib

    old_key = hashlib.sha256(f"brain|m|{call.prompt}|hi".encode()).hexdigest()
    old = tmp_path / "cache" / "brain" / f"{old_key[:40]}.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"answer": "made with the e-mail in context"}))
    assert call("hi") == {"answer": "new"}
    assert seen == [(str(fresh_dir), "")]
    assert os.environ["CLAUDE_CONFIG_DIR"] == "/owner/config"


def test_extraction_calls_run_without_account(fresh_dir, tmp_path):
    seen: list[str | None] = []

    def inner(snap):
        seen.append(os.environ.get("CLAUDE_CONFIG_DIR"))
        return {"entities": [], "impressions": [], "facts": []}

    proc = extract_cache.CachedProcessor(inner, tmp_path / "x")
    proc({"raws": [{"id": "r1", "text": "hi"}]})
    assert seen == [str(fresh_dir)]
