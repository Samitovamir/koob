from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from koob.cli import app

from .helpers import scenario_dict

runner = CliRunner()


def test_eval_dev_runs_end_to_end_and_names_what_is_not_built(tmp_path):
    out = tmp_path / "r.json"
    res = runner.invoke(app, ["memory", "eval", "--set", "dev", "--json", str(out)])
    assert res.exit_code == 0, res.output
    assert "not built" in res.output and "gate poisoning=0 failures:" in res.output
    doc = json.loads(out.read_text())
    scn = doc["dev"]["scenarios"]
    assert len(scn) >= 29 and all("probes_graded" in s for s in scn)
    assert doc["dev"]["gate_states"] and set(doc["dev"]["gate_states"].values()) <= {
        "pass",
        "fail",
        "incomplete",
    }


def test_eval_group_filter_and_live_needs_a_brain():
    res = runner.invoke(app, ["memory", "eval", "--group", "c3"])
    assert res.exit_code == 0 and "c3" in res.output and "c1 " not in res.output
    res = runner.invoke(app, ["memory", "eval", "--live"])
    assert res.exit_code == 2


def test_hidden_set_by_path_prints_group_scores_only(tmp_path):
    root = tmp_path / "hidden" / "memory"
    (root / "knows").mkdir(parents=True)
    (root / "personas.yaml").write_text(yaml.safe_dump({"mira": {"contacts": {}, "chats": {}}}))
    d = scenario_dict(id="knows-h01", set="hidden", title="SENTINEL-CLI-TITLE")
    (root / "knows" / "knows-h01.yaml").write_text(yaml.safe_dump(d, allow_unicode=True))
    out = tmp_path / "h.json"
    res = runner.invoke(app, ["memory", "eval", "--set", str(root), "--json", str(out), "-v"])
    assert res.exit_code == 0, res.output
    blob = res.output + out.read_text()
    assert "SENTINEL" not in blob and "knows-h01" not in blob
    assert "knows" in res.output
    assert (
        runner.invoke(app, ["memory", "eval", "--set", str(root), "--scenario", "x"]).exit_code == 2
    )


def test_hidden_set_format_error_is_redacted(tmp_path):
    root = tmp_path / "hidden" / "memory"
    (root / "knows").mkdir(parents=True)
    (root / "personas.yaml").write_text(yaml.safe_dump({"mira": {"contacts": {}}}))
    d = scenario_dict(id="knows-h01", set="hidden", leak="SENTINEL-ERR")
    (root / "knows" / "knows-h01.yaml").write_text(yaml.safe_dump(d, allow_unicode=True))
    res = runner.invoke(app, ["memory", "eval", "--set", str(root)])
    assert res.exit_code == 2 and "SENTINEL" not in res.output and "knows-h01" not in res.output


def test_hidden_scenario_may_use_a_dev_persona(tmp_path):
    root = tmp_path / "private"
    (root / "knows").mkdir(parents=True)
    d = scenario_dict(id="knows-h02", set="hidden", title="SENTINEL-P")
    (root / "knows" / "knows-h02.yaml").write_text(yaml.safe_dump(d, allow_unicode=True))
    res = runner.invoke(app, ["memory", "eval", "--set", str(root)])
    assert res.exit_code == 0, res.output
    assert "== hidden ==" in res.output and "SENTINEL" not in res.output
