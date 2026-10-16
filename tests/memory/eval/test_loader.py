from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from koob.memory.eval.loader import ScenarioError, load_scenario, load_set
from koob.memory.eval.schema import GROUPS, Scenario

from .helpers import scenario_dict

DEV = Path(__file__).resolve().parents[3] / "evals" / "memory" / "dev"


def test_whole_dev_set_loads():
    s = load_set(DEV)
    assert len(s.scenarios) >= 29
    assert not s.errors
    assert {sc.group for sc in s.scenarios} <= set(GROUPS)
    assert all(sc.set == "dev" for sc in s.scenarios)


def test_every_dev_probe_has_expectations():
    for sc in load_set(DEV).scenarios:
        assert sc.probes
        for p in sc.probes:
            assert p.expect or p.expect_any


def test_unknown_top_level_field_fails():
    with pytest.raises(ValidationError, match="surprise"):
        Scenario.model_validate(scenario_dict(surprise=1))


def test_unknown_event_and_expect_fields_fail():
    bad = scenario_dict()
    bad["timeline"][0]["colour"] = "red"
    with pytest.raises(ValidationError):
        Scenario.model_validate(bad)
    bad = scenario_dict()
    bad["probes"][0]["expect"]["must_includ"] = ["x"]
    with pytest.raises(ValidationError):
        Scenario.model_validate(bad)


def test_unknown_subgroup_channel_behaviour_fail():
    with pytest.raises(ValidationError, match="subgroup"):
        Scenario.model_validate(scenario_dict(subgroup="nope"))
    bad = scenario_dict()
    bad["probes"][0]["expect"] = {"forbidden": [{"values": ["x"], "channels": ["nowhere"]}]}
    with pytest.raises(ValidationError, match="channels"):
        Scenario.model_validate(bad)
    bad = scenario_dict()
    bad["probes"][0]["expect"] = {"behaviour": "shout"}
    with pytest.raises(ValidationError, match="behaviour"):
        Scenario.model_validate(bad)


def test_dangling_references_fail():
    bad = scenario_dict()
    bad["timeline"].append({"ev": "redeliver", "of": "ghost"})
    with pytest.raises(ValidationError, match="ghost"):
        Scenario.model_validate(bad)
    bad = scenario_dict()
    bad["probes"][0]["after"] = "ghost"
    with pytest.raises(ValidationError, match="ghost"):
        Scenario.model_validate(bad)


def test_undeclared_canary_template_fails():
    bad = scenario_dict()
    bad["timeline"][0]["text"] = "код {{c9.lat}}"
    with pytest.raises(ValidationError, match="c9"):
        Scenario.model_validate(bad)


def test_file_name_must_match_id_and_folder_group(tmp_path):
    d = tmp_path / "knows"
    d.mkdir()
    f = d / "wrong.yaml"
    f.write_text(yaml.safe_dump(scenario_dict(), allow_unicode=True))
    with pytest.raises(ScenarioError, match="file name"):
        load_scenario(f)


def test_hidden_errors_are_redacted(tmp_path):
    root = tmp_path / "hidden" / "memory"
    (root / "knows").mkdir(parents=True)
    bad = scenario_dict(
        id="knows-h01",
        set="hidden",
        title="SENTINEL-SECRET-TITLE",
        surprise="SENTINEL-SECRET-VALUE",
    )
    (root / "knows" / "knows-h01.yaml").write_text(yaml.safe_dump(bad, allow_unicode=True))
    (root / "personas.yaml").write_text(yaml.safe_dump({"mira": {"contacts": {}}}))
    with pytest.raises(ScenarioError) as ei:
        load_set(root, redact=True)
    msg = str(ei.value)
    assert "SENTINEL" not in msg and "knows-h01" not in msg
    assert "1 scenario file" in msg and "knows: 1" in msg


def test_set_must_match_tree(tmp_path):
    root = tmp_path / "hidden" / "memory"
    (root / "knows").mkdir(parents=True)
    (root / "personas.yaml").write_text(yaml.safe_dump({"mira": {"contacts": {}}}))
    (root / "knows" / "knows-d90.yaml").write_text(
        yaml.safe_dump(scenario_dict(), allow_unicode=True)
    )
    with pytest.raises(ScenarioError, match="does not match the tree"):
        load_set(root)
