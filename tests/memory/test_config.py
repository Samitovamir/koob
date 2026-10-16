"""The `memory:` settings block (the memory spec §5.4, §5.5, §20)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from koob import config
from koob.memory.model import LABEL_NAMES, SENSITIVE_DEFAULT, Label, label_from_names


def test_label_names_match_the_model():
    assert config.MEMORY_LABELS == LABEL_NAMES
    assert [Label[n.upper()].value for n in LABEL_NAMES] == [1 << i for i in range(9)]


def test_absent_block_is_off_and_local_only():
    m = config.MemoryCfg()
    assert (m.engine, m.shadow) == ("legacy", "off")
    assert m.routing["unlabeled"] == ["cloud"]
    assert all(m.routing[n] == ["local"] for n in LABEL_NAMES)
    assert label_from_names(m.sensitive_labels) == SENSITIVE_DEFAULT


def test_settings_file_keeps_the_engine_off():
    m = config.settings().memory
    assert (m.engine, m.shadow) == ("legacy", "off")
    for name in ("health", "intimate", "identity_docs", "money", "unknown_sensitive"):
        assert m.routing[name] == ["local"]  # §5.4 local only until the extraction test


def test_yaml_booleans_and_partial_routing():
    m = config.MemoryCfg.model_validate({"shadow": True, "routing": {"legal": ["local", "cloud"]}})
    assert m.shadow == "on"
    assert m.routing["legal"] == ["local", "cloud"]
    assert m.routing["health"] == ["local"]


@pytest.mark.parametrize(
    "bad",
    [
        {"engine": "new"},
        {"routing": {"healthy": ["local"]}},
        {"sensitive_labels": ["secret"]},
        {"routing": {"health": ["anywhere"]}},
    ],
)
def test_bad_values_fail_loudly(bad):
    with pytest.raises(ValidationError):
        config.MemoryCfg.model_validate(bad)
