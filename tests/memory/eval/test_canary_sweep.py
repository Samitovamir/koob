from __future__ import annotations

import json
import random

from koob.memory.eval import canary as cn
from koob.memory.eval.schema import Canary

TOKEN = Canary(id="c1", lat="ZEPHYRA-7731", cyr="ЗЕФИРА-7731", dig="7731-0942", allowed=["none"])


def stores(tmp):
    return cn.StoreMap(
        [
            cn.StoreSpec("memory_db", [str(tmp / "memory" / "*.db*")]),
            cn.StoreSpec("career_db", [str(tmp / "koob.db*")]),
            cn.StoreSpec("sdk_transcripts", [str(tmp / "sessions")]),
            cn.StoreSpec("workspace", [str(tmp / "workspace")]),
        ]
    )


def test_fresh_token_keeps_shape_and_changes_chars():
    t = cn.fresh_token(TOKEN, random.Random(1))
    assert t.lat != TOKEN.lat and t.cyr != TOKEN.cyr and t.dig != TOKEN.dig
    assert len(t.lat) == len(TOKEN.lat) and t.lat[-5] == "-" and t.lat[-4:].isdigit()
    assert t.lat.split("-")[1] == t.cyr.split("-")[1]  # a shared digit run stays shared
    assert t == cn.fresh_token(TOKEN, random.Random(1))
    assert t != cn.fresh_token(TOKEN, random.Random(2))


def test_encodings_cover_utf8_json_escape_utf16():
    enc = cn.encodings("ЗЕФИРА")
    assert {"utf8", "utf16le", "json_escape"} <= set(enc)
    assert b"\\u0417" in enc["json_escape"]


def test_scan_finds_every_spelling_and_encoding(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "sessions").mkdir()
    (tmp_path / "memory" / "memory.db-wal").write_bytes(b"xx " + TOKEN.lat.encode() + b" yy")
    (tmp_path / "sessions" / "s.jsonl").write_text(
        json.dumps({"t": TOKEN.cyr}), encoding="ascii"
    )  # \uXXXX escapes
    (tmp_path / "koob.db").write_bytes(TOKEN.dig.encode("utf-16-le"))
    smap = stores(tmp_path)
    hits = cn.scan(smap, [TOKEN])
    assert {(h.store, h.spelling, h.encoding) for h in hits} == {
        ("memory_db", "lat", "utf8"),
        ("sdk_transcripts", "cyr", "json_escape"),
        ("career_db", "dig", "utf16le"),
    }


def test_clean_stores_have_zero_hits(tmp_path):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "a.txt").write_text("nothing here")
    assert cn.scan(stores(tmp_path), [TOKEN]) == []


def test_unlisted_uses_allow_rules(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "sessions" / "s.jsonl").write_text(TOKEN.lat)
    (tmp_path / "memory" / "memory.db").write_text(TOKEN.lat)
    smap = stores(tmp_path)
    hits = cn.scan(smap, [TOKEN])
    # nothing allowed: both stores are violations
    assert len(cn.unlisted(hits, [TOKEN], smap)["c1"]) == 2
    turn = TOKEN.model_copy(update={"allowed": ["turn_conversation:e2"]})
    assert [h.store for h in cn.unlisted(hits, [turn], smap)["c1"]] == ["memory_db"]
    trash = TOKEN.model_copy(update={"allowed": ["turn_conversation:e2", "trash_until_purge"]})
    assert cn.unlisted(hits, [trash], smap)["c1"] == []
    assert [h.store for h in cn.unlisted(hits, [trash], smap, purged=True)["c1"]] == ["memory_db"]


def test_default_stores_follow_seams_part_b(tmp_path):
    smap = cn.default_stores(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "ws",
        claude_projects_dir=tmp_path / "proj",
        vault_dir=tmp_path / "vault",
    )
    names = {s.name for s in smap.stores}
    assert {
        "memory_db",
        "career_db",
        "llm_logs",
        "server_log",
        "workspace",
        "sdk_transcripts",
        "vault",
    } <= names
    assert "audio" in smap.excluded and "webkit_store" in smap.excluded
