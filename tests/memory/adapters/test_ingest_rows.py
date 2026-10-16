"""the memory spec §6.2 (v0.4): the ingest row shape (the eval harness's), chat ids of
Cube-turn rows and the `source_kind` vocabulary.
Spec marker: `[T adapters/test_ingest_rows::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import api, ingest
from koob.memory.model import SOURCE_FAMILIES, IngestRow

TS = "2026-10-12T09:00:00.000000Z"


def test_harness_shape_validates():
    other = IngestRow.model_validate(
        {
            "ts": "2026-10-12T09:00:00.000000Z",
            "role": "other",
            "source": "telegram",
            "chat_id": "tg:777",
            "msg_id": 42,
            "sender": "Дима",
            "sender_tg_id": 777,
            "text": "буду в субботу",
            "reply_to": None,
            "noise": "chatter",
        }
    )
    assert other.sender_id == "tg:777" and other.msg_id == "42"
    tool = IngestRow(
        ts="t", role="tool", tool="tg_read", source="telegram", chat_id="tg:777", text="…",
        author_tier="other",
    )  # fmt: skip
    assert tool.author_tier is not None
    with pytest.raises(ValueError):
        IngestRow.model_validate({"ts": "t", "role": "assistant", "text": "x"})


def test_island_chat_is_null():
    row = IngestRow(ts="t", role="owner", text="запомни", chat_id="island", channel="voice")
    assert row.chat_id is None


def test_source_kind_vocabulary(mem):
    """Stored `raw_chunks.source_kind` is `<family>[:<detail>]` from `SOURCE_FAMILIES`;
    legacy `cube_turn` maps to `owner` / `cube` by role."""
    cases = [
        ({"role": "owner", "source": "cube_turn"}, "owner"),
        ({"role": "cube", "source": "cube_turn"}, "cube"),
        ({"role": "cube", "source": "bot"}, "bot"),
        ({"role": "tool", "tool": "web_read"}, "tool:web_read"),
        ({"role": "tool", "tool": "tg_read", "source": "telegram"}, "telegram"),
        ({"role": "tool", "tool": "screen_look", "source": "capture:general"}, "screen:general"),
        ({"role": "tool", "tool": "x", "source": "nonsense"}, "tool:x"),
        ({"role": "other", "source": "telegram"}, "telegram"),
        ({"role": "other", "source": "mail"}, "mail"),
    ]
    for extra, want in cases:
        assert ingest.source_kind_of(IngestRow(ts="t", text="x", **extra)) == want, extra
    rows = [
        {"role": "owner", "ts": TS, "text": "купи билеты на субботу", "source": "cube_turn"},
        {"role": "cube", "ts": TS, "text": "Посмотрю билеты на субботу", "source": "cube_turn"},
        {"role": "tool", "ts": TS, "tool": "web_read", "text": "Билеты на субботу: 3 рейса"},
    ]
    api.ingest_turn(
        turn_id="t1", thread_id="th1", origin="owner", status="done", rows=rows, store=mem
    )
    kinds = {k for (k,) in mem.execute("SELECT source_kind FROM raw_chunks")}
    assert kinds == {"owner", "cube", "tool:web_read"}
    assert all(k.partition(":")[0] in SOURCE_FAMILIES for k in kinds)


def test_cube_turn_rows_have_no_chat(mem):
    """The user's words and the Cube's answer store `chat_id` NULL and `source_ref` `thread:<id>`;
    a tool row that read a chat carries that chat's id."""
    rows = [
        {"role": "owner", "ts": TS, "text": "что там у Димы?", "chat_id": "island"},
        {"role": "cube", "ts": TS, "text": "Дима будет в субботу к семи.", "chat_id": "tg:777"},
        {
            "role": "tool", "ts": TS, "tool": "tg_read", "source": "telegram", "chat_id": "tg:777",
            "msg_id": 42, "author_tier": "other", "text": "Дима: буду в субботу к семи",
        },
    ]  # fmt: skip
    api.ingest_turn(
        turn_id="t1", thread_id="th9", origin="owner", status="done", rows=rows, store=mem
    )
    got = {
        k: (c, ref)
        for k, c, ref in mem.execute("SELECT source_kind, chat_id, source_ref FROM raw_chunks")
    }
    assert got == {
        "owner": (None, "thread:th9"),
        "cube": (None, "thread:th9"),
        "telegram": ("tg:777", "msg:42"),
    }
