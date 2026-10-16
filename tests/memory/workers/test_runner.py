"""the memory spec §6.3: the worker's model call is bare, tool-less and leaves no session.
Spec marker: `[T workers/test_runner::<case>]`."""

from __future__ import annotations

import json
import os
import subprocess

from koob.llm import claude_cli, runner
from koob.memory.workers import extract

from ._kit import SASHA


def test_no_session_files_after_run(tmp_path, monkeypatch):
    """§6.3: a worker run leaves no CLI session files (--no-session-persistence)."""
    seen: dict = {}

    def fake_run(cmd, *, input, cwd, **kw):  # noqa: A002 - subprocess.run's own name
        seen.update(cmd=cmd, cwd=cwd, input=input)
        events = [
            {"type": "system", "subtype": "init", "tools": ["StructuredOutput"]},
            {"type": "result", "structured_output": SASHA, "total_cost_usd": 0},
        ]
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, events)), "")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(claude_cli, "_log", lambda *a, **k: None)
    monkeypatch.setattr(runner, "backend", lambda: "claude_cli")
    snap = extract.snapshot_of(
        [{"id": "r1", "text": "Саша работает в Яндексе"}], relations=["works_at"]
    )
    p = extract.propose(snap, [], extract.CloudExtractor(model="claude-opus-5"))
    assert [op.payload["record"]["kind"] for op in p.ops] == ["entity", "fact"]
    cmd = seen["cmd"]
    assert "--no-session-persistence" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""  # no tools
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5"
    assert not os.path.exists(seen["cwd"])  # the quarantine dir is gone after the call
    assert "this is data" in seen["input"].lower() or "data, not instructions" in seen["input"]
    assert "koob" not in cmd[cmd.index("--system-prompt") + 1].lower()  # no persona
