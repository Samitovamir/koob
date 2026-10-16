"""What a probe run produced: the graders read this, never the system under test directly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

# Channels whose text exists at memory level (no brain). The rest need --live or a hook.
MEMORY_CHANNELS = {
    "core", "basket", "update_block", "store_tools", "summaries", "episodes", "assoc", "criteria",
    "intentions", "outbox", "actions", "memory_files", "worker_input", "cloud_input",
}  # fmt: skip


class Inspector(Protocol):
    """Read-only view of memory state for `records:` checks and derived-channel texts."""

    def records(self, where: dict[str, Any]) -> list[dict[str, Any]]: ...

    def channel_text(self, channel: str) -> str | None:
        """Text held in a derived channel (`summaries`, `episodes`, `criteria`, `intentions`, `outbox`...);
        None when this inspector cannot observe it."""
        ...


@dataclass
class Check:
    name: str
    status: str  # pass | fail | skip
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "fail"


@dataclass
class Observation:
    """One wording's run (or a state-only run) as seen by the graders."""

    wording: str | None = None
    level: str = "memory"  # memory | e2e
    # Text a must_include / must_not_include applies to: the answer (e2e/scripted) or, at memory
    # level, the union of core, basket and store-tool results.
    primary: str = ""
    answer: str | None = None
    questions_text: str = ""  # text the one-question check reads (answer, or basket `ask` items)
    channels: dict[str, str] = field(default_factory=dict)  # observed channel -> text
    behaviours: set[str] = field(default_factory=set)
    offers: list[dict[str, Any]] = field(default_factory=list)  # {text, grounds_text}
    candidates_text: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)  # {tool, target, strings}
    denials: list[dict[str, Any]] = field(default_factory=list)  # {tool, target, reason}
    lines: dict[str, str] = field(
        default_factory=dict
    )  # event id (or "_probe") -> user-facing text
    notifications: list[dict[str, Any]] = field(default_factory=list)  # {at: datetime, text}
    forget_journal_entries: int | None = None
    chat_policy: dict[str, str] | None = None
    scan: dict[str, list[str]] | None = None  # canary id -> unlisted hits (path:encoding)
    inspector: Inspector | None = None
    basket_ms: float | None = None
    basket_tokens: int | None = None
    first_word_s: float | None = None
    model_calls: int = 0
    cost_usd: float = 0.0
    not_built: list[str] = field(default_factory=list)  # components whose api raised NotImplementedError
    # Hooks the probe needed that raised NotBuilt: hook name -> component. A check that reads one
    # of them is reported `skip: not built <lane>`; the other checks of the wording are still graded.
    # Hook names: basket, core, update_block, store_tools, offers, actions, denials,
    # notifications, records.
    unavailable: dict[str, str] = field(default_factory=dict)
    contact_ids: dict[str, list[str]] = field(default_factory=dict)  # contact key -> stable ids
    tz: str = "UTC"
    time: datetime | None = None
