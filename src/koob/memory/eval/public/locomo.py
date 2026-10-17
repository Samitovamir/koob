"""LoCoMo (snap-research/locomo, `locomo10.json`) as input to the memory: data, arrival, contexts.

The file is untrusted data: it is read with `json` only and its strings only ever become message
text, questions and gold answers.

Arrival follows the product path. Each conversation goes into its own fresh store through
`api.sweep` (the same call a Telegram chat's new messages take), one sweep per session, in session
timestamp order. Every message carries a real timestamp in the row's `ts` field: the session's
`session_N_date_time`, read as wall-clock time in the memory's own zone, plus `PACE` per turn
(LoCoMo dates sessions, not turns). Timestamps never go into the message text. After each session
the job timer runs (`ApiSUT.run_worker("tick")`: labels, then the extract worker), and a night pass
runs between sessions held on different days.

Speaker mapping (`SPEAKER_MAPPING`): both speakers are contacts in one shared Telegram chat and the
memory's user is absent; each speaker keeps a role of its own (`other`, with its own sender id and
display name). The questions ask about both speakers by name, so neither may be cast as the user:
the user's words are the user's orders and facts, read by different rules, and a speaker filed
as the user would be answered as «me», not by the name the question uses.

Image turns (`img_url`, `blip_caption`, `query`): images are not in the release. The caption is
kept the way LoCoMo's own scripts show it to a reader, `[shares a photo: <caption>]` after the
turn's text; the image URL and the search query are dropped. All three systems see the same text.

Contexts per question:
- `ours`: what the Cube's turn gets from memory (`adapters.turn.compute`: the core view, the
  update block and the recall basket), in a fresh thread and session, the store rolled back after;
- `ours_search`: the same basket plus what the memory's own `search` tool returns for the
  question (the call the brain makes on a turn), the two together within `SEARCH_TOKENS`
  (about what the RAG control spends); the per-question search of the published protocols;
- `full`: the whole conversation, session by session, with each session's date;
- `rag`: the top-k messages by cosine similarity under the memory's own embedder, no ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

SPEAKER_MAPPING = "contacts"  # both speakers are contacts in one chat; the user is absent
PACE = timedelta(seconds=30)  # between two turns of one session (LoCoMo dates sessions only)
ASK_AFTER = timedelta(days=1)  # questions are asked this long after the last session
RAG_K = 20
SEARCH_TOKENS = 1100  # `ours_search`: the basket plus the search result within this many tokens
CATEGORIES = (1, 2, 3, 4)  # the standard set; category 5 (adversarial) is reported apart
ADVERSARIAL = 5
NOT_MENTIONED = "Not mentioned in the conversation"

_WHEN = re.compile(
    r"^\s*(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ap>am|pm)\s+on\s+(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+),?"
    r"\s+(?P<y>\d{4})\s*$",
    re.I,
)
_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}


@dataclass(frozen=True)
class Turn:
    speaker: str
    dia_id: str
    text: str
    caption: str | None = None

    def body(self) -> str:
        """The message text as every system sees it."""
        text = self.text.strip()
        if self.caption:
            text = f"{text} [shares a photo: {self.caption.strip()}]".strip()
        return text


@dataclass(frozen=True)
class Session:
    n: int
    date_text: str
    when: datetime  # aware, UTC
    turns: tuple[Turn, ...]

    def turn_time(self, i: int) -> datetime:
        return self.when + PACE * i

    @property
    def end(self) -> datetime:
        return self.turn_time(max(len(self.turns) - 1, 0))


@dataclass(frozen=True)
class Question:
    qid: str
    category: int
    question: str
    answer: str | None
    evidence: tuple[str, ...] = ()
    adversarial_answer: str | None = None


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    questions: list[Question] = field(default_factory=list)

    @property
    def asked_at(self) -> datetime:
        return self.sessions[-1].end + ASK_AFTER

    def scored(self, cats: tuple[int, ...] = CATEGORIES) -> list[Question]:
        return [q for q in self.questions if q.category in cats]


# ---------------------------------------------------------------------- data


def parse_when(text: str, zone: tzinfo | None = None) -> datetime:
    """«4:04 pm on 20 January, 2023» as an aware UTC instant. The wall-clock time is read in
    `zone` (default: this process's local zone, the one the memory renders dates in)."""
    m = _WHEN.match(text or "")
    if not m:
        raise ValueError(f"unreadable session date: {text!r}")
    hour = int(m["h"]) % 12 + (12 if m["ap"].lower() == "pm" else 0)
    month = _MONTHS.get(m["mon"].lower())
    if month is None:
        raise ValueError(f"unreadable month: {text!r}")
    naive = datetime(int(m["y"]), month, int(m["d"]), hour, int(m["m"]))
    local = naive.replace(tzinfo=zone) if zone is not None else naive.astimezone()
    return local.astimezone(UTC)


def _sessions(conv: dict[str, Any], zone: tzinfo | None) -> list[Session]:
    out = []
    for key, val in conv.items():
        m = re.fullmatch(r"session_(\d+)", key)
        if not m or not isinstance(val, list):
            continue
        date_text = str(conv.get(f"{key}_date_time") or "")
        turns = tuple(
            Turn(
                speaker=str(t.get("speaker") or ""),
                dia_id=str(t.get("dia_id") or ""),
                text=str(t.get("text") or ""),
                caption=str(t["blip_caption"]) if t.get("blip_caption") else None,
            )
            for t in val
            if isinstance(t, dict)
        )
        out.append(Session(int(m.group(1)), date_text, parse_when(date_text, zone), turns))
    return sorted(out, key=lambda s: (s.when, s.n))  # session timestamp order


def _questions(sample_id: str, qa: list[dict[str, Any]]) -> list[Question]:
    out = []
    for i, q in enumerate(qa):
        ans = q.get("answer")
        out.append(
            Question(
                qid=f"{sample_id}/q{i:03d}",
                category=int(q.get("category") or 0),
                question=str(q.get("question") or ""),
                answer=None if ans is None else str(ans),
                evidence=tuple(str(e) for e in q.get("evidence") or ()),
                adversarial_answer=(
                    str(q["adversarial_answer"]) if q.get("adversarial_answer") else None
                ),
            )
        )
    return out


def parse(raw: list[dict[str, Any]], zone: tzinfo | None = None) -> list[Conversation]:
    out = []
    for item in raw:
        conv = item.get("conversation") or {}
        sid = str(item.get("sample_id") or f"conv-{len(out)}")
        out.append(
            Conversation(
                sample_id=sid,
                speaker_a=str(conv.get("speaker_a") or ""),
                speaker_b=str(conv.get("speaker_b") or ""),
                sessions=_sessions(conv, zone),
                questions=_questions(sid, item.get("qa") or []),
            )
        )
    return out


def load(path: Path | str, zone: tzinfo | None = None) -> list[Conversation]:
    with open(path, encoding="utf-8") as fh:
        return parse(json.load(fh), zone)


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------- arrival rows


def _stable_int(*parts: str) -> int:
    return 10**9 + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) % 10**9


def chat_id(conv: Conversation) -> str:
    return f"tg:{_stable_int('chat', conv.sample_id)}"


def sender_ids(conv: Conversation) -> dict[str, str]:
    """One stable Telegram user id per speaker (the user has none here: the user is absent)."""
    return {
        name: str(_stable_int("user", conv.sample_id, name))
        for name in (conv.speaker_a, conv.speaker_b)
    }


def session_rows(conv: Conversation, session: Session) -> list[dict[str, Any]]:
    """The session's messages as arrival rows: each speaker a contact in one chat."""
    ids, chat = sender_ids(conv), chat_id(conv)
    rows = []
    for i, t in enumerate(session.turns):
        body = t.body()
        if not body:
            continue
        rows.append(
            {
                "ts": session.turn_time(i).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "role": "other",
                "source": "telegram",
                "chat_id": chat,
                "msg_id": f"{conv.sample_id}:{t.dia_id}",
                "sender": t.speaker,
                "sender_tg_id": ids.get(t.speaker),
                "text": body,
            }
        )
    return rows


# ---------------------------------------------------------------------- controls


def transcript(conv: Conversation) -> str:
    """The whole conversation (the full-context control)."""
    blocks = []
    for s in conv.sessions:
        lines = [f"Session {s.n} - {s.date_text}"]
        lines += [f"{t.speaker}: {t.body()}" for t in s.turns if t.body()]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@dataclass(frozen=True)
class Chunk:
    order: int
    text: str
    dia_id: str = ""


def chunks(conv: Conversation) -> list[Chunk]:
    """One chunk per message, with its session's date (the plain RAG control's unit)."""
    out = []
    for s in conv.sessions:
        for t in s.turns:
            if t.body():
                out.append(Chunk(len(out), f"({s.date_text}) {t.speaker}: {t.body()}", t.dia_id))
    return out


class RagIndex:
    """Top-k messages by cosine similarity under the given embedder; no ledger, no policy."""

    def __init__(self, conv: Conversation, embedder: Any, k: int = RAG_K) -> None:
        import numpy as np

        self.k = k
        self.chunks = chunks(conv)
        self.embedder = embedder
        m = np.asarray(embedder.embed_docs([c.text for c in self.chunks]), dtype="float32")
        self.m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def top(self, question: str) -> list[Chunk]:
        """The k nearest messages, in conversation order."""
        import numpy as np

        q = np.asarray(self.embedder.embed_query(question), dtype="float32").reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        top = np.argsort(-(self.m @ q))[: self.k]
        return sorted((self.chunks[int(i)] for i in top), key=lambda c: c.order)

    def context(self, question: str) -> str:
        return "\n".join(c.text for c in self.top(question))


# ---------------------------------------------------------------------- our memory


class Arrival:
    """One conversation through the real arrival path into a fresh store under `workdir`."""

    def __init__(
        self, conv: Conversation, workdir: Path, processor: Any, labeler: Any = "auto"
    ) -> None:
        from ..sut import ApiSUT

        self.conv, self.workdir = conv, Path(workdir)
        self.sut = ApiSUT(processor=processor, labeler=labeler)
        self.clock: Any = None
        self.log: list[dict[str, Any]] = []

    def start(self) -> None:
        from ...clock import ManualClock
        from ...workers.scheduler import ManualScheduler

        first = self.conv.sessions[0].when - timedelta(minutes=5)
        self.clock = ManualClock(first)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.sut.start(None, self.clock, ManualScheduler(self.clock), self.workdir)  # type: ignore[arg-type]

    def _to(self, when: datetime) -> None:
        delta = (when - self.clock.now()).total_seconds()
        if delta > 0:
            self.clock.advance(delta)

    def ingest(self) -> None:
        """Every session in timestamp order: its messages arrive, then the job timer runs; a
        night pass runs before a session on a later day."""
        sessions = self.conv.sessions
        for i, s in enumerate(sessions):
            rows = session_rows(self.conv, s)
            self._to(s.end + timedelta(minutes=1))
            stored = self.sut.api.sweep(rows=rows, store=self.sut.store())
            self.sut.run_worker("tick")
            self.log.append({"session": s.n, "rows": len(rows), "stored": stored})
            nxt = sessions[i + 1].when if i + 1 < len(sessions) else self.conv.asked_at
            night = night_after(s.end)
            if night < nxt:
                self._to(night)
                self.sut.run_worker("night")
        self._to(self.conv.asked_at)
        self.sut.run_worker("tick")

    def resume(self) -> None:
        """Open the store an earlier `ingest` left under `workdir`, at question time: the
        questions meet that store again without a second arrival (and no extraction)."""
        if not (self.workdir / "memory" / "memory.db").exists():
            raise FileNotFoundError(f"no store under {self.workdir}")
        self.start()
        self._to(self.conv.asked_at)

    def context(self, q: Question, n: int) -> tuple[str, dict[str, Any]]:
        """What the Cube's turn gets from memory for the question (see `views`)."""
        v = self.views(q, n, search_tokens=None)
        return v["ours"], v["ours_meta"]

    def views(self, q: Question, n: int, *, search_tokens: int | None = SEARCH_TOKENS) -> dict:
        """The question's memory contexts in one fresh thread and session: `ours`, the turn's
        pre-turn injection, and with `search_tokens`, `ours_search`, that injection plus the
        memory's `search` tool called with the question. The store is rolled back afterwards,
        so every question meets the same store."""
        from ... import clock as clockmod
        from ...adapters import mcp
        from ...adapters import turn as turn_adapter
        from ...model import TurnCtx

        now = self.clock.now()
        turn = TurnCtx(
            turn_id=f"locomo-{n}",
            thread_id=f"th-locomo-{n}",
            session_id=f"sess-locomo-{n}",
            owner_words=q.question,
            clock=now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        )
        token = self.sut.snapshot()
        found = None
        try:
            with clockmod.scoped(self.clock):
                inj = turn_adapter.compute(
                    self.sut.store(), turn, first_question=True, resumed=False
                )
                if search_tokens is not None:
                    out = mcp.call(
                        "search", {"query": q.question[:300]}, turn, store=self.sut.store()
                    )
                    found = "" if out.get("is_error") else out["content"][0]["text"]
        finally:
            self.sut.rollback(token)
            if token is not None:
                token.close()
        text = "\n\n".join(p for p in (inj.core, inj.update, inj.basket) if p)
        res: dict[str, Any] = {
            "ours": text,
            "ours_meta": {"basket_items": inj.basket_items, "ms": inj.timings_ms.get("total")},
        }
        if search_tokens is not None:
            extra = budgeted_tool_text(
                found or "", set(_handles(text)), search_tokens - tokens(text)
            )
            res["ours_search"] = "\n\n".join(p for p in (text, extra) if p)
            res["search_meta"] = {"items": len(_handles(found)), "kept": len(_handles(extra))}
        return res

    def stop(self) -> None:
        self.sut.stop()


def tokens(text: str | None) -> int:
    """The harness's token measure (four characters a token), the same for every system."""
    return len(text or "") // 4


def _handles(text: str | None) -> list[str]:
    from .evidence import handles

    return handles(text)


def budgeted_tool_text(text: str, shown: set[str], budget: int) -> str:
    """A memory tool's output cut to `budget` tokens: its item lines (`[m:<id>] ...`) in the
    tool's order, without the items the context already shows, while they fit; the lines around
    them (the fence and its note) are kept. Nothing when no item fits."""
    lines = text.splitlines()
    frame = [ln for ln in lines if not ln.startswith("[m:")]
    left = budget - tokens("\n".join(frame))
    kept: list[str] = []
    for ln in lines:
        if not ln.startswith("[m:"):
            continue
        hs = _handles(ln)
        if hs and hs[0] in shown:
            continue
        cost = tokens(ln) + 1
        if cost > left:
            break
        kept.append(ln)
        left -= cost
    if not kept:
        return ""
    keep = set(kept)
    return "\n".join(ln for ln in lines if not ln.startswith("[m:") or ln in keep)


def night_after(end: datetime) -> datetime:
    """03:00 local on the day after `end` (the night pass of that night)."""
    local = end.astimezone()
    nxt = (local + timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
    return nxt.astimezone(UTC)


__all__ = [
    "ADVERSARIAL",
    "CATEGORIES",
    "NOT_MENTIONED",
    "SPEAKER_MAPPING",
    "Arrival",
    "Conversation",
    "Question",
    "RagIndex",
    "SEARCH_TOKENS",
    "Session",
    "Turn",
    "budgeted_tool_text",
    "chat_id",
    "chunks",
    "load",
    "parse",
    "parse_when",
    "session_rows",
    "tokens",
    "transcript",
]
