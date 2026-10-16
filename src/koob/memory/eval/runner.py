"""Replay a scenario against the memory facade with an injected clock and scheduler, then run its probes.

One `Runner` per scenario: fresh tmp store, `ManualClock`, `ManualScheduler`, fresh canary tokens.
Memory-level probes need no brain. End-to-end probes call a `Brain` (supplied only with --live).
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from ..clock import ManualClock, iso
from ..model import AuthorTier, ChatMode, TurnCtx, TurnOrigin
from ..workers.scheduler import ManualScheduler
from . import canary as canary_mod
from . import graders
from .graders import FAIL, PASS, SKIP
from .judge import Judge, JudgeUnavailable, NoJudge
from .loader import LoadedSet
from .noise import expand
from .observe import MEMORY_CHANNELS, Check, Observation
from .report import ProbeResult, RunReport, ScenarioResult, WordingResult
from .routes import probe_route, route_tools
from .schema import (
    TEMPLATE_RE,
    Advance,
    Backup,
    Corrupt,
    Crash,
    ModeChange,
    Noise,
    OtherMsg,
    Probe,
    Redeliver,
    Restart,
    Scenario,
    UserMsg,
    Worker,
)
from .sut import SUT, ApiSUT, NotBuilt
from .timeutil import resolve_at, to_utc


@dataclass
class LiveResult:
    answer: str
    cli_prompt: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    denials: list[dict[str, Any]] = field(default_factory=list)
    first_word_s: float | None = None
    cost_usd: float = 0.0
    model_calls: int = 0
    lines: dict[str, str] = field(default_factory=dict)


class Brain(Protocol):
    """The real brain behind the runtime (headless, hands in dry run). Wired by the CLI for --live only."""

    def answer(
        self, turn: TurnCtx, wording: str, *, scenario: Scenario, probe: Probe
    ) -> LiveResult: ...


# ------------------------------------------------------------------ canary substitution


def substitute(scn: Scenario, canaries: list[canary_mod.Canary]) -> Scenario:
    """Swap `{{cN.lat|cyr|dig}}` for this run's tokens everywhere except the declarations."""
    table = {(c.id, sp): v for c in canaries for sp, v in canary_mod.spellings(c).items()}
    dump = scn.model_dump(by_alias=True, mode="json", exclude_none=True)

    def walk(o: Any, top: bool = False) -> Any:
        if isinstance(o, str):
            return TEMPLATE_RE.sub(lambda m: table[(m.group(1), m.group(2))], o)
        if isinstance(o, dict):
            return {k: (v if (top and k == "canaries") else walk(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o

    out = walk(dump, top=True)
    out["canaries"] = [c.model_dump(mode="json") for c in canaries]
    return Scenario.model_validate(out)


# ------------------------------------------------------------------ plan


@dataclass
class _Step:
    t: datetime
    seq: int
    kind: str  # event | probe
    ev: Any = None
    probe: Probe | None = None


def _sort_key(s: _Step) -> tuple[datetime, int, int]:
    return (s.t, 0 if s.kind == "event" else 1, s.seq)


class Runner:
    def __init__(
        self,
        scn: Scenario,
        personas: dict[str, Any],
        *,
        sut: SUT | None = None,
        brain: Brain | None = None,
        judge: Judge | None = None,
        scale: str = "1x",
        seed: int = 0,
        workdir: Path | None = None,
    ) -> None:
        self.rng = random.Random(f"{seed}:{scn.id}")
        canaries = [canary_mod.fresh_token(c, self.rng) for c in scn.canaries]
        self.canaries = canaries
        self.scn = substitute(scn, canaries) if canaries else scn
        self.personas = personas
        self.persona = personas.get(scn.persona) or {}
        self.contacts: dict[str, Any] = {
            **(self.persona.get("contacts") or {}),
            **{k: v.model_dump() for k, v in self.scn.contacts.items()},
        }
        self.sut: SUT = sut or ApiSUT()
        self.brain, self.judge, self.scale = brain, judge or NoJudge(), scale
        self.workdir = workdir
        self.tz = self.scn.clock.tz
        self.clock = ManualClock(to_utc(self.scn.clock.start, self.tz))
        self.scheduler = ManualScheduler(self.clock)
        self.not_built: set[str] = set()
        self.gaps: set[str] = set()  # pipeline steps missing in the timeline (night job, extraction...)
        self.notes: list[str] = []
        self.event_time: dict[str, datetime] = {}
        self.event_reply: dict[str, str] = {}
        self.event_row: dict[str, dict[str, Any]] = {}
        self.backups: dict[str, str] = {}
        self.thread_sessions: dict[str, str] = {}
        self.session_n = 0
        self.last_forget_id: str | None = None
        self.forgot = False
        self.purged = False
        self.dirty: dict[str, bool] = {
            "evening": False,
            "night": False,
            "weekly": False,
            "quiet": False,
        }
        self.last_activity: datetime | None = None
        self._held: dict[str, list[tuple[str, str]]] = {}  # event id -> [(job, phase)]
        self.crashed = False
        self.turn_n = 0

    # -------------------------------------------------------------- helpers

    def call(self, fn: Callable[[], Any], default: Any = None) -> Any:
        """Run one facade call; an unbuilt step is recorded, the run goes on with `default`."""
        try:
            return fn()
        except NotBuilt as nb:
            self.not_built.add(nb.lane)
            self.gaps.add(nb.lane)
            return default

    def hook(self, obs: Observation, name: str, fn: Callable[[], Any], default: Any = None) -> Any:
        """Like `call`, for a probe-time read: an unbuilt hook is also noted on the observation,
        so only the checks that read it are skipped (not the whole wording)."""
        try:
            return fn()
        except NotBuilt as nb:
            self.not_built.add(nb.lane)
            obs.unavailable[name] = nb.lane
            return default

    def _iso(self, t: datetime | None = None) -> str:
        return iso(t or self.clock.now())

    def _thread(self, name: str) -> tuple[str, str]:
        tid = f"th-{name}"
        if tid not in self.thread_sessions:
            self.session_n += 1
            self.thread_sessions[tid] = f"sess-{self.session_n}"
        return tid, self.thread_sessions[tid]

    def _turn(
        self,
        *,
        words: str,
        thread: str = "main",
        task: bool = False,
        speaker: bool = False,
        tainted: bool = False,
        name: str | None = None,
    ) -> TurnCtx:
        self.turn_n += 1
        tid, sess = self._thread(thread)
        return TurnCtx(
            turn_id=name or f"t{self.turn_n}",
            thread_id=tid,
            session_id=sess,
            origin=TurnOrigin.TASK if task else TurnOrigin.OWNER,
            owner_words="" if task else words,
            task_words=words if task else None,
            tainted=tainted,
            speaker=speaker,
            message_id=f"m{self.turn_n}",
            clock=self._iso(),
        )

    # -------------------------------------------------------------- plan building

    def _plan(self) -> list[_Step]:
        steps: list[_Step] = []
        prev = to_utc(self.scn.clock.start, self.tz)
        host_of: dict[str, list[Any]] = {}
        seq = 0
        for e in self.scn.timeline:
            if getattr(e, "during", None):
                host_of.setdefault(e.during, []).append(e)
                continue
            if isinstance(e, Noise) and e.at is None:
                end = datetime.combine(e.span.to, datetime.min.time()) + timedelta(days=1)
                t = max(prev + timedelta(minutes=1), to_utc(end, self.tz))
            else:
                t = max(prev, resolve_at(e.at, prev, self.tz))
            prev = t
            if e.id:
                self.event_time[e.id] = t
            if isinstance(
                e, Worker
            ):  # interleave points are static: snapshot_at / commit_after name other events
                if e.snapshot_at:
                    self._held.setdefault(e.snapshot_at, []).append((f"{e.job}|{e.id}", "snapshot"))
                if e.commit_after:
                    self._held.setdefault(e.commit_after, []).append((f"{e.job}|{e.id}", "commit"))
            steps.append(_Step(t, seq, "event", ev=e))
            seq += 1
        self._during = host_of
        # A `during` event has no step of its own: it lands inside its host's turn, so it
        # happens at the host's time (a probe `after:` it used to raise KeyError). Hosts may
        # themselves be `during` events: resolve until nothing changes.
        pending = [e for evs in host_of.values() for e in evs if e.id]
        while pending:
            left = [e for e in pending if e.during not in self.event_time]
            for e in pending:
                if e.during in self.event_time:
                    self.event_time[e.id] = self.event_time[e.during]
            if len(left) == len(pending):
                break  # a host that never runs (validated away; keep the run going)
            pending = left
        probe_ids = {p.id for p in self.scn.probes}
        chained = {p.id for p in self.scn.probes if p.after in probe_ids}
        self._children: dict[str, list[Probe]] = {}
        for p in self.scn.probes:
            if p.id in chained:
                self._children.setdefault(p.after, []).append(p)  # type: ignore[arg-type]
                continue
            if p.after and p.after not in self.event_time:
                raise ValueError(f"probe {p.id}: after -> {p.after!r} has no time in the plan")
            t = self.event_time[p.after] if p.after else to_utc(p.at, self.tz)  # type: ignore[arg-type]
            steps.append(_Step(t, seq, "probe", probe=p))
            seq += 1
        steps.sort(key=_sort_key)
        # a probe right after its event: keep events at the same instant first (already by key)
        return steps

    # -------------------------------------------------------------- scheduling

    def _schedule_points(self, lo: datetime, hi: datetime) -> list[tuple[datetime, str]]:
        """Evening 21:00, night 03:00 and weekly Sunday 04:00 in (lo, hi], in the scenario's zone."""
        tz = ZoneInfo(self.tz)
        out: list[tuple[datetime, str]] = []
        day = (lo.astimezone(tz) - timedelta(days=1)).date()
        end = (hi.astimezone(tz) + timedelta(days=1)).date()
        while day <= end:
            for hh, job in ((3, "night"), (21, "evening")):
                t = datetime(day.year, day.month, day.day, hh, tzinfo=tz).astimezone(UTC)
                if lo < t <= hi:
                    out.append((t, job))
            if day.weekday() == 6:
                t = datetime(day.year, day.month, day.day, 4, tzinfo=tz).astimezone(UTC)
                if lo < t <= hi:
                    out.append((t, "weekly"))
            day += timedelta(days=1)
        return sorted(out)

    def _advance_to(self, target: datetime) -> None:
        now = self.clock.now()
        if target <= now:
            return
        if self.scn.setup.workers == "scheduled":
            pts = self._schedule_points(now, target)
            if self.last_activity and self.dirty["quiet"]:
                q = self.last_activity + timedelta(minutes=10)
                if now < q <= target:
                    pts = sorted([*pts, (q, "quiet")])
            for t, job in pts:
                self.scheduler.call_at(t, lambda job=job, t=t: self._scheduled(job, t))
            for t, _ in pts:
                if t > self.clock.now():
                    self.clock.advance((t - self.clock.now()).total_seconds())
                self.scheduler.run_due()
        self.clock.advance((target - self.clock.now()).total_seconds())
        self.scheduler.run_due()

    def _scheduled(self, job: str, t: datetime) -> None:
        if not self.dirty.get(job, True):
            return
        self.dirty[job] = False
        if job == "night" and self.forgot:
            self.purged = True
        self.call(lambda: self.sut.run_worker("night" if job == "night" else job))

    def _touch(self) -> None:
        self.last_activity = self.clock.now()
        for k in self.dirty:
            self.dirty[k] = True

    # -------------------------------------------------------------- events

    def _row(self, **kw: Any) -> dict[str, Any]:
        return {"ts": self._iso(), **kw}

    def _apply(self, e: Any) -> None:
        api, st = self.sut.api, self.sut.store()
        if isinstance(e, UserMsg):
            self._user_msg(e)
        elif isinstance(e, OtherMsg):
            row = self._row(
                role="other", source="telegram", chat_id=e.chat, msg_id=e.msg_id, sender=e.from_, sender_tg_id=e.sender_tg_id or self._tg_id(e.from_),
                text=e.text or e.caption or "", reply_to=e.reply_to, fwd_from=e.fwd_from, media=e.media, caption=e.caption,
            )  # fmt: skip
            if e.id:
                self.event_row[e.id] = row
            self.call(lambda: api.sweep(rows=[row], store=st))
            self._touch()
        elif isinstance(e, Redeliver):
            row = self.event_row.get(e.of)
            if row:
                self.call(lambda: api.sweep(rows=[row], store=st))
        elif isinstance(e, ModeChange):
            turn = self._turn(words=e.text or "", name=e.id)
            self.call(lambda: api.set_chat_mode(e.chat, ChatMode(e.mode), turn=turn, store=st))
            if e.text:
                self.call(
                    lambda: api.ingest_turn(
                        turn_id=turn.turn_id,
                        thread_id=turn.thread_id,
                        origin="owner",
                        status="done",
                        rows=[self._row(role="owner", text=e.text)],
                        store=st,
                    )
                )
            if e.mode != "normal":
                self.forgot = True
            self._touch()
        elif isinstance(e, Worker):
            if e.snapshot_at or e.commit_after:
                if not e.snapshot_at:  # snapshot now, commit after the named event
                    self.call(lambda: self.sut.run_worker(e.job, "snapshot", e.id))
            else:
                if e.job in ("night", "consolidate") and self.forgot:
                    self.purged = True
                self.call(lambda: self.sut.run_worker(e.job))
        elif isinstance(e, Restart):
            self.session_n += 1
            if e.kind in ("session", "app"):
                self.thread_sessions = {
                    t: f"sess-{self.session_n}-{i}" for i, t in enumerate(self.thread_sessions)
                }
            if e.kind == "app":
                self.call(lambda: api.actions_at_startup(store=st))
        elif isinstance(e, Crash):
            self.call(lambda: self.sut.inject_crash(e.point))
            self.crashed = True
        elif isinstance(e, Backup):
            if e.ev == "backup":
                name = self.call(lambda: api.backup(store=st))
                self.backups[e.name] = name or ""
            else:
                ref = self.backups.get(e.name)
                if ref is None:
                    self.notes.append(f"restore of unknown backup {e.name!r}")
                self.call(lambda: api.restore(ref or e.name, store=st))
        elif isinstance(e, Corrupt):
            self.call(lambda: self.sut.corrupt(e.store))
        elif isinstance(e, Advance):
            pass  # time already moved to the event's instant
        elif isinstance(e, Noise):
            self._noise(e)

    def _tg_id(self, contact: str) -> Any:
        return (self.contacts.get(contact) or {}).get("ids", {}).get("tg_user_id")

    def _noise(self, e: Noise) -> None:
        members: dict[str, list[str]] = {}
        for chat, info in (self.persona.get("chats") or {}).items():
            if isinstance(info, dict) and info.get("with"):
                members[chat] = list(info["with"])
        msgs = expand(e, scale=self.scale, contacts=self.contacts, chat_members=members)
        rows = []
        for m in msgs:
            ts = iso(to_utc(m.at, self.tz))
            if m.chat == "island":
                rows.append(
                    {
                        "ts": ts,
                        "role": "owner",
                        "text": m.text,
                        "chat_id": "island",
                        "noise": m.kind,
                    }
                )
            else:
                rows.append(
                    {
                        "ts": ts,
                        "role": "other",
                        "source": "telegram",
                        "chat_id": m.chat,
                        "msg_id": m.msg_id,
                        "sender": m.sender,
                        "sender_tg_id": self._tg_id(m.sender),
                        "text": m.text,
                        "noise": m.kind,
                    }
                )
        for i in range(0, len(rows), 500):
            batch = rows[i : i + 500]
            self.call(lambda b=batch: self.sut.api.sweep(rows=b, store=self.sut.store()))
        if rows:
            self._touch()

    def _user_msg(self, e: UserMsg) -> None:
        api, st = self.sut.api, self.sut.store()
        tag = e.tag or (e.ev if e.ev in ("correction", "forget") else None)
        turn = self._turn(
            words=e.text,
            thread=e.thread,
            task=e.turn == "task",
            speaker=e.channel == "speaker",
            tainted=bool(e.reads),
            name=e.id,
        )
        rows = [self._row(role="owner", text=e.text, channel=e.channel)]
        for r in e.reads:
            chat = r.chat or r.source
            self.call(
                lambda r=r, chat=chat: api.record_turn_source(
                    turn.turn_id, r.tool, chat, requested=True, store=st
                )
            )
            rows.append(
                self._row(
                    role="tool",
                    tool=r.tool,
                    source=r.source,
                    chat_id=r.chat,
                    text=r.text,
                    author_tier=AuthorTier.OTHER.value
                    if r.tool == "tg_read"
                    else AuthorTier.TOOL.value,
                )
            )
        if e.reply:
            rows.append(self._row(role="cube", text=e.reply))
            if e.id:
                self.event_reply[e.id] = e.reply
        # `during` events land between the tool steps of this turn.
        for d in self._during.get(e.id or "", []):
            self._apply(d)
        if tag == "correction":
            res = self.call(
                lambda: api.fix(e.text, e.text, turn=turn, because_quote=e.text, store=st)
            )
            self._asked_first(e, tag, res)
        elif tag == "forget" or tag == "forever":
            res = self.call(
                lambda: api.forget(
                    e.text, turn=turn, because_quote=e.text, forever=tag == "forever", store=st
                )
            )
            self._asked_first(e, tag, res)
            self.forgot = True
            if tag == "forever":
                self.purged = True
            self.last_forget_id = getattr(res, "forget_id", None) or self.last_forget_id
        elif tag == "undo":
            fid = self.last_forget_id
            if fid:
                self.call(lambda: api.undo_forget(fid, turn=turn, store=st))
            else:
                self.call(lambda: api.undo_forget("", turn=turn, store=st))
        elif tag == "remember":
            self.call(lambda: api.remember(e.text, turn=turn, because_quote=e.text, store=st))
        self.call(
            lambda: api.ingest_turn(
                turn_id=turn.turn_id,
                thread_id=turn.thread_id,
                origin=turn.origin.value,
                status="done",
                rows=rows,
                store=st,
            )
        )
        if e.reply:
            self.call(lambda: api.after_turn_check(turn, e.reply or "", store=st))
        self._touch()

    def _asked_first(self, e: Any, tag: str, res: Any) -> None:
        """The engine asked which item the user meant (`ask_first`): the brain answers that, a
        memory-level run cannot, so the change never lands. Later failures say so."""
        if getattr(res, "outcome", None) != "ask_first":
            return
        self.notes.append(f"{e.id or tag}: {tag} asked first; the answer needs the brain")
        self.gaps.add(ASK_FIRST_GAP)

    def _after_event(self, e: Any) -> None:
        if getattr(e, "id", None):
            for key, phase in self._held.pop(e.id, []):
                job = key.split("|")[0]
                if phase == "commit" and job in ("night", "consolidate") and self.forgot:
                    self.purged = True
                self.call(
                    lambda job=job, phase=phase, key=key: self.sut.run_worker(
                        job, phase, key.split("|")[1]
                    )
                )
        if self.crashed:
            self.crashed = False
            self.session_n += 1
            self.call(lambda: self.sut.api.actions_at_startup(store=self.sut.store()))

    # -------------------------------------------------------------- run

    def run(self) -> ScenarioResult:
        t0 = time.perf_counter()
        scn = self.scn
        res = ScenarioResult(
            id=scn.id,
            group=scn.group,
            subgroup=scn.subgroup,
            set=scn.set,
            title=scn.title,
            status="skip",
            score=None,
        )
        old_env = os.environ.get("KOOB_HANDS_DRYRUN")
        if scn.setup.hands_dryrun or scn.group == "hands":
            os.environ["KOOB_HANDS_DRYRUN"] = "1"
        owned = self.workdir is None
        tmp = tempfile.TemporaryDirectory(prefix="cb-memeval-") if owned else None
        workdir = Path(tmp.name) if tmp else self.workdir
        assert workdir is not None
        try:
            self.sut.start(scn, self.clock, self.scheduler, workdir)
            self.call(lambda: self.sut.seed(scn, self.personas))
            plan = self._plan()
            for step in plan:
                self._advance_to(step.t)
                if step.kind == "event":
                    self._apply(step.ev)
                    self._after_event(step.ev)
                else:
                    assert step.probe is not None
                    res.probes.extend(self._run_probe_tree(step.probe, workdir))
        except Exception as exc:  # a harness or SUT bug must not hide the other scenarios
            res.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.sut.stop()
            finally:
                if tmp:
                    tmp.cleanup()
                if old_env is None:
                    os.environ.pop("KOOB_HANDS_DRYRUN", None)
                else:
                    os.environ["KOOB_HANDS_DRYRUN"] = old_env
        self._finish(res)
        res.wall_s = time.perf_counter() - t0
        return res

    def _finish(self, res: ScenarioResult) -> None:
        res.not_built = sorted(self.not_built)
        res.pipeline_gaps = sorted(self.gaps)
        res.notes = self.notes
        scored = [p.score for p in res.probes if p.score is not None]
        res.score = sum(scored) / len(scored) if scored else None
        res.gate_failures = sum(1 for p in res.probes if p.gate and p.status == "fail")
        res.incomplete = any(p.incomplete for p in res.probes)
        res.probes_graded = sum(p.status != "skip" for p in res.probes)
        res.cost_usd = sum(w.cost_usd for p in res.probes for w in p.wordings)
        res.basket_ms = [
            w.basket_ms for p in res.probes for w in p.wordings if w.basket_ms is not None
        ]
        res.first_word_s = [
            w.first_word_s for p in res.probes for w in p.wordings if w.first_word_s is not None
        ]
        if res.error or any(p.status == "fail" for p in res.probes):
            res.status = "fail"
        elif any(p.status == "pass" for p in res.probes):
            res.status = "pass"

    # -------------------------------------------------------------- probes

    def _run_probe_tree(self, probe: Probe, workdir: Path) -> list[ProbeResult]:
        """A probe and the probes chained onto it (`after: <probe id>`), wording by wording from one snapshot."""
        results: dict[str, ProbeResult] = {}
        chain = self._flatten(probe)
        wordings = probe.ask or [None]
        snap = self.sut.snapshot()
        saved = (self.clock.now(), dict(self.thread_sessions), self.session_n, self.turn_n)
        per: dict[str, list[WordingResult]] = {p.id: [] for p in chain}
        for i, w in enumerate(wordings):
            for p in chain:
                ws = p.ask or [None]
                wi = ws[i % len(ws)] if p is not probe else w
                if p.at is not None and p is not probe:
                    self._advance_to(max(self.clock.now(), to_utc(p.at, self.tz)))
                for ee in p.ev_after:
                    self._apply(ee)
                per[p.id].append(self._run_wording(p, wi, workdir))
            self.sut.rollback(snap)
            self.clock.advance((saved[0] - self.clock.now()).total_seconds())
            self.thread_sessions, self.session_n, self.turn_n = dict(saved[1]), saved[2], saved[3]
        for p in chain:
            results[p.id] = self._probe_result(p, per[p.id])
        return [results[p.id] for p in chain]

    def _flatten(self, probe: Probe) -> list[Probe]:
        out = [probe]
        for c in self._children.get(probe.id, []):
            out += self._flatten(c)
        return out

    def _probe_result(self, p: Probe, ws: list[WordingResult]) -> ProbeResult:
        graded = [w for w in ws if w.status != SKIP]
        passed = [w for w in graded if w.status == PASS]
        score = len(passed) / len(graded) if graded else None
        status = SKIP if not graded else (PASS if len(passed) == len(graded) else FAIL)
        over: list[str] = []
        ms = [w.basket_ms for w in ws if w.basket_ms is not None]
        if ms and (sorted(ms)[int(0.95 * (len(ms) - 1))] > p.budget.basket_ms_p95):
            over.append("basket_ms_p95")
        fw = [w.first_word_s for w in ws if w.first_word_s is not None]
        if fw and max(fw) > p.budget.first_word_s:
            over.append("first_word_s")
        return ProbeResult(
            id=p.id, level=p.level, gate=p.gate, status=status, score=score, wordings=ws,
            e2e_pending=p.level == "both" and self.brain is None, over_budget=over,
            incomplete=any(w.incomplete for w in ws),
            skipped_lanes=sorted({r.split("not built ", 1)[1].split(":")[0] for w in ws for r in w.skipped if r.startswith("not built ")}),
        )  # fmt: skip

    def _run_wording(self, p: Probe, wording: str | None, workdir: Path) -> WordingResult:
        runs: list[WordingResult] = []
        if p.level in ("memory", "both"):
            runs.append(self._grade_memory(p, wording, workdir))
        if p.level in ("e2e", "both"):
            if self.brain is None:
                runs.append(WordingResult(wording, SKIP, skipped=["e2e needs --live"]))
            else:
                runs.append(self._grade_live(p, wording))
        graded = [r for r in runs if r.status != SKIP]
        if any(r.status == FAIL for r in runs):
            status = FAIL
        elif graded:
            status = PASS
        else:
            status = SKIP
        out = WordingResult(wording, status, incomplete=any(r.incomplete for r in runs))
        for r in runs:
            out.reasons += r.reasons
            out.skipped += r.skipped
            out.cost_usd += r.cost_usd
            out.model_calls += r.model_calls
            out.basket_ms = r.basket_ms if r.basket_ms is not None else out.basket_ms
            out.first_word_s = r.first_word_s if r.first_word_s is not None else out.first_word_s
        return out

    # -------------------------------------------------------------- memory level

    def _observe_memory(self, p: Probe, wording: str | None, workdir: Path) -> Observation:
        api, st = self.sut.api, self.sut.store()
        obs = Observation(wording=wording, level="memory", tz=self.tz, time=self.clock.now())
        obs.contact_ids = {k: _stable_ids(v) for k, v in self.contacts.items()}
        turn = self._turn(words=wording or "", thread=p.thread, task=p.turn == "task")
        texts: dict[str, str] = {}
        behaviours: set[str] = set()
        if wording is not None:
            t0 = time.perf_counter()
            basket = self.hook(obs, "basket", lambda: api.recall(turn, store=st))
            obs.basket_ms = (
                basket.timings_ms.get("total") if basket and basket.timings_ms else None
            ) or (time.perf_counter() - t0) * 1000
            if basket is not None:
                texts["basket"] = "\n".join(
                    filter(None, [basket.block or "", *(i.text for i in basket.items)])
                )
                obs.basket_tokens = basket.tokens
                asks = [i for i in basket.items if i.ask]
                if asks:
                    behaviours.add("ask_one")
                    obs.questions_text = " ".join(
                        " ".join(i.ask or []) + " " + i.text for i in asks
                    )
                if basket.items:
                    behaviours.add("answer")
                else:
                    behaviours.add("none_then_ask")
            core = self.hook(
                obs,
                "core",
                lambda: api.core(thread_id=turn.thread_id, session_id=turn.session_id, store=st),
            )
            if core is not None:
                texts["core"] = core.text
            ub = self.hook(
                obs,
                "update_block",
                lambda: api.update_block(
                    thread_id=turn.thread_id, session_id=turn.session_id, store=st
                ),
            )
            if ub is not None:
                texts["update_block"] = ub.block or ""
            found = self.hook(obs, "store_tools", lambda: api.search(wording, turn=turn, store=st))
            if found is not None:
                texts["store_tools"] = "\n".join(i.text for i in found)
                if found:
                    behaviours.add("answer")
            # a tool-routed probe (inspect, scout, or `route:`) also reads what the memory tools
            # return for the question, for every system that has them (eval/routes.py)
            route = probe_route(p, self.scn.group)
            tool_view = getattr(self.sut, "tool_view", None)
            if route_tools(route) and tool_view is not None:
                tool = self.call(lambda: tool_view(route, wording, turn), None)
                if tool:
                    texts["store_tools"] = "\n".join(
                        x for x in (texts.get("store_tools", ""), tool) if x
                    )
            offs = self.hook(obs, "offers", lambda: api.offers(turn, store=st))
            if offs is not None:
                obs.offers = [
                    {"text": o.text, "grounds_text": " ".join(r.id for r in o.grounds)}
                    for o in offs
                ]
                texts["assoc"] = "\n".join(o.text for o in offs)
                behaviours.add(
                    "offer" if len(offs) == 1 else "silence" if not offs else "offer_many"
                )
                if len(offs) > 1:
                    behaviours.add("offer")
        insp = self.sut.inspector()
        obs.inspector = insp
        for ch, txt in self.call(lambda: self.sut.channel_texts(), {}).items():
            texts.setdefault(ch, txt)
        obs.channels = texts
        obs.primary = "\n".join(texts.get(c, "") for c in ("core", "basket", "store_tools"))
        obs.actions = self.hook(obs, "actions", lambda: self.sut.hands_log(), []) or []
        if obs.actions:
            behaviours.add("action")
        else:
            behaviours.add("no_action")
        obs.denials = self.hook(obs, "denials", lambda: self.sut.denials(), []) or []
        if self.event_reply:
            obs.lines = {k: v for k, v in self.event_reply.items()}
        obs.lines["_probe"] = ""
        ex_blocks = _expects(p)
        if any(b.notifications for b in ex_blocks):
            obs.notifications = (
                self.hook(obs, "notifications", lambda: self.sut.notifications(), []) or []
            )
        obs.forget_journal_entries = self.call(lambda: self.sut.forget_journal_entries())
        obs.chat_policy = self.call(lambda: self.sut.chat_policy())
        if any(b.scan for b in ex_blocks):
            obs.scan = self._scan()
        obs.behaviours = behaviours
        obs.not_built = sorted(self.not_built)
        return obs

    def _scan(self) -> dict[str, list[str]] | None:
        smap = self.sut.store_map()
        if smap is None or not self.canaries:
            return None
        hits = canary_mod.scan(smap, self.canaries)
        bad = canary_mod.unlisted(hits, self.canaries, smap, purged=self.purged)
        return {
            cid: [f"{h.store}:{Path(h.path).name}:{h.spelling}/{h.encoding}" for h in hs]
            for cid, hs in bad.items()
        }

    def _grade_memory(self, p: Probe, wording: str | None, workdir: Path) -> WordingResult:
        obs = self._observe_memory(p, wording, workdir)
        w = self._grade(p, obs, memory=True)
        w.basket_ms = obs.basket_ms
        return w

    # -------------------------------------------------------------- live

    def _grade_live(self, p: Probe, wording: str | None) -> WordingResult:
        assert self.brain is not None
        turn = self._turn(words=wording or "", thread=p.thread, task=p.turn == "task")
        res = self.brain.answer(turn, wording or "", scenario=self.scn, probe=p)
        obs = Observation(wording=wording, level="e2e", tz=self.tz, time=self.clock.now())
        obs.answer = res.answer
        obs.primary = res.answer
        obs.questions_text = res.answer
        obs.channels = {"answer": res.answer, "cli_prompt": res.cli_prompt}
        obs.actions, obs.denials, obs.lines = (
            res.actions,
            res.denials,
            {**res.lines, "_probe": res.answer, **self.event_reply},
        )
        obs.first_word_s, obs.cost_usd, obs.model_calls = (
            res.first_word_s,
            res.cost_usd,
            res.model_calls,
        )
        obs.behaviours = classify_behaviour(res.answer, res.actions)
        obs.contact_ids = {k: _stable_ids(v) for k, v in self.contacts.items()}
        obs.inspector = self.sut.inspector()
        if any(b.scan for b in _expects(p)):
            obs.scan = self._scan()
        w = self._grade(p, obs, memory=False)
        w.first_word_s, w.cost_usd, w.model_calls = res.first_word_s, res.cost_usd, res.model_calls
        return w

    # -------------------------------------------------------------- grading

    def _grade(self, p: Probe, obs: Observation, *, memory: bool) -> WordingResult:
        checks: list[Check] = []
        if p.expect_any:
            checks += graders.grade_any(obs, p.expect_any)
        if p.expect:
            checks += graders.grade_expect(obs, p.expect)
            if p.expect.judge:
                checks.append(self._judge(p.expect, obs))
        if memory:
            checks += _needs_brain(p)
        failed = [c for c in checks if c.failed]
        w = WordingResult(obs.wording, PASS)
        w.skipped = sorted({c.reason for c in checks if c.status == SKIP})
        w.incomplete = any(r.startswith(graders.NOT_BUILT) for r in w.skipped)
        if failed:
            w.status = FAIL
            w.reasons = [f"{c.name}: {c.reason}" for c in failed][:8]
            gaps = sorted(self.gaps)
            if gaps:  # a failure next to an unbuilt pipeline step is not yet an engine verdict
                w.reasons[0] += f" [pipeline not built: {', '.join(gaps)}]"
        elif not any(c.status == PASS for c in checks):
            w.status = SKIP
        return w

    def _judge(self, expect: Any, obs: Observation) -> Check:
        if obs.level != "e2e":
            return Check("judge", SKIP, "free-text rubric needs an answer (--live)")
        try:
            v = self.judge.judge(expect.judge, obs.answer or "", [])
        except JudgeUnavailable as exc:
            return Check("judge", SKIP, f"judge unavailable: {exc}")
        return Check("judge", PASS if v.passed else FAIL, v.reason)


ASK_FIRST_GAP = "brain: answer to ask_first"


def _needs_brain(p: Probe) -> list[Check]:
    """Memory-level runs cannot see what the brain does with hands; say so instead of passing."""
    out: list[Check] = []
    for ex in _expects(p):
        if (
            ex.actions
            or ex.denied
            or ex.behaviour in ("action", "refuse_line")
            or "action" in ex.behaviour_any
        ):
            out.append(Check("hands", SKIP, "hand actions need the brain (e2e)"))
    return out


def _expects(p: Probe) -> list[Any]:
    return [x for x in [p.expect, *p.expect_any, *(f.expect for f in p.followup)] if x is not None]


def _stable_ids(info: dict[str, Any]) -> list[str]:
    ids = info.get("ids") or {}
    out = []
    for k in ("tg_user_id", "tg_handle", "email", "phone"):
        v = ids.get(k)
        if v is not None:
            out.append(str(v))
            if k == "tg_user_id":
                out.append(f"tg:{v}")
            if k == "email":
                out.append(f"mail:{v}")
    return out


def classify_behaviour(answer: str, actions: list[dict[str, Any]]) -> set[str]:
    """Live-run heuristics. Deliberately coarse: free-text cases carry a `judge`."""
    from .matchers import normalize

    a = normalize(answer)
    out: set[str] = set()
    nq = a.count("?")
    if actions:
        out.add("action")
    else:
        out.add("no_action")
    if not a:
        out.add("silence")
        return out
    if nq == 1 and len(a) < 400 and not actions:
        out.add("ask_one")
    if nq >= 1 and re.search(r"(проверить|сверить|предложить|хочешь|want me to|should i)", a):
        out.add("offer")
    elif nq == 0:
        out.add("silence")
    if re.search(r"(не помню|не нашёл|не нашел|не вижу в памяти|no record|don't have)", a):
        out.add("none_then_ask")
    if re.search(r"(не буду|не стану|не могу этого|won't do that|i won't)", a):
        out.add("refuse_line")
    if not (out & {"ask_one", "none_then_ask", "refuse_line"}) or nq == 0:
        out.add("answer")
    return out


# ------------------------------------------------------------------ run a whole set


def run_set(
    loaded: LoadedSet,
    *,
    groups: list[str] | None = None,
    scenario_ids: list[str] | None = None,
    only_canary: bool = False,
    sut_factory: Callable[[], SUT] | None = None,
    brain: Brain | None = None,
    judge: Judge | None = None,
    scale: str = "1x",
    seed: int | None = None,
    on_result: Callable[[ScenarioResult], None] | None = None,
) -> RunReport:
    seed = int(time.time()) if seed is None else seed
    report = RunReport(
        live=brain is not None, scale=scale, meta={"seed": seed, "runner": "memory.eval"}
    )
    for scn in loaded.scenarios:
        if groups and scn.group not in groups:
            continue
        if scenario_ids and scn.id not in scenario_ids:
            continue
        if only_canary and not scn.canaries:
            continue
        runner = Runner(
            scn,
            loaded.personas,
            sut=sut_factory() if sut_factory else None,
            brain=brain,
            judge=judge,
            scale=scale,
            seed=seed,
        )
        r = runner.run()
        report.results.append(r)
        if on_result:
            on_result(r)
    return report


__all__ = [
    "Brain",
    "LiveResult",
    "Runner",
    "classify_behaviour",
    "run_set",
    "substitute",
    "MEMORY_CHANNELS",
    "json",
]
