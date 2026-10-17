"""The system under test, as the runner sees it.

`SUT` is the hook surface the harness needs beyond the public facade: seeding, worker control,
snapshots for paraphrase isolation, derived-channel text, the dry-run hands log, fault injection.
`ApiSUT` is the real thing: it drives `koob.memory.api` (the only entry, spec G13) against a
tmp `memory.db`, and reads state through a small SQL inspector. A hook that is not built yet
raises `NotBuilt`, which the report shows as "not built: <lane>".

A replay of the same scenario must give the same store and the same answers, so `ApiSUT` takes
away what the wall clock and thread timing decide in the app:
- row ids (`ledger.new_id`) come in creation order from the scenario clock and a counter: a
  ULID's random part orders ids minted in the same millisecond at random, and extract batches,
  entities and facts are read `ORDER BY ... id`, which moved the extraction snapshots;
- the finder deadlines are lifted (`EVAL_DEADLINE_MS`): a lane cut by a busy CPU returned
  nothing on one run and its hits on the next (latency has its own tests);
- the vector index is built and loaded in the caller's thread at every job pass and before
  every probe (`scheduler.embed`), never by a background load a search kicks off;
- `settle` waits until the label pass queue is really empty, including passes queued by a
  pass that was running.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from ..clock import Clock
from ..workers.scheduler import Scheduler
from .canary import StoreMap
from .observe import Inspector
from .schema import Scenario

SETTLE_S = 120.0  # the longest a label pass may keep a step waiting before the run fails
WARM_S = 300.0  # the labeler warm-up wait, once per process (the embedder load is the slow part)
EVAL_DEADLINE_MS = 10_000.0  # every finder deadline during a run: no lane is cut by CPU load


def settle(timeout: float = SETTLE_S) -> None:
    """Wait for the engine's background label passes (`labeler.finalize_later`, one worker
    thread). The app lets them land a few milliseconds after a write; in the eval an unwaited
    pass raced the next step: record versions, `labels_final` and so the extraction snapshot
    changed from run to run, a pass queued in one scenario delayed the next one's (the run order
    leaked), and a pass landing after a probe rollback leaked into the next wording.
    Every step therefore starts after the passes queued by the steps before it."""
    from .. import labeler

    end = time.monotonic() + timeout
    while True:
        labeler.drain(max(0.0, end - time.monotonic()))
        with labeler._jobs_lock:  # noqa: SLF001 - the queue a running pass may have grown
            left = [f for f in labeler._jobs if not f.done()]  # noqa: SLF001
        if not left:
            return
        if time.monotonic() >= end:
            raise TimeoutError(f"{len(left)} label passes still queued after {timeout:.0f} s")


class OrderedIds:
    """`ledger.new_id` for a replay: a ULID whose time is the scenario clock (never going
    back) and whose random part is a counter, so ids sort in creation order on every run."""

    def __init__(self, clock: Clock | None) -> None:
        self.clock = clock
        self._ms = 0
        self._n = 0
        self._lock = threading.Lock()

    def __call__(self) -> str:
        from ..ledger import _CROCKFORD

        with self._lock:
            if self.clock is not None:
                self._ms = max(self._ms, int(self.clock.now().timestamp() * 1000))
            self._n += 1
            n = (self._ms << 80) | self._n
        out = []
        for _ in range(26):
            out.append(_CROCKFORD[n & 31])
            n >>= 5
        return "".join(reversed(out))


def warm_labeler(lab: Any, timeout: float = WARM_S) -> bool:
    """Load the embedder and wait until `lab` is ready (or `timeout`). True when warm."""
    import time

    ready = getattr(lab, "ready", None)
    if ready is None:
        return True  # a plain classifier object (tests) has no warm-up
    if ready():
        return True
    from ..recall import vectors

    vectors.current_embedder(wait=True)
    t0 = time.monotonic()
    while not ready() and time.monotonic() - t0 < timeout:
        time.sleep(0.5)
    return bool(ready())


class NotBuilt(Exception):
    """A piece of the system the harness needs does not exist yet."""

    def __init__(self, lane: str) -> None:
        super().__init__(lane)
        self.lane = lane


class Guarded:
    """Wrap an api object: NotImplementedError and missing functions become NotBuilt."""

    def __init__(self, api: Any, after: Any = None) -> None:
        self._api = api
        self.clock: Any = None  # the scenario clock: every facade call reads it (scoped)
        self.after = after  # run after every facade call (ApiSUT: `settle`)

    def __getattr__(self, name: str) -> Any:
        try:
            fn = getattr(self._api, name)
        except AttributeError as exc:
            raise NotBuilt(f"api.{name} missing") from exc

        def call(*a: Any, **kw: Any) -> Any:
            from contextlib import nullcontext

            from .. import clock as clockmod

            scope = clockmod.scoped(self.clock) if self.clock is not None else nullcontext()
            try:
                with scope:
                    return fn(*a, **kw)
            except NotImplementedError as exc:
                raise NotBuilt(str(exc.args[0]) if exc.args else name) from exc
            finally:
                if self.after is not None:
                    self.after()

        return call


class SUT(Protocol):
    api: Any  # a Guarded facade

    def start(self, scn: Scenario, clock: Clock, scheduler: Scheduler, workdir: Path) -> None: ...
    def stop(self) -> None: ...
    def store(self) -> Any: ...  # value for the facade's `store=` argument
    def seed(self, scn: Scenario, personas: dict[str, Any]) -> None: ...
    def run_worker(self, job: str, phase: str = "run", token: str | None = None) -> None: ...
    def snapshot(self) -> Any: ...
    def rollback(self, token: Any) -> None: ...
    def inspector(self) -> Inspector | None: ...
    def channel_texts(self) -> dict[str, str]: ...
    def hands_log(self) -> list[dict[str, Any]]: ...
    def denials(self) -> list[dict[str, Any]]: ...
    def notifications(self) -> list[dict[str, Any]]: ...
    def forget_journal_entries(self) -> int | None: ...
    def chat_policy(self) -> dict[str, str] | None: ...
    def inject_crash(self, point: str) -> None: ...
    def corrupt(self, store: str) -> None: ...
    def store_map(self) -> StoreMap | None: ...


# Kind rows beside `facts` (migrations/memory/0001_core.sql), aliased apart: several kinds
# share column names (`state`, `applicability`, `text_en`, `status`, `outcome`).
_KIND_COLS: tuple[tuple[str, str], ...] = (
    ("k_d_text", "d.chosen_text"),
    ("k_d_domain", "d.domain"),
    ("k_d_alts", "d.alternatives"),
    ("k_d_revise", "d.revision_conditions"),
    ("k_d_at", "d.decided_at"),
    ("k_d_matter", "d.matter_id"),
    ("k_i_trigger", "i.trigger"),
    ("k_i_completion", "i.completion"),
    ("k_i_action", "i.action"),
    ("k_i_state", "i.state"),
    ("k_i_fired", "i.fired_at"),
    ("k_i_outcome", "i.outcome"),
    ("k_c_text", "c.text_en"),
    ("k_c_appl", "c.applicability"),
    ("k_c_evidence", "c.evidence"),
    ("k_c_exceptions", "c.exceptions"),
    ("k_c_status", "c.status"),
    ("k_im_text", "im.text_en"),
    ("k_im_quote", "im.quote_orig"),
    ("k_im_at", "im.observed_at"),
    ("k_about_name", "ea.display_name"),
    ("k_w_matter", "w.matter_id"),
    ("k_w_state", "w.state"),
    ("k_l_owner", "lp.owner"),
    ("k_l_text", "lp.text_en"),
    ("k_l_due", "lp.due"),
    ("k_l_state", "lp.state"),
    ("k_s_rule", "ls.rule_en"),
    ("k_s_appl", "ls.applicability"),
    ("k_s_helpful", "ls.helpful"),
    ("k_s_harmful", "ls.harmful"),
    ("k_e_type", "ep.event_type"),
    ("k_e_start", "ep.span_start"),
    ("k_e_end", "ep.span_end"),
    ("k_e_place", "ep.place"),
    ("k_e_outcome", "ep.outcome"),
    ("k_e_summary", "ep.summary_en"),
)
_KIND_JOINS = (
    "LEFT JOIN decisions d ON d.id = r.id LEFT JOIN intentions i ON i.id = r.id "
    "LEFT JOIN criteria c ON c.id = r.id LEFT JOIN impressions im ON im.id = r.id "
    "LEFT JOIN entities ea ON ea.id = im.about_entity_id LEFT JOIN work_states w ON w.id = r.id "
    "LEFT JOIN loops lp ON lp.id = r.id LEFT JOIN lessons ls ON ls.id = r.id "
    "LEFT JOIN episodes ep ON ep.id = r.id "
)


# A trigger's words (model.Trigger); its kind, ids and timestamps are not text.
_TRIGGER_TEXT = ("goal", "semantic", "to")


def _json(v: Any) -> Any:
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _strings(v: Any) -> list[str]:
    """Every string inside a JSON value, keys excluded."""
    v = _json(v)
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [s for x in v.values() for s in _strings(x)]
    if isinstance(v, list):
        return [s for x in v for s in _strings(x)]
    return []


def _shape(kind: str, k: dict[str, Any], domains: dict[str, str]) -> dict[str, Any]:
    """The kind row's fields under the names the matcher contract uses."""
    if kind == "decision":
        return {
            "text": k["k_d_text"],
            "domain": k["k_d_domain"],
            "alternatives": _json(k["k_d_alts"]),
            "revision_conditions": _json(k["k_d_revise"]),
            "decided_at": k["k_d_at"],
            "matter_id": k["k_d_matter"],
        }
    if kind == "intention":
        trig = _json(k["k_i_trigger"])
        said = [trig.get(f) for f in _TRIGGER_TEXT] if isinstance(trig, dict) else []
        parts = [k["k_i_action"], k["k_i_completion"], *said, k["k_i_outcome"]]
        return {
            "text": " ".join(s for p in parts for s in _strings(p)),
            "trigger": _json(k["k_i_trigger"]),
            "completion": _json(k["k_i_completion"]),
            "action": _json(k["k_i_action"]),
            "state": k["k_i_state"],
            "fired_at": k["k_i_fired"],
            "outcome": k["k_i_outcome"],
        }
    if kind == "criterion":
        evidence = _json(k["k_c_evidence"]) or []
        exceptions = _json(k["k_c_exceptions"]) or []
        return {
            "text": k["k_c_text"],
            "applicability": k["k_c_appl"],
            "evidence": evidence,
            "evidence_count": len(evidence),
            "domains": sorted({domains[e] for e in evidence if e in domains}),
            "exceptions": exceptions,
            "exceptions_count": len(exceptions),
            "criterion_status": k["k_c_status"],
        }
    if kind == "impression":
        return {"text": k["k_im_text"], "quote_orig": k["k_im_quote"], "observed_at": k["k_im_at"]}
    if kind == "work_state":
        state = _json(k["k_w_state"])
        return {"text": " ".join(_strings(state)), "state": state, "matter_id": k["k_w_matter"]}
    if kind == "loop":
        return {
            "text": k["k_l_text"],
            "owner": k["k_l_owner"],
            "due": k["k_l_due"],
            "state": k["k_l_state"],
        }
    if kind == "lesson":
        return {
            "text": k["k_s_rule"],
            "applicability": k["k_s_appl"],
            "helpful": k["k_s_helpful"],
            "harmful": k["k_s_harmful"],
        }
    if kind == "episode":
        return {
            "text": k["k_e_summary"],
            "summary_en": k["k_e_summary"],
            "event_type": k["k_e_type"],
            "span_start": k["k_e_start"],
            "span_end": k["k_e_end"],
            "place": k["k_e_place"],
            "outcome": k["k_e_outcome"],
        }
    return {}


class SqlInspector:
    """`records:` checks over a migrated memory.db (tables of migrations/memory/0001_core.sql)."""

    def __init__(self, conn: sqlite3.Connection, names: dict[str, list[str]]) -> None:
        self.conn = conn
        self.names = names  # contact key -> display names and aliases (normalised by the caller)

    def records(self, where: dict[str, Any]) -> list[dict[str, Any]]:
        kinds = where.get("kind")
        kinds = [kinds] if isinstance(kinds, str) else (kinds or [])
        self.conn.row_factory = sqlite3.Row
        try:
            if "derived" in kinds or where.get("dkind"):
                rows = self.conn.execute(
                    "SELECT id, dkind, dkey, content AS text, read_set FROM derived WHERE valid = 1"
                ).fetchall()
                return [{"kind": "derived", **dict(r)} for r in rows]
            if "action" in kinds:
                return [
                    {"kind": "action", **dict(r)}
                    for r in self.conn.execute("SELECT * FROM actions")
                ]
            if "entity_alias" in kinds:
                return [
                    {"kind": "entity_alias", **dict(r)}
                    for r in self.conn.execute("SELECT * FROM aliases")
                ]
            out = self._kind_rows(usable_only=not where.get("status"))
            if "justification" in kinds:
                return self._justifications(out) + [r for r in out if r["kind"] in kinds]
            return out
        except sqlite3.OperationalError as exc:
            raise NotBuilt(f"inspector: {exc}") from exc

    def _kind_rows(self, *, usable_only: bool) -> list[dict[str, Any]]:
        """One row per record with its kind row read the way the README's matcher contract
        names the fields: `text` (what `value` matches), `domain`, `state`, `trigger`,
        `evidence`/`evidence_count`/`domains`, `about`, `provenance` (its justifications)."""
        from ..model import LABEL_NAMES

        q = (
            "SELECT r.id, r.kind, r.status, r.review, r.labels, r.author_tier, f.relation, rel.family, "
            "f.value_text, f.value_norm, f.text_en, f.fkind, f.polarity, f.stability, f.valid_from, f.valid_to, "
            "f.applicability, f.quote_orig, e.display_name AS subject_name, "
            + ", ".join(f"{col} AS {alias}" for alias, col in _KIND_COLS)
            + " FROM records r LEFT JOIN facts f ON f.id = r.id LEFT JOIN relations rel ON rel.relation = f.relation "
            "LEFT JOIN entities e ON e.id = f.subject_id "
            + _KIND_JOINS
            + ("WHERE r.status IN ('active', 'needs_review') " if usable_only else "")
        )
        prov: dict[str, list[str]] = {}
        for t, pv in self.conn.execute("SELECT target_record_id, provenance FROM justifications"):
            if pv not in prov.setdefault(t, []):
                prov[t].append(pv)
        domains = {
            r[0]: r[1] for r in self.conn.execute("SELECT id, domain FROM decisions") if r[1]
        }
        out = []
        for r in self.conn.execute(q):
            d = dict(r)
            bits = int(d.get("labels") or 0)
            d["labels"] = [n for i, n in enumerate(LABEL_NAMES) if bits & (1 << i)]
            d["review"] = bool(d["review"])
            d["subject"] = self._subject(d.pop("subject_name", None))
            kind_cols = {a: d.pop(a) for a, _ in _KIND_COLS}
            d.update(_shape(d["kind"], kind_cols, domains))
            if d["kind"] == "impression":
                d["about"] = self._subject(kind_cols["k_about_name"])
            d["applicability"] = _json(d.get("applicability"))
            d["provenance"] = prov.get(d["id"]) or None
            out.append(d)
        return out

    def _justifications(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """`kind: justification`: one row per member link, `target` = the conclusion's text."""
        by_id = {r["id"]: r for r in rows}
        out = []
        q = "SELECT id, target_record_id, group_no, member_record_id, provenance FROM justifications"
        for jid, tid, group, member, pv in self.conn.execute(q).fetchall():
            t = by_id.get(tid)
            if t is None:
                continue  # its conclusion is not usable (or not asked for by status)
            out.append(
                {
                    "kind": "justification",
                    "id": jid,
                    "target": t.get("text") or t.get("text_en") or t.get("value_text") or "",
                    "target_id": tid,
                    "target_kind": t["kind"],
                    "member": member,
                    "group_no": group,
                    "provenance": pv,
                    "status": t["status"],
                    "author_tier": t.get("author_tier"),
                }
            )
        return out

    def _subject(self, display: str | None) -> str | None:
        if display is None:
            return None
        from .matchers import normalize

        nd = normalize(display)
        for key, names in self.names.items():
            if nd in [normalize(n) for n in names]:
                return key
        return display

    def channel_text(self, channel: str) -> str | None:
        table = {
            "summaries": "SELECT content FROM derived WHERE dkind = 'summary' AND valid = 1",
            "assoc": "SELECT content FROM derived WHERE dkind = 'assoc_link' AND valid = 1",
            "episodes": "SELECT summary_en FROM episodes",
        }.get(channel)
        if table is None:
            return None
        try:
            return "\n".join(str(r[0]) for r in self.conn.execute(table))
        except sqlite3.OperationalError as exc:
            raise NotBuilt(f"inspector: {exc}") from exc


class ApiSUT:
    """Default SUT: the real facade on a tmp store. Hooks that are not built yet raise NotBuilt."""

    def __init__(
        self,
        api: Any | None = None,
        *,
        stores: StoreMap | None = None,
        extract: str = "off",
        processor: Any | None = None,
        labeler: Any = "auto",
    ) -> None:
        """`extract`: off | cloud | local - which extract processor the worker ticks run
        (cached on disk, see extract_cache). `processor` overrides it (tests: a fake).
        `labeler`: "auto" (on, except under pytest) | True | False | a classifier object; the
        real one is installed and waited warm in `start` (eval is not a turn path)."""
        if api is None:
            from .. import api as real_api

            api = real_api
        self.api = Guarded(api, after=settle)
        self._stores = stores
        self.extract = extract
        self._processor = processor
        self._proc: Any = processor
        self._labeler = labeler
        self._old_classifier: Any = None
        self._classifier_set = False
        self.notes: list[str] = []
        self.conn: sqlite3.Connection | None = None
        self.workdir: Path | None = None
        self._names: dict[str, list[str]] = {}
        self._real_new_id: Any = None
        self._deadlines: dict[str, float] | None = None

    def start(self, scn: Scenario, clock: Clock, scheduler: Scheduler, workdir: Path) -> None:
        from .. import store as store_mod

        settle()  # nothing queued by an earlier scenario may land in this one's time
        self.workdir, self._clock, self._scheduler = workdir, clock, scheduler
        self.api.clock = clock  # notices, denials and stamps carry scenario time, not today
        self._pin_timing(clock)
        self.conn = store_mod.open(workdir / "memory" / "memory.db", home=workdir / "home")
        self._attach_vectors(workdir / "memory" / "vectors.db")
        self._configure_extract()
        self._install_labeler()
        self.embed_now()  # an index (empty) in RAM before the first search: no background load

    def _pin_timing(self, clock: Clock) -> None:
        """Ids in creation order and finder deadlines out of reach (see the module doc)."""
        from .. import ledger
        from ..recall import basket

        if self._real_new_id is None:
            self._real_new_id = ledger.new_id
        ledger.new_id = OrderedIds(clock)
        if self._deadlines is None:
            self._deadlines = dict(basket.FINDER_DEADLINES_MS)
        basket.FINDER_DEADLINES_MS.update(
            {k: max(v, EVAL_DEADLINE_MS) for k, v in self._deadlines.items()}
        )

    def _unpin_timing(self) -> None:
        from .. import ledger
        from ..recall import basket

        if self._real_new_id is not None:
            ledger.new_id, self._real_new_id = self._real_new_id, None
        if self._deadlines is not None:
            basket.FINDER_DEADLINES_MS.update(self._deadlines)
            self._deadlines = None

    def embed_now(self) -> None:
        """Vectors for every record and raw message the store holds now, loaded into RAM in
        this thread (`scheduler.embed`, which the app's job timer runs every tick). Called at
        start and before each probe: in the app the timer ticks many times between a message
        and a question asked hours later."""
        from ..recall import vectors

        if self.conn is None or not vectors.embedder_ready():
            return  # no embedder in RAM (a unit test): nothing to embed with, nothing to load
        from .. import clock as clockmod
        from ..workers import scheduler as sched

        if self._clock is None:
            sched.embed(self.conn)
            return
        with clockmod.scoped(self._clock):
            sched.embed(self.conn)

    @staticmethod
    def _attach_vectors(path: Path | None) -> None:
        """Point the vector hot path at THIS scenario's vectors.db. `vectors` attaches the
        first file it sees for the whole process, so without this every scenario after the
        first searched the first one's index: vector search was blind for the run."""
        from ..recall import vectors

        vectors.attach(path)

    def _configure_extract(self) -> None:
        from ..workers import extract as ex

        proc = self._processor
        if proc is None and self.extract != "off":
            from .extract_cache import make_processor

            proc = make_processor(self.extract)
        self._proc = proc
        if proc is not None:
            ex.configure(**{getattr(proc, "route", self.extract): proc})
        else:
            ex.configure()

    def _install_labeler(self) -> None:
        """Install the label classifier and WAIT until it is warm: a turn must never wait for
        it, but an eval run may (a cold labeler would leave every label undecided)."""
        from .. import policy

        if not self.labeler_on():
            return
        want = True if self._labeler == "auto" else self._labeler
        self._old_classifier, self._classifier_set = policy.classifier(), True
        if want is True:
            from .. import labeler as lab_mod

            want = lab_mod.Labeler(background=False)
        policy.set_classifier(want)
        if not warm_labeler(want):
            self.notes.append("labeler cold after the wait: labels stay undecided")

    def labeler_on(self) -> bool:
        """Whether this run classifies labels (the bench numbers are only real when it does)."""
        import os

        want = self._labeler
        if want == "auto":
            return "PYTEST_CURRENT_TEST" not in os.environ
        return bool(want)

    def stop(self) -> None:
        try:
            settle()  # no pass may write into this store after it closes (or into the next)
        except Exception as exc:  # noqa: BLE001 - stop must still close the store
            self.notes.append(f"label passes did not settle: {type(exc).__name__}")
        if self._classifier_set:
            from .. import policy

            policy.set_classifier(self._old_classifier)
            self._classifier_set = False
        if self._processor is not None or self.extract != "off":
            from ..workers import extract as ex

            ex.configure()
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self._attach_vectors(None)
        self._unpin_timing()

    def store(self) -> Any:
        return self.conn

    def tool_view(self, route: str, wording: str, turn: Any = None) -> str | None:
        """What the memory tools of `route` return for the question (eval/routes.py)."""
        from ..model import TurnCtx
        from .routes import api_tool_text, route_tools

        tools = route_tools(route)
        if not tools or self.conn is None:
            return None
        turn = turn or TurnCtx(
            turn_id="bench", thread_id="th-main", session_id="sess-bench", owner_words=wording
        )
        return api_tool_text(self.conn, wording, turn, tools)

    def seed(self, scn: Scenario, personas: dict[str, Any]) -> None:
        from ..model import ChatMode, TurnCtx

        pers = personas.get(scn.persona) or {}
        for key, info in {
            **(pers.get("contacts") or {}),
            **{k: v.model_dump() for k, v in scn.contacts.items()},
        }.items():
            self._names[key] = [
                str(info.get("en") or ""),
                str(info.get("ru") or ""),
                *map(str, info.get("aliases") or []),
            ]
        turn = TurnCtx(turn_id="seed", thread_id="seed", owner_words="")
        for chat, mode in scn.setup.chat_policy.items():
            self.api.set_chat_mode(chat, ChatMode(mode), turn=turn, store=self.store())
        for item in scn.setup.known:
            text = item.get("text") if isinstance(item, dict) else str(item)
            self.api.remember(
                text,
                turn=turn.model_copy(update={"owner_words": text}),
                because_quote=text,
                store=self.store(),
            )

    def run_worker(self, job: str, phase: str = "run", token: str | None = None) -> None:
        if phase != "run":
            raise NotBuilt("extract: worker snapshot/commit hooks")
        from ..workers import scheduler as sched

        if self._proc is not None:  # extraction runs at every schedule point, before the night pass
            self._jobs(sched)
        if job in ("night", "consolidate"):
            self.api.run_nightly(store=self.store())
            self.api.run_purges(store=self.store())
            return
        if self._proc is not None:
            return
        self._jobs(sched)

    def _jobs(self, sched: Any) -> None:
        """One pass of the app's job timer (`adapters.turn`: `api.run_jobs` every few
        seconds): the label sweep (`labeler.finalize_pending`), then `scheduler.tick` on this
        run's clock (run_jobs itself reads the wired process clock). Schedule points are hours
        apart, and in the app the next pass comes seconds later, so its sweep runs here too:
        the worker's records are final before the next turn, as they would be."""
        import contextlib

        from .. import clock as clockmod
        from .. import labeler

        with clockmod.scoped(self._clock):
            settle()
            with contextlib.suppress(sqlite3.Error):
                labeler.finalize_pending(self.conn)
            try:
                sched.tick(self.conn, self._clock)
            except NotImplementedError as exc:
                raise NotBuilt(str(exc.args[0]) if exc.args else "extract") from exc
            finally:
                settle()  # the worker's commits queued their passes: they land first
            with contextlib.suppress(sqlite3.Error):
                labeler.finalize_pending(self.conn)

    def snapshot(self) -> Any:
        """The store before a probe's wordings (each one rolls back to it). The job timer's
        vector pass runs first, outside the rollback: what arrived is searchable by meaning."""
        if self.conn is None:
            return None
        settle()
        self.embed_now()
        copy = sqlite3.connect(":memory:")
        self.conn.backup(copy)
        return copy

    def rollback(self, token: Any) -> None:
        if token is not None and self.conn is not None:
            settle()  # a pass queued by this wording must not land after the restore
            token.backup(self.conn)

    def inspector(self) -> Inspector | None:
        return SqlInspector(self.conn, self._names) if self.conn is not None else None

    def channel_texts(self) -> dict[str, str]:
        out: dict[str, str] = {}
        insp = self.inspector()
        for ch in ("summaries", "episodes", "assoc"):
            try:
                t = insp.channel_text(ch) if insp else None
            except NotBuilt:
                t = None
            if t is not None:
                out[ch] = t
        return out

    def hands_log(self) -> list[dict[str, Any]]:
        if self.conn is None:
            return []
        try:
            rows = self.conn.execute(
                "SELECT tool, stable_target, summary_en FROM actions ORDER BY seq"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"tool": r[0], "target": r[1], "strings": [r[2] or ""]} for r in rows]

    def denials(self) -> list[dict[str, Any]]:
        """What the runtime guard refused. The engine keeps no such log; a facade that records
        one exposes `recorded_denials(store=)`. Until then it is not built, never `[]`."""
        return self._recorded("recorded_denials", "runtime: guard denial log")

    def notifications(self) -> list[dict[str, Any]]:
        """What was shown to the user. Same rule: `recorded_notifications(store=)` or not built."""
        return self._recorded("recorded_notifications", "runtime: owner-facing notification log")

    def _recorded(self, fn_name: str, lane: str) -> list[dict[str, Any]]:
        fn = getattr(self.api._api, fn_name, None)  # noqa: SLF001 - Guarded wraps the facade
        if fn is None:
            raise NotBuilt(lane)
        try:
            return list(fn(store=self.store()))
        except NotImplementedError as exc:
            raise NotBuilt(str(exc.args[0]) if exc.args else lane) from exc

    def forget_journal_entries(self) -> int | None:
        if self.workdir is None:
            return None
        j = self.workdir / "memory" / "journal" / "forget.jsonl"
        return sum(1 for _ in j.open(encoding="utf-8")) if j.exists() else 0

    def chat_policy(self) -> dict[str, str] | None:
        if self.conn is None:
            return None
        try:
            return {r[0]: r[1] for r in self.conn.execute("SELECT chat_id, mode FROM chat_policy")}
        except sqlite3.OperationalError:
            return None

    def inject_crash(self, point: str) -> None:
        raise NotBuilt(f"crash point {point}: fault-injection hook (owning module)")

    def corrupt(self, store: str) -> None:
        raise NotBuilt("store: corruption injection / recovery")

    def store_map(self) -> StoreMap | None:
        if self._stores is not None:
            return self._stores
        if self.workdir is None:
            return None
        from .canary import StoreSpec

        return StoreMap(
            [
                StoreSpec("memory_db", [str(self.workdir / "memory" / "*")]),
                StoreSpec("workspace", [str(self.workdir / "workspace")]),
            ]
        )
