"""Episodes and daily summaries rebuilt from usable records (raw only for quotes, through the
retraction overlay); participants, place, time, outcome and cause links for C3.

Spec: §4.3, §6.4, §6.5, §9.6, §15.3.

- **Episodes are code, not a model.** A seed is a usable fact of kind `event` or `plan` (a
  bounded happening or a plan that may fall through). Seeds join one episode when they come
  from one message and name the same kind of happening, or when they name the same kind of
  happening with a shared person within `CLUSTER_DAYS`. The episode row takes event type,
  span (the facts' valid time only: unknown stays NULL, §4.1), place, outcome and an English
  summary built from the seeds' texts, their stated causes and their quotes. `episode_links`
  carry participants (people, never the user), places, `about` entities, causes (the
  justification members of the seeds, §8.1) and the record that gave the outcome.
- **Ops (§6.5).** A new episode is an ADD (`derived_from` = seeds and causes, `mentions` =
  the linked entities); a changed one is a SUPERSEDE of the old episode by a new record
  (history kept, the old one never edited in place, G10); an episode whose seeds all left use
  goes to `needs_review` by FLAG. One commit group per episode. The ledger's ADD has no link
  field, so `apply_op` (this module's applier) writes the `episode_links` rows inside the same
  transaction, recomputed from the committed sources (the read set guarantees they are the
  ones `propose` saw).
- **Daily summaries (§6.4, G10).** `propose_daily_summary(day)` builds the day's `summary`
  derived item (ADD `derived`, id `sum-<day>`) from the usable records said that day (the
  day of their raw message, else of their recording), each with its quote through the
  retraction overlay. The read set holds every record and raw chunk it used, so labels are
  inherited (§5.3) and a change to any of them invalidates the summary (§7.3). It never
  reads the previous summary: no `derived` ref, no old content.
- **Schedule.** The night step (`night_step`, registered with `workers.nightly`) builds
  episodes for the facts changed since its own watermark and the summaries of the days those
  changes touch, newest first, capped by the night. Invalidated summaries come back through
  the `rebuild_derived` job (`rebuild_summary`, registered with the core view's dispatcher).
- **Day.** A summary day is the Mac's local date (`DAY_TZ`, None = the system zone).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Any

from . import ledger, store
from .model import (
    OWNER_ENTITY_ID,
    AddPayload,
    AuthorTier,
    DerivedIn,
    DerivedKind,
    EpisodeRole,
    FlagPayload,
    OpKind,
    Proposal,
    ProposalOp,
    ReadRef,
    ReadSrc,
    SupersedePayload,
    payload_dict,
)

NIGHT_STEP = "episodes"
CLUSTER_DAYS = 21  # same happening with the same person: plan, then what became of it
SUMMARY_LINES = 80  # a busy day keeps its first lines; the rest stay in the ledger
LINE_CHARS = 240
EPISODE_CHARS = 600
DAY_TZ: list[tzinfo | None] = [None]  # None: the Mac's local zone

_USABLE = ("active", "archived")
_SEED_FKINDS = ("event", "plan")
_SUMMARY_SKIP_KINDS = ("entity", "episode")
_MARK_EPISODES = "episodes_version"
_MARK_SUMMARIES = "summaries_version"

# Names match recall/cue.py `_EVENT_TYPES` / `_OUTCOMES` (the C3 finder compares them); the
# patterns are tuned for stored text (English canonical text plus the user's quote).
_EVENT_TYPES: tuple[tuple[str, str], ...] = (
    ("meeting", r"встрет|встреч|увидеться|\bmeet"),
    ("dinner", r"ужин|обед|завтрак|кофе|dinner|lunch|coffee|breakfast"),
    ("trip", r"поездк|поехал|съездил|путешеств|командиров|отпуск|\btrip|travel|vacation"),
    ("call", r"созвон|звонил|звонок|позвонил|\bcall(?!ed off)"),
    ("event", r"концерт|выставк|вечеринк|праздник|день рождени|concert|exhibition|party"),
)
_RELATION_TYPES = {"travels_to": "trip", "visited": "trip", "wants_to_meet": "meeting"}
_OUTCOMES: tuple[tuple[str, str], ...] = (
    (
        "cancelled",
        r"\bcancel|fell through|called off|didn.t happen|did not happen|сорвал|отменил"
        r"|отмен[аеиуы]|не получилось|не вышло|не состоял",
    ),
    ("postponed", r"postpon|reschedul|put off|перенес|перенёс|отложил|отложен"),
    ("done", r"\bhappened|went well|took place|состоял|получилось|прошл[аио] (?:хорошо|отлично)"),
)


# ---------------------------------------------------------------- small helpers


def _marks(n: int) -> str:
    return ", ".join("?" for _ in range(n))


def _chunks(ids: Sequence[str], n: int = 400) -> Iterable[list[str]]:
    for i in range(0, len(ids), n):
        yield list(ids[i : i + n])


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def day_of(ts: str) -> str:
    """The local date (YYYY-MM-DD) of a stored UTC timestamp."""
    return _parse(ts).astimezone(DAY_TZ[0]).date().isoformat()


def _day_window(day: str) -> tuple[str, str]:
    """UTC bounds [lo, hi) of a local day, with a margin; callers check `day_of` exactly."""
    d = date.fromisoformat(day)
    start = datetime(d.year, d.month, d.day)
    start = start.replace(tzinfo=DAY_TZ[0]) if DAY_TZ[0] is not None else start.astimezone()
    lo = (start - timedelta(hours=2)).astimezone(UTC)
    hi = (start + timedelta(days=1, hours=2)).astimezone(UTC)
    return lo.strftime("%Y-%m-%dT%H:%M:%S"), hi.strftime("%Y-%m-%dT%H:%M:%S")


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _quote_ok(conn: sqlite3.Connection, quote: str, raw_id: str | None, record_id: str) -> bool:
    """§9.6 retraction overlay: a quote holding a corrected value is left out."""
    try:
        ledger.check_retracted(conn, [quote], raws=[raw_id] if raw_id else [], records=[record_id])
    except ledger.CommitRejected:
        return False
    return True


# ---------------------------------------------------------------- seeds


@dataclass
class _Seed:
    id: str
    version: int
    relation: str
    text_en: str
    quote: str | None
    fkind: str
    polarity: str
    valid_from: str | None
    valid_to: str | None
    raw_id: str | None
    anchor: str
    place_value: str | None
    entities: set[str] = field(default_factory=set)
    causes: list[str] = field(default_factory=list)
    event_type: str | None = None
    outcome: str | None = None


def _classify(text: str, table: tuple[tuple[str, str], ...]) -> str | None:
    t = text.casefold()
    for name, rx in table:
        if re.search(rx, t):
            return name
    return None


def _outcome(s: _Seed) -> str | None:
    said = _classify(f"{s.text_en} {s.quote or ''}", _OUTCOMES)
    if said:
        return said
    if s.fkind == "event":
        return "cancelled" if s.polarity == "negated" else "done"
    return None  # a plan with no word on what became of it


def _load_seeds(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, _Seed]:
    """Usable event and plan facts among `ids`, with their entities and stated causes."""
    ids = list(dict.fromkeys(ids))
    out: dict[str, _Seed] = {}
    for chunk in _chunks(ids):
        rows = conn.execute(
            "SELECT r.id, r.version, r.recorded_at, f.subject_id, f.value_entity_id, f.relation,"
            " f.value_text, f.text_en, f.quote_orig, f.fkind, f.polarity, f.valid_from,"
            " f.valid_to, f.source_raw_id, f.keywords_en, f.keywords_ru, rc.ts AS raw_ts,"
            " rel.value_type FROM records r JOIN facts f ON f.id = r.id"
            " LEFT JOIN raw_chunks rc ON rc.id = f.source_raw_id"
            " LEFT JOIN relations rel ON rel.relation = f.relation"
            f" WHERE r.id IN ({_marks(len(chunk))}) AND r.kind = 'fact'"  # noqa: S608
            f" AND r.status IN {_USABLE} AND f.fkind IN {_SEED_FKINDS}",
            chunk,
        ).fetchall()
        for r in rows:
            s = _Seed(
                id=r["id"],
                version=r["version"],
                relation=r["relation"],
                text_en=r["text_en"] or "",
                quote=r["quote_orig"],
                fkind=r["fkind"],
                polarity=r["polarity"],
                valid_from=r["valid_from"],
                valid_to=r["valid_to"],
                raw_id=r["source_raw_id"],
                anchor=r["valid_from"] or r["raw_ts"] or r["recorded_at"],
                place_value=r["value_text"] if r["value_type"] == "place" else None,
            )
            s.entities = {e for e in (r["subject_id"], r["value_entity_id"]) if e}
            words = " ".join(
                x or "" for x in (s.text_en, s.quote, r["keywords_en"], r["keywords_ru"])
            )
            s.event_type = _RELATION_TYPES.get(s.relation) or _classify(words, _EVENT_TYPES)
            s.outcome = _outcome(s)
            out[s.id] = s
    if not out:
        return out
    keys = list(out)
    for chunk in _chunks(keys):
        for fid, eid in conn.execute(
            "SELECT e.from_id, e.to_id FROM edges e JOIN entities n ON n.id = e.to_id"
            f" WHERE e.etype IN ('mentions', 'about') AND e.from_id IN ({_marks(len(chunk))})",  # noqa: S608
            chunk,
        ):
            out[fid].entities.add(eid)
        for target, member in conn.execute(
            "SELECT j.target_record_id, j.member_record_id FROM justifications j"
            " JOIN records m ON m.id = j.member_record_id"
            f" WHERE j.target_record_id IN ({_marks(len(chunk))})"  # noqa: S608
            f" AND m.status IN {_USABLE} AND m.kind != 'entity'"
            " ORDER BY j.group_no, j.member_record_id",
            chunk,
        ):
            if member not in out[target].causes:
                out[target].causes.append(member)
    return out


def _entity_types(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, tuple[str, str]]:
    """entity id -> (etype, display_name), usable entities only."""
    ids = list(dict.fromkeys(ids))
    out: dict[str, tuple[str, str]] = {}
    for chunk in _chunks(ids):
        for eid, etype, name in conn.execute(
            "SELECT n.id, n.etype, n.display_name FROM entities n JOIN records r ON r.id = n.id"
            f" WHERE n.id IN ({_marks(len(chunk))}) AND r.status IN {_USABLE}",  # noqa: S608
            chunk,
        ):
            out[eid] = (etype, name)
    return out


def _people(seed: _Seed, types: dict[str, tuple[str, str]]) -> set[str]:
    return {e for e in seed.entities if e != OWNER_ENTITY_ID and types.get(e, ("",))[0] == "person"}


def _neighbours(conn: sqlite3.Connection, seeds: dict[str, _Seed]) -> set[str]:
    """Fact ids that may join these seeds: same message, or the same person named."""
    raws = sorted({s.raw_id for s in seeds.values() if s.raw_id})
    ents = sorted({e for s in seeds.values() for e in s.entities if e != OWNER_ENTITY_ID})
    found: set[str] = set()
    for chunk in _chunks(raws):
        found |= {
            r[0]
            for r in conn.execute(
                f"SELECT id FROM facts WHERE source_raw_id IN ({_marks(len(chunk))})"  # noqa: S608
                f" AND fkind IN {_SEED_FKINDS}",
                chunk,
            )
        }
    for chunk in _chunks(ents):
        q = _marks(len(chunk))
        found |= {
            r[0]
            for r in conn.execute(
                f"SELECT id FROM facts WHERE fkind IN {_SEED_FKINDS}"  # noqa: S608
                f" AND (subject_id IN ({q}) OR value_entity_id IN ({q}))"
                " UNION SELECT e.from_id FROM edges e JOIN facts f ON f.id = e.from_id"
                f" WHERE e.etype = 'mentions' AND e.to_id IN ({q}) AND f.fkind IN {_SEED_FKINDS}",
                [*chunk, *chunk, *chunk],
            )
        }
    return found - set(seeds)


def _joins(a: _Seed, b: _Seed, types: dict[str, tuple[str, str]]) -> bool:
    """Same kind of happening, never a seed and its own cause; then the same message, or a
    shared person within `CLUSTER_DAYS`."""
    if a.event_type is None or a.event_type != b.event_type:
        return False
    if a.id in b.causes or b.id in a.causes:
        return False  # a cause is its own happening, linked, never merged
    if a.raw_id and a.raw_id == b.raw_id:
        return True
    if not (_people(a, types) & _people(b, types)):
        return False
    gap = abs((_parse(a.anchor) - _parse(b.anchor)).total_seconds())
    return gap <= CLUSTER_DAYS * 86400


def _clusters(seeds: dict[str, _Seed], types: dict[str, tuple[str, str]]) -> list[list[_Seed]]:
    """Union-find over buckets (event type + message, event type + person), each sorted by
    time: only bucket neighbours are compared, so a backfill of years stays linear-ish."""
    parent = {k: k for k in seeds}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    order = sorted(seeds.values(), key=lambda s: (s.anchor, s.id))
    buckets: dict[tuple[str, str, str], list[_Seed]] = {}
    for s in order:
        if s.event_type is None:
            continue
        if s.raw_id:
            buckets.setdefault(("raw", s.event_type, s.raw_id), []).append(s)
        for person in _people(s, types):
            buckets.setdefault(("person", s.event_type, person), []).append(s)
    for members in buckets.values():
        for a, b in zip(members, members[1:], strict=False):
            if _joins(a, b, types):
                ra, rb = find(a.id), find(b.id)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[_Seed]] = {}
    for s in order:
        groups.setdefault(find(s.id), []).append(s)
    return sorted(groups.values(), key=lambda g: g[0].id)


# ---------------------------------------------------------------- one episode


@dataclass
class _Built:
    fields: dict[str, Any]
    seeds: list[str]
    causes: list[str]
    mentions: list[str]
    links: set[tuple[str, str]]
    read_set: list[ReadRef]


def _cause_texts(conn: sqlite3.Connection, ids: list[str]) -> list[tuple[str, str, str | None]]:
    """(id, English text, quote) of usable cause records."""
    out = []
    for cid in ids:
        rec = ledger.get_record(conn, cid)
        if rec is None or not rec.usable:
            continue
        text = next(
            (
                getattr(rec, a)
                for a in ("text_en", "summary_en", "chosen_text", "rule_en")
                if getattr(rec, a, None)
            ),
            "",
        )
        if text:
            out.append((cid, text, getattr(rec, "quote_orig", None)))
    return out


def _versions(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, int]:
    ids = list(dict.fromkeys(ids))
    out: dict[str, int] = {}
    for chunk in _chunks(ids):
        out.update(
            conn.execute(
                f"SELECT id, version FROM records WHERE id IN ({_marks(len(chunk))})",  # noqa: S608
                chunk,
            ).fetchall()
        )
    return out


def _build(
    conn: sqlite3.Connection, cluster: list[_Seed], types: dict[str, tuple[str, str]]
) -> _Built:
    """An episode from its seeds (sorted by anchor): records only, raw only as their quotes."""
    in_cluster = {s.id for s in cluster}
    causes = [c for s in cluster for c in s.causes if c not in in_cluster]
    causes = list(dict.fromkeys(causes))
    cause_rows = _cause_texts(conn, causes)
    causes = [c[0] for c in cause_rows]
    event_type = next((s.event_type for s in cluster if s.event_type), None)
    starts = sorted(s.valid_from for s in cluster if s.valid_from)
    ends = sorted(s.valid_to for s in cluster if s.valid_to)
    with_outcome = [s for s in cluster if s.outcome]
    outcome_src = with_outcome[-1] if with_outcome else None
    ents = sorted({e for s in cluster for e in s.entities if e != OWNER_ENTITY_ID})
    people = sorted(e for e in ents if types.get(e, ("",))[0] == "person")
    places = sorted(e for e in ents if types.get(e, ("",))[0] == "place")
    about = sorted(e for e in ents if e in types and e not in people and e not in places)
    place = next((s.place_value for s in cluster if s.place_value), None)
    if place is None and places:
        place = types[places[0]][1]

    texts = list(dict.fromkeys(s.text_en.strip() for s in cluster if s.text_en.strip()))
    summary = "; ".join(texts)
    if cause_rows:
        summary += ". Because of: " + "; ".join(t for _, t, _ in cause_rows)
    quotes = [
        q
        for q, raw, rid in [(s.quote, s.raw_id, s.id) for s in cluster]
        + [(q, None, cid) for cid, _, q in cause_rows]
        if q and _quote_ok(conn, q, raw, rid)
    ]
    if quotes:
        summary += ". Said: " + " ".join(f"«{q.strip()}»" for q in dict.fromkeys(quotes))
    fields = {
        "event_type": event_type,
        "span_start": starts[0] if starts else None,
        "span_end": ends[-1] if ends else None,
        "place": place,
        "outcome": outcome_src.outcome if outcome_src else None,
        "summary_en": _clip(summary, EPISODE_CHARS),
    }
    links = {(EpisodeRole.PARTICIPANT.value, e) for e in people}
    links |= {(EpisodeRole.PLACE.value, e) for e in places}
    links |= {(EpisodeRole.ABOUT.value, e) for e in about}
    links |= {(EpisodeRole.CAUSE.value, c) for c in causes}
    if outcome_src is not None:
        links.add((EpisodeRole.OUTCOME.value, outcome_src.id))
    mentions = [*people, *places, *about]
    seeds = [s.id for s in cluster]
    vers = _versions(conn, [*seeds, *causes, *mentions])
    read_set = [ReadRef(id=i, version=vers[i]) for i in dict.fromkeys([*seeds, *causes, *mentions])]
    return _Built(fields, seeds, causes, mentions, links, read_set)


def links_for(conn: sqlite3.Connection, derived_from: Sequence[str]) -> set[tuple[str, str]]:
    """The `episode_links` of an episode built from these sources (seeds and causes), as
    `_build` computes them; the applier calls it inside the commit."""
    seeds = _load_seeds(conn, derived_from)
    causes_of_others = {c for s in seeds.values() for c in s.causes if c in derived_from}
    cluster = sorted(
        (s for k, s in seeds.items() if k not in causes_of_others), key=lambda s: (s.anchor, s.id)
    )
    if not cluster:
        return set()
    types = _entity_types(conn, {e for s in cluster for e in s.entities})
    return _build(conn, cluster, types).links


# ---------------------------------------------------------------- existing episodes


@dataclass
class _Old:
    id: str
    version: int
    fields: dict[str, Any]
    sources: set[str]
    links: set[tuple[str, str]]

    @property
    def seed_sources(self) -> set[str]:
        return self.sources - {t for role, t in self.links if role == EpisodeRole.CAUSE.value}


def _episodes_from(conn: sqlite3.Connection, source_ids: Iterable[str]) -> dict[str, _Old]:
    """Usable episodes built from any of these records."""
    src = sorted(set(source_ids))
    ids: set[str] = set()
    for chunk in _chunks(src):
        ids |= {
            r[0]
            for r in conn.execute(
                "SELECT e.from_id FROM edges e JOIN records r ON r.id = e.from_id"
                f" WHERE e.etype = 'derived_from' AND e.to_id IN ({_marks(len(chunk))})"  # noqa: S608
                f" AND r.kind = 'episode' AND r.status IN {_USABLE}",
                chunk,
            )
        }
    out: dict[str, _Old] = {}
    for eid in sorted(ids):
        row = conn.execute(
            "SELECT r.version, e.event_type, e.span_start, e.span_end, e.place, e.outcome,"
            " e.summary_en FROM records r JOIN episodes e ON e.id = r.id WHERE r.id = ?",
            (eid,),
        ).fetchone()
        sources = {
            r[0]
            for r in conn.execute(
                "SELECT to_id FROM edges WHERE from_id = ? AND etype = 'derived_from'", (eid,)
            )
        }
        links = {
            (r[0], r[1])
            for r in conn.execute(
                "SELECT role, target_id FROM episode_links WHERE episode_id = ?", (eid,)
            )
        }
        fields = {k: row[k] for k in row.keys() if k != "version"}  # noqa: SIM118
        out[eid] = _Old(eid, row["version"], fields, sources, links)
    return out


def _superseded_by(conn: sqlite3.Connection, ids: Iterable[str]) -> set[str]:
    """Records these ids replaced (`supersedes` new → old), so an episode follows its facts."""
    ids = sorted(set(ids))
    out: set[str] = set()
    for chunk in _chunks(ids):
        out |= {
            r[0]
            for r in conn.execute(
                "SELECT to_id FROM edges WHERE etype = 'supersedes'"
                f" AND from_id IN ({_marks(len(chunk))})",  # noqa: S608
                chunk,
            )
        }
    return out


# ---------------------------------------------------------------- propose / commit


def _changed_facts(
    conn: sqlite3.Connection, *, since_ts: str = "", since_version: int = 0
) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT id FROM records WHERE kind = 'fact' AND version > ? AND updated_at > ?"
            " ORDER BY version, id",
            (since_version, since_ts),
        )
    ]


def _propose(conn: sqlite3.Connection, changed: Sequence[str]) -> Proposal:
    if not changed:
        return Proposal(ops=[], author_tier=AuthorTier.SYSTEM)
    pool = _load_seeds(conn, changed)
    olds: dict[str, _Old] = {}
    for _ in range(6):  # grow to whole clusters (same message, same person) and their episodes
        new = {k: v for k, v in _episodes_from(conn, {*changed, *pool}).items() if k not in olds}
        olds.update(new)
        want = _neighbours(conn, pool) | {s for o in olds.values() for s in o.seed_sources}
        more = {k: v for k, v in _load_seeds(conn, want - set(pool)).items() if k not in pool}
        if not more and not new:
            break
        pool.update(more)
    types = _entity_types(conn, {e for s in pool.values() for e in s.entities})
    ops: list[ProposalOp] = []
    taken: set[str] = set()
    for cluster in _clusters(pool, types):
        built = _build(conn, cluster, types)
        lineage = set(built.seeds) | _superseded_by(conn, built.seeds)
        mine = sorted(
            (o for o in olds.values() if o.id not in taken and o.seed_sources & lineage),
            key=lambda o: o.id,
        )
        sources = set(built.seeds) | set(built.causes)
        if (
            len(mine) == 1
            and mine[0].fields == built.fields
            and mine[0].sources == sources
            and mine[0].links == built.links
        ):
            taken.add(mine[0].id)
            continue
        group = f"episode:{built.seeds[0]}"
        new_id = ledger.new_id()
        record = {"kind": "episode", "id": new_id, "author_tier": AuthorTier.SYSTEM.value}
        record.update({k: v for k, v in built.fields.items() if v is not None})
        extra = {"derived_from": [*built.seeds, *built.causes], "mentions": built.mentions}
        if not mine:
            ops.append(
                ProposalOp(
                    op=OpKind.ADD,
                    group=group,
                    read_set=built.read_set,
                    payload=payload_dict(AddPayload(record=record, rule_match=False, **extra)),
                )
            )
            continue
        first, *rest = mine
        ops.append(
            ProposalOp(
                op=OpKind.SUPERSEDE,
                target=first.id,
                group=group,
                read_set=[*built.read_set, ReadRef(id=first.id, version=first.version)],
                payload=payload_dict(SupersedePayload(record=record, **extra)),
            )
        )
        for o in rest:
            ops.append(
                ProposalOp(
                    op=OpKind.SUPERSEDE,
                    target=o.id,
                    group=group,
                    read_set=[ReadRef(id=o.id, version=o.version)],
                    payload=payload_dict(SupersedePayload(by=new_id)),
                )
            )
        taken |= {o.id for o in mine}
    for o in sorted(olds.values(), key=lambda o: o.id):
        if o.id in taken or o.seed_sources & set(pool):
            continue  # FLAG only when every seed left use: lost support (§4.2)
        ops.append(
            ProposalOp(
                op=OpKind.FLAG,
                target=o.id,
                group=f"episode:{o.id}",
                read_set=[ReadRef(id=o.id, version=o.version)],
                payload=payload_dict(FlagPayload(status="needs_review")),
            )
        )
    return Proposal(ops=ops, author_tier=AuthorTier.SYSTEM)


def propose_episodes(conn: sqlite3.Connection, *, since: str) -> Proposal:
    """Episode ops for the facts changed after `since` (an ISO UTC timestamp; "" = all).
    Pure: reads the ledger, writes nothing."""
    return _propose(conn, _changed_facts(conn, since_ts=since or ""))


def _write_links(ctx: store.CommitCtx, episode_id: str, sources: Sequence[str]) -> None:
    ctx.conn.execute("DELETE FROM episode_links WHERE episode_id = ?", (episode_id,))
    ctx.conn.executemany(
        "INSERT OR IGNORE INTO episode_links (episode_id, role, target_id) VALUES (?, ?, ?)",
        [(episode_id, role, target) for role, target in sorted(links_for(ctx.conn, sources))],
    )


def apply_op(ctx: store.CommitCtx, op: ProposalOp) -> Sequence[str]:
    """`ledger.apply_op`, plus the `episode_links` of a new episode record (the ADD payload
    has no link field, §6.5)."""
    touched = ledger.apply_op(ctx, op)
    rec = op.payload.get("record")
    if isinstance(rec, dict) and rec.get("kind") == "episode" and rec.get("id"):
        _write_links(ctx, str(rec["id"]), list(op.payload.get("derived_from") or []))
    return touched


def commit(conn: sqlite3.Connection, proposal: Proposal) -> dict[str, int]:
    """Commit group by group (background priority): a stale or rejected group stays as it
    was for the next pass; counts only, never content."""
    out = {"committed": 0, "stale": 0, "rejected": 0}
    groups: dict[str, list[ProposalOp]] = {}
    for op in proposal.ops:
        groups.setdefault(op.group, []).append(op)
    for ops in groups.values():
        part = proposal.model_copy(update={"ops": ops})
        try:
            res = store.commit(conn, part, apply_op, priority="background")
        except store.OpRejected:
            out["rejected"] += 1
            continue
        out["committed"] += len(res.committed)
        out["stale"] += len(res.stale)
        out["rejected"] += len(res.rejected)
    return out


# ---------------------------------------------------------------- daily summaries


def summary_id(day: str) -> str:
    return f"sum-{day}"


def _anchors(conn: sqlite3.Connection, ids: Sequence[str]) -> dict[str, tuple[str, list[str]]]:
    """record id -> (anchor ts, active raw ids it was built from): the raw message's time,
    else the recording time."""
    out: dict[str, tuple[str, list[str]]] = {}
    for chunk in _chunks(list(dict.fromkeys(ids))):
        rec = dict(
            conn.execute(
                f"SELECT id, recorded_at FROM records WHERE id IN ({_marks(len(chunk))})",  # noqa: S608
                chunk,
            ).fetchall()
        )
        raws: dict[str, list[tuple[str, str, str]]] = {}
        for rid, raw_id, ts, status in conn.execute(
            "SELECT e.from_id, rc.id, rc.ts, rc.status FROM edges e"
            " JOIN raw_chunks rc ON rc.id = e.to_id"
            f" WHERE e.etype = 'derived_from' AND e.from_id IN ({_marks(len(chunk))})",  # noqa: S608
            chunk,
        ):
            raws.setdefault(rid, []).append((ts, raw_id, status))
        for rid, recorded in rec.items():
            got = sorted(raws.get(rid, []))
            anchor = got[0][0] if got else recorded
            out[rid] = (anchor, [r for _, r, st in got if st == "active"])
    return out


def _day_candidates(conn: sqlite3.Connection, day: str) -> list[str]:
    lo, hi = _day_window(day)
    raw_ids = [
        r[0] for r in conn.execute("SELECT id FROM raw_chunks WHERE ts >= ? AND ts < ?", (lo, hi))
    ]
    ids: set[str] = set()
    for chunk in _chunks(raw_ids):
        ids |= {
            r[0]
            for r in conn.execute(
                "SELECT from_id FROM edges WHERE etype = 'derived_from'"
                f" AND to_id IN ({_marks(len(chunk))})",  # noqa: S608
                chunk,
            )
        }
    ids |= {
        r[0]
        for r in conn.execute(
            "SELECT id FROM records WHERE recorded_at >= ? AND recorded_at < ?", (lo, hi)
        )
    }
    out: list[str] = []
    for chunk in _chunks(sorted(ids)):
        out += [
            r[0]
            for r in conn.execute(
                f"SELECT id FROM records WHERE id IN ({_marks(len(chunk))})"  # noqa: S608
                f" AND status IN {_USABLE} AND kind NOT IN {_SUMMARY_SKIP_KINDS}",
                chunk,
            )
        ]
    return out


def _line(conn: sqlite3.Connection, record_id: str, raws: list[str]) -> str | None:
    rec = ledger.get_record(conn, record_id)
    if rec is None or not rec.usable:
        return None
    kind = rec.kind.value
    text = next(
        (getattr(rec, a) for a in ("text_en", "chosen_text", "rule_en") if getattr(rec, a, None)),
        None,
    )
    if text is None and kind == "work_state":
        text = rec.state.goal  # type: ignore[union-attr]
    if not text:
        return None
    prefix = {"impression": "Impression: ", "decision": "Decided: ", "lesson": "Lesson: "}
    line = prefix.get(kind, "") + _clip(text, LINE_CHARS)
    quote = getattr(rec, "quote_orig", None)
    if quote and _quote_ok(conn, quote, raws[0] if raws else None, record_id):
        line += f" («{_clip(quote, LINE_CHARS)}»)"
    return line


def propose_daily_summary(conn: sqlite3.Connection, day: str) -> Proposal:
    """Built from usable records, never from the previous summary."""
    empty = Proposal(ops=[], author_tier=AuthorTier.SYSTEM)
    anchors = _anchors(conn, _day_candidates(conn, day))
    picked = sorted(
        ((a, rid, raws) for rid, (a, raws) in anchors.items() if day_of(a) == day),
        key=lambda x: (x[0], x[1]),
    )
    lines: list[str] = []
    used: list[tuple[str, list[str]]] = []
    for _, rid, raws in picked:
        if len(lines) >= SUMMARY_LINES:
            break
        line = _line(conn, rid, raws)
        if line:
            lines.append(f"- {line}")
            used.append((rid, raws))
    if not lines:
        return empty
    vers = _versions(conn, [rid for rid, _ in used])
    raw_ids = list(dict.fromkeys(r for _, raws in used for r in raws))
    revs: dict[str, int] = {}
    for chunk in _chunks(raw_ids):
        revs.update(
            conn.execute(
                f"SELECT id, rev FROM raw_chunks WHERE id IN ({_marks(len(chunk))})",  # noqa: S608
                chunk,
            ).fetchall()
        )
    read_set = [ReadRef(id=rid, version=vers[rid]) for rid, _ in used]
    read_set += [ReadRef(id=r, version=revs[r], src=ReadSrc.RAW) for r in raw_ids if r in revs]
    content = f"{day}\n" + "\n".join(lines)
    did = summary_id(day)
    old = conn.execute(
        "SELECT content, read_set, valid FROM derived WHERE id = ?", (did,)
    ).fetchone()
    if old is not None and old["valid"] == 1 and old["content"] == content:
        want = sorted(
            [r.id, r.version] if r.src is ReadSrc.RECORD else [r.id, r.version, r.src.value]
            for r in read_set
        )
        if sorted(json.loads(old["read_set"] or "[]")) == want:
            return empty  # nothing it rests on changed
    derived = DerivedIn(id=did, dkind=DerivedKind.SUMMARY, dkey=day, content=content)
    op = ProposalOp(
        op=OpKind.ADD,
        group=f"summary:{day}",
        read_set=read_set,
        payload=payload_dict(AddPayload(derived=derived)),
    )
    return Proposal(ops=[op], author_tier=AuthorTier.SYSTEM)


def rebuild_summary(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    """The `rebuild_derived` runner for `summary` items (§7.3): rebuilt from the ledger and
    raw. A stale read set raises, so the job goes back to the queue (attempts + 1)."""
    day = payload.get("dkey")
    if not day:
        return
    got = commit(conn, propose_daily_summary(conn, str(day)))
    if got["stale"]:
        raise RuntimeError("summary read set went stale")


# ---------------------------------------------------------------- the night


def _changed_records(conn: sqlite3.Connection, since: int, upto: int) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT id FROM records WHERE version > ? AND version <= ?"
            f" AND kind NOT IN {_SUMMARY_SKIP_KINDS} ORDER BY version DESC, id",
            (since, upto),
        )
    ]


def night_step(night: Any) -> dict[str, int]:
    """Episodes for the facts changed since this step's own watermark, then the summaries of
    the days those changes touch (and any invalid one), newest first, at most `night.cap`."""
    from .workers import nightly

    conn = night.conn
    prev = nightly.last_mark(conn) or {}
    ep_since = int(prev.get(_MARK_EPISODES) or 0)  # type: ignore[arg-type]
    sum_since = int(prev.get(_MARK_SUMMARIES) or 0)  # type: ignore[arg-type]
    upto = int(night.upto_version)
    got = commit(conn, _propose(conn, _changed_facts(conn, since_version=ep_since)))
    out = {"episodes": got["committed"], "episodes_stale": got["stale"]}
    days = {day_of(a) for a, _ in _anchors(conn, _changed_records(conn, sum_since, upto)).values()}
    days |= {
        r[0]
        for r in conn.execute(
            "SELECT dkey FROM derived WHERE dkind = 'summary' AND valid = 0 AND dkey IS NOT NULL"
        )
    }
    built = stale = 0
    for day in sorted(days, reverse=True)[: max(0, int(night.cap))]:
        res = commit(conn, propose_daily_summary(conn, day))
        built += res["committed"]
        stale += res["stale"]
    out.update({"summaries": built, "summaries_stale": stale})
    night.mark[_MARK_EPISODES] = upto if not got["stale"] else ep_since
    night.mark[_MARK_SUMMARIES] = upto if not stale else sum_since
    return out


def install() -> None:
    """Register the night step and the summary rebuild (idempotent; runs at import)."""
    from .recall import core_view
    from .workers import nightly

    nightly.register_step(NIGHT_STEP, night_step)
    core_view.register_rebuild(DerivedKind.SUMMARY.value, rebuild_summary)


install()
