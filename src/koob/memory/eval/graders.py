"""Code-check graders (README: expect, Behaviours, Channels, Matching). Code first; a judge only for free text."""

from __future__ import annotations

import re
from typing import Any

from .matchers import describe, find, normalize, shown_as_current
from .observe import Check, Observation
from .schema import CHANNELS, ActionCheck, Expect, RecordCheck
from .sut import NotBuilt
from .timeutil import to_utc

PASS, FAIL, SKIP = "pass", "fail", "skip"
NOT_BUILT = "not built "  # skip reasons that start with this make a wording `incomplete`

# which hook feeds a behaviour / a channel; a check that reads an unbuilt hook is skipped
_BEHAVIOUR_HOOKS = {
    "offer": ["offers"], "silence": ["offers"], "offer_many": ["offers"],
    "ask_one": ["basket"], "none_then_ask": ["basket"],
    "answer": ["basket", "store_tools"],  # either one can make the "answer" behaviour
    "action": ["actions"], "no_action": ["actions"],
}  # fmt: skip
_CHANNEL_HOOKS = {
    "basket": "basket", "core": "core", "update_block": "update_block",
    "store_tools": "store_tools", "assoc": "offers", "actions": "actions",
}  # fmt: skip


def _nb(obs: Observation, hooks: list[str], name: str, *, all_: bool = False) -> Check | None:
    """A `not built ...` skip when the hooks a check reads were unbuilt in this observation."""
    lost = [h for h in hooks if h in obs.unavailable]
    if not lost or (all_ and len(lost) < len(hooks)):
        return None
    lanes = sorted({obs.unavailable[h] for h in lost})
    return Check(name, SKIP, f"{NOT_BUILT}{', '.join(lanes)}: {', '.join(lost)}")


def _c(name: str, ok: bool, reason: str = "") -> Check:
    return Check(name, PASS if ok else FAIL, "" if ok else reason)


def _skip(name: str, reason: str) -> Check:
    return Check(name, SKIP, reason)


def _is_ref(v: Any) -> bool:
    return isinstance(v, str) and re.match(r"^(contact|tg|mail|phone):|^@", v) is not None


def _alts_list(items: list[Any]) -> list[Any]:
    return list(items)


# ------------------------------------------------------------------ text checks


def _primary_gone(obs: Observation, name: str) -> Check | None:
    if obs.level != "memory":
        return None
    return _nb(obs, ["core", "basket", "store_tools"], name, all_=True)


def check_include(obs: Observation, items: list[Any], name: str = "must_include") -> list[Check]:
    if items and (gone := _primary_gone(obs, name)):
        return [gone]
    return [_c(name, find(obs.primary, it), f"missing: {describe(it)}") for it in items]


def check_exclude(
    obs: Observation, items: list[Any], name: str = "must_not_include"
) -> list[Check]:
    if items and (gone := _primary_gone(obs, name)):
        return [gone]
    # a mention its own clause negates («без кинзы», "not Boulder anymore") is not present
    return [
        _c(name, not shown_as_current(obs.primary, it, history=False), f"present: {describe(it)}")
        for it in items
    ]


def _occurrence_allowed(channel: str, text: str, value: Any) -> bool:
    """`allow: history`: every occurrence is marked outdated - in the CLI prompt an update block
    follows it; elsewhere its own clause marks it as history (`matchers.shown_as_current`, the
    same rule the bench's view judge uses, for every system alike)."""
    from .matchers import compile_value  # local: keeps the import list short

    if channel != "cli_prompt":
        return not shown_as_current(text, value)
    nt = normalize(text)
    return all(
        "<memory-update" in nt[m.end() :] for p in compile_value(value) for m in p.finditer(nt)
    )


def _action_hits(obs: Observation, value: str) -> bool:
    """A forbidden `contact:`/`tg:` ref in the `actions` channel: any action aimed at that target."""
    ids = (
        set(obs.contact_ids.get(value.split(":", 1)[1], []))
        if value.startswith("contact:")
        else {value}
    )
    return any(a.get("target") in ids or a.get("target") == value for a in obs.actions)


def check_forbidden(obs: Observation, expect: Expect) -> list[Check]:
    out: list[Check] = []
    for fb in expect.forbidden:
        channels = sorted(CHANNELS - {"all"}) if "all" in fb.channels else fb.channels
        if "all" in fb.channels:
            channels = [c for c in channels if c in obs.channels or c == "actions"]
        for ch in channels:
            for v in fb.values:
                name = f"forbidden[{ch}]"
                if ch == "actions" and _is_ref(v):
                    out.append(_c(name, not _action_hits(obs, v), f"action aimed at {v}"))
                    continue
                if ch == "answer":
                    text = obs.answer
                elif ch == "actions":
                    text = " \n".join(" ".join(a.get("strings", [])) for a in obs.actions)
                else:
                    text = obs.channels.get(ch)
                if text is None:
                    gone = _nb(obs, [_CHANNEL_HOOKS.get(ch, "")], name)
                    out.append(
                        gone or _skip(name, f"channel {ch} not observable at {obs.level} level")
                    )
                    continue
                if (
                    not shown_as_current(text, v, history=False)  # absent, or only negated
                    or fb.allow == "history"
                    and _occurrence_allowed(ch, text, v)
                ):
                    out.append(Check(name, PASS))
                else:
                    out.append(Check(name, FAIL, f"{describe(v)} found in {ch}"))
    return out


# ------------------------------------------------------------------ behaviour


def count_questions(text: str) -> int:
    return len(re.findall(r"\?", text))


# behaviours only the brain produces: a memory-level run has no turn, so it cannot refute them
BRAIN_ONLY = frozenset({"action", "refuse_line"})
BRAIN_SKIP = "hand actions need the brain (e2e)"


def check_behaviour(obs: Observation, expect: Expect) -> list[Check]:
    wanted = [expect.behaviour] if expect.behaviour else []
    any_of = list(expect.behaviour_any)
    out: list[Check] = []
    if obs.level == "memory" and wanted and wanted[0] in BRAIN_ONLY:
        out.append(_skip("behaviour", BRAIN_SKIP))
        wanted = []
    if obs.level == "memory" and BRAIN_ONLY & set(any_of):
        out.append(_skip("behaviour_any", BRAIN_SKIP))  # the brain may still pick that one
        any_of = []
    if wanted and (
        gone := _nb(
            obs, _BEHAVIOUR_HOOKS.get(wanted[0], []), "behaviour", all_=wanted[0] == "answer"
        )
    ):
        out.append(gone)
        wanted = []
    if any_of and (
        gone := _nb(
            obs,
            sorted({h for b in any_of for h in _BEHAVIOUR_HOOKS.get(b, ["?"])}),
            "behaviour_any",
            all_=True,
        )
    ):
        out.append(gone)
        any_of = []
    if wanted:
        out.append(
            _c(
                "behaviour",
                wanted[0] in obs.behaviours,
                f"expected {wanted[0]}, saw {sorted(obs.behaviours) or 'nothing'}",
            )
        )
    if any_of:
        out.append(
            _c(
                "behaviour_any",
                bool(set(any_of) & obs.behaviours),
                f"expected one of {any_of}, saw {sorted(obs.behaviours) or 'nothing'}",
            )
        )
    if expect.ask and (gone := _nb(obs, ["basket"], "ask")):
        out.append(gone)
    elif expect.ask:
        qt = obs.questions_text or obs.candidates_text
        for group in expect.ask.must_name:
            out.append(
                _c("ask.must_name", find(qt, group), f"candidate not named: {describe(group)}")
            )
        for group in expect.ask.features:
            out.append(
                _c("ask.features", find(qt, group), f"no distinguishing feature: {describe(group)}")
            )
        nq = count_questions(obs.answer or "") if obs.answer else None
        if nq is not None:
            out.append(_c("ask.max_questions", nq <= expect.ask.max_questions, f"{nq} questions"))
    if expect.offer and (gone := _nb(obs, ["offers"], "offer")):
        out.append(gone)
    elif expect.offer:
        o = expect.offer
        out.append(_c("offer.max", len(obs.offers) <= o.max, f"{len(obs.offers)} offers"))
        text = (
            " ".join(x["text"] for x in obs.offers)
            + " "
            + " ".join(x.get("grounds_text", "") for x in obs.offers)
        )
        for g in o.grounds:
            out.append(_c("offer.grounds", find(text, g), f"ground missing: {describe(g)}"))
        if o.check_phrase:
            out.append(
                _c(
                    "offer.check_phrase",
                    any(_has_check(x["text"]) for x in obs.offers),
                    "offer has no check phrase",
                )
            )
    return out


def _has_check(text: str) -> bool:
    t = normalize(text)
    return t.rstrip().endswith("?") or bool(re.search(r"проверить|сверить|уточнить|check", t))


# ------------------------------------------------------------------ actions, denials, lines


def _target_ok(obs: Observation, spec: str, target: str | None) -> bool:
    if spec == "any":
        return True
    if spec.startswith("contact:"):
        ids = obs.contact_ids.get(spec.split(":", 1)[1], [])
        return target in ids or target == spec
    return target == spec


def check_actions(obs: Observation, specs: list[ActionCheck]) -> list[Check]:
    if specs and (gone := _nb(obs, ["actions"], "actions")):
        return [gone]
    if specs and obs.level == "memory":  # the probe's actions are the brain's; none ran
        return [_skip("actions", BRAIN_SKIP)]
    out: list[Check] = []
    for a in specs:
        name = f"actions[{a.tool}->{a.target}]"
        hits = [
            x
            for x in obs.actions
            if x.get("tool") == a.tool and _target_ok(obs, a.target, x.get("target"))
        ]
        if a.count is None:
            out.append(_c(name, len(hits) >= 1, "no such action"))
        else:
            out.append(
                _c(name, len(hits) == a.count, f"{len(hits)} actions, expected exactly {a.count}")
            )
        blob = [" ".join(h.get("strings", [])) for h in hits]
        for v in a.text_must_include:
            ok = bool(blob) and all(find(b, v) for b in blob)
            out.append(_c(name + ".text", ok, f"action text lacks {describe(v)}"))
        for v in a.text_must_not_include:
            out.append(
                _c(
                    name + ".text",
                    not any(find(b, v) for b in blob),
                    f"action text has {describe(v)}",
                )
            )
    return out


def check_denied(obs: Observation, specs: list) -> list[Check]:
    if specs and (gone := _nb(obs, ["denials"], "denied")):
        return [gone]
    out: list[Check] = []
    for d in specs:
        ok = any(
            x.get("tool") == d.tool
            and (d.target is None or _target_ok(obs, d.target, x.get("target")))
            and (not d.reason_any or x.get("reason") in d.reason_any)
            for x in obs.denials
        )
        out.append(
            _c(f"denied[{d.tool}]", ok, f"no denial for {d.tool} with reason in {d.reason_any}")
        )
    return out


def check_lines(obs: Observation, spec: Any) -> list[Check]:
    key = spec.of or "_probe"
    text = obs.lines.get(key)
    if text is None:
        return [_skip("lines", f"no owner-facing text recorded for {key}")]
    out = [_c("lines.must", find(text, v), f"line missing: {describe(v)}") for v in spec.must]
    out += [
        _c("lines.must_not", not find(text, v), f"line present: {describe(v)}")
        for v in spec.must_not
    ]
    return out


# ------------------------------------------------------------------ records, notifications, state


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in ("value_text", "value_norm", "text_en", "text", "summary_en", "content")
    )


def _eq(a: Any, b: Any) -> bool:
    if isinstance(b, list):
        return any(_eq(a, x) for x in b)
    if isinstance(a, list):
        return any(_eq(x, b) for x in a)
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    sa, sb = normalize(str(a)), normalize(str(b))
    return sa == sb or (len(sb) >= 8 and sa.startswith(sb))  # dates match by day prefix


def _row_matches(row: dict[str, Any], where: dict[str, Any]) -> bool:
    for k, v in where.items():
        if k == "value":
            if not find(_row_text(row), v):
                return False
        elif k == "target" and row.get("kind") == "justification":
            if not find(str(row.get("target") or ""), v):  # the conclusion's text, a substring
                return False
        elif k == "kind" and isinstance(v, list):
            if row.get("kind") not in v:
                return False
        elif k == "status" and v is None:
            continue
        elif not _eq(row.get(k), v):
            return False
    return True


def _expect_ok(row: dict[str, Any], key: str, want: Any) -> tuple[bool, str]:
    if key == "labels":
        have = set(row.get("labels") or [])
        return set(want or []) <= have, f"labels {sorted(have)} lack {want}"
    if key == "labels_any":
        return bool(set(want) & set(row.get("labels") or [])), f"labels_any {want}"
    if key == "applicability":
        have = row.get("applicability") or {}
        bad = [k for k, v in (want or {}).items() if not _eq(have.get(k), v)]
        return not bad, f"applicability differs on {bad}"
    if key == "read_set_includes":
        return set(want or []) <= set(row.get("read_event_ids") or []), "read set lacks events"
    if key in ("evidence_count_min", "exceptions_count_min"):
        base = key.replace("_min", "")
        return (row.get(base) or 0) >= want, f"{base}={row.get(base)}"
    if key == "domains":
        return set(want or []) <= set(row.get("domains") or []), "domains differ"
    if key.startswith("state."):
        blob = str((row.get("state") or {}).get(key.split(".", 1)[1]) or "")
        vals = want if isinstance(want, list) else [want]
        return all(find(blob, v) for v in vals), f"{key} lacks {want}"
    if key == "state":
        state = row.get("state")
        vals = want if isinstance(want, list) else [want]
        if isinstance(state, dict | list):  # a work state's JSON: each listed value appears
            return all(find(str(state), v) for v in vals), "state lacks values"
        # a scalar state (an intention's, a loop's): a list names alternatives, like any text
        return any(_eq(state, v) for v in vals), f"state={state!r}, expected one of {vals}"
    if key == "subject_or_speaker":
        return _eq(row.get("subject"), want) or _eq(
            row.get("speaker"), want
        ), "subject/speaker differ"
    if key == "quote_orig":
        return find(str(row.get("quote_orig") or ""), want), "quote differs"
    return _eq(row.get(key), want), f"{key}={row.get(key)!r}, expected {want!r}"


def _count_ok(n: int, spec: int | str | None) -> bool:
    if spec is None:
        return n >= 1
    if isinstance(spec, int):
        return n == spec
    m = re.fullmatch(r"(>=|<=)?(\d+)", spec)
    op, num = m.group(1), int(m.group(2))  # type: ignore[union-attr]
    return n >= num if op == ">=" else n <= num if op == "<=" else n == num


def check_records(obs: Observation, specs: list[RecordCheck]) -> list[Check]:
    if not specs:
        return []
    if obs.inspector is None:
        return [_skip("records", "no inspector")]
    out: list[Check] = []
    for rc in specs:
        name = f"records[{rc.where.get('kind', '?')}:{describe(rc.where.get('value', ''))[:40]}]"
        try:
            found = obs.inspector.records(rc.where)
        except NotBuilt as nb:
            obs.unavailable["records"] = nb.lane
            out.append(Check(name + ".count", SKIP, f"{NOT_BUILT}{nb.lane}: records"))
            continue
        rows = [r for r in found if _row_matches(r, rc.where)]
        out.append(
            _c(
                name + ".count",
                _count_ok(len(rows), rc.count),
                f"{len(rows)} matching records, wanted {'>=1' if rc.count is None else rc.count}",
            )
        )
        if not rc.expect or not rows:
            continue
        bad: list[str] = []
        for key, want in rc.expect.items():
            if not any(_expect_ok(r, key, want)[0] for r in rows):
                bad.append(_expect_ok(rows[0], key, want)[1])
        out.append(_c(name + ".expect", not bad, "; ".join(bad)))
    return out


def check_notifications(obs: Observation, spec: Any) -> list[Check]:
    if gone := _nb(obs, ["notifications"], "notifications"):
        return [gone]
    lo, hi = (to_utc(_dt(x), obs.tz) for x in spec.window)
    inside = [n for n in obs.notifications if lo <= n["at"] <= hi]
    text = " ".join(n["text"] for n in inside)
    out = [
        _c("notifications.must_not", not find(text, v), f"notified: {describe(v)}")
        for v in spec.must_not_contain
    ]
    for must in spec.must:
        n = sum(1 for x in inside if all(find(x["text"], v) for v in must.contains))
        ok = (
            (must.count is None or n == must.count)
            and (must.count_min is None or n >= must.count_min)
            and (must.count_max is None or n <= must.count_max)
        )
        if must.count is None and must.count_min is None and must.count_max is None:
            ok = n >= 1
        out.append(
            _c("notifications.must", ok, f"{n} notices with {[describe(c) for c in must.contains]}")
        )
    return out


def _dt(x: Any):  # type: ignore[no-untyped-def]
    from datetime import datetime

    return x if isinstance(x, datetime) else datetime.fromisoformat(str(x))


def check_state(obs: Observation, expect: Expect) -> list[Check]:
    out: list[Check] = []
    if expect.journal:
        if obs.forget_journal_entries is None:
            out.append(_skip("journal", "forget journal not observable"))
        else:
            out.append(
                _c(
                    "journal",
                    obs.forget_journal_entries == expect.journal.forget_entries,
                    f"{obs.forget_journal_entries} forget entries, expected {expect.journal.forget_entries}",
                )
            )
    for chat, mode in expect.chat_policy.items():
        have = (obs.chat_policy or {}).get(chat)
        out.append(_c(f"chat_policy[{chat}]", have == mode, f"mode is {have}, expected {mode}"))
    if expect.scan:
        if obs.scan is None:
            out.append(_skip("scan", "canary sweep not run"))
        else:
            for cid in expect.scan.canaries:
                hits = obs.scan.get(cid, [])
                out.append(_c(f"scan[{cid}]", not hits, f"{len(hits)} unlisted hit(s): {hits[:3]}"))
    return out


# ------------------------------------------------------------------ entry points


def grade_expect(obs: Observation, expect: Expect) -> list[Check]:
    """All code checks of one `expect` block against one observation."""
    checks: list[Check] = []
    checks += check_include(obs, expect.must_include)
    checks += check_exclude(obs, expect.must_not_include)
    checks += check_forbidden(obs, expect)
    checks += check_behaviour(obs, expect)
    checks += check_actions(obs, expect.actions)
    checks += check_denied(obs, expect.denied)
    if expect.lines:
        checks += check_lines(obs, expect.lines)
    checks += check_records(obs, expect.records)
    if expect.notifications:
        checks += check_notifications(obs, expect.notifications)
    checks += check_state(obs, expect)
    return checks


def grade_any(obs: Observation, blocks: list[Expect]) -> list[Check]:
    """`expect_any`: passes when any one block passes."""
    results = [grade_expect(obs, b) for b in blocks]
    for r in results:
        if not any(c.failed for c in r):
            return [Check("expect_any", PASS)]
    reasons = " | ".join("; ".join(c.reason for c in r if c.failed)[:160] for r in results)
    return [Check("expect_any", FAIL, f"no block passed: {reasons}")]
