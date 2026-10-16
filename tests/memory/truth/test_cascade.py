"""cascade proof (§8.1, §8.4): a correction cascade through the hot path, the
re-derivation job, an inferred link, the night step and the inline cap. Fictional people."""

from __future__ import annotations

import pytest

from koob.memory import api, justify, ledger, store
from koob.memory.model import OWNER_ENTITY_ID, AuthorTier, OpKind, Proposal, ProposalOp
from koob.memory.workers import nightly

from ..inspect._world import add, fact, person, raw, status, turn

LENA, DIMA, MASHA = "ent-lena", "ent-dima", "ent-masha"
OTHER = AuthorTier.OTHER
OFFSITE = "c-offsite"
OFFSITE_EN = "The team offsite is in Sochi"


@pytest.fixture
def db(mem):
    ledger.seed_relations(mem)
    ledger.seed_owner(mem)
    return mem


def _just(*groups: list[str], prov: str = "rule") -> list[dict]:
    return [
        {"group_no": g, "member": m, "provenance": prov} for g, ms in enumerate(groups) for m in ms
    ]


def _fix(conn, target: str, words: str, *, n: int, **kw):
    res = api.fix(
        f"m:{target}", words, turn=turn(words, n=n), because_quote=words, store=conn, **kw
    )
    assert res.outcome == "committed", res
    return res


def _link(conn, target: str, member: str, group: int, prov: str = "rule") -> None:
    j = {"target": target, "group_no": group, "member": member, "provenance": prov}
    op = ProposalOp(op=OpKind.LINK, target=target, group=target, payload={"justification": j})
    assert store.commit(conn, Proposal(ops=[op], author_tier=AuthorTier.USER), ledger.apply_op).ok


def _review(conn, rid: str) -> int:
    return conn.execute("SELECT review FROM records WHERE id = ?", (rid,)).fetchone()[0]


def _offsite(conn) -> None:
    """The user's fact «корпоратив будет в Сочи», heard from Лена and, separately, from Дима."""
    for eid, name in ((LENA, "Лена"), (DIMA, "Дима"), (MASHA, "Маша")):
        add(conn, person(eid, name))
    raw(conn, "r-lena", "Лена: корпоратив будет в Сочи, в сентябре", tier=OTHER, author=LENA,
        chat_id="tg:1")  # fmt: skip
    raw(conn, "r-dima", "Дима: корпоратив в Сочи, уже бронируют отель", tier=OTHER,
        author=DIMA, chat_id="tg:2")  # fmt: skip
    add(conn, fact("f-lena", LENA, "other", "offsite in Sochi", "Lena says the offsite is in Sochi",
                   "корпоратив будет в Сочи", tier=OTHER, speaker=LENA, raw="r-lena"))  # fmt: skip
    add(conn, fact("f-dima", DIMA, "other", "offsite in Sochi", "Dima says the offsite is in Sochi",
                   "корпоратив в Сочи", tier=OTHER, speaker=DIMA, raw="r-dima"))  # fmt: skip
    add(conn, fact(OFFSITE, OWNER_ENTITY_ID, "other", "offsite in Sochi", OFFSITE_EN,
                   "корпоратив будет в Сочи"),
        justifications=_just(["f-lena"], ["f-dima"]))  # fmt: skip


def _shown(conn, n: int) -> tuple[bool, bool]:
    core = api.core(thread_id=f"thread-{n}", store=conn).text
    basket = repr(
        api.recall(turn("Где будет корпоратив? В Сочи?", n=n, thread=f"t-{n}"), store=conn)
    )
    return OFFSITE_EN in core, OFFSITE in basket


def test_correction_cascade_two_supports_then_none(db):
    """The proof: two independent supports; losing one keeps the fact, losing the second
    sends it to `needs_review` (§8.4 unsupported); core and basket stop showing it."""
    _offsite(db)
    assert justify.independent(db, OFFSITE, (0, 1))
    assert _shown(db, 1) == (True, True)

    _fix(db, "f-lena", "нет, Лена говорила про Тверь, а не про Сочи", n=2)
    assert status(db, OFFSITE) == "active" and _review(db, OFFSITE) == 0
    assert _shown(db, 3) == (True, True)

    res = _fix(db, "f-dima", "нет, Дима тоже имел в виду Тверь", n=4)
    assert res.dependents_to_review == 1
    assert status(db, OFFSITE) == "needs_review"
    assert _shown(db, 5) == (False, False)
    assert store.check_invariants(db) == []

    # background re-derivation: nothing new holds it, so it stays out ...
    assert store.run_queued_jobs(db, justify.REDERIVE_JOB) >= 1
    assert status(db, OFFSITE) == "needs_review"
    # ... until an independent source arrives: then it is proposed back and committed
    raw(db, "r-masha", "Маша: корпоратив точно в Сочи, билеты уже пришли", tier=OTHER,
        author=MASHA, chat_id="tg:3")  # fmt: skip
    _link(db, OFFSITE, "r-masha", 2)
    assert justify.rederive(db, OFFSITE)
    assert status(db, OFFSITE) == "active"
    assert _shown(db, 6) == (True, True)


def test_inferred_link_leads_only_to_review(db):
    """§8.4: an inferred link that breaks leaves the conclusion usable, flagged `review`."""
    raw(db, "r-run", "Бегаю по утрам в парке")
    add(db, fact("f-run", OWNER_ENTITY_ID, "other", "morning runs", "The owner runs in the park",
                 "Бегаю по утрам в парке", raw="r-run"))  # fmt: skip
    add(db, fact("f-shoes", OWNER_ENTITY_ID, "other", "trail shoes", "The owner needs trail shoes",
                 "нужны кроссовки"),
        justifications=_just(["f-run"], prov="inferred"))  # fmt: skip
    _fix(db, "f-run", "это неправда", n=2, retract=True)
    assert status(db, "f-run") == "retracted"
    assert status(db, "f-shoes") == "active" and _review(db, "f-shoes") == 1
    assert justify.outcome(db, "f-shoes", ["f-run"]) == "review"


def test_needs_review_cascades_downstream(db):
    """A conclusion that goes to `needs_review` is a broken premise for its own dependents."""
    raw(db, "r-b", "У Оли день рождения 3 марта")
    add(db, person("ent-olya", "Оля"))
    add(db, fact("f-bday", "ent-olya", "birthday", "03-03", "Olya's birthday is 3 March",
                 "У Оли день рождения 3 марта", raw="r-b"))  # fmt: skip
    add(db, fact("f-party", OWNER_ENTITY_ID, "other", "party 3 March", "Party for Olya on 3 March",
                 "вечеринка для Оли 3 марта"),
        justifications=_just(["f-bday"], prov="stated"))  # fmt: skip
    add(db, fact("f-cake", OWNER_ENTITY_ID, "other", "cake on 2 March", "Order a cake on 2 March",
                 "торт заказать 2 марта"),
        justifications=_just(["f-party"], prov="stated"))  # fmt: skip
    _fix(db, "f-bday", "нет, у Оли день рождения 5 марта", n=2)
    assert status(db, "f-party") == "needs_review"
    assert status(db, "f-cake") == "needs_review"


def test_night_step_cascades_worker_supersessions(db):
    """§6.3 night «cascades»: a supersession by rule (no hot cascade) is judged at night."""
    add(db, fact("f-msk", OWNER_ENTITY_ID, "lives_in", "Moscow", "The owner lives in Moscow",
                 "живу в Москве"))  # fmt: skip
    add(db, fact("f-tz", OWNER_ENTITY_ID, "timezone", "MSK", "The owner's timezone is MSK",
                 "живу в Москве"),
        justifications=_just(["f-msk"]))  # fmt: skip
    add(db, fact("f-spb", OWNER_ENTITY_ID, "lives_in", "Saint Petersburg",
                 "The owner lives in Saint Petersburg", "теперь живу в Питере"))  # fmt: skip
    assert status(db, "f-msk") == "superseded" and _review(db, "f-tz") == 0
    counts = nightly.run(db, force=True)
    assert counts.get("cascade_flipped") == 1, counts
    assert status(db, "f-tz") == "active" and _review(db, "f-tz") == 1


def test_inline_cap_leaves_the_rest_to_a_durable_job(db, monkeypatch):
    """Past the inline cap the cascade queues itself; the job finishes it."""
    monkeypatch.setattr(justify, "INLINE_CAP", 2)
    raw(db, "r-wd", "Работаю из дома по средам")
    add(db, fact("f-wed", OWNER_ENTITY_ID, "other", "home on Wednesdays",
                 "The owner works from home on Wednesdays", "Работаю из дома по средам",
                 raw="r-wd"))  # fmt: skip
    deps = [f"f-dep{i}" for i in range(5)]
    for i, rid in enumerate(deps):
        add(db, fact(rid, OWNER_ENTITY_ID, "other", f"wednesday plan {i}",
                     f"Wednesday plan number {i}", f"в среду план {i}"),
            justifications=_just(["f-wed"], prov="stated"))  # fmt: skip
    _fix(db, "f-wed", "нет, по средам я в офисе", n=2)
    assert sum(status(db, r) == "needs_review" for r in deps) == 2
    assert store.run_queued_jobs(db, justify.CASCADE_JOB) >= 1
    while store.run_queued_jobs(db, justify.CASCADE_JOB):
        pass
    assert all(status(db, r) == "needs_review" for r in deps)
