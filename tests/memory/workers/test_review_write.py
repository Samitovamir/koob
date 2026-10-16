"""Regression tests for the memory write path:
labelled known facts never ride to the cloud; a fact's tier follows where its VALUE came
from (§13.1); a mixed batch routes on the union of its labels (§5.4)."""

from __future__ import annotations

from koob.memory import policy
from koob.memory.model import AuthorTier, Label, ReadSrc, Route
from koob.memory.workers import extract, scheduler

from .. import _ledger as L
from ._kit import SASHA, Gold, batch, fact, jobs, raw


def _tier_of(conn, relation: str) -> str:
    return conn.execute(
        "SELECT r.author_tier FROM facts f JOIN records r ON r.id = f.id WHERE f.relation = ?",
        (relation,),
    ).fetchone()[0]


def test_labelled_known_fact_never_sent_to_cloud(env):
    """§5.4: a HEALTH fact about the user stays out of a cloud batch's known facts;
    an unlabeled final one rides along."""
    conn, clk = env
    L.add(conn, L.fact("fh", "self", "works_at", "HIV clinic on Lenina", labels=Label.HEALTH,
                       labels_final=True))  # fmt: skip
    L.add(conn, L.fact("fu", "self", "lives_in", "Tver", labels_final=True))
    raw(conn, "r1", "Саша работает в Яндексе", tier="other")
    cloud = Gold(SASHA)
    extract.configure(cloud=cloud)
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    assert len(cloud.snaps) == 1
    sent = {f["id"] for f in cloud.snaps[0]["facts"]}
    assert "fh" not in sent and "fu" in sent
    assert "HIV" not in extract.user_text(cloud.snaps[0])


def test_unknown_labels_fact_treated_as_labelled(env):
    """Fail-closed: a foreign fact whose labels are not final never goes to the cloud."""
    conn, _clk = env
    L.add(conn, L.fact("fx", "self", "lives_in", "Tver", tier=AuthorTier.OTHER,
                       labels_final=False))  # fmt: skip
    raw(conn, "r1", "Саша работает в Яндексе", tier="other")
    snap, _ = extract.snapshot(conn, ["r1"])
    assert "fx" not in {f["id"] for f in extract.cloud_safe(snap)["facts"]}
    assert "fx" in {f["id"] for f in snap["facts"]}  # the local route still sees it


def test_value_from_foreign_row_gets_foreign_tier(env):
    """Quoting four letters of the user's row cannot make a tool row's value the user's word."""
    conn, clk = env
    raw(conn, "r1", "что там Олег пишет?", tier="user", turn="t1")
    raw(conn, "r2", "Олег: Денис теперь работает в Яндексе", tier="tool", turn="t1")
    gold = {"facts": [fact("f1", "self", "works_at", "Yandex", "Олег", "r1")]}
    extract.configure(cloud=Gold(gold), local=Gold(gold, route="local"))
    batch(conn, ["r1", "r2"])
    scheduler.tick(conn, clk)
    tiers = [
        r[0]
        for r in conn.execute("SELECT r.author_tier FROM facts f JOIN records r ON r.id = f.id")
    ]
    assert tiers and all(t != "user" for t in tiers)


def test_value_in_his_quote_keeps_his_tier(env):
    """Control: a value inside the user's own quote stays the user's (`user`)."""
    conn, clk = env
    raw(conn, "r1", "я теперь работаю в Яндексе", tier="user", turn="t1")
    gold = {"facts": [fact("f1", "self", "works_at", "Яндексе", "работаю в Яндексе", "r1")]}
    extract.configure(cloud=Gold(gold))
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    assert _tier_of(conn, "works_at") == "user"


def test_value_found_nowhere_is_system_tier(env):
    """A value in none of the batch's rows is the model's own (`system`), never the user's."""
    conn, clk = env
    raw(conn, "r1", "я сменил работу", tier="user", turn="t1")
    gold = {"facts": [fact("f1", "self", "works_at", "Google", "сменил работу", "r1")]}
    extract.configure(cloud=Gold(gold))
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    assert _tier_of(conn, "works_at") == "system"


def test_mixed_labelled_and_unlabeled_turn_routes_on_union(env):
    """§5.4: another person's HEALTH row and a plain row form one batch; the route
    follows the union of labels (local), never an empty intersection that parks. The user's
    own HEALTH words beside the Cube's plain reply count as unlabeled: cloud."""
    conn, clk = env
    raw(conn, "r1", "у меня болит спина, врач сказал грыжа", tier="user",
        labels=int(Label.HEALTH), turn="t1")  # fmt: skip
    raw(conn, "r2", "Понял, береги себя.", tier="system", turn="t1")
    raw(conn, "r3", "у Оли опять мигрень", tier="other", labels=int(Label.HEALTH), turn="t2")
    raw(conn, "r4", "ок, до вечера", tier="other", turn="t2")
    cloud, local = Gold({"facts": []}), Gold({"facts": []}, route="local")
    extract.configure(cloud=cloud, local=local)
    assert scheduler.finalize_labels(conn, ["r1", "r2", "r3", "r4"]) == 4

    def raws(ids):  # the scheduler routes on the batch's raw members (`run_batch`)
        return [r for r in extract.snapshot(conn, ids)[1] if r.src is ReadSrc.RAW]

    assert policy.route(conn, raws(["r1", "r2"])) == {Route.CLOUD}
    assert policy.route(conn, raws(["r3", "r4"])) == {Route.LOCAL}
    batch(conn, ["r3", "r4"])
    scheduler.tick(conn, clk)
    assert jobs(conn)[0]["state"] == "done" and len(local.snaps) == 1 and cloud.snaps == []


def test_parked_no_route_job_rearmed_when_routing_changes(env):
    """A job parked for `no_route` comes back once the table allows a route."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе", tier="other")
    policy.configure(routing={"unlabeled": frozenset()})
    cloud = Gold(SASHA)
    extract.configure(cloud=cloud)
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    assert jobs(conn)[0]["payload"]["waiting_for"] == "no_route"
    clk.advance(8 * 86400)
    scheduler.tick(conn, clk)
    assert jobs(conn)[0]["state"] == "parked"
    policy.configure(routing={"unlabeled": frozenset({Route.CLOUD})})
    scheduler.tick(conn, clk)
    assert jobs(conn)[0]["state"] == "done" and len(cloud.snaps) == 1
