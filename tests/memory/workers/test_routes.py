"""the memory spec §5.4: processing routes for workers (config, not code).
Spec marker: `[T workers/test_routes::<case>]`."""

from __future__ import annotations

from koob.memory import policy
from koob.memory.model import Label
from koob.memory.workers import extract, scheduler

from ._kit import SASHA, Gold, audit, batch, jobs, raw

DAY = 86400


def test_local_only_parks_not_downgrades(env):
    """§5.4: a local-only job with no local processor waits, then parks after 7 days; never downgrades."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе", tier="other", labels=int(Label.HEALTH))
    cloud, local = Gold(SASHA), Gold(SASHA, route="local", up=False)
    extract.configure(cloud=cloud, local=local)
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)
    (job,) = jobs(conn)
    assert job["state"] == "queued" and job["payload"]["waiting_for"] == "no_local"
    clk.advance(8 * DAY)
    scheduler.tick(conn, clk)
    assert jobs(conn)[0]["state"] == "parked" and audit(conn) == ["job_parked:no_local"]
    local.up = True  # a local processor appears: re-armed, run locally
    scheduler.tick(conn, clk)
    assert jobs(conn)[0]["state"] == "done" and len(local.snaps) == 1
    assert cloud.snaps == []  # never downgraded to the cloud
    assert conn.execute("SELECT labels FROM records r JOIN facts f ON f.id = r.id").fetchone()[0]


def test_route_is_config_and_needs_final_labels(env):
    """The table decides (config); unlabeled raw goes to the cloud only once code finalized
    its labels (no classifier wired → local only)."""
    conn, clk = env
    raw(conn, "r1", "Саша работает в Яндексе")
    cloud, local = Gold(SASHA), Gold(SASHA, route="local")
    extract.configure(cloud=cloud, local=local)
    policy.set_classifier(None)
    batch(conn, ["r1"])
    scheduler.tick(conn, clk)  # unlabeled routes cloud only, and the cloud needs final labels
    assert local.snaps == [] and cloud.snaps == []
    assert jobs(conn)[0]["payload"]["waiting_for"] == "no_route"
    policy.set_classifier(lambda text: {})
    clk.advance(601)
    scheduler.tick(conn, clk)
    assert len(cloud.snaps) == 1 and local.snaps == []
    raw(conn, "r2", "Саша любит Тверь")
    policy.configure(routing={"unlabeled": ["local"]})  # the switch is config, not code
    batch(conn, ["r2"])
    scheduler.tick(conn, clk)
    assert len(local.snaps) == 1 and len(cloud.snaps) == 1
    final = conn.execute("SELECT labels_final FROM raw_chunks WHERE id = 'r2'").fetchone()[0]
    assert final == 1
