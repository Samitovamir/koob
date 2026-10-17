"""§5.3, §5.4: another person's message the label classifier is unsure about goes to the cloud
extraction like the user's own words; a confidently sensitive one never rides in a cloud batch,
stays stored with its labels, and the route is decided from the labels known when the batch runs."""

from __future__ import annotations

import json

from koob.memory import policy
from koob.memory.model import Label
from koob.memory.workers import extract, scheduler

from ._kit import Gold, batch, jobs, raw

UNSURE = "Марк сказал, что последнюю неделю плохо спит и всё время устаёт"
HEALTH = "Марк: врач подтвердил диабет второго типа, теперь на инсулине"
MONEY = "Марк: должен банку за квартиру ещё 900 тысяч"
OWN = "Денис: завтра в девять созвон с Марком по проекту"


def _scores(text: str) -> dict[Label, float]:
    """A stand-in classifier: unsure about one message, confident about another."""
    if text == UNSURE:
        return {Label.HEALTH: 0.3}
    if text == HEALTH:
        return {Label.HEALTH: 0.9}
    return {}


def _cloud_texts(proc: Gold) -> str:
    return json.dumps([r["text"] for s in proc.snaps for r in s["raws"]], ensure_ascii=False)


def _labels(conn, rid: str) -> tuple[Label, bool]:
    lab, final = conn.execute(
        "SELECT labels, labels_final FROM raw_chunks WHERE id = ?", (rid,)
    ).fetchone()
    return Label(lab), bool(final)


def test_an_uncertain_foreign_message_reaches_the_cloud_batch(env):
    conn, clk = env
    policy.set_classifier(_scores)
    raw(conn, "u1", UNSURE, tier="other")
    cloud = Gold({"facts": []})
    extract.configure(cloud=cloud)  # no local processor: a local-only route would wait
    batch(conn, ["u1"], turn=None)
    scheduler.tick(conn, clk)
    assert _labels(conn, "u1") == (Label.UNKNOWN_SENSITIVE, True)
    assert jobs(conn)[0]["state"] == "done"
    assert UNSURE in _cloud_texts(cloud)


def test_a_confident_foreign_message_never_rides_in_a_cloud_batch(env):
    """Cloud only: the uncertain message and the user's own are read now, the labelled ones wait."""
    conn, clk = env
    policy.set_classifier(_scores)
    raw(conn, "m1", OWN, tier="user")
    raw(conn, "m2", UNSURE, tier="other")
    raw(conn, "m3", HEALTH, tier="other")
    raw(conn, "m4", MONEY, tier="other")
    cloud = Gold({"facts": []})
    extract.configure(cloud=cloud)
    batch(conn, ["m1", "m2", "m3", "m4"], turn=None)
    scheduler.tick(conn, clk)
    sent = _cloud_texts(cloud)
    assert OWN in sent and UNSURE in sent
    assert HEALTH not in sent and MONEY not in sent
    # stored with their labels, waiting for a local processor
    assert _labels(conn, "m3")[0] & Label.HEALTH
    assert _labels(conn, "m4")[0] & Label.MONEY
    waiting = [j for j in jobs(conn) if j["state"] in ("queued", "parked")]
    assert [sorted(j["payload"]["raw_ids"]) for j in waiting] == [["m3", "m4"]]


def test_with_both_processors_a_confident_foreign_message_still_never_reaches_the_cloud(env):
    conn, clk = env
    policy.set_classifier(_scores)
    raw(conn, "b1", UNSURE, tier="other")
    raw(conn, "b2", HEALTH, tier="other")
    cloud, local = Gold({"facts": []}), Gold({"facts": []}, route="local")
    extract.configure(cloud=cloud, local=local)
    batch(conn, ["b1", "b2"], turn=None)
    scheduler.tick(conn, clk)
    assert HEALTH not in _cloud_texts(cloud)
    assert HEALTH in _cloud_texts(local)  # the same snapshot shape on the local route


def test_the_route_follows_the_labels_known_when_the_batch_runs(env):
    """Enqueued unlabelled and not final, labelled confident by the classifier at run time:
    the route is read after the labelling pass, right before the send, so it stays local."""
    conn, clk = env
    raw(conn, "r1", HEALTH, tier="other")
    cloud = Gold({"facts": []})
    extract.configure(cloud=cloud)
    batch(conn, ["r1"], turn=None)  # at enqueue the row carries no label at all
    policy.set_classifier(_scores)
    scheduler.tick(conn, clk)
    assert _labels(conn, "r1") == (Label.HEALTH, True)
    assert cloud.snaps == []
    assert jobs(conn)[0]["state"] in ("queued", "parked")
