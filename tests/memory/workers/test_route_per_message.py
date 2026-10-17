"""§5.4: the route is chosen per MESSAGE, not per extract job. A batch no processor may
take whole is split: the messages that may be read now are read now, only the blocking ones
wait. Another person's labelled text stays local only; the user's labelled words never share a
cloud read with anyone else's text; nothing unfinal is split (fail-closed)."""

from __future__ import annotations

from koob.memory import policy
from koob.memory.model import Label, ReadSrc, Route
from koob.memory.workers import extract, scheduler

from ._kit import Gold, batch, jobs, raw

EMPTY = {"facts": []}


def _seen(proc: Gold) -> list[set[str]]:
    return [{r["id"] for r in s["raws"]} for s in proc.snaps]


def _texts(proc: Gold) -> str:
    return " ".join(r["text"] for s in proc.snaps for r in s["raws"])


def test_clean_messages_go_now_and_only_the_labelled_one_waits(env):
    conn, clk = env
    raw(conn, "m1", "Lena moved her standup to Thursdays at nine", tier="other")
    raw(
        conn,
        "m2",
        "у Ромы опять обострение, лежит в стационаре",
        tier="other",
        labels=int(Label.HEALTH),
    )
    raw(conn, "m3", "Тимур взял билеты в Калининград на майские", tier="other")
    cloud = Gold(EMPTY)
    extract.configure(cloud=cloud)  # no local processor: the labelled one cannot run now
    batch(conn, ["m1", "m2", "m3"], turn=None)
    scheduler.tick(conn, clk)
    assert _seen(cloud) == [{"m1", "m3"}]
    assert "стационаре" not in _texts(cloud)
    by_ids = {tuple(j["payload"]["raw_ids"]): j for j in jobs(conn)}
    assert by_ids[("m2",)]["state"] == "queued"
    assert by_ids[("m2",)]["payload"]["waiting_for"] == "no_local"
    assert by_ids[("m1", "m3")]["state"] == "done"


def test_a_tool_read_follows_the_same_per_message_rule(env):
    conn, clk = env
    raw(conn, "t1", "найди, когда у Миры дедлайн по гранту", tier="user", turn="q7")
    raw(
        conn,
        "t2",
        "[page] Grant portal: the deadline for Mira's proposal is 14 March",
        tier="tool",
        turn="q7",
    )
    raw(
        conn,
        "t3",
        "Mira: my lawyer says the custody hearing moved",
        tier="tool",
        labels=int(Label.LEGAL),
        turn="q7",
    )
    cloud = Gold(EMPTY)
    extract.configure(cloud=cloud)
    batch(conn, ["t1", "t2", "t3"], turn="q7")
    scheduler.tick(conn, clk)
    assert {"t1", "t2"} <= set().union(*_seen(cloud))
    assert "custody" not in _texts(cloud)
    waiting = [j for j in jobs(conn) if j["state"] == "queued"]
    assert [j["payload"]["raw_ids"] for j in waiting] == [["t3"]]


def test_his_labelled_words_never_share_a_cloud_read_with_another_persons_text(env):
    """The union rule inside a part: the user's HEALTH words beside a clean reply would be a local
    job; split, each side is read on its own and neither ever beside the other."""
    conn, clk = env
    raw(
        conn,
        "h1",
        "I've had a fever since Sunday, staying in",
        tier="user",
        labels=int(Label.HEALTH),
    )
    raw(conn, "h2", "Ок, тогда созвон переносим на среду", tier="other")
    assert policy.route(conn, _refs(conn, ["h1", "h2"])) == {Route.LOCAL}
    cloud = Gold(EMPTY)
    extract.configure(cloud=cloud)
    batch(conn, ["h1", "h2"], turn=None)
    scheduler.tick(conn, clk)
    assert sorted(map(sorted, _seen(cloud))) == [["h1"], ["h2"]]
    assert all(j["state"] == "done" for j in jobs(conn))


def test_unfinal_labels_split_nothing(env):
    """A cold classifier: no message is final, so none may go to the cloud; the batch waits
    whole, as before."""
    conn, clk = env
    policy.set_classifier(None)
    raw(conn, "u1", "Arseniy starts at the new studio on Monday", tier="other")
    raw(conn, "u2", "у Арсения долг по ипотеке", tier="other", labels=int(Label.MONEY))
    cloud = Gold(EMPTY)
    extract.configure(cloud=cloud)
    batch(conn, ["u1", "u2"], turn=None)
    scheduler.tick(conn, clk)
    assert cloud.snaps == []
    (job,) = jobs(conn)
    assert job["state"] == "queued" and job["payload"]["raw_ids"] == ["u1", "u2"]


def test_a_parked_mixed_batch_rearms_its_clean_part_and_the_rest_parks_again(env):
    conn, clk = env
    raw(conn, "p1", "Злата теперь ведёт бэкенд в новой команде", tier="other")
    raw(
        conn,
        "p2",
        "Zlata's therapist changed her meds again",
        tier="other",
        labels=int(Label.HEALTH),
    )
    extract.configure()  # nothing configured: the job waits and, after a week, parks
    jid = batch(conn, ["p1", "p2"], turn=None)
    scheduler.tick(conn, clk)
    clk.advance(8 * 86400)
    for _ in range(2):
        conn.execute("UPDATE jobs SET not_before = NULL WHERE id = ?", (jid,))
        conn.commit()
        scheduler.tick(conn, clk)
    assert jobs(conn)[0]["state"] == "parked"
    cloud = Gold(EMPTY)
    extract.configure(cloud=cloud)  # a cloud processor appears: the clean part may go now
    scheduler.tick(conn, clk)
    assert _seen(cloud) == [{"p1"}]
    rest = next(j for j in jobs(conn) if j["id"] == jid)
    assert rest["state"] == "parked" and rest["payload"]["raw_ids"] == ["p2"]
    scheduler.tick(conn, clk)  # the rest alone has nothing to read now: not re-armed
    assert len(cloud.snaps) == 1


def test_a_safety_net_does_not_close_a_batch_whose_rest_still_waits(env):
    conn, clk = env
    raw(conn, "s1", "запомни: офис теперь на Лиговском, третий этаж", tier="user", turn="q9")
    raw(
        conn,
        "s2",
        "Gleb: the bank froze my card after the overdraft",
        tier="tool",
        labels=int(Label.MONEY),
        turn="q9",
    )
    cloud = Gold(EMPTY)
    extract.configure(cloud=cloud)
    jid = batch(conn, ["s1", "s2"], turn="q9")
    scheduler.enqueue(conn, scheduler.SAFETY, {"turn_id": "q9"}, idempotency_key="safety:q9")
    scheduler.tick(conn, clk)
    assert _seen(cloud) == [{"s1"}]
    state = {j["id"]: j["state"] for j in jobs(conn)}
    assert state[jid] == "queued"
    safety = [j for j in jobs(conn) if j["jkind"] == scheduler.SAFETY]
    assert safety[0]["state"] == "done"


def _refs(conn, ids: list[str]):
    scheduler.finalize_labels(conn, ids)
    return [r for r in extract.snapshot(conn, ids)[1] if r.src is ReadSrc.RAW]
