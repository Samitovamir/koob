"""Route half (§5.3, §5.4): the user's own words are not refused the cloud by the
Cube's reply in the same batch, and everyday words are not labelled sensitive at arrival.
The foreign rule stays: another person's labelled text keeps every label local only."""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import Label, ReadSrc, Route
from koob.memory.workers import extract, scheduler

from ._kit import Gold, batch, jobs, raw


def _route(conn, ids: list[str]) -> set[Route]:
    scheduler.finalize_labels(conn, ids)
    return policy.route(conn, [r for r in extract.snapshot(conn, ids)[1] if r.src is ReadSrc.RAW])


def test_his_words_beside_the_cubes_plain_reply_go_to_the_cloud(env):
    conn, clk = env
    raw(conn, "a1", "мне выписали антибиотики на неделю, пью утром", tier="user", turn="q1")
    raw(conn, "a2", "Хорошо, напомню утром в восемь.", tier="system", turn="q1")
    cloud, local = Gold({"facts": []}), Gold({"facts": []}, route="local")
    extract.configure(cloud=cloud)  # no local processor: a local-only route would wait
    assert _route(conn, ["a1", "a2"]) == {Route.CLOUD}
    batch(conn, ["a1", "a2"], turn="q1")
    scheduler.tick(conn, clk)
    assert jobs(conn)[0]["state"] == "done" and len(cloud.snaps) == 1 and local.snaps == []


def test_the_cubes_own_label_still_follows_the_table(env):
    conn, _ = env
    raw(conn, "b1", "завтра у меня созвон с Пашей", tier="user", turn="q2")
    raw(conn, "b2", "Паша просил никому не говорить про его кредит", tier="system", turn="q2")
    assert _route(conn, ["b1", "b2"]) == {Route.LOCAL}


@pytest.mark.parametrize("tier", ["other", "tool"])
def test_anyone_elses_text_keeps_the_union_with_his_words(env, tier):
    """The user's HEALTH words give the context another person's reply is read in: local only."""
    conn, _ = env
    raw(conn, "c1", "у меня опять мигрень третий день", tier="user", labels=int(Label.HEALTH))
    raw(conn, "c2", "у меня тоже, держись", tier=tier)
    assert _route(conn, ["c1", "c2"]) == {Route.LOCAL}


def test_foreign_labelled_text_stays_local_and_parks_without_local(env):
    conn, clk = env
    raw(conn, "d1", "Вера: должна банку за машину ещё 300 тысяч", tier="other")
    cloud = Gold({"facts": []})
    extract.configure(cloud=cloud)
    assert _route(conn, ["d1"]) == {Route.LOCAL}
    batch(conn, ["d1"], turn=None)
    scheduler.tick(conn, clk)
    assert jobs(conn)[0]["state"] in ("queued", "parked") and cloud.snaps == []


@pytest.mark.parametrize(
    "text",
    [
        "Окей, договорились, созвонимся в пятницу",
        "мы договорились встретиться у вокзала",
        "Сняли квартиру в центре Самары, переезжаем в июле",
        "живу на последнем этаже, вид классный",
        "She found an apartment near campus",
    ],
)
def test_everyday_words_get_no_label_at_arrival(text):
    assert policy.label_topics(policy.words(text)) == Label.NONE
    assert policy.label(text) & (Label.LEGAL | Label.LOCATION_PRECISE) == Label.NONE


@pytest.mark.parametrize(
    ("text", "bit"),
    [
        ("Подписали договор с подрядчиком на год", Label.LEGAL),
        ("Подписали договор аренды на год", Label.LEGAL | Label.MONEY),
        ("мой адрес: квартира 47, код 1290", Label.LOCATION_PRECISE),
        ("поднимайся на 7 этаж, третья дверь", Label.LOCATION_PRECISE),
        ("I'm in apt 4B, buzz twice", Label.LOCATION_PRECISE),
        ("we're on the 12th floor of the library", Label.LOCATION_PRECISE),
    ],
)
def test_the_sensitive_forms_still_label(text, bit):
    got = policy.label(text)
    assert got & (bit | Label.UNKNOWN_SENSITIVE), text
