"""the memory spec §5.3-§5.5: the uncertain band of anyone's message routes as unlabeled;
every confident label keeps the table (local only by default) and its use-now rule; banned and
«не копать» chats are decided at admission, before any route."""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import AuthorTier, ChatMode, Label, ReadRef, ReadSrc, Route

from .conftest import owner


def _ref(rid: str) -> ReadRef:
    return ReadRef(id=rid, version=1, src=ReadSrc.RAW)


@pytest.mark.parametrize("tier", ["other", "tool", "system"])
def test_an_uncertain_message_from_anyone_routes_like_his_own_words(db, tier):
    db.raw("r-unsure", text="Марк опять плохо спит", tier=tier, labels=Label.UNKNOWN_SENSITIVE)
    assert policy.route(db.conn, [_ref("r-unsure")]) == {Route.CLOUD}
    assert policy.route_each(db.conn, [_ref("r-unsure")]) == {"r-unsure": {Route.CLOUD}}


def test_an_uncertain_message_still_needs_final_labels_for_the_cloud(db):
    db.raw("r-pending", text="Марк опять плохо спит", labels=Label.UNKNOWN_SENSITIVE, final=False)
    assert policy.route(db.conn, [_ref("r-pending")]) == {Route.LOCAL}
    assert policy.route_each(db.conn, [_ref("r-pending")]) == {"r-pending": {Route.LOCAL}}


@pytest.mark.parametrize(
    "labels",
    [
        Label.HEALTH,
        Label.MONEY,
        Label.IDENTITY_DOCS,
        Label.INTIMATE,
        Label.THIRD_PARTY_PRIVATE,
        Label.HEALTH | Label.UNKNOWN_SENSITIVE,
        Label.LOCATION_PRECISE | Label.UNKNOWN_SENSITIVE,
        Label(1 << 12),
        Label(1 << 12) | Label.UNKNOWN_SENSITIVE,
    ],
)
def test_a_confident_label_keeps_another_persons_message_local(db, labels):
    db.raw("r-sens", text="Марк: подробности", labels=labels)
    assert policy.route(db.conn, [_ref("r-sens")]) == {Route.LOCAL}
    assert policy.route_each(db.conn, [_ref("r-sens")]) == {"r-sens": {Route.LOCAL}}


def test_a_job_with_one_confident_member_stays_local(db):
    db.raw("r-unsure", text="Марк опять плохо спит", labels=Label.UNKNOWN_SENSITIVE)
    db.raw("r-plain", text="Марк переехал в Тбилиси")
    db.raw("r-health", text="Марк: у меня диабет", labels=Label.HEALTH)
    assert policy.route(db.conn, [_ref("r-unsure"), _ref("r-plain")]) == {Route.CLOUD}
    assert policy.route(db.conn, [_ref("r-unsure"), _ref("r-health")]) == {Route.LOCAL}


def test_the_uncertain_band_follows_the_unlabeled_row_of_the_table(db):
    db.raw("r-unsure", text="Марк опять плохо спит", labels=Label.UNKNOWN_SENSITIVE)
    policy.configure(routing={"unlabeled": ["local"]})
    assert policy.route(db.conn, [_ref("r-unsure")]) == {Route.LOCAL}


def test_a_confident_foreign_raw_is_served_only_on_a_direct_question(db):
    """The raw lane: another person's labelled message stays stored verbatim and reaches the
    brain only when the user asks about that person and that topic."""
    db.entity("e-mark", "Марк")
    db.raw("r-health", text="у меня диабет, сижу на инсулине", labels=Label.HEALTH, author="e-mark")
    db.raw("r-money", text="должен банку 900 тысяч", labels=Label.MONEY, author="e-mark")
    conn = db.conn
    for words in ("", "Что нового у Марка?", "Закажи пиццу", "Как у меня со здоровьем?"):
        assert policy.allowed(conn, ["r-health", "r-money"], owner(words)) == [], words
    assert policy.allowed(conn, ["r-health", "r-money"], owner("Как у Марка со здоровьем?")) == [
        "r-health"
    ]
    assert policy.allowed(conn, ["r-health", "r-money"], owner("Сколько Марк должен банку?")) == [
        "r-money"
    ]
    assert db.count("raw_chunks", "id IN ('r-health', 'r-money') AND status = 'active'") == 2


@pytest.mark.parametrize(
    ("mode", "reason"),
    [(ChatMode.BANNED, "chat_banned"), (ChatMode.NO_DIG, "no_dig_outside_window")],
)
@pytest.mark.parametrize("tier", [AuthorTier.OTHER, AuthorTier.USER])
def test_a_barred_chat_admits_nothing_to_route(db, mode, reason, tier):
    """Banned and «не копать» chats are decided before storage: an unsure message there is
    never stored, so no route can read it."""
    policy.set_mode(db.conn, "tg:mark", mode)
    ok, why = policy.admit(
        db.conn,
        source_kind="telegram",
        chat_id="tg:mark",
        text="Марк опять плохо спит",
        author_tier=tier,
        turn=None,
    )
    assert not ok and why == reason
    assert db.count("raw_chunks") == 0
    assert db.count("jobs", "jkind = 'extract'") == 0
