"""the memory spec §5.4: processing routes per label.
Spec marker: `[T policy/test_routes::<case>]`."""

from __future__ import annotations

from koob.memory import policy
from koob.memory.model import Label, ReadRef, ReadSrc, Route


class _Stub:
    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts):
        self.seen.extend(texts)


def _dispatch(conn, jobs, *, cloud, local):
    """A worker runner's gate: route first, cloud preferred, never a downgrade."""
    waiting = []
    for read_set, texts in jobs:
        routes = policy.route(conn, read_set)
        if Route.CLOUD in routes:
            cloud(texts)
        elif Route.LOCAL in routes:
            local(texts)
        else:
            waiting.append(texts)
    return waiting


def test_lexicon_miss_never_reaches_cloud_stub(db):
    """§5.4: cloud requires labels_final; a lexicon miss never reaches the cloud stub."""
    conn = db.conn
    # a lexicon miss: nothing in the text names its topic; the classifier is uncertain
    miss_text = "опять скачет сахар, вечером мерил дважды"
    assert not policy.label(miss_text) & policy.sensitive_mask()
    policy.set_classifier(lambda t: {Label.HEALTH: 0.3} if "сахар" in t else {})
    labels = policy.label(miss_text)
    assert labels == Label.UNKNOWN_SENSITIVE

    db.raw("r-plain", text="Саша переехал в Ригу", chat="tg:sasha", final=True)
    db.raw("r-pending", text=miss_text, chat="tg:sasha", final=False)  # labelling not finished
    db.raw("r-uncertain", text=miss_text, chat="tg:sasha", labels=labels, final=True)
    db.raw("r-health", text="Саша: у меня диабет", chat="tg:sasha", labels=Label.HEALTH)

    def job(rid):
        return [ReadRef(id=rid, version=1, src=ReadSrc.RAW)], [rid]

    cloud, local = _Stub(), _Stub()
    waiting = _dispatch(
        conn,
        [job("r-plain"), job("r-pending"), job("r-uncertain"), job("r-health")],
        cloud=cloud,
        local=local,
    )
    assert cloud.seen == ["r-plain"]
    assert local.seen == ["r-uncertain", "r-health"]
    assert waiting == [["r-pending"]]  # waits for its labels; never downgraded to the cloud
    # a job mixing a pending member with a plain one never goes to the cloud either
    mixed = [
        ReadRef(id="r-plain", version=1, src=ReadSrc.RAW),
        ReadRef(id="r-pending", version=1, src=ReadSrc.RAW),
    ]
    assert Route.CLOUD not in policy.route(conn, mixed)


def test_own_words_cloud_whatever_labels(db):
    """§5.4: the user's own words route as unlabeled whatever their labels; other people's keep
    the table; a mixed job follows the table."""
    conn = db.conn
    db.entity("e-self", "Тимур")
    own = db.fact(
        "f-own",
        subject="e-self",
        text_en="Gastritis; ferritin 9",
        quote="Обострился гастрит, ферритин 9",
        labels=Label.HEALTH,
        tier="user",
    )
    conf = db.fact(
        "f-conf",
        subject="e-self",
        text_en="Iron for three months",
        labels=Label.HEALTH | Label.MONEY,
        tier="user_confirmed",
    )
    other = db.fact(
        "f-other",
        subject="e-sasha",
        text_en="Sasha has diabetes",
        labels=Label.HEALTH,
        tier="other",
    )
    raw_own = db.raw("r-own", text="у меня аллергия на орехи", tier="user", labels=Label.HEALTH)

    def refs(*ids):
        return [ReadRef(id=i, version=1) for i in ids]

    assert policy.route(conn, refs(own)) == {Route.CLOUD}
    assert policy.route(conn, refs(own, conf)) == {Route.CLOUD}
    assert policy.route(conn, [ReadRef(id=raw_own, version=1, src=ReadSrc.RAW)]) == {Route.CLOUD}
    assert policy.route(conn, refs(other)) == {Route.LOCAL}
    assert policy.route(conn, refs(own, other)) == {Route.LOCAL}


def test_unusable_or_missing_member_allows_nothing(db):
    db.fact("f-gone", subject="e-x", text_en="x", status="forgotten")
    assert policy.route(db.conn, [ReadRef(id="f-gone", version=1)]) == set()
    assert policy.route(db.conn, [ReadRef(id="nope", version=1)]) == set()


def test_table_is_the_config(db):
    """The table comes from `memory.routing`; an empty list means "wait"."""
    db.fact("f-money", subject="e-x", text_en="x owes y", labels=Label.MONEY, tier="other")
    db.fact("f-plain", subject="e-x", text_en="x likes jazz", tier="other")
    ref = [ReadRef(id="f-money", version=1)]
    policy.configure(routing={"money": ["local", "cloud"]})
    assert policy.route(db.conn, ref) == {Route.LOCAL, Route.CLOUD}
    policy.configure(routing={"money": []})
    assert policy.route(db.conn, ref) == set()
    policy.configure(routing={"unlabeled": ["local"]})
    assert policy.route(db.conn, [ReadRef(id="f-plain", version=1)]) == {Route.LOCAL}
    # reserved bits fail closed like unknown_sensitive
    assert policy._routes_for(Label(1 << 12)) == {Route.LOCAL}


def test_settings_file_defaults():
    """With no override the table is `config/settings.yaml`'s: every label local only."""
    policy.configure()
    table = policy.routing_table()
    assert table["unlabeled"] == frozenset({Route.CLOUD})
    for name in policy.LABEL_NAMES:
        assert table[name] == frozenset({Route.LOCAL})
    assert policy.sensitive_mask() == policy.ALL_LABELS & ~Label.LOCATION_PRECISE
