"""the memory spec §2: the pipeline order admit -> store -> process -> use now -> rank.
Spec marker: `[T policy/test_order::<case>]`."""

from __future__ import annotations

from koob.memory import policy
from koob.memory.model import Label

from .conftest import owner


def _seed(db):
    db.entity("e-self", "Денис")
    db.entity("e-sasha", "Саша", "Александр")
    db.fact(
        "f-sasha-health",
        subject="e-sasha",
        text_en="Sasha has diabetes",
        quote="у меня диабет",
        labels=Label.HEALTH,
        tier="other",
        pinned="health",
    )
    db.fact("f-sasha-riga", subject="e-sasha", text_en="Sasha moved to Riga", tier="other")
    db.fact(
        "f-sasha-money",
        subject="e-sasha",
        text_en="Sasha owes the bank 2M",
        quote="должен банку два миллиона",
        labels=Label.MONEY,
        tier="other",
    )
    db.fact(
        "f-own-allergy",
        subject="e-self",
        text_en="Allergic to nuts",
        quote="у меня аллергия на орехи",
        labels=Label.HEALTH,
        tier="user",
        pinned="health",
    )
    db.fact("f-own-jazz", subject="e-self", text_en="Likes jazz", tier="user", pinned="taste")
    db.fact(
        "f-addr",
        subject="e-sasha",
        text_en="Sasha lives at Lenina 5",
        labels=Label.LOCATION_PRECISE,
        tier="other",
    )


def _finders(conn, turn):
    """Every finder applies the label mask in its SQL (§10.2 step 2)."""
    clause, params = policy.mask_clause(turn, alias="r")
    return [
        row[0]
        for row in conn.execute(
            "SELECT r.id FROM records r WHERE r.kind <> 'entity'"
            f" AND r.status IN ('active', 'archived') AND {clause} ORDER BY r.id",
            params,
        )
    ]


def test_rank_never_sees_excluded(db):
    """§2: ranking never sees an item use-now excluded."""
    _seed(db)
    conn = db.conn
    seen: list[list[str]] = []

    def rank(ids):
        seen.append(list(ids))
        pinned = {
            r[0] for r in conn.execute("SELECT id FROM records WHERE pinned_category IS NOT NULL")
        }
        return sorted(ids, key=lambda i: (i not in pinned, i))

    turn = owner("Что нового у Саши? И какая завтра погода?")
    found = _finders(conn, turn)
    # the mask applies "use now" earlier: others' sensitive items never leave the SQL
    assert "f-sasha-health" not in found and "f-sasha-money" not in found
    # the user's own items pass the SQL; use-now is the second fence
    assert "f-own-allergy" in found
    ranked = rank(policy.allowed(conn, found, turn))
    assert seen == [["f-addr", "f-own-jazz", "f-sasha-riga"]]
    # pinned categories rank first only among allowed items
    assert ranked[0] == "f-own-jazz"
    for excluded in ("f-sasha-health", "f-sasha-money", "f-own-allergy"):
        assert all(excluded not in batch for batch in seen)


def test_mask_never_cuts_what_use_now_allows(db):
    """The SQL mask is a superset of use-now: it applies the rule earlier, never stricter."""
    _seed(db)
    conn = db.conn
    items = [r[0] for r in conn.execute("SELECT id FROM records WHERE kind <> 'entity'")]
    turns = [
        owner(""),
        owner("Что нового у Саши?"),
        owner("Как у Саши со здоровьем?"),
        owner("Сколько Саша должен банку?"),
        owner("Закажи пиццу"),
        owner("Напиши Диме адрес Саши"),
    ]
    for turn in turns:
        found = set(_finders(conn, turn))
        for item in items:
            if policy.use_now(conn, item, turn):
                assert item in found, (turn.owner_words, item)
    # and a direct question does reach the ranker
    direct = owner("Как у Саши со здоровьем?")
    assert "f-sasha-health" in policy.allowed(conn, _finders(conn, direct), direct)


def test_label_mask_shape():
    sens = policy.sensitive_mask()
    neutral = policy.label_mask(owner(""))
    assert not neutral & sens and neutral & Label.LOCATION_PRECISE
    health = policy.label_mask(owner("как у Саши со здоровьем?"))
    assert health & Label.HEALTH and not health & Label.MONEY
    sql, params = policy.mask_clause(owner(""), alias="d", tiered=False)
    assert "author_tier" not in sql and params == [int(neutral)]
