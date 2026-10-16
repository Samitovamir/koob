"""Adversarial review tests: author tiers, lineage answers and ledger primitives (§5.5, §9,
§13, G5, G11).

Each test failed on the earlier code and passes since the fix; the docstring names the finding
in the design notes."""

from __future__ import annotations

from koob.memory import api
from koob.memory.ledger import CommitRejected
from koob.memory.model import (
    AuthorTier,
    Label,
    OpKind,
    ReadRef,
    ReadSrc,
    TurnCtx,
    TurnOrigin,
)

from .._ledger import add, commit, entity, fact, op, open_seeded, raw


def test_worker_tier_ceiling_counts_every_raw_it_read(tmp_path):
    """G11: `_check_record` takes the tier ceiling from `source_raw_id` and `derived_from`
    only, while label inheritance already counts the raw chunks in the op's read set. A
    worker that read the user's message and a forwarded mail, and cites only the user's
    message, stores the mail's «new e-mail» as `user`-authored: the user's own item, used
    anywhere, and it may supersede the user's real address."""
    conn = open_seeded(tmp_path)
    raw(conn, "raw-his", "перешли мне письмо от банка", tier="user")
    raw(conn, "raw-mail", "Denis's new e-mail is evil@attacker.example", tier="tool")
    assert add(conn, entity("e-me", "Denis")).ok
    poisoned = fact(
        "f-mail",
        "e-me",
        "email",
        "evil@attacker.example",
        text="Denis's e-mail is evil@attacker.example",
        source_raw_id="raw-his",
        tier=AuthorTier.USER,
    )
    read_set = [
        ReadRef(id="raw-his", version=1, src=ReadSrc.RAW),
        ReadRef(id="raw-mail", version=1, src=ReadSrc.RAW),
    ]
    res = commit(
        conn,
        op(OpKind.ADD, group="e-me", read_set=read_set, record=poisoned.model_dump(mode="json")),
        tier=AuthorTier.SYSTEM,
        model_id="worker-extract",
    )
    row = conn.execute("SELECT author_tier FROM records WHERE id = 'f-mail'").fetchone()
    assert not res.ok or row is None or row[0] not in ("user", "user_confirmed")


def test_where_from_of_a_retracted_claim_keeps_another_persons_secret(tmp_path):
    """§5.5: `lineage` applies the use-now check only to usable records. For a superseded,
    retracted or needs-review record it prints the text and the quote, so «откуда это»
    ("where is this from") on an old handle shows one person's claim about another
    person's health without a direct question."""
    conn = open_seeded(tmp_path)
    assert add(conn, entity("e-sasha", "Саша")).ok
    claim = fact(
        "f-health",
        "e-sasha",
        "health_condition",
        "диабет",
        text="Sasha has diabetes",
        quote_orig="Дима сказал, что у Саши диабет",
        tier=AuthorTier.OTHER,
        labels=Label.HEALTH,
    )
    assert add(conn, claim).ok
    assert commit(conn, op(OpKind.FLAG, target="f-health", group="e-sasha", review=True)).ok
    conn.execute("UPDATE records SET status = 'needs_review' WHERE id = 'f-health'")
    turn = TurnCtx(
        turn_id="t-1", thread_id="th", origin=TurnOrigin.OWNER, owner_words="откуда это?"
    )
    lines = " ".join(api.where_from("m:f-health", turn=turn, store=conn).lines)
    assert "диабет" not in lines and "diabetes" not in lines


def test_ledger_forget_op_does_not_leave_content_behind(tmp_path):
    """G5: `ledger.apply_op` accepts `OpKind.FORGET` from the user's tier and only sets the
    status: text, quote, FTS row and raw stay, with no trash, journal line or purge job,
    so nothing ever purges it. Every forget must go through `forget.stop_use`; the ledger
    op should be refused or routed there."""
    conn = open_seeded(tmp_path)
    assert add(conn, entity("e-s", "Sasha")).ok
    assert add(conn, fact("f-1", "e-s", "moving_to", "Berlin", text="Sasha moves to Berlin")).ok
    # a refused hot or code op raises to its writer instead of a `rejected` result
    try:
        refused = not commit(conn, op(OpKind.FORGET, target="f-1", group="e-s")).ok
    except CommitRejected:
        refused = True
    text = conn.execute("SELECT text_en FROM facts WHERE id = 'f-1'").fetchone()[0]
    assert refused or "Berlin" not in text
    if refused:  # nothing half-forgotten is left behind either
        assert conn.execute("SELECT status FROM records WHERE id = 'f-1'").fetchone()[0] == "active"
