"""the memory spec §5.3-§5.4: labels become final after the classifier pass, only grow,
stay not final when the classifier is cold or fails, and the cloud gate needs them final.
Plus what the facade records (denials, notices) for the eval's ApiSUT."""

from __future__ import annotations

from koob.memory import api, labeler, ledger, policy, store
from koob.memory.model import AuthorTier, Label, OpKind, Proposal, ProposalOp, ReadRef
from koob.memory.workers import extract
from koob.memory.writers import hot

from ..hotpath._world import turn_of


def _health_if(word: str):
    return lambda t: {Label.HEALTH: 0.9} if word in (t or "") else {}


class _Cold:
    def __call__(self, text):
        return {}

    def ready(self):
        return False


class _Broken:
    def __call__(self, text):
        raise RuntimeError("embedder died")


def _row(conn, rid):
    return conn.execute(
        "SELECT labels, labels_final, version FROM records WHERE id = ?", (rid,)
    ).fetchone()


def _remember(conn, words, n=1):
    r = api.remember(words, turn=turn_of(words, n=n), because_quote=words, store=conn)
    assert r.outcome == "committed"
    hot.drain_labels()
    return r.record_ids[0]


def test_background_pass_makes_labels_final(mem):
    policy.set_classifier(_health_if("зюзя"))
    rid = _remember(mem, "запомни: у Саши зюзя")
    labels, final, _v = _row(mem, rid)
    assert final == 1 and Label(labels) & Label.HEALTH


def test_cold_classifier_leaves_not_final_then_the_sweep_finalizes(mem):
    policy.set_classifier(_Cold())
    rid = _remember(mem, "запомни: у Димы новая работа")
    assert _row(mem, rid)[1] == 0  # fail-closed: no label is final on a guess
    assert labeler.finalize_pending(mem) == (0, 0)  # still cold: the sweep does nothing
    policy.set_classifier(_health_if("никогда"))
    api.run_jobs(store=mem)
    assert _row(mem, rid)[1] == 1


def test_failing_classifier_leaves_not_final(mem):
    policy.set_classifier(_Broken())
    rid = _remember(mem, "запомни: у Пети диабет")
    labels, final, _v = _row(mem, rid)
    assert final == 0
    assert Label(labels) & Label.HEALTH  # the lexicon label still landed (labels only grow)


def test_pass_goes_through_the_ledger_and_labels_only_grow(mem):
    policy.set_classifier(None)
    rid = _remember(mem, "запомни: я должен Диме 5000 руб")
    labels0, final0, v0 = _row(mem, rid)
    assert final0 == 0 and Label(labels0) & Label.MONEY
    policy.set_classifier(_health_if("Диме"))
    assert labeler.finalize_record(mem, rid)
    labels, final, v = _row(mem, rid)
    assert final == 1 and Label(labels) & Label.MONEY and Label(labels) & Label.HEALTH
    assert v > v0  # a label that grew is a new version (§7.1), committed as an UPDATE op
    ev = mem.execute("SELECT item_ids FROM events ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert rid in ev


def test_race_edit_under_the_pass_is_rescored(mem):
    """An edit committed between the pass's read and its commit makes the pass stale; it
    scores the new text, so a final label never comes from text that is no longer there."""
    policy.set_classifier(None)
    rid = _remember(mem, "запомни: Саша любит чай")
    seen: list[str] = []

    def racing(text):
        seen.append(text)
        if len(seen) == 1:  # the concurrent "fix" edit lands while the pass scores
            rec = ledger.get_record(mem, rid)
            op = ProposalOp(
                op=OpKind.UPDATE,
                target=rid,
                group=rid,
                read_set=[ReadRef(id=rid, version=rec.version)],
                payload={"fields": {"text_en": "Sasha has diabetes"}},
            )
            res = store.commit(
                mem, Proposal(ops=[op], author_tier=AuthorTier.USER), ledger.apply_op
            )
            assert res.ok
        return {Label.HEALTH: 0.9} if "diabetes" in text else {}

    policy.set_classifier(racing)
    assert labeler.finalize_record(mem, rid)
    labels, final, _v = _row(mem, rid)
    assert "Sasha has diabetes" in seen  # the second attempt scored the edited text
    assert final == 1 and Label(labels) & Label.HEALTH


def test_raw_pass_or_grows_and_respects_cold(db):
    db.raw("r1", text="скачет сахар", labels=Label.MONEY, final=False)
    db.raw("r2", text="встреча в пятницу", final=False)
    policy.set_classifier(_Cold())
    assert labeler.finalize_raws(db.conn, ["r1", "r2"]) == 0
    policy.set_classifier(_health_if("сахар"))
    assert labeler.finalize_raws(db.conn, ["r1", "r2"]) == 2
    rows = dict(db.conn.execute("SELECT id, labels FROM raw_chunks WHERE labels_final = 1"))
    assert Label(rows["r1"]) == Label.MONEY | Label.HEALTH and Label(rows["r2"]) == Label.NONE


def test_cloud_gate_needs_final_labels_even_for_his_own_words():
    own = {"labels": 0, "labels_final": False, "tier": "user"}
    assert not extract.cloud_safe_fact(own)  # not final yet -> stays home
    assert extract.cloud_safe_fact({**own, "labels_final": True})
    assert not extract.cloud_safe_fact({"labels": 1, "labels_final": True, "tier": "user"})
    assert not extract.cloud_safe_fact({"labels": 0, "tier": "other"})  # finality unknown -> not cloud-safe
    snap = {"facts": [{**own, "id": "a"}, {**own, "id": "b", "labels_final": True}]}
    assert [f["id"] for f in extract.cloud_safe(snap)["facts"]] == ["b"]


def test_facade_records_notices_and_the_stop_hook_denial(mem):
    policy.set_classifier(None)
    words = "запомни: у Кати кошка"
    api.remember(words, turn=turn_of(words, n=7), because_quote=words, store=mem)
    notes = api.recorded_notifications(store=mem)
    assert notes and notes[-1]["tool"] == "memory_remember" and notes[-1]["text"]
    assert notes[-1]["at"].tzinfo is not None
    t = turn_of("как дела", n=8)
    blocked = api.claims_unbacked_change(t, "Запомнил, исправил.", store=mem)
    denials = api.recorded_denials(store=mem)
    assert blocked
    assert {"tool": "stop", "target": None, "reason": "claim_without_commit"}.items() <= denials[
        -1
    ].items()
