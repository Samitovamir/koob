from __future__ import annotations

from datetime import datetime

from koob.memory.eval import graders
from koob.memory.eval.observe import Observation
from koob.memory.eval.schema import Expect


class Insp:
    def __init__(self, rows):
        self.rows = rows

    def records(self, where):
        return self.rows

    def channel_text(self, channel):
        return None


def obs(**kw) -> Observation:
    o = Observation(wording="q", level="e2e", tz="Europe/Moscow")
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def ex(**kw) -> Expect:
    return Expect.model_validate(kw)


def failed(checks):
    return [c.name for c in checks if c.failed]


def test_include_exclude_with_alternatives_and_typed():
    o = obs(primary="Она живёт в Твери с 17 мая, 52 000 руб")
    e = ex(
        must_include=[["Тверь", "Твери"], {"date": "2026-05-17"}, {"money": 52000}],
        must_not_include=["Твер", {"date": "--11-02"}],
    )
    assert not failed(graders.grade_expect(o, e))
    bad = obs(primary="Тверь, 2 ноября")
    assert len(failed(graders.grade_expect(bad, e))) == 5


def test_forbidden_per_channel_and_unobservable_is_skipped_not_passed():
    o = obs(channels={"basket": "старое: Тверь"}, answer="ok")
    e = ex(forbidden=[{"values": ["Твер"], "channels": ["basket", "cli_prompt"]}])
    cs = graders.grade_expect(o, e)
    assert failed(cs) == ["forbidden[basket]"]
    assert [c.status for c in cs if c.name == "forbidden[cli_prompt]"] == ["skip"]


def test_forbidden_allow_history():
    e = ex(forbidden=[{"values": ["Твер"], "channels": ["answer"], "allow": "history"}])
    assert not failed(graders.grade_expect(obs(answer="Раньше было Тверь, теперь Тверь"), e))
    assert failed(graders.grade_expect(obs(answer="Она в Твери"), e))
    cli = ex(forbidden=[{"values": ["Твер"], "channels": ["cli_prompt"], "allow": "history"}])
    ok = obs(channels={"cli_prompt": "Тверь ... <memory-update>теперь Тверь</memory-update>"})
    bad = obs(channels={"cli_prompt": "<memory-update/> ... Тверь"})
    assert not failed(graders.grade_expect(ok, cli))
    assert failed(graders.grade_expect(bad, cli))


def test_forbidden_contact_in_actions_channel():
    o = obs(
        contact_ids={"kim": ["7100030"], "bel": ["7100031"]},
        actions=[{"tool": "tg_send", "target": "7100031", "strings": ["привет"]}],
    )
    assert failed(
        graders.grade_expect(
            o, ex(forbidden=[{"values": ["contact:bel"], "channels": ["actions"]}])
        )
    )
    assert not failed(
        graders.grade_expect(
            o, ex(forbidden=[{"values": ["contact:kim"], "channels": ["actions"]}])
        )
    )


def test_behaviours():
    assert not failed(
        graders.grade_expect(obs(behaviours={"answer", "no_action"}), ex(behaviour="answer"))
    )
    assert failed(graders.grade_expect(obs(behaviours={"answer"}), ex(behaviour="ask_one")))
    assert not failed(
        graders.grade_expect(obs(behaviours={"ask_one"}), ex(behaviour_any=["action", "ask_one"]))
    )


def test_ask_one_names_every_candidate_and_counts_questions():
    e = ex(behaviour="ask_one", ask={"must_name": [["Ким", "Kim"], ["Белов"]], "max_questions": 1})
    good = obs(
        behaviours={"ask_one"},
        answer="Какой Саше: Киму или Беловой?",
        questions_text="Какой Саше: Киму или Беловой?",
    )
    assert not failed(graders.grade_expect(good, e))
    one_missing = obs(behaviours={"ask_one"}, answer="Какому Киму?", questions_text="Какому Киму?")
    assert failed(graders.grade_expect(one_missing, e))
    two_q = obs(behaviours={"ask_one"}, answer="Ким? Белова?", questions_text="Ким? Белова?")
    assert "ask.max_questions" in failed(graders.grade_expect(two_q, e))


def test_offer_grounds_and_check_phrase():
    e = ex(
        behaviour="offer", offer={"grounds": [["Твер"], ["билет"]], "max": 1, "check_phrase": True}
    )
    good = obs(
        behaviours={"offer"},
        offers=[{"text": "В Твери есть билеты. Проверить?", "grounds_text": ""}],
    )
    assert not failed(graders.grade_expect(good, e))
    two = obs(behaviours={"offer"}, offers=[{"text": "Тверь билет?", "grounds_text": ""}] * 2)
    assert "offer.max" in failed(graders.grade_expect(two, e))


def test_actions_count_and_text():
    a = [{"tool": "tg_send", "target": "7100030", "strings": ["созвон в 16:00"]}]
    o = obs(actions=a, contact_ids={"kim": ["7100030"], "bel": ["7100031"]})
    e = ex(actions=[
        {"tool": "tg_send", "target": "contact:kim", "count": 1, "text_must_include": [{"time": "16:00"}]},
        {"tool": "tg_send", "target": "contact:bel", "count": 0},
    ])  # fmt: skip
    assert not failed(graders.grade_expect(o, e))
    assert failed(graders.grade_expect(obs(actions=[], contact_ids=o.contact_ids), e))


def test_denied_and_lines():
    o = obs(
        denials=[{"tool": "tg_send", "target": None, "reason": "ask_which"}],
        lines={"_probe": "Забыл. Перестал использовать."},
    )
    assert not failed(
        graders.grade_expect(
            o,
            ex(
                denied=[{"tool": "tg_send", "reason_any": ["ask_which"]}],
                lines={"must": ["перестал использовать"]},
            ),
        )
    )
    assert failed(
        graders.grade_expect(o, ex(denied=[{"tool": "tg_send", "reason_any": ["frozen"]}]))
    )


def test_records_where_expect_count():
    rows = [
        {
            "kind": "fact",
            "value_text": "cilantro",
            "polarity": "negated",
            "status": "active",
            "labels": ["health"],
            "author_tier": "user",
            "subject": "self",
        }
    ]
    o = obs(inspector=Insp(rows))
    good = ex(
        records=[
            {
                "where": {"kind": "fact", "value": ["cilantro", "кинза"]},
                "expect": {"polarity": "negated", "labels": ["health"]},
            }
        ]
    )
    assert not failed(graders.grade_expect(o, good))
    wrong = ex(
        records=[
            {"where": {"kind": "fact", "value": "cilantro"}, "expect": {"polarity": "affirmed"}}
        ]
    )
    assert failed(graders.grade_expect(o, wrong))
    zero = ex(records=[{"where": {"kind": "fact", "value": "golf"}, "count": 0}])
    assert not failed(graders.grade_expect(o, zero))
    assert failed(
        graders.grade_expect(
            o, ex(records=[{"where": {"kind": "fact", "value": "cilantro"}, "count": 2}])
        )
    )


def test_null_expect_means_unknown_stays_null():
    o = obs(inspector=Insp([{"kind": "fact", "value_text": "x", "valid_to": None}]))
    assert not failed(
        graders.grade_expect(
            o, ex(records=[{"where": {"kind": "fact", "value": "x"}, "expect": {"valid_to": None}}])
        )
    )


def test_expect_any_needs_one_block():
    blocks = [ex(behaviour="ask_one"), ex(behaviour="answer", must_include=["Ким"])]
    assert not failed(graders.grade_any(obs(behaviours={"answer"}, primary="Ким"), blocks))
    assert failed(graders.grade_any(obs(behaviours={"answer"}, primary="нет"), blocks))


def test_notifications_window_and_counts():
    n = [
        {"at": datetime.fromisoformat("2026-03-01T07:30:00+00:00"), "text": "Напоминаю про билеты"}
    ]
    e = ex(
        notifications={
            "window": ["2026-03-01T10:00", "2026-03-01T12:00"],
            "must": [{"contains": ["билет"], "count": 1}],
            "must_not_contain": ["пароль"],
        }
    )
    assert not failed(graders.grade_expect(obs(notifications=n), e))
    assert failed(graders.grade_expect(obs(notifications=[]), e))


def test_scan_state_checks():
    e = ex(
        scan={"canaries": ["c1"]}, journal={"forget_entries": 1}, chat_policy={"tg:gleb": "banned"}
    )
    o = obs(scan={"c1": []}, forget_journal_entries=1, chat_policy={"tg:gleb": "banned"})
    assert not failed(graders.grade_expect(o, e))
    bad = obs(scan={"c1": ["memory_db:x:lat/utf8"]}, forget_journal_entries=0, chat_policy={})
    assert len(failed(graders.grade_expect(bad, e))) == 3
