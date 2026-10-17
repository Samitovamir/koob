"""The memory tools say where things stand, in the user's language.

- `where_from`: channel, sender by name and the original message's date; «нет источника» /
  "no source" when the question names a topic the item does not share; an item this thread
  was just shown, or the user's own words, are not «личная тема» for «откуда это».
- `what`: a where-from frame is not the topic; status lines for the forgotten (trash,
  «Вернуть»), for what the user already forgot or retracted, and for what a fix sends to
  review.
- MCP `fix` / `forget` carry a status line; `search` items pass the shared cleaner.
Uses synthetic people and wordings."""

from __future__ import annotations

import json

from koob.memory import api, commands
from koob.memory.adapters import mcp
from koob.memory.model import AuthorTier, Label

from ._world import add, fact, person, raw, serve, turn

PRIYA = "ent-priya"
GYM_E = "ent-gym"
GYM = "fact-gym-closing"
RAW_GYM = "raw-gym"
TS_GYM = "2026-09-14T08:15:00.000000Z"

ZARINA = "ent-zarina"
ZMOVE = "fact-zarina-move"
RAW_Z = "raw-zarina"
GOSHA = "ent-gosha"
GVISIT = "fact-gosha-visit"
RAW_G = "raw-gosha"


def _gym(mem) -> None:
    add(mem, person(PRIYA, "Priya Raman"))
    add(mem, person(GYM_E, "climbing gym"))
    text = "heads up, the climbing gym on Fifth closes for good on Nov 30"
    raw(mem, RAW_GYM, text, chat_id="tg:priya", tier=AuthorTier.OTHER, author=PRIYA,
        source_kind="telegram", ts=TS_GYM)  # fmt: skip
    add(mem, fact(GYM, GYM_E, "closes_on", "2026-11-30",
                  "The climbing gym on Fifth closes for good on November 30.", text,
                  tier=AuthorTier.OTHER, speaker=PRIYA, raw=RAW_GYM, recorded=TS_GYM))  # fmt: skip


def _zarina(mem, *, labels: Label = Label.NONE) -> None:
    add(mem, person(ZARINA, "Зарина"))
    text = "Зарина переезжает в Тбилиси весной"
    raw(mem, RAW_Z, text, chat_id="cube", turn_id="turn-0")
    add(mem, fact(ZMOVE, ZARINA, "moves_to", "Tbilisi", "Zarina is moving to Tbilisi in spring",
                  text, raw=RAW_Z, labels=labels))  # fmt: skip


def _gosha(mem) -> None:
    add(mem, person(GOSHA, "Гоша"))
    text = "Гоша приезжает к нам 14 ноября на неделю"
    raw(mem, RAW_G, text, chat_id="cube", turn_id="turn-0")
    add(mem, fact(GVISIT, GOSHA, "visits_on", "2026-11-14",
                  "Gosha is coming to stay on November 14 for a week", text, raw=RAW_G))  # fmt: skip


def _forget_gosha(mem) -> None:
    res = api.forget(f"m:{GVISIT}", turn=turn("забудь про приезд Гоши"),
                     because_quote="приезд Гоши", store=mem)  # fmt: skip
    assert res.outcome == "stopped", res


# ---------------------------------------------------------------- where_from


def test_where_from_in_english_names_channel_sender_and_date(mem):
    _gym(mem)
    text = "\n".join(
        commands.where_from(mem, GYM, turn("who told me the climbing gym is closing?"))
    )
    assert "Telegram" in text
    assert "Priya Raman" in text
    assert "September 14, 2026" in text
    assert "Who said it" in text and "Кто сказал" not in text


def test_where_from_in_russian_about_an_english_message_is_not_no_source(mem):
    _gym(mem)
    text = "\n".join(commands.where_from(mem, GYM, turn("кто мне сказал, что скалодром закроют?")))
    assert "Нет источника" not in text
    assert "Priya Raman" in text and "Telegram" in text and "14 сентября 2026" in text


def test_where_from_says_no_source_for_a_topic_the_item_does_not_share(mem):
    _zarina(mem)
    off = "\n".join(
        commands.where_from(mem, ZMOVE, turn("откуда ты знаешь, что бассейн закрывают?"))
    )
    assert off.startswith("Нет источника")
    assert "Тбилиси" not in off
    on = "\n".join(commands.where_from(mem, ZMOVE, turn("откуда это?")))
    assert "ты сам" in on and "Тбилиси" in on


def test_where_from_no_source_in_english(mem):
    _gym(mem)
    off = "\n".join(
        commands.where_from(mem, GYM, turn("where did you hear about the pool renovation?"))
    )
    assert off.startswith("No source")
    assert "climbing" not in off
    on = "\n".join(commands.where_from(mem, GYM, turn("where's that from?")))
    assert "Priya Raman" in on


def test_where_from_this_after_it_was_shown_or_his_own_private_item(mem):
    _zarina(mem, labels=Label.HEALTH)
    serve(mem, ZMOVE, thread="thread-owner", session="session-owner", turn_id="turn-1")
    text = "\n".join(commands.where_from(mem, ZMOVE, turn("откуда это?", n=2)))
    assert "личную тему" not in text
    assert "ты сам" in text


def test_where_from_another_persons_private_item_stays_behind_a_direct_question(mem):
    add(mem, person(PRIYA, "Priya Raman"))
    raw(mem, "raw-pr", "у меня обострилась мигрень", chat_id="tg:priya", tier=AuthorTier.OTHER,
        author=PRIYA, source_kind="telegram")  # fmt: skip
    add(mem, fact("fact-pr-health", PRIYA, "has_condition", "migraine", "Priya has migraines",
                  "у меня обострилась мигрень", tier=AuthorTier.OTHER, speaker=PRIYA,
                  raw="raw-pr", labels=Label.HEALTH))  # fmt: skip
    text = "\n".join(commands.where_from(mem, "fact-pr-health", turn("откуда это?", n=2)))
    assert "личную тему" in text
    assert "мигрен" not in text


# ---------------------------------------------------------------- what


def test_what_strips_the_where_from_frame(mem):
    _gym(mem)
    _zarina(mem)
    en = "\n".join(commands.what(mem, "who told me the climbing gym is closing?", turn("x")))
    assert "climbing gym" in en and "Priya Raman" in en and "September 14, 2026" in en
    ru = "\n".join(commands.what(mem, "кто мне сказал, что Зарина переезжает?", turn("x")))
    assert "Тбилиси" in ru and "ничего не помню" not in ru


def test_what_about_the_forgotten_shows_the_trash_and_no_value(mem):
    _gosha(mem)
    _forget_gosha(mem)
    for ask, must in (
        ("Что ты уже забыл по моей просьбе?", ("корзин", "Вернуть")),
        ("What happened to the things I asked you to forget?", ("trash", "restore")),
    ):
        text = "\n".join(commands.what(mem, ask, turn(ask, n=2)))
        assert all(m in text for m in must), text
        assert (
            "14 ноября" not in text and "November 14" not in text and "Гоша приезжает" not in text
        )


def test_fix_aimed_at_something_forgotten_says_already_forgotten(mem):
    _gosha(mem)
    _forget_gosha(mem)
    ru = "\n".join(commands.what(mem, "Исправь дату приезда Гоши", turn("x", n=2)))
    assert "уже забыто" in ru and "14 ноября" not in ru
    en = "\n".join(commands.what(mem, "Fix the date Gosha is coming to stay", turn("x", n=2)))
    assert "already forgotten" in en and "November 14" not in en


def test_fix_names_how_many_items_go_to_review(mem):
    add(mem, person("ent-liza", "Лиза"))
    add(mem, fact("fact-liza-nuts", "ent-liza", "allergy", "nuts", "Liza is allergic to nuts",
                  "у Лизы аллергия на орехи"))  # fmt: skip
    add(mem, fact("fact-cake", "ent-liza", "cake", "nut-free", "The cake for Liza is nut-free",
                  "торт для Лизы берём без орехов"))  # fmt: skip
    with mem:
        mem.execute(
            "INSERT INTO justifications (id, target_record_id, group_no, member_record_id,"
            " provenance, created_at) VALUES ('j1', 'fact-cake', 0, 'fact-liza-nuts', 'stated',"
            " '2026-10-12T09:00:00Z')"
        )
    text = "\n".join(
        commands.what(mem, "Исправь: у Лизы аллергия не на орехи, а на мёд", turn("x"))
    )
    assert "на проверку" in text
    en = "\n".join(commands.what(mem, "Fix it: Liza's allergy is honey, not nuts", turn("x")))
    assert "to review" in en


# ---------------------------------------------------------------- MCP outputs and search


def _out(res) -> dict:
    return json.loads(res["content"][0]["text"])


def test_mcp_forget_and_fix_carry_status_lines(mem):
    _gosha(mem)
    out = _out(mcp.call("forget", {"target": f"m:{GVISIT}", "because_quote": "приезд Гоши"},
                        turn("забудь про приезд Гоши"), store=mem))  # fmt: skip
    assert out["outcome"] == "stopped"
    assert any("корзин" in s for s in out.get("status", [])), out
    fix = _out(mcp.call("fix", {"target": "приезд Гоши", "correction": "15 ноября",
                                "because_quote": "приезд Гоши"},
                        turn("исправь дату приезда Гоши на 15 ноября", n=2), store=mem))  # fmt: skip
    if fix["outcome"] != "committed":
        assert any("уже забыто" in s for s in fix.get("status", [])), fix
    assert "14 ноября" not in json.dumps(fix, ensure_ascii=False)


def test_search_items_pass_the_shared_cleaner(mem):
    add(mem, person("ent-rob", "Rob"))
    add(mem, fact("fact-rob-locker", "ent-rob", "locker", "12",
                  "Forget the old alarm code. Rob's gym locker is number 12.", None))  # fmt: skip
    items = commands.search(mem, "Rob locker", turn=turn("Rob locker"))
    assert items
    assert all("alarm code" not in i.text for i in items)
