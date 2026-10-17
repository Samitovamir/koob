"""The last ways an overtaken value of the user's reached the brain as current.

- A record the hot path wrote («запомни…») has no `source_raw_id` (the user's message is
  ingested after the write); a fix that retires it links the user's old message to it
  (`raw_retractions`), so the raw lane never quotes the retired value as current.
- The user's reason after a forget's target («…, don't want to jinx it») is not part of the
  target.
- «Ничего не помню про …» echoes the user's words with the side their own «не X, а Y» retired
  masked.
Synthetic names and wordings (the mechanism, not a probe)."""

from __future__ import annotations

from koob.memory import api, commands
from koob.memory.model import TurnCtx, TurnOrigin
from koob.memory.recall import vectors

TS = "2026-03-02T09:00:00.000000Z"


def _turn(words: str, n: int, *, clock: str = TS) -> TurnCtx:
    return TurnCtx(
        turn_id=f"turn-{n}",
        thread_id="thread-1",
        session_id="session-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        message_id=f"msg-{n}",
        clock=clock,
    )


def _ingest(mem, turn: TurnCtx, text: str, ts: str) -> None:
    api.ingest_turn(
        turn_id=turn.turn_id,
        thread_id=turn.thread_id,
        origin="owner",
        status="done",
        rows=[{"role": "owner", "ts": ts, "text": text}],
        store=mem,
    )


def _seen(mem, words: str, n: int) -> str:
    turn = _turn(words, n, clock="2026-03-04T09:00:00.000000Z")
    b = api.recall(turn, store=mem)
    found = api.search(words, turn=turn, store=mem) or []
    return "\n".join([b.block or "", *(i.text for i in b.items), *(i.text for i in found)])


def test_fix_masks_the_raw_a_hot_write_came_from(mem):
    """«Запомни: приём у ортодонта в четверг 5-го» (hot write), later «перенесли на пятницу 6-е»
    (a fix without «не X»): the old message never comes back with «5-го» as current."""
    vectors.reset()
    vectors.set_embedder(None)
    said = "запомни: приём у ортодонта в четверг 5-го в 16:30, кабинет 212"
    t1 = _turn(said, 1)
    w = api.remember(said, turn=t1, because_quote=said, store=mem)
    assert w.outcome == "committed"
    _ingest(mem, t1, said, "2026-03-02T09:00:00Z")
    fix = "приём у ортодонта перенесли на пятницу 6-е, в то же время"
    t2 = _turn(fix, 2, clock="2026-03-03T09:00:00.000000Z")
    r = api.fix(fix, fix, turn=t2, because_quote=fix, store=mem)
    assert r.outcome == "committed", r
    _ingest(mem, t2, fix, "2026-03-03T09:00:00Z")
    old_raw = mem.execute("SELECT id FROM raw_chunks WHERE turn_id = 'turn-1'").fetchone()[0]
    new_raw = mem.execute("SELECT id FROM raw_chunks WHERE turn_id = 'turn-2'").fetchone()[0]
    linked = {row[0] for row in mem.execute("SELECT raw_id FROM raw_retractions")}
    assert old_raw in linked and new_raw not in linked  # the fixing turn is the user's new word
    for n, words in enumerate(["когда приём у ортодонта?", "ортодонт", "кабинет 212"], 3):
        seen = _seen(mem, words, n)
        assert "четверг 5-го" not in seen, seen
    assert "пятницу 6-е" in _seen(mem, "когда приём у ортодонта?", 9)


def test_a_retired_record_from_another_message_is_not_linked(mem):
    """Only the user's raws that hold the record's own quote are linked: another message of the
    same turn stays as it is."""
    said = "запомни: пароль от вайфая на даче спрошу у соседа"
    t1 = _turn(said, 1)
    assert api.remember(said, turn=t1, because_quote=said, store=mem).outcome == "committed"
    api.ingest_turn(
        turn_id=t1.turn_id, thread_id=t1.thread_id, origin="owner", status="done",
        rows=[{"role": "owner", "ts": "2026-03-02T09:00:00Z", "text": said},
              {"role": "owner", "ts": "2026-03-02T09:00:05Z", "text": "а ещё купить лампочки"}],
        store=mem,
    )  # fmt: skip
    fix = "про вайфай на даче: спрошу у сторожа, не у соседа"
    t2 = _turn(fix, 2, clock="2026-03-03T09:00:00.000000Z")
    assert api.fix(fix, fix, turn=t2, because_quote=fix, store=mem).outcome == "committed"
    texts = {
        row[0]
        for row in mem.execute(
            "SELECT c.text FROM raw_retractions x JOIN raw_chunks c ON c.id = x.raw_id"
        )
    }
    assert texts == {said}


def test_forget_reason_is_not_the_target():
    """A dropped subject ("don't want…", «а то…», "since…") still ends the target."""
    for said, want in [
        ("forget what i told you about the scholarship, don't wanna jinx it", "scholarship"),
        ("forget the marathon plan, since my knee is bad", "marathon plan"),
        ("Забудь про поездку в Тбилиси, а то сглажу", "поездку в Тбилиси"),
        ("Забудь про ремонт, не хочется об этом", "ремонт"),
    ]:
        q, _whole = commands._forget_query(said)
        assert q.rstrip().endswith(want), (said, q)
    # a comma inside the target, with no reason word, keeps the rest
    q, _ = commands._forget_query("forget the trip to Porto, Lisbon and Faro")
    assert "Faro" in q


def test_not_found_echo_masks_his_retired_side(mem):
    """«Поправка: Лейла прилетает не 14–18 марта, а 2–6 апреля» asked as «что помнишь про…»
    with nothing stored: the echo never quotes 14–18 марта as current."""
    said = "Поправка: Лейла прилетает не 14–18 марта, а 2–6 апреля."
    lines = commands.render_what(commands._What(about=said))
    assert len(lines) == 1 and "ничего не помню" in lines[0]
    assert "14–18 марта" not in lines[0] and "2–6 апреля" in lines[0], lines
