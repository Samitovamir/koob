"""Memory spec §5.1 rule 3: never-store classes. Secrets, passwords and keys never
enter memory or a model (absolute rule). Spec marker: `[T policy/test_never_store]`."""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import AuthorTier

from .conftest import owner

SECRETS = [
    "мой пароль от почты: hunter2!",
    "пароль от вайфая Kx9!pz2Q",
    "пин-код от карты 4821",
    "password = s3cr3t-Pa55",
    "api key: AIzaSyD3x4mpl3K3yV4lu3F0rT3st1ngPurp0s3s",
    "sk-ant-api03-Zx8vQ2mN7pL4kR9tW1yB6cD3fG5hJ0",
    "ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
    "AKIAIOSFODNN7EXAMPLE",
    "bot 123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw1",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
    "Ваш код подтверждения: 482913",
    "482913 — ваш код для входа",
    "карта 4111 1111 1111 1111, срок 12/28",
    "IBAN DE89 3704 0044 0532 0130 00",
    "р/с 40817810099910004312",
    "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----",
    "postgres://denis:Tr0ub4dor3@db.local:5432/app",
    "otpauth://totp/GitHub:denis?secret=JBSWY3DPEHPK3PXP",
    "токен такой: 9fK2mQx7Lp3ZrT8wVb5N",
    "random Zq9XwT4vB7nM2kLp8RfY3dHs here",
]
ORDINARY = [
    "у Саши день рождения 12 мая",
    "коммит 849293ed и ветка mem/l1-policy",
    "id 3f2b8c1e-9d4a-4e6b-8f1a-2c3d4e5f6a7b",
    "позвони Диме +7 916 123-45-67",
    "надо сменить пароль на почте",
    "Новый код от домофона 5521, квартира 12",
    "встреча в 2026-10-14 в 19:30, зал 4B",
    "test_own_sensitive_used_where_relevant_with_note",
    "ключ от квартиры под ковриком",
    "the token is valid until Friday",
    "Error: password is required",
    "password: ********",
]


@pytest.mark.parametrize("text", SECRETS)
def test_secret_shapes_found(text):
    spans = policy.never_store_spans(text)
    assert spans, text
    red = policy.redact(text)
    assert policy.SECRET_MARK in red
    for s, e in spans:
        assert text[s:e] not in red


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_text_untouched(text):
    assert policy.never_store_spans(text) == []


def test_never_store(db, turn):
    """§5.1: never-store classes are dropped; in the user's own words only the span becomes
    [secret removed]."""
    conn = db.conn
    before = db.table_counts()

    # foreign text (a tool result, an SMS, another person) carrying a secret: the chunk is dropped
    for tier in (AuthorTier.TOOL, AuthorTier.OTHER, AuthorTier.SYSTEM, AuthorTier.USER_ROOM):
        ok, reason = policy.admit(
            conn,
            source_kind="tool_result",
            chat_id=None,
            text="Ваш код подтверждения: 482913. Никому не сообщайте.",
            author_tier=tier,
            turn=turn,
        )
        assert (ok, reason) == (False, "never_store")

    # the user's own words: admitted, only the span replaced
    a = policy.admit_ex(
        conn,
        source_kind="cube_turn",
        chat_id=None,
        text="запомни: пароль от вайфая Kx9!pz2Q, роутер в коридоре",
        author_tier=AuthorTier.USER,
        turn=owner("запомни: пароль от вайфая Kx9!pz2Q, роутер в коридоре"),
    )
    assert a.admit and a.reason == "secret_redacted"
    assert a.text == "запомни: пароль от вайфая [secret removed], роутер в коридоре"
    assert "Kx9" not in a.text

    # nothing but content-free audit lines was written
    after = db.table_counts()
    assert {t: after[t] - before[t] for t in after if after[t] != before[t]} == {"audit": 4}
    for event, actor, record_id in conn.execute("SELECT event, actor, record_id FROM audit"):
        assert event == "admit.reject.never_store" and actor == "policy" and record_id is None


def test_password_store_kinds_never(db):
    """Rule 7: password managers and the Keychain never, whoever's words."""
    for kind in ("keychain", "password_manager", "1password"):
        ok, reason = policy.admit(
            db.conn,
            source_kind=kind,
            chat_id=None,
            text="anything",
            author_tier=AuthorTier.USER,
            turn=None,
        )
        assert (ok, reason) == (False, "never_store")


def test_clean_text_admitted_unchanged(db, turn):
    a = policy.admit_ex(
        db.conn,
        source_kind="cube_turn",
        chat_id=None,
        text=turn.owner_words,
        author_tier=AuthorTier.USER,
        turn=turn,
    )
    assert a.admit and a.reason is None and a.text == turn.owner_words



# the other direction of the absolute rule: talk about passwords is not a password
TALK_ABOUT_PASSWORDS = [
    "сбросил пароль от почты",
    "забыл пароль, напомни как восстановить",
    "PASS: test_login",
    "PASS: tests/test_api.py::test_login",
    "pass: 12/12",
    "boarding pass: 12B",
    "надо сменить пароль на почте",
    "поменял пароль на почте, всё ок",
    "поменял пароль на госуслугах",
    "пароль поменял, теперь всё работает",
    "пароль: теперь другой",
    "changed the password to something stronger",
    "reset the password to default",
    "пароль от почты denis@example.com — поменять",
]


@pytest.mark.parametrize("text", TALK_ABOUT_PASSWORDS)
def test_talk_about_passwords_stays_uncut(text):
    assert policy.never_store_spans(text) == []


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ('поменял пароль на "correct horse"', 'поменял пароль на "[secret removed]"'),
        ("password was changed to Hunter2024 yesterday", "password was changed to [secret removed] yesterday"),
        ("the password is now Hunter2024", "the password is now [secret removed]"),
        ("пароль на вайфай: sunflower", "пароль на вайфай: [secret removed]"),
        ("пароль от вайфая: «мой кот Барсик 2019»", "пароль от вайфая: «[secret removed]»"),
    ],
)  # fmt: skip
def test_change_and_quoted_values_redacted_whole(text, kept):
    assert policy.redact(text) == kept


def _say(conn, text, *, chat="tg:lena", tier=AuthorTier.USER, ts="2026-10-12T09:00:00.000000Z"):
    return policy.admit_ex(
        conn, source_kind="telegram", chat_id=chat, text=text, author_tier=tier, turn=None, ts=ts
    )


def test_lone_password_after_the_keyword_is_redacted_in_his_words(db):
    """«пароль от вайфая:», then «sunflower» alone: the user's own value becomes the mark."""
    assert _say(db.conn, "пароль от вайфая:").text == "пароль от вайфая:"
    a = _say(db.conn, "sunflower", ts="2026-10-12T09:01:00.000000Z")
    assert a.admit and a.reason == "secret_redacted" and a.text == policy.SECRET_MARK


def test_lone_password_from_another_person_after_his_question_is_dropped(db):
    """The user asks «какой пароль от вайфая?», another person answers «Kx9pz2Q7»: the chunk
    is dropped."""
    assert _say(db.conn, "какой пароль от вайфая?").admit
    a = _say(db.conn, "Kx9pz2Q7", tier=AuthorTier.OTHER, ts="2026-10-12T09:00:30.000000Z")
    assert (a.admit, a.reason) == (False, "never_store")


@pytest.mark.parametrize(
    ("first", "second", "ts"),
    [
        ("сбросил пароль от почты", "спасибо", "2026-10-12T09:01:00.000000Z"),
        ("забыл пароль, напомни как восстановить", "Barsik2019", "2026-10-12T09:01:00.000000Z"),
        ("пароль от вайфая:", "Barsik2019", "2026-10-12T09:30:00.000000Z"),  # the flag is stale
        ("пароль от вайфая:", "ну и где он записан?", "2026-10-12T09:01:00.000000Z"),
    ],
)
def test_a_keyword_does_not_cut_ordinary_next_messages(db, first, second, ts):
    assert _say(db.conn, first).admit
    a = _say(db.conn, second, ts=ts)
    assert a.admit and a.text == second


def test_the_flag_is_per_chat_and_holds_no_content(db):
    _say(db.conn, "пароль от вайфая:", chat="tg:lena")
    a = _say(db.conn, "Barsik2019", chat="tg:gleb", ts="2026-10-12T09:00:10.000000Z")
    assert a.text == "Barsik2019"
    for at, left, strong in policy._EXPECT.values():
        assert isinstance(left, int) and isinstance(strong, bool) and at.tzinfo is not None
