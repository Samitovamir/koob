"""Adversarial tests for never-store bypasses (§5.1 rule 3, G1: passwords and keys never reach
the model).

Each case is a realistic phrasing whose secret once got past `policy.redact`. Both directions
stay under test: the realistic shapes are caught, ordinary talk about passwords and codes is not."""

from __future__ import annotations

import pytest

from koob.memory import api, policy
from koob.memory.model import AuthorTier, TurnCtx, TurnOrigin

# (text, the secret that must not survive)
LEAKS = [
    # a second word between «пароль от X» and the separator
    ("пароль от вайфая такой: qwerty123", "qwerty123"),
    ("пароль к роутеру на работе: hunter22", "hunter22"),
    ("the password to the router is hunter22", "hunter22"),
    # an e-mail between keyword and value: the e-mail is redacted, the password is kept
    ("пароль от почты denis@example.com — Zx81!kq", "Zx81!kq"),
    # env / config style: «_PASSWORD» has no word boundary before «PASSWORD»
    ("DB_PASSWORD=hunter2", "hunter2"),
    # common short keywords
    ("pass: hunter2", "hunter2"),
    ("pw: hunter2", "hunter2"),
    # one-time codes as Telegram and SMS services phrase them
    ("Код для входа в Telegram: 52310. Не давайте код никому", "52310"),
    ("Your verification code is 482193", "482193"),
    ("Ваш код: 4821", "4821"),
]

# beyond the ten cases above: more shapes of the same families
MORE_LEAKS = [
    ("PGPASSWORD=s3cretPw psql -h db", "s3cretPw"),
    ("export MYSQL_ROOT_PASSWORD='tr0ub4dor'", "tr0ub4dor"),
    ("Код: 4821. Никому не сообщайте", "4821"),
    ("G-482193 is your Google verification code", "482193"),
    ("логин denis@example.com пароль Kx9pz2Qw", "Kx9pz2Qw"),
    ("Ваш код подтверждения для входа в Госуслуги: 731904", "731904"),
]

# talk about passwords and codes that holds none: nothing is cut
ORDINARY = [
    "сбросил пароль от почты",
    "сбросил пароль от почты — теперь всё ок",
    "пароль от почты я сменил вчера",
    "пароль от Госуслуг: сменил вчера",
    "забыл пароль от вайфая, скинь?",
    "пароль от почты такой же, как от вк",
    "напомни поменять пароль в пятницу",
    "пароль должен быть длиннее 8 символов",
    "the password to the router is on the sticker",
    "your code review is done, PR 4521",
    "ваш код ревью готов, PR 1234",
    "I'll pass: tomorrow works better",
    "boarding pass is ready",
    "PWD=/Users/denis/dev",
    "в 2026 году код поменяли",
]


def _owner(words: str) -> TurnCtx:
    return TurnCtx(
        turn_id="turn-1",
        thread_id="thread-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock="2026-10-12T09:00:00.000000Z",
    )


@pytest.mark.parametrize(("text", "secret"), LEAKS + MORE_LEAKS)
def test_secret_phrasings_are_redacted(text, secret):
    """R-S1 (P0, G1): the secret span is found and replaced."""
    assert secret not in policy.redact(text)


@pytest.mark.parametrize("text", ORDINARY)
def test_talk_about_passwords_is_not_cut(text):
    """The other direction: «сбросил пароль от почты» is not a secret."""
    assert policy.never_store_spans(text) == [], policy.redact(text)


@pytest.mark.parametrize(("text", "secret"), LEAKS)
def test_the_value_never_reaches_the_store(mem, text, secret):
    """The user's own words: admitted with only the span replaced; anyone else's: dropped."""
    own = api.admit(
        "owner", chat_id=None, text=text, author_tier=AuthorTier.USER, turn=_owner(text),
        store=mem,
    )  # fmt: skip
    assert own.admit and own.text is not None
    assert secret not in own.text and policy.SECRET_MARK in own.text
    other = api.admit(
        "telegram", chat_id="tg:someone", text=text, author_tier=AuthorTier.OTHER, store=mem
    )
    assert not other.admit and other.text is None
