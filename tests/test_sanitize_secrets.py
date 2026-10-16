"""The secret scrub for vault notes (sanitize.scrub_secrets, PENTEST-MAP §3.3).

Every fake key here is assembled at runtime (`"ghp_" + "a" * 36`), so no key-shaped
string is ever committed. The scrub guarantees that a key or a password from a note never
reaches the reader on screen or the model."""

from __future__ import annotations

import pytest

from koob import sanitize
from koob.assistant import memory
from koob.sanitize import SCRUB_KEY, SCRUB_VALUE, scrub_secrets


def _keys() -> dict[str, str]:
    return {
        "github": "ghp_" + "a1B2" * 9,
        "github_pat": "github_pat_" + "Z9y8" * 6,
        "anthropic": "sk-" + "ant-" + "api03-" + "q" * 30,
        "aws": "AKIA" + "ABCDEFGHIJKLMNOP",
        "slack": "xox" + "b-" + "1234567890-abcdefghij",
        "google": "AIza" + "S" * 35,
        "jwt": "eyJ" + "a" * 20 + ".eyJ" + "b" * 20 + "." + "c" * 20,
        "bot": "123456789:" + "A" * 35,
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\n"
        + "M" * 64
        + "\n-----END RSA PRIVATE KEY-----",
    }


@pytest.mark.parametrize("name", list(_keys()))
def test_each_key_shape_is_hidden(name):
    secret = _keys()[name]
    text = "# Setup\n\n" + "Some words about the lab. " * 80 + f"\nkey here: {secret}\n"
    out = scrub_secrets(text)
    assert secret not in out.text
    assert SCRUB_KEY in out.text
    assert out.strong_hits == 1 and not out.skipped
    assert out.text.startswith("# Setup")


@pytest.mark.parametrize(
    "line,kept,value",
    [
        ("password: Summer2024!", "password: " + SCRUB_VALUE, "Summer2024!"),
        ("Пароль = qwerty-9981", "Пароль = " + SCRUB_VALUE, "qwerty-9981"),
        ("api_key=4f1c2e9a7b", "api_key=" + SCRUB_VALUE, "4f1c2e9a7b"),
        ('"token": "tok_live_998877"', '"token": "' + SCRUB_VALUE + '"', "tok_live_998877"),
        ("**Password:** Winter!2026", "**Password:** " + SCRUB_VALUE, "Winter!2026"),
        ("secret: hunter2hunter2", "secret: " + SCRUB_VALUE, "hunter2hunter2"),
        # the shapes the memory detector already flagged (one list for both)
        ("api_hash: " + "1a2b3c4d" * 4, "api_hash: " + SCRUB_VALUE, "1a2b3c4d" * 4),
        ("Password: $up3rS3cret!", "Password: " + SCRUB_VALUE, "$up3rS3cret!"),
        ("password: (Hunter2)", "password: " + SCRUB_VALUE, "Hunter2"),
        ("- пароль от VPN: Zx9!kLm2#pQ", "пароль от VPN: " + SCRUB_VALUE, "Zx9!kLm2#pQ"),
        ("ssh root@10.10.10.5 -p 2222 # pw: Zx9kLm2pQ", "# pw: " + SCRUB_VALUE, "Zx9kLm2pQ"),
        ("user: admin pass: Adm1n!2026", "pass: " + SCRUB_VALUE, "Adm1n!2026"),
    ],
)
def test_the_key_name_stays_and_the_value_goes(line, kept, value):
    text = "Notes on the box.\n" * 30 + line + "\n"
    out = scrub_secrets(text)
    assert kept in out.text
    assert value not in out.text
    assert out.strong_hits == 0 and out.hits == 1


def test_a_hydra_example_keeps_its_placeholders():
    lines = [
        "hydra -l admin -P /usr/share/wordlists/rockyou.txt 10.10.10.5 http-post-form "
        '"/login:username=^USER^&password=^PASS^:F=incorrect"',
        "hydra -l admin -p password123 ssh://10.10.10.5",
        "curl -H 'Authorization: Bearer <TOKEN>' http://10.10.10.5/api",
        "export PASSWORD=$PASSWORD",
        "password: <your password>",
        "password=${PASS}",
        "pass: $(cat pass.txt)",
        "password: (see the vault)",
        "Password attacks: hydra, medusa, crackmapexec",
        "bypass: the WAF with chunked encoding",
    ]
    text = "\n".join(lines) + "\n"
    out = scrub_secrets(text)
    assert out.text == text and out.hits == 0


def test_a_literal_value_in_a_command_still_goes():
    text = (
        "Login:\n"
        + "context line\n" * 20
        + "sqlmap -u 'http://x/?id=1' --auth-cred password=Tr0ub4dor\n"
    )
    out = scrub_secrets(text)
    assert "Tr0ub4dor" not in out.text and "password=" + SCRUB_VALUE in out.text


def test_more_than_three_strong_hits_skip_the_note():
    k = _keys()
    body = "Long enough note body. " * 200
    four = body + "\n".join([k["github"], k["aws"], k["anthropic"], k["slack"]])
    three = body + "\n".join([k["github"], k["aws"], k["anthropic"]])
    assert scrub_secrets(four).skipped and scrub_secrets(four).text == ""
    assert not scrub_secrets(three).skipped


def test_over_ten_percent_hidden_skips_the_note():
    text = "x\n" + "\n".join(f"password: Value{i:04d}Longer" for i in range(10))
    out = scrub_secrets(text)
    assert out.ratio > 0.10 and out.skipped and out.text == ""


def test_a_note_with_no_secrets_is_byte_identical():
    text = (
        "---\nstatus: active\ntags: [networking]\n---\n# TCP и UDP\n\n"
        "> [!danger] Не путай\n> TCP — с установкой соединения.\n\n"
        "```bash\nnmap -sS -p- 10.10.10.5\n```\n"
        "| порт | служба |\n|---|---|\n| 22 | ssh |\n"
        "Хэш MD5 пустой строки: d41d8cd98f00b204e9800998ecf8427e\n"
        "Токены и пароли обсуждаем словами, без значений.\n"
    )
    out = scrub_secrets(text)
    assert out.text == text and out.text is text
    assert (out.hits, out.strong_hits, out.ratio, out.skipped) == (0, 0, 0.0, False)


def test_an_env_line_with_a_random_value_is_hidden():
    value = "a9F3k2L8q1Z7x4C6v0B5"
    text = "Config:\n" + "line\n" * 30 + f"SERVICE_KEY={value}\nDEBUG=true\n"
    out = scrub_secrets(text)
    assert value not in out.text and "DEBUG=true" in out.text


def test_the_scrub_never_returns_what_it_hid():
    k = _keys()["github"]
    out = scrub_secrets("x " * 400 + k)
    assert k not in repr(out) and k not in out.text


def test_memory_looks_secret_as_before():
    # one shared list lives in sanitize.py; memory imports it
    assert memory.SECRET_PATTERNS is sanitize.SECRET_PATTERNS
    assert memory.looks_secret("token: " + "abcdef123456")
    assert memory.looks_secret(_keys()["github"])
    assert memory.looks_secret("a" * 40)  # long hex-ish blob
    assert not memory.looks_secret("Сегодня прочитал про TCP и DNS")
    with pytest.raises(memory.MemoryError):
        memory.check_text("key " + _keys()["aws"])


@pytest.mark.parametrize(
    "text",
    [
        "Код из Telegram\n12345",
        "Enter this verification code in Authenticator\n482 913",
        "Enter code in Authenticator\n482 913",
        "Код подтверждения\n\n  4 8 2 9 1 3",
    ],
)
def test_a_code_on_the_line_after_its_label_is_hidden(text):
    out, hits = sanitize.redact_secrets(text)
    assert hits == 1 and "[скрыто: похоже на код]" in out
    assert not any(ch.isdigit() for ch in out.split("\n")[-1])


def test_a_number_that_is_not_alone_under_the_label_stays():
    assert sanitize.redact_secrets("source code\n1234 lines")[1] == 0
    assert sanitize.redact_secrets("Код курса\n2026-09-24")[1] == 0


# shapes that previously got through the scrub
@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("Your verification code:482913", "Your verification code:"),
        (
            "Login code: 48213. Do not give this code to anyone",
            "Login code: . Do not give this code to anyone",
        ),
        (
            "Enter this verification code in Authenticator: 4 8 2 9 1 3",
            "Enter this verification code in Authenticator: ",
        ),
        ("Код из Telegram\n\n\n12345", "Код из Telegram\n\n\n"),
        ("code\n\n\n\n482913.", "code\n\n\n\n."),
        ("Код\n12 345", "Код\n"),
        ("code\n4829-13", "code\n"),
        ("verification: 9876 5432", "verification: "),
    ],
)
def test_each_recheck_code_shape_is_hidden(text, kept):
    out, hits = sanitize.redact_secrets(text)
    assert hits == 1
    assert out.replace("[скрыто: похоже на код]", "") == kept


@pytest.mark.parametrize(
    "text",
    [
        "source code\n1234 lines",
        "Код подтверждения\n2026-09-24",
        "code 2026-09-24",
        "код 14:30:00, code 12:3456",
        "экзамен 2026-09-24 в 14:30, код 10:45",
        "Assignment code: CS-101 due 2026-09-24 14:30",
        "Код курса\n09/2026",
        "code\n\n\n\n\n482913",  # four blank lines: too far from the label
    ],
)
def test_dates_times_and_far_numbers_stay(text):
    assert sanitize.redact_secrets(text) == (text, 0)
