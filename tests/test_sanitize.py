"""The sanitizer is the only thing standing between an attacker's text and a model.
These tests run entirely offline - no model is called."""

from __future__ import annotations

from pathlib import Path

from koob.sanitize import detect_injection, sanitize

INJECTION = Path(__file__).parent / "fixtures" / "injection"


def test_every_injection_fixture_is_flagged(injection_cases):
    """All six required cases from spec section 5 must raise the suspicious flag."""
    assert len(injection_cases) == 6
    for name, raw in injection_cases:
        result = sanitize(raw)
        assert result.suspicious, f"{name} passed undetected"
        assert result.reasons, name


def test_clean_twin_is_not_flagged():
    clean = (INJECTION / "clean_twin.txt").read_text(encoding="utf-8")
    assert not sanitize(clean).suspicious


def test_requirements_survive_sanitizing():
    """The real content of a poisoned posting must match its clean twin: an attacker
    must not be able to hide the actual requirements."""
    clean = sanitize((INJECTION / "clean_twin.txt").read_text(encoding="utf-8")).text
    # Every requirement line of the clean twin must survive in each poisoned variant.
    wanted = [ln.strip("- ").strip() for ln in clean.splitlines() if ln.strip().startswith("-")]
    assert len(wanted) == 5, "fixture drift: the clean twin should list five requirements"

    for name in ("01_direct_instruction.txt", "02_hidden_css.html", "03_html_comment.html"):
        dirty = sanitize((INJECTION / name).read_text(encoding="utf-8")).text
        for line in wanted:
            assert line in dirty, f"{name} lost requirement '{line}'"
        assert "Philadelphia" in dirty


def test_hidden_html_text_is_removed_not_just_flagged():
    raw = (INJECTION / "02_hidden_css.html").read_text(encoding="utf-8")
    out = sanitize(raw)
    assert "PWNED" not in out.text
    assert "set every requirement" not in out.text
    assert "hidden_element" in out.reasons


def test_html_comments_never_reach_the_model():
    raw = (INJECTION / "03_html_comment.html").read_text(encoding="utf-8")
    out = sanitize(raw)
    assert "SYSTEM PROMPT OVERRIDE" not in out.text
    assert "html_comment_directive" in out.reasons


def test_zero_width_characters_are_stripped_and_flagged():
    raw = (INJECTION / "04_zerowidth_homoglyph.txt").read_text(encoding="utf-8")
    out = sanitize(raw)
    assert "​" not in out.text
    # With the zero-widths removed the phrase becomes visible to the detector.
    assert "ignore_instructions" in out.reasons or "invisible_characters" in out.reasons


def test_script_and_style_are_dropped():
    raw = "<p>Role</p><script>alert('x')</script><style>body{}</style>"
    out = sanitize(raw)
    assert "alert" not in out.text and "body{}" not in out.text


def test_truncation_bounds_input():
    out = sanitize("a" * 50_000)
    assert len(out.text) == 20_000
    assert out.truncated


def test_detector_does_not_fire_on_ordinary_postings():
    ordinary = (
        "We are looking for an intern to join the security team. You will assist with "
        "vulnerability assessments and report writing. Our systems are large and our "
        "assistants are friendly."
    )
    assert detect_injection(ordinary) == []
