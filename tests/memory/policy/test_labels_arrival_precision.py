"""Arrival precision of `minor` and `health` (§5.3): a child word sets `minor` beside a detail
about the child, a bare mention stays fail-closed (`unknown_sensitive`), and one's own childhood
or things made for children carry no label; `health` needs a condition, an injury or a
treatment, while the field («mental health jobs») is a weak hit and figurative therapy none.
The topic lexicons still read the user's questions whole."""

from __future__ import annotations

import pytest

from koob.memory import policy
from koob.memory.model import Label


@pytest.fixture(autouse=True)
def _no_classifier():
    prev = policy.classifier()
    policy.set_classifier(None)
    yield
    policy.set_classifier(prev)


@pytest.mark.parametrize(
    "text",
    [
        "Denis Smith's son got hurt at football practice and needed stitches",
        "my daughter's teacher says she keeps getting into fights at recess",
        "our son is 6 years old and starts first grade in the fall",
        "the kids have a fever since Tuesday, staying home from school",
        "my son was in a car accident last night but he is okay",
        "the toddler is allergic to peanuts",
        "his daughter goes to the kindergarten on Maple Street",
        "a stranger tried to talk to my kid at the playground",
        "сын опять подрался в школе, вызывают к директору",
        "дочке 4 года, ходит в садик у дома",
        "у ребенка температура второй день",
        "малыш упал с горки, ушиб коленку",
    ],
)
def test_child_with_a_personal_detail_is_minor(text):
    assert policy.lexicon_label(text) & Label.MINOR, text


@pytest.mark.parametrize(
    "text",
    [
        "Yesterday I took the kids to the museum, they loved it",
        "We went camping with my husband and kids last weekend",
        "My kids are so excited about summer break",
        "гуляли с детьми в парке весь вечер",
    ],
)
def test_bare_mention_of_children_stays_fail_closed(text):
    got = policy.lexicon_label(text)
    assert not got & Label.MINOR and got & Label.UNKNOWN_SENSITIVE, text


@pytest.mark.parametrize(
    "text",
    [
        "I loved riding bikes with my dad when I was a kid",
        "Denis Smith has a shelf of kids' books from many cultures",
        "My favourite memory from childhood is the sea",
        "We are putting together a talent show for the kids at the center",
        "Researching adoption agencies to give a home to kids who need it",
        "когда я был ребенком, мы жили у моря",
        "детские книги стоят на нижней полке",
    ],
)
def test_own_childhood_and_things_for_children_carry_no_minor(text):
    got = policy.lexicon_label(text)
    assert not got & (Label.MINOR | Label.UNKNOWN_SENSITIVE), text


@pytest.mark.parametrize(
    "text",
    [
        "I started seeing a therapist for my anxiety",
        "the doctor put my arm in a cast after the injury",
        "Denis Smith was diagnosed with diabetes last year",
        "physical therapy twice a week after the surgery",
        "у мамы болит спина, записались к врачу",
    ],
)
def test_condition_injury_or_treatment_is_health(text):
    assert policy.lexicon_label(text) & Label.HEALTH, text


@pytest.mark.parametrize(
    "text",
    [
        "I want to work in mental health and help people",
        "ran a charity race for mental health awareness",
        "trying to eat healthier this month",
    ],
)
def test_health_as_a_field_or_habit_is_a_weak_hit(text):
    got = policy.lexicon_label(text)
    assert not got & Label.HEALTH and got & Label.UNKNOWN_SENSITIVE, text


@pytest.mark.parametrize(
    "text",
    [
        "pottery class is like therapy for me",
        "painting with the family was fun and therapeutic",
    ],
)
def test_figurative_therapy_carries_no_health(text):
    assert not policy.lexicon_label(text) & Label.HEALTH, text


def test_questions_still_hit_the_whole_topic_lexicons():
    for label, words in (
        (Label.MINOR, "what did Denis Smith say about his kids"),
        (Label.HEALTH, "how is Denis Smith's mental health"),
    ):
        assert policy.topic_hit(label, policy.words(words)), words
