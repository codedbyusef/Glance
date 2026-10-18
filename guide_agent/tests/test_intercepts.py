"""Section 7: normalization and phrase matching."""
import pytest

from guide_agent.intercepts import CANCEL, HALT, REPEAT, PHRASES, classify, normalize

WAKE = "Nova"


@pytest.mark.parametrize("cls", [HALT, CANCEL, REPEAT])
def test_every_phrase_matches_its_class(cls):
    for phrase in PHRASES[cls]:
        assert classify(phrase, WAKE) == cls, phrase


def test_normalization_drops_case_punctuation_wake_word_and_please():
    assert normalize("Nova, STOP please!", WAKE) == ["stop"]
    assert classify("Nova, STOP please!", WAKE) == HALT
    assert classify("Never-mind.", WAKE) == CANCEL
    assert classify("Pardon?", WAKE) == REPEAT
    assert classify("Please, say that again.", WAKE) == REPEAT


def test_repeated_phrases_match():
    assert classify("stop stop", WAKE) == HALT
    assert classify("hold on, hold on", WAKE) == HALT
    assert classify("sorry, what?", WAKE) == REPEAT


@pytest.mark.parametrize("text", [
    "What does that stop sign say",
    "stop, I got it",
    "forget it, what does this sign say",
    "forget the can, find the door",
    "stop cancel",  # mixed classes go to the model
    "wait for me",
])
def test_non_matches(text):
    assert classify(text, WAKE) is None


def test_empty_after_normalization_is_not_an_intercept():
    assert normalize("Nova, please.", WAKE) == []
    assert classify("Nova, please.", WAKE) is None
