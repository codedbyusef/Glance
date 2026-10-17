"""Utterance intercepts (CLAUDE.md section 7), checked before any model call."""
from __future__ import annotations

import re

HALT = "halt"
CANCEL = "cancel"
REPEAT = "repeat"

PHRASES = {
    HALT: ("stop", "wait", "hold on", "pause"),
    CANCEL: ("cancel", "never mind", "forget it"),
    REPEAT: ("repeat", "repeat that", "say that again", "what", "sorry", "pardon"),
}


def normalize(text: str, wake_word: str) -> list[str]:
    """Lowercase, drop punctuation, the wake word and "please". Returns the remaining words."""
    text = text.lower().replace("'", "").replace("’", "")
    words = re.sub(r"[^\w\s]", " ", text).split()
    wake = re.sub(r"[^\w\s]", " ", wake_word.lower()).split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if wake and words[i:i + len(wake)] == wake:
            i += len(wake)
            continue
        if words[i] != "please":
            out.append(words[i])
        i += 1
    return out


def _covered(words: list[str], phrases: tuple[str, ...]) -> bool:
    """True if the words are one or more phrases of the same class back to back."""
    if not words:
        return True
    for phrase in phrases:
        p = phrase.split()
        if words[:len(p)] == p and _covered(words[len(p):], phrases):
            return True
    return False


def classify(text: str, wake_word: str) -> str | None:
    """Return HALT, CANCEL or REPEAT when the whole utterance is that intercept, else None."""
    words = normalize(text, wake_word)
    if not words:
        return None
    for name, phrases in PHRASES.items():
        if _covered(words, phrases):
            return name
    return None
