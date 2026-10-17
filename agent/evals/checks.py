"""Matchers and global checks (CLAUDE.md section 17.3). All text matching is case-insensitive."""
from __future__ import annotations

import re

BANNED_PHRASES = ("look at", "you can see", "you'll see", "over there", "arm's length")
MARKDOWN = re.compile(r"[*_#`~|<>\[\]{}\\]|^\s*(?:[-•]|\d+[.)])\s", re.M)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
MATCHER_KEYS = ("equals", "in", "contains_any", "contains_all", "not_contains_any", "max_words", "regex")


def norm(text: str) -> str:
    """Lowercase, straight apostrophes, punctuation dropped (apostrophes kept), single spaces."""
    text = str(text).lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def _contains(haystack: str, needle: str) -> bool:
    return f" {norm(needle)} " in f" {norm(haystack)} "


def match(value, matcher: dict, label: str) -> list[str]:
    """Return failure messages; empty when every key of the matcher passes."""
    failures = []
    unknown = set(matcher) - set(MATCHER_KEYS)
    if unknown:
        return [f"{label}: unknown matcher keys {sorted(unknown)}"]
    if "equals" in matcher:
        want = matcher["equals"]
        ok = norm(value) == norm(want) if isinstance(want, str) and isinstance(value, str) else value == want
        if not ok:
            failures.append(f"{label}: expected {want!r}, got {value!r}")
    if "in" in matcher:
        options = matcher["in"]
        ok = any(norm(value) == norm(o) if isinstance(o, str) and isinstance(value, str) else value == o
                 for o in options)
        if not ok:
            failures.append(f"{label}: expected one of {options!r}, got {value!r}")
    text = "" if value is None else str(value)
    if "contains_any" in matcher and not any(_contains(text, n) for n in matcher["contains_any"]):
        failures.append(f"{label}: contains none of {matcher['contains_any']!r}: {text!r}")
    if "contains_all" in matcher:
        missing = [n for n in matcher["contains_all"] if not _contains(text, n)]
        if missing:
            failures.append(f"{label}: missing {missing!r}: {text!r}")
    if "not_contains_any" in matcher:
        found = [n for n in matcher["not_contains_any"] if _contains(text, n)]
        if found:
            failures.append(f"{label}: must not contain {found!r}: {text!r}")
    if "max_words" in matcher and len(text.split()) > matcher["max_words"]:
        failures.append(f"{label}: {len(text.split())} words, max {matcher['max_words']}: {text!r}")
    if "regex" in matcher and not re.search(matcher["regex"], text, re.I):
        failures.append(f"{label}: does not match /{matcher['regex']}/: {text!r}")
    return failures


def spoken_violations(text: str) -> list[str]:
    """Global checks on one spoken output: banned phrases and markdown or emoji."""
    lowered = text.lower().replace("’", "'")
    found = [f"banned phrase {p!r}" for p in BANNED_PHRASES if p in lowered]
    if MARKDOWN.search(text):
        found.append("markdown symbol")
    if EMOJI.search(text):
        found.append("emoji")
    return [f"{v}: {text!r}" for v in found]
