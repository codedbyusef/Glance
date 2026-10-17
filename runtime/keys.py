"""Keyboard input for the live scripts. One reader of stdin for the whole process, plus the push-to-talk toggle.

Terminal lines are routed like this (route() below):
    empty line (Enter)          push-to-talk toggle (audio decision 4: press to start, press again to stop)
    exactly one known key       that key, e.g. "s" + Enter = the same as pressing s in the window
    anything else               a typed utterance in --typed mode, otherwise ignored
In --view window the same keys also work in the window, and Enter there toggles push-to-talk too.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable


def route(line: str, keys: set[str], typed: bool) -> tuple[str, str] | None:
    """One terminal line -> ("enter", "") | ("key", k) | ("line", text) | None (ignored)."""
    text = line.strip()
    if not text:
        return ("enter", "")
    if len(text) == 1 and text.lower() in keys:
        return ("key", text.lower())
    if typed:
        return ("line", text)
    return None


class TogglePtt:
    """A push-to-talk source for audio's Mic (start(put) / close()). toggle() flips it from any thread."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.down = False
        self._put: Callable[[tuple], None] | None = None
        self._lock = threading.Lock()

    def start(self, put: Callable[[tuple], None]) -> None:
        self._put = put

    def close(self) -> None:
        self._put = None

    def toggle(self) -> bool:
        """Flip the key. Returns the new state (True = talking)."""
        with self._lock:
            self.down = not self.down
            down, put = self.down, self._put
        if put is not None:
            put(("ptt", down, self.clock()))
        return down


class StdinRouter:
    """Reads stdin on a daemon thread and routes each line (see route())."""

    def __init__(self, keys: set[str], on_key: Callable[[str], None], on_enter: Callable[[], None] | None = None,
                 on_line: Callable[[str], None] | None = None, stream=None):
        self.keys = keys
        self.on_key = on_key
        self.on_enter = on_enter
        self.on_line = on_line
        self.stream = stream or sys.stdin
        self._thread: threading.Thread | None = None

    def start(self) -> "StdinRouter":
        self._thread = threading.Thread(target=self._run, name="stdin", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while True:
            try:
                line = self.stream.readline()
            except (OSError, ValueError):
                return
            if not line:  # stdin closed
                return
            self.handle(line)

    def handle(self, line: str) -> None:
        r = route(line, self.keys, typed=self.on_line is not None)
        if r is None:
            return
        kind, value = r
        if kind == "enter":
            if self.on_enter is not None:
                self.on_enter()
        elif kind == "key":
            self.on_key(value)
        else:
            self.on_line(value)
