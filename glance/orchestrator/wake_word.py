"""openWakeWord + VAD, with push-to-talk as a fallback when venue noise breaks detection."""

from __future__ import annotations


class WakeWordListener:
    def __init__(self, phrase: str = "hey jarvis"):
        self.phrase = phrase

    def listen(self):
        raise NotImplementedError
