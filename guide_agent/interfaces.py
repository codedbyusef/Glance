"""External interfaces (CLAUDE.md section 5). Stubs live in stubs.py; real components implement the same methods."""
from __future__ import annotations

import time
from typing import Callable, Protocol

from .schemas import Frame

PlaybackListener = Callable[[str, str], None]  # (speech_id, "completed" | "interrupted")


class AudioOut(Protocol):
    def say(self, text: str, expect: str) -> str:
        """Queue text for playback (decision 7) and return a speech id.

        expect is none, done or answer. Completion is reported later through the listener, never from inside say().
        """

    def stop_speech(self) -> None:
        """Stop playback now, including anything queued. Each stopped speech reports interrupted."""

    def cue(self, name: str) -> None: ...

    def set_listener(self, listener: PlaybackListener) -> None: ...


class HapticStartError(Exception):
    pass


class HapticLoop(Protocol):
    def start(self, target: str, refer: str) -> str:
        """Start guidance and return the session id. Raises on failure."""

    def pause(self, session_id: str) -> None: ...

    def resume(self, session_id: str) -> None: ...

    def cancel(self, session_id: str) -> None: ...


class Camera(Protocol):
    def latest_frame(self) -> Frame | None: ...


class Clock(Protocol):
    def now(self) -> float: ...


class RealClock:
    def now(self) -> float:
        return time.time()
