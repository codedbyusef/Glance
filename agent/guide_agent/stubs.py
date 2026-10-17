"""Stubs for every external interface, plus a fake clock. Used by tests, the harness and the eval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .interfaces import HapticStartError, PlaybackListener
from .schemas import Frame


@dataclass
class OutputCall:
    name: str  # say, stop_speech, cue, haptic.start, haptic.pause, haptic.resume, haptic.cancel, model
    args: dict
    at: float


class Recorder:
    """Ordered record of output calls across all stubs, so tests can check ordering."""

    def __init__(self, clock, echo: Callable[[OutputCall], None] | None = None):
        self.clock = clock
        self.calls: list[OutputCall] = []
        self.echo = echo

    def add(self, call_name: str, /, **args) -> None:
        call = OutputCall(call_name, args, self.clock.now())
        self.calls.append(call)
        if self.echo:
            self.echo(call)

    def names(self, start: int = 0) -> list[str]:
        return [c.name for c in self.calls[start:]]

    def of(self, name: str, start: int = 0) -> list[OutputCall]:
        return [c for c in self.calls[start:] if c.name == name]


class StubAudio:
    """Playback never finishes on its own: call finish() to complete everything queued, in order."""

    def __init__(self, recorder: Recorder):
        self.rec = recorder
        self.listener: PlaybackListener | None = None
        self.playing: list[tuple[str, str, str]] = []  # (speech_id, text, expect), oldest first
        self._n = 0

    def set_listener(self, listener: PlaybackListener) -> None:
        self.listener = listener

    def say(self, text: str, expect: str) -> str:
        self._n += 1
        speech_id = f"sp-{self._n}"
        self.rec.add("say", text=text, expect=expect)
        self.playing.append((speech_id, text, expect))
        return speech_id

    def stop_speech(self) -> None:
        self.rec.add("stop_speech")
        stopped, self.playing = self.playing, []
        for speech_id, _, _ in stopped:
            self.listener(speech_id, "interrupted")

    def cue(self, name: str) -> None:
        self.rec.add("cue", name=name)

    def finish(self) -> None:
        """Complete all queued playback, including speech queued by completion handlers."""
        while self.playing:
            speech_id, _, _ = self.playing.pop(0)
            self.listener(speech_id, "completed")

    @property
    def last_expect(self) -> str | None:
        says = self.rec.of("say")
        return says[-1].args["expect"] if says else None


class StubHaptic:
    def __init__(self, recorder: Recorder):
        self.rec = recorder
        self.sessions: dict[str, str] = {}  # session_id -> running | paused | cancelled
        self.fail_next_start = False
        self._n = 0

    def start(self, target: str, refer: str) -> str:
        self.rec.add("haptic.start", target=target, refer=refer)
        if self.fail_next_start:
            self.fail_next_start = False
            raise HapticStartError("stub: guidance service unavailable")
        self._n += 1
        session_id = f"s-{self._n}"
        self.sessions[session_id] = "running"
        return session_id

    def pause(self, session_id: str) -> None:
        self.rec.add("haptic.pause", session_id=session_id)
        self.sessions[session_id] = "paused"

    def resume(self, session_id: str) -> None:
        self.rec.add("haptic.resume", session_id=session_id)
        self.sessions[session_id] = "running"

    def cancel(self, session_id: str) -> None:
        self.rec.add("haptic.cancel", session_id=session_id)
        self.sessions[session_id] = "cancelled"

    @property
    def current(self) -> str | None:
        """The newest session id, if any."""
        return f"s-{self._n}" if self._n else None


class RecordingModel:
    """Wraps a model adapter and records each call as a "model" output, so ordering checks can see it."""

    def __init__(self, inner, recorder: Recorder):
        self.inner = inner
        self.rec = recorder

    async def generate(self, system, messages, tools, temperature):
        self.rec.add("model")
        return await self.inner.generate(system, messages, tools, temperature)


@dataclass
class StubCamera:
    frame: Frame | None = None

    def latest_frame(self) -> Frame | None:
        return self.frame


@dataclass
class FakeClock:
    t: float = field(default=1_800_000_000.0)

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds
