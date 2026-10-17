"""The agent-facing input contract (CLAUDE.md section 5.2) and stand-in sinks for isolated runs."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol


class AudioEvents(Protocol):
    def on_speech_start(self, at: float | None = None) -> None: ...

    def on_utterance(self, text: str, ended_at: float | None = None, frame: object | None = None) -> None: ...


class PrintSink:
    def __init__(self):
        self._started: float | None = None

    def on_speech_start(self, at: float | None = None) -> None:
        self._started = time.time()
        print(f"[speech_start] at={at:.3f}" if at is not None else "[speech_start]", flush=True)

    def on_utterance(self, text: str, ended_at: float | None = None, frame: object | None = None) -> None:
        lag = f" delivered {time.time() - ended_at:.2f}s after end of speech" if ended_at else ""
        print(f"[utterance] {text!r}{lag}", flush=True)


@dataclass
class SinkCall:
    name: str  # speech_start | utterance
    args: tuple
    thread: str


class RecordingSink:
    """Records every call, in order, with the thread it arrived on."""

    def __init__(self):
        self.calls: list[SinkCall] = []
        self._cond = threading.Condition()

    def on_speech_start(self, at: float | None = None) -> None:
        self._add("speech_start", (at,))

    def on_utterance(self, text: str, ended_at: float | None = None, frame: object | None = None) -> None:
        self._add("utterance", (text, ended_at, frame))

    def _add(self, name: str, args: tuple) -> None:
        with self._cond:
            self.calls.append(SinkCall(name, args, threading.current_thread().name))
            self._cond.notify_all()

    def names(self) -> list[str]:
        return [c.name for c in self.calls]

    def utterances(self) -> list[str]:
        return [c.args[0] for c in self.calls if c.name == "utterance"]

    def wait_for(self, n_calls: int, timeout: float = 2.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: len(self.calls) >= n_calls, timeout)
