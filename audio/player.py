"""TTS player (CLAUDE.md sections 5.1 and 11, decisions 9 and 10).

Threads:
- synth: turns queued text into audio, working ahead of the writer.
- writer: owns the output stream and plays the queue head in 50 ms chunks, checking a stop generation between chunks.

Reports go through an outbox that is drained by dispatch(self._drain), so the listener always runs on the dispatch
thread. stop_speech drains the outbox and then reports interrupted speeches synchronously, on the caller's thread.
"""
from __future__ import annotations

import itertools
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .capture import resample_to_float
from .listen import EXPECTS, ReplyWindow
from .log import EventLog, null_log

CHUNK_S = 0.05
REOPEN_S = 2.0
CACHE_SIZE = 16
FIXED_PHRASES = (
    "Okay, cancelled.",
    "Sorry, I had a problem. Please say that again.",
    "Sorry, I had a problem. Do you want to keep going?",
)

Listener = Callable[[str, str], None]
Dispatch = Callable[..., None]


def direct_dispatch(fn, *args) -> None:
    fn(*args)


def clean_text(text: str) -> str:
    return " ".join(re.sub(r"[*_#`]", " ", text).split())


# ------------------------------------------------------------------ synthesis

class Synth(Protocol):
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Returns (float32 mono audio in [-1, 1], sample rate)."""


class PiperSynth:
    def __init__(self, voice_path, speaking_rate: float = 1.0):
        from piper import PiperVoice, SynthesisConfig

        self.voice = PiperVoice.load(str(voice_path))
        # speaking_rate 1.2 means 20% faster; Piper's length_scale below 1 is faster.
        self.syn_config = SynthesisConfig(length_scale=1.0 / max(0.25, speaking_rate))

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks, rate = [], 22050
        for chunk in self.voice.synthesize(text, syn_config=self.syn_config):
            rate = chunk.sample_rate
            chunks.append(np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1))
        return (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)), rate


class FakeSynth:
    """10 ms of quiet audio per character. Counts calls; set fail_on to a text to make it raise."""

    def __init__(self, rate: int = 16000, per_char_s: float = 0.01):
        self.rate = rate
        self.per_char_s = per_char_s
        self.calls: list[str] = []
        self.fail_on: set[str] = set()
        self.delay_s = 0.0

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.calls.append(text)
        if self.delay_s:
            time.sleep(self.delay_s)
        if text in self.fail_on:
            raise RuntimeError("fake synth failure")
        return np.full(int(len(text) * self.per_char_s * self.rate), 0.01, dtype=np.float32), self.rate


# ------------------------------------------------------------------ output

class OutputBackend(Protocol):
    rate: int

    def open(self) -> None: ...  # raises on failure

    def write(self, chunk: np.ndarray) -> None: ...  # float32 mono, blocks until queued; raises on device failure

    def abort(self) -> None: ...  # drop buffered audio now, stay open

    def latency(self) -> float: ...  # seconds from write returning to the sound reaching the ear

    def close(self) -> None: ...


class SoundDeviceOutput:
    """One sounddevice OutputStream at the device's default rate."""

    def __init__(self, device=None):
        self.device = device
        self.rate = 0
        self.stream = None

    def open(self) -> None:
        import sounddevice as sd

        self.close()
        self.rate = int(sd.query_devices(self.device, "output")["default_samplerate"])
        self.stream = sd.OutputStream(device=self.device, samplerate=self.rate, channels=1, dtype="float32",
                                      latency="low")
        self.stream.start()

    def write(self, chunk: np.ndarray) -> None:
        self.stream.write(np.ascontiguousarray(chunk, dtype=np.float32).reshape(-1, 1))

    def abort(self) -> None:
        self.stream.abort()
        self.stream.start()

    def latency(self) -> float:
        return float(self.stream.latency)

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.close(ignore_errors=True)
            finally:
                self.stream = None


class FakeOutput:
    """Records written samples. Writes take real time (so stop tests work) unless realtime is False.

    gate: when cleared, write blocks until it is set again. fail_writes/fail_opens make the next N calls raise.
    """

    def __init__(self, rate: int = 16000, latency_s: float = 0.0, realtime: bool = True):
        self.rate = rate
        self.latency_s = latency_s
        self.realtime = realtime
        self.gate = threading.Event()
        self.gate.set()
        self.written: list[np.ndarray] = []
        self.aborts = 0
        self.opens = 0
        self.fail_writes = 0
        self.fail_opens = 0
        self.wrote = threading.Event()

    def open(self) -> None:
        if self.fail_opens:
            self.fail_opens -= 1
            raise OSError("fake device missing")
        self.opens += 1

    def write(self, chunk: np.ndarray) -> None:
        self.gate.wait()
        if self.fail_writes:
            self.fail_writes -= 1
            raise OSError("fake device unplugged")
        self.written.append(chunk.copy())
        self.wrote.set()
        if self.realtime:
            time.sleep(len(chunk) / self.rate)

    def abort(self) -> None:
        self.aborts += 1

    def latency(self) -> float:
        return self.latency_s

    def close(self) -> None:
        pass

    @property
    def samples_written(self) -> int:
        return sum(len(c) for c in self.written)


def tone(rate: int, freq: float = 660.0, dur_s: float = 0.15, amp: float = 0.12) -> np.ndarray:
    """A short soft tone with raised-cosine edges, for the thinking cue."""
    t = np.arange(int(rate * dur_s)) / rate
    env = np.minimum(1.0, np.minimum(t, dur_s - t) / 0.03)
    env = 0.5 - 0.5 * np.cos(np.pi * np.clip(env, 0, 1))
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ------------------------------------------------------------------ player

@dataclass(eq=False)
class _Speech:
    id: str
    text: str
    expect: str
    queued_at: float
    audio: np.ndarray | None = None  # float32
    rate: int = 0
    failed: bool = False
    synthesizing: bool = False


class Player:
    """Implements the agent's AudioOut (section 5.1)."""

    def __init__(self, synth: Synth, output: OutputBackend, dispatch: Dispatch = direct_dispatch,
                 log: EventLog | None = None, window: ReplyWindow | None = None,
                 fixed_phrases: tuple[str, ...] = FIXED_PHRASES):
        self.synth = synth
        self.output = output
        self.dispatch = dispatch
        self.log = log or null_log()
        self.window = window or ReplyWindow()
        self.fixed_phrases = fixed_phrases
        self._listener: Listener | None = None
        self._cond = threading.Condition()
        self._queue: deque[_Speech] = deque()  # head is the speech playing or next to play
        self._outbox: deque[tuple[str, str]] = deque()
        self._gen = 0  # bumped by stop_speech
        self._stop_t = 0.0  # monotonic time of the last stop_speech, for the playback_cut latency
        self._cue: np.ndarray | None = None
        self._writing = False
        self._device_ok = False
        self._closed = False
        self._ids = itertools.count(1)
        self._fixed: dict[str, tuple[np.ndarray, int]] = {}
        self._lru: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._threads: list[threading.Thread] = []
        self.ready = threading.Event()  # set once the output device has been tried
        self.warm = threading.Event()  # set once the fixed phrases are cached
        self._next_open = 0.0

    # ---------------------------------------------------------------- AudioOut

    def set_listener(self, listener: Listener) -> None:
        self._listener = listener

    def say(self, text: str, expect: str) -> str:
        if expect not in EXPECTS:
            self.log.write("error", where="player.say", error=f"invalid expect {expect!r}, using none")
            expect = "none"
        sp = _Speech(id=f"sp-{next(self._ids)}", text=clean_text(text), expect=expect, queued_at=time.monotonic())
        self.window.on_say(expect)
        self.log.write("say_queued", speech_id=sp.id, chars=len(sp.text), expect=expect)
        with self._cond:
            self._queue.append(sp)
            self._cond.notify_all()
        return sp.id

    def stop_speech(self) -> None:
        with self._cond:
            self._gen += 1
            self._stop_t = time.monotonic()
            stopped = list(self._queue)
            self._queue.clear()
            done = list(self._outbox)
            self._outbox.clear()
            self._cond.notify_all()
        self.log.write("stop_speech", stopped=[s.id for s in stopped])
        for speech_id, status in done:
            self._report(speech_id, status)
        for s in stopped:
            self.log.write("playback_done", speech_id=s.id, status="interrupted")
            self._report(s.id, "interrupted")

    def cue(self, name: str) -> None:
        if name != "thinking":
            self.log.write("error", where="player.cue", error=f"unknown cue {name!r}")
            return
        with self._cond:
            if self._writing or self._queue or not self._device_ok:
                self.log.write("cue", name=name, played=False)
                return
            self._cue = tone(self.output.rate)
            self._cond.notify_all()
        self.log.write("cue", name=name, played=True)

    @property
    def busy(self) -> bool:
        """True while anything is playing or queued."""
        with self._cond:
            return bool(self._queue)

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> "Player":
        for name, target in (("tts-synth", self._synth_loop), ("tts-writer", self._writer_loop)):
            t = threading.Thread(target=self._run_forever, args=(name, target), name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def close(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout)
        try:
            self.output.close()
        except Exception as e:
            self.log.error("player.close", e)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued (CLI and tests)."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._queue, timeout)

    # ---------------------------------------------------------------- reports

    def _report(self, speech_id: str, status: str) -> None:
        listener = self._listener
        if listener is None:
            return
        try:
            listener(speech_id, status)
        except Exception as e:
            self.log.error("player.listener", e, speech_id=speech_id)

    def _drain(self) -> None:
        with self._cond:
            done = list(self._outbox)
            self._outbox.clear()
        for speech_id, status in done:
            self._report(speech_id, status)

    def _finish(self, sp: _Speech, status: str) -> bool:
        """Under the lock: report sp if it is still the queue head. Returns whether a drain is needed."""
        if self._queue and self._queue[0] is sp:
            self._queue.popleft()
            self._outbox.append((sp.id, status))
            self._cond.notify_all()
            return True
        return False

    def _fail_all(self, reason: str) -> None:
        with self._cond:
            stopped = list(self._queue)
            self._queue.clear()
            self._outbox.extend((s.id, "interrupted") for s in stopped)
            self._cond.notify_all()
        for s in stopped:
            self.log.write("playback_done", speech_id=s.id, status="interrupted", reason=reason)
        if stopped:
            self.dispatch(self._drain)

    # ---------------------------------------------------------------- threads

    def _run_forever(self, name: str, loop: Callable[[], None]) -> None:
        while not self._closed:
            try:
                loop()
                return
            except Exception as e:
                self.log.error(name, e)
                time.sleep(0.2)

    def _lookup(self, text: str) -> tuple[np.ndarray, int] | None:
        if text in self._fixed:
            return self._fixed[text]
        if text in self._lru:
            self._lru.move_to_end(text)
            return self._lru[text]
        return None

    def _synth_loop(self) -> None:
        import scipy.signal  # noqa: F401  first import takes seconds; pay it here, not at first playback

        for phrase in self.fixed_phrases:
            if phrase not in self._fixed:
                key = clean_text(phrase)
                try:
                    self._fixed[key] = self.synth.synthesize(key)
                except Exception as e:
                    self.log.error("player.synth_fixed", e)
        self.warm.set()
        while True:
            with self._cond:
                sp = None
                while not self._closed:
                    sp = next((s for s in self._queue if s.audio is None and not s.synthesizing), None)
                    if sp is not None:
                        break
                    self._cond.wait()
                if self._closed:
                    return
                sp.synthesizing = True
            t0 = time.monotonic()
            audio, rate, failed, cached = np.zeros(0, dtype=np.float32), self.output.rate or 16000, False, False
            if sp.text:
                hit = self._lookup(sp.text)
                cached = hit is not None
                try:
                    audio, rate = hit if hit is not None else self.synth.synthesize(sp.text)
                    if hit is None:
                        self._lru[sp.text] = (audio, rate)
                        while len(self._lru) > CACHE_SIZE:
                            self._lru.popitem(last=False)
                except Exception as e:
                    failed = True
                    self.log.error("player.synth", e, speech_id=sp.id)
            self.log.write("synth", speech_id=sp.id, ms=int((time.monotonic() - t0) * 1000), cached=cached,
                           failed=failed)
            out_rate = self.output.rate
            if out_rate and rate != out_rate and len(audio):
                audio, rate = resample_to_float(audio, rate, out_rate), out_rate  # ahead of playback
            with self._cond:
                sp.audio, sp.rate, sp.failed, sp.synthesizing = audio, rate, failed, False
                self._cond.notify_all()

    def _open_output(self) -> bool:
        try:
            self.output.open()
        except Exception as e:
            self.log.error("player.open_output", e)
            return False
        with self._cond:
            self._device_ok = True
        self.log.write("output_open", rate=self.output.rate)
        return True

    def _writer_loop(self) -> None:
        while not self._closed:
            if not self._device_ok:
                if time.monotonic() >= self._next_open:
                    self._next_open = time.monotonic() + REOPEN_S
                    self._open_output()
                if not self._device_ok:
                    # Device down: fail whatever is queued, then wait for a new say() or the next reopen attempt.
                    self._fail_all("output_device")
                    self.ready.set()
                    with self._cond:
                        if not self._queue and not self._closed:
                            self._cond.wait(max(0.0, self._next_open - time.monotonic()))
                    continue
            self.ready.set()
            with self._cond:
                while not self._closed and self._cue is None and not (self._queue and self._queue[0].audio is not None):
                    self._cond.wait(0.5)
                if self._closed:
                    return
                cue, self._cue = self._cue, None
                sp = None if cue is not None else self._queue[0]
                gen = self._gen
                self._writing = True
            try:
                if cue is not None:
                    self._play(cue, self.output.rate, gen)
                else:
                    self._play_speech(sp, gen)
            except Exception as e:
                self.log.error("player.output", e)
                with self._cond:
                    self._device_ok = False
                    self._writing = False
                self._next_open = time.monotonic() + REOPEN_S
                try:
                    self.output.close()
                except Exception:
                    pass
                self._fail_all("output_device")
            finally:
                with self._cond:
                    self._writing = False

    def _play(self, audio: np.ndarray, rate: int, gen: int) -> bool:
        """Write audio and wait out the output latency. Returns False if stop_speech cut it off."""
        out_rate = self.output.rate
        if rate != out_rate:
            audio = resample_to_float(audio, rate, out_rate)
        step = max(1, int(out_rate * CHUNK_S))
        for i in range(0, len(audio), step):
            if self._gen != gen or self._closed:
                return self._cut(gen)
            self.output.write(audio[i:i + step])
        deadline = time.monotonic() + (self.output.latency() if len(audio) else 0.0)
        while time.monotonic() < deadline:
            if self._gen != gen or self._closed:
                return self._cut(gen)
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return True

    def _cut(self, gen: int) -> bool:
        """Abort the stream (drops buffered audio) and log how long after stop_speech the sound stopped."""
        self.output.abort()
        if self._gen != gen:  # cut by stop_speech, not by close
            self.log.write("playback_cut", ms=int((time.monotonic() - self._stop_t) * 1000))
        return False

    def _play_speech(self, sp: _Speech, gen: int) -> None:
        if sp.failed:
            status, played = "interrupted", False
        else:
            self.log.write("playback_start", speech_id=sp.id, wait_ms=int((time.monotonic() - sp.queued_at) * 1000))
            played = self._play(sp.audio, sp.rate, gen)
            status = "completed"
        with self._cond:
            # stop_speech bumps gen and reports the speech itself; only report if we still own it.
            report = played or sp.failed
            needs_drain = report and self._gen == gen and self._finish(sp, status)
        if needs_drain:
            self.log.write("playback_done", speech_id=sp.id, status=status)
            self.dispatch(self._drain)


def build_player(cfg, log: EventLog | None = None, dispatch: Dispatch = direct_dispatch,
                 window: ReplyWindow | None = None) -> Player:
    """The real Player from config. Raises if the Piper voice can't be loaded."""
    synth = PiperSynth(cfg.path(cfg.piper_voice), cfg.speaking_rate)
    return Player(synth, SoundDeviceOutput(cfg.output_device), dispatch=dispatch, log=log,
                  window=window or ReplyWindow(cfg.assistant_name))
