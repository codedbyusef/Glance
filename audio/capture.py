"""Microphone capture (CLAUDE.md sections 7 and 12): 16 kHz mono int16 in 80 ms blocks, with device recovery."""
from __future__ import annotations

import threading
import time
from math import gcd
from typing import Callable, Protocol

import numpy as np

from .log import EventLog, null_log

SAMPLE_RATE = 16000
BLOCK_S = 0.08
BLOCK_SAMPLES = 1280
REOPEN_S = 2.0
STALL_S = 1.5  # no audio for this long means the device is gone

# Items put on the Mic's queue: ("audio", block, t) with t the wall time of the block's first sample,
# ("device_lost", None, t) when the stream dies.
Put = Callable[[tuple], None]


def resample_to(samples: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    """int16 mono at src_rate to int16 mono at dst_rate (scipy resample_poly)."""
    if src_rate == dst_rate:
        return samples.astype(np.int16, copy=False)
    from scipy.signal import resample_poly

    g = gcd(src_rate, dst_rate)
    out = resample_poly(samples.astype(np.float32), dst_rate // g, src_rate // g)
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


def resample_to_float(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """float32 mono resampling, for playback."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def device_rate(device, kind: str) -> int:
    """The device's default sample rate. kind is input or output."""
    import sounddevice as sd

    return int(sd.query_devices(device, kind)["default_samplerate"])


class CaptureSource(Protocol):
    def start(self, put: Put) -> None: ...

    def close(self) -> None: ...


class SoundDeviceCapture:
    """Opens the mic at 16 kHz if it can, else at its native rate and resamples each block.

    A supervisor thread reopens the stream every 2 s after a failure or a stall, and reports device_lost.
    """

    def __init__(self, device=None, log: EventLog | None = None):
        self.device = device
        self.log = log or null_log()
        self._closed = threading.Event()
        self._last_audio = 0.0
        self._thread: threading.Thread | None = None

    def start(self, put: Put) -> None:
        self._thread = threading.Thread(target=self._supervise, args=(put,), name="mic-capture", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(3)

    def _open(self, put: Put):
        import sounddevice as sd

        try:
            sd.check_input_settings(device=self.device, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
            rate = SAMPLE_RATE
        except Exception:
            rate = device_rate(self.device, "input")
        blocksize = int(round(rate * BLOCK_S))
        if rate != SAMPLE_RATE:
            import scipy.signal  # noqa: F401  import before the callback needs it

        def callback(indata, frames, t, status):
            now = time.time()
            self._last_audio = time.monotonic()
            block = indata[:, 0].copy()
            if rate != SAMPLE_RATE:
                block = resample_to(block, rate)
                if len(block) != BLOCK_SAMPLES:
                    block = np.resize(block, BLOCK_SAMPLES)
            put(("audio", block, now - BLOCK_S))

        stream = sd.InputStream(device=self.device, samplerate=rate, channels=1, dtype="int16",
                                blocksize=blocksize, callback=callback)
        stream.start()
        self.log.write("mic_open", rate=rate, resampling=rate != SAMPLE_RATE)
        return stream

    def _supervise(self, put: Put) -> None:
        while not self._closed.is_set():
            try:
                stream = self._open(put)
            except Exception as e:
                self.log.error("capture.open", e)
                self._closed.wait(REOPEN_S)
                continue
            self._last_audio = time.monotonic()
            try:
                while not self._closed.wait(0.25):
                    if time.monotonic() - self._last_audio > STALL_S or not stream.active:
                        self.log.write("error", where="capture", error="mic stalled or lost; reopening")
                        put(("device_lost", None, time.time()))
                        break
            finally:
                try:
                    stream.abort(ignore_errors=True)
                    stream.close(ignore_errors=True)
                except Exception:
                    pass
            self._closed.wait(REOPEN_S)


def wav_items(path, t0: float | None = None) -> list[tuple]:
    """A 16 kHz mono WAV file as ("audio", block, t) items, for replaying recordings through the Mic."""
    from .stt_client import read_wav

    with open(path, "rb") as f:
        samples, rate, channels, width = read_wav(f.read())
    if channels != 1 or width != 2:
        raise ValueError("expected mono PCM16")
    samples = resample_to(samples, rate)
    pad = (-len(samples)) % BLOCK_SAMPLES
    samples = np.concatenate([samples, np.zeros(pad + BLOCK_SAMPLES * 30, np.int16)])  # 2.4 s of trailing silence
    t0 = time.time() if t0 is None else t0
    return [("audio", samples[i:i + BLOCK_SAMPLES], t0 + i / SAMPLE_RATE)
            for i in range(0, len(samples), BLOCK_SAMPLES)]


class FakeCapture:
    """Puts scripted items on start, from a thread, optionally paced in real time."""

    def __init__(self, items: list[tuple], realtime: bool = False):
        self.items = items
        self.realtime = realtime
        self.done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, put: Put) -> None:
        def run():
            for item in self.items:
                if self.realtime and item[0] == "audio":  # stamp with the real time, as the mic would
                    time.sleep(BLOCK_S)
                    item = (item[0], item[1], time.time() - BLOCK_S)
                put(item)
            self.done.set()

        self._thread = threading.Thread(target=run, name="fake-capture", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(2)
