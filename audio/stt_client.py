"""Speech-to-text client (CLAUDE.md section 10): Vultr faster-whisper first, local tiny.en on failure, then filters."""
from __future__ import annotations

import io
import re
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx
import numpy as np

from .log import EventLog, null_log

SAMPLE_RATE = 16000
# The server's default initial prompt (section 9). Whisper sometimes "transcribes" noise as the prompt itself.
DEFAULT_PROMPT = "done, yes, no, stop, wait, cancel, repeat"
PROMPT_WORDS = frozenset(DEFAULT_PROMPT.replace(",", " ").split())


@dataclass
class SttResult:
    text: str
    backend: str  # vultr | local | none
    ms: int
    dropped_reason: str | None = None  # no_speech | empty | non_speech_tag | blocklist | prompt_echo | stt_failed


@dataclass
class RawTranscript:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


class SttError(Exception):
    pass


class Backend(Protocol):
    name: str

    def transcribe(self, wav_bytes: bytes) -> RawTranscript: ...


class Transcriber(Protocol):
    def transcribe(self, wav_bytes: bytes) -> SttResult: ...


# ------------------------------------------------------------------ WAV and Whisper helpers (shared with the server)

def to_wav(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    """Mono int16 samples to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


def read_wav(wav_bytes: bytes) -> tuple[np.ndarray, int, int, int]:
    """Returns (int16 samples, rate, channels, sample width). Raises wave.Error or EOFError on a bad file."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if w.getcomptype() != "NONE":
            raise wave.Error("compressed WAV")
        frames = w.readframes(w.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16) if width == 2 else np.frombuffer(frames, dtype=np.uint8)
    return samples, rate, channels, width


def to_float(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / 32768.0


def summarize_segments(segments) -> RawTranscript:
    """no_speech_prob is the max over segments; avg_logprob is the duration-weighted mean (section 9)."""
    segments = list(segments)
    if not segments:
        return RawTranscript(text="", no_speech_prob=1.0, avg_logprob=0.0)
    text = "".join(s.text for s in segments).strip()
    no_speech = max(float(s.no_speech_prob) for s in segments)
    durations = [max(0.0, float(s.end) - float(s.start)) for s in segments]
    total = sum(durations)
    if total > 0:
        avg = sum(float(s.avg_logprob) * d for s, d in zip(segments, durations)) / total
    else:
        avg = sum(float(s.avg_logprob) for s in segments) / len(segments)
    return RawTranscript(text=text, no_speech_prob=no_speech, avg_logprob=avg)


# ------------------------------------------------------------------ filters

_TAGS_ONLY = re.compile(r"^(\s*(\[[^\]]*\]|\([^)]*\))\s*)+$")


def normalize_text(text: str) -> str:
    """Trim and collapse whitespace. Case and punctuation are left alone: the agent normalizes."""
    return " ".join(text.split())


def _phrase_key(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def is_prompt_echo(text: str) -> bool:
    """More than 2 words that all come from the prompt, or any prompt word 3+ times (decision 23)."""
    words = _phrase_key(text).split()
    counts = {w: words.count(w) for w in set(words) & PROMPT_WORDS}
    return (len(words) > 2 and all(w in PROMPT_WORDS for w in words)) or any(n >= 3 for n in counts.values())


def drop_reason(raw: RawTranscript, text: str, blocklist: list[str]) -> str | None:
    if raw.no_speech_prob > 0.6 and raw.avg_logprob < -1.0:
        return "no_speech"
    if not text:
        return "empty"
    if _TAGS_ONLY.match(text):
        return "non_speech_tag"
    # The whole utterance must be a blocklisted phrase; case and surrounding punctuation are ignored ("You.").
    if _phrase_key(text) in {_phrase_key(b) for b in blocklist}:
        return "blocklist"
    if is_prompt_echo(text):
        return "prompt_echo"
    return None


# ------------------------------------------------------------------ backends

class VultrWhisper:
    name = "vultr"

    def __init__(self, url: str, token: str | None, timeout_s: float, transport: httpx.BaseTransport | None = None):
        base = url.rstrip("/")
        if base.endswith("/transcribe"):
            base = base[: -len("/transcribe")]
        self.base = base
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # One persistent client, so the TLS handshake isn't paid on every utterance.
        self.client = httpx.Client(timeout=httpx.Timeout(timeout_s), headers=headers, transport=transport)

    def transcribe(self, wav_bytes: bytes) -> RawTranscript:
        try:
            r = self.client.post(f"{self.base}/transcribe", content=wav_bytes,
                                 headers={"Content-Type": "audio/wav"})
        except httpx.TimeoutException as e:
            raise SttError("timeout") from e
        except httpx.HTTPError as e:
            raise SttError(f"connection: {type(e).__name__}") from e
        if r.status_code != 200:
            raise SttError(f"http {r.status_code}")
        try:
            body = r.json()
            return RawTranscript(text=str(body["text"]), no_speech_prob=float(body["no_speech_prob"]),
                                 avg_logprob=float(body["avg_logprob"]))
        except (ValueError, KeyError, TypeError) as e:
            raise SttError("bad response") from e

    def health(self) -> dict:
        """GET /health. Raises SttError when unreachable or not ready."""
        try:
            r = self.client.get(f"{self.base}/health")
        except httpx.HTTPError as e:
            raise SttError(f"connection: {type(e).__name__}") from e
        if r.status_code != 200:
            raise SttError(f"http {r.status_code}")
        return r.json()


class LocalWhisper:
    """faster-whisper on the Pi CPU, int8. The model loads on first use unless preload() is called."""

    name = "local"

    def __init__(self, model_name: str = "tiny.en", model_factory: Callable[[], object] | None = None):
        self.model_name = model_name
        self._factory = model_factory or self._load
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        from faster_whisper import WhisperModel

        return WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def preload(self) -> None:
        with self._lock:
            if self._model is None:
                self._model = self._factory()

    def transcribe(self, wav_bytes: bytes) -> RawTranscript:
        try:
            samples, rate, channels, width = read_wav(wav_bytes)
            if rate != SAMPLE_RATE or channels != 1 or width != 2:
                raise SttError("local: expected 16 kHz mono PCM16")
            with self._lock:
                if self._model is None:
                    self._model = self._factory()
                segments, _info = self._model.transcribe(
                    to_float(samples), language="en", beam_size=1, vad_filter=True,
                    condition_on_previous_text=False)
                return summarize_segments(segments)
        except SttError:
            raise
        except Exception as e:  # model load or inference failure
            raise SttError(f"local: {type(e).__name__}: {e}") from e


# ------------------------------------------------------------------ the Transcriber used by the Mic

class FallbackTranscriber:
    def __init__(self, primary: Backend | None, fallback: Backend | None, blocklist: list[str],
                 log: EventLog | None = None):
        self.primary = primary
        self.fallback = fallback
        self.blocklist = blocklist
        self.log = log or null_log()

    def transcribe(self, wav_bytes: bytes) -> SttResult:
        t0 = time.monotonic()
        raw, backend = None, "none"
        for b in (self.primary, self.fallback):
            if b is None:
                continue
            try:
                raw, backend = b.transcribe(wav_bytes), b.name
                break
            except SttError as e:
                nxt = self.fallback.name if b is self.primary and self.fallback is not None else None
                self.log.write("stt_fallback", backend=b.name, reason=str(e), next=nxt)
            except Exception as e:  # never let STT crash the mic worker
                self.log.error(f"stt.{b.name}", e)
        ms = int((time.monotonic() - t0) * 1000)
        if raw is None:
            result = SttResult("", "none", ms, "stt_failed")
        else:
            text = normalize_text(raw.text)
            reason = drop_reason(raw, text, self.blocklist)
            result = SttResult("" if reason else text, backend, ms, reason)
        self.log.write("stt", backend=result.backend, ms=result.ms, chars=len(result.text),
                       dropped_reason=result.dropped_reason)
        return result


def build_transcriber(cfg, log: EventLog | None = None, url: str | None = None, token: str | None = None,
                      ) -> FallbackTranscriber:
    """The real transcriber from config and env vars. Doesn't preload; see LocalWhisper.preload."""
    from . import config

    url = url if url is not None else config.stt_url()
    token = token if token is not None else config.stt_token()
    primary = VultrWhisper(url, token, cfg.stt_timeout_s) if url else None
    return FallbackTranscriber(primary, LocalWhisper(cfg.fallback_model), cfg.hallucination_blocklist, log)


def check_health(transcriber: FallbackTranscriber, log: EventLog) -> None:
    """Section 12: GET /health at startup, logged, in the background so startup never waits on it."""
    primary = transcriber.primary
    if not isinstance(primary, VultrWhisper):
        log.write("stt_health", ok=False, error="GLANCE_STT_URL not set; local STT only")
        return

    def run():
        try:
            log.write("stt_health", ok=True, **primary.health())
        except Exception as e:
            log.write("stt_health", ok=False, error=str(e))

    threading.Thread(target=run, name="stt-health", daemon=True).start()


def preload_fallback(transcriber: FallbackTranscriber, log: EventLog) -> None:
    """Load the local fallback model in the background (fallback_preload)."""
    fallback = transcriber.fallback
    if not isinstance(fallback, LocalWhisper):
        return

    def run():
        t0 = time.monotonic()
        try:
            fallback.preload()
            log.write("stt_fallback_ready", model=fallback.model_name, ms=int((time.monotonic() - t0) * 1000))
        except Exception as e:
            log.error("stt.preload", e)

    threading.Thread(target=run, name="stt-preload", daemon=True).start()
