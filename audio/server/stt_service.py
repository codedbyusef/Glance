"""Vultr STT service (CLAUDE.md section 9, decision 5).

Run from the repo root:  uvicorn audio.server.stt_service:app --host 0.0.0.0 --port 8001
Env: GLANCE_STT_TOKEN (required), GLANCE_STT_MODEL (default small.en),
     GLANCE_STT_INITIAL_PROMPT (default the command list; "none" or empty means no prompt),
     GLANCE_STT_DEVICE (cuda, the default, runs float16; cpu runs int8 for laptop testing).
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..stt_client import SAMPLE_RATE, read_wav, summarize_segments, to_float

logger = logging.getLogger("audio.stt_service")

DEFAULT_PROMPT = "done, yes, no, stop, wait, cancel, repeat"
BEAM_SIZE = 5
MAX_SECONDS = 15
MAX_BODY_BYTES = MAX_SECONDS * SAMPLE_RATE * 2 + 4096  # PCM16 mono plus header slack
# A request that waited this long for the lock has been given up on by the client (its timeout is about 2 s),
# so it is answered 503 without inference instead of delaying the requests behind it.
STALE_S = 10.0
# Warm-up: two passes on a real speech clip with the request settings, so the first real request doesn't pay for
# CUDA kernel setup, the VAD model or the decoder's first run (decision 21). Silence is the fallback.
WARMUP_CLIP = Path(__file__).resolve().parent.parent / "assets" / "stt_check.wav"
WARMUP_PASSES = 2


def add_cuda_dll_dirs() -> list[str]:
    """Windows only: make the pip nvidia-cublas-cu12 / nvidia-cudnn-cu12 DLLs loadable by CTranslate2.

    Must run before faster_whisper (ctranslate2) is imported. Returns the directories added.
    """
    if os.name != "nt":
        return []
    added = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc"):
        try:
            mod = __import__(pkg, fromlist=["__path__"])
        except ImportError:
            continue
        for base in mod.__path__:
            d = os.path.join(base, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")  # for plain LoadLibrary calls
                added.append(d)
    return added


def _load_whisper(model_name: str, device: str):
    if device == "cuda":
        for d in add_cuda_dll_dirs():
            logger.info("CUDA DLL directory: %s", d)
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")


def create_app(model_factory: Callable[[str], object] | None = None, token: str | None = None,
               model_name: str | None = None, initial_prompt: str | None = None,
               load_in_background: bool = True, device: str | None = None, stale_s: float = STALE_S,
               warmup_clip: Path | None = WARMUP_CLIP) -> FastAPI:
    """Settings default to the env vars. Tests pass a stub model_factory."""
    token = token if token is not None else os.environ.get("GLANCE_STT_TOKEN", "")
    model_name = model_name or os.environ.get("GLANCE_STT_MODEL", "small.en")
    if initial_prompt is None:
        initial_prompt = os.environ.get("GLANCE_STT_INITIAL_PROMPT", DEFAULT_PROMPT)
    if initial_prompt.strip().lower() in ("", "none"):  # PowerShell can't set an env var to ""
        initial_prompt = ""
    device = device or os.environ.get("GLANCE_STT_DEVICE", "cuda")
    factory = model_factory or (lambda name: _load_whisper(name, device))
    state = {"model": None, "ready": False, "error": None}
    settings = {"model": model_name, "device": device, "beam_size": BEAM_SIZE, "initial_prompt": initial_prompt or None}
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    lock = threading.Lock()  # one transcription at a time

    def run(model, audio: np.ndarray, vad_filter: bool = True):
        segments, _info = model.transcribe(
            audio, language="en", beam_size=BEAM_SIZE, vad_filter=vad_filter, condition_on_previous_text=False,
            initial_prompt=initial_prompt or None)
        return summarize_segments(segments)  # consumes the segment generator

    def load():
        try:
            logger.info("loading: %s", settings)
            t0 = time.monotonic()
            model = factory(model_name)
            warm_up(model)
            state["model"] = model
            state["ready"] = True
            logger.info("ready in %.1fs: %s", time.monotonic() - t0, settings)
        except Exception as e:
            state["error"] = repr(e)
            logger.exception("model load failed")

    def warm_up(model) -> None:
        audio, vad_filter = None, True
        try:
            samples, rate, channels, width = read_wav(Path(warmup_clip).read_bytes())
            if rate == SAMPLE_RATE and channels == 1 and width == 2:
                audio = to_float(samples)
        except Exception as e:  # missing or bad clip: warm up on silence instead
            logger.warning("warm-up clip %s unusable (%s); warming up on silence", warmup_clip, e)
        if audio is None:
            # vad_filter off, or the VAD would drop the silence and the decoder would never run.
            audio, vad_filter = np.zeros(SAMPLE_RATE, dtype=np.float32), False
        for i in range(WARMUP_PASSES):
            t0 = time.monotonic()
            r = run(model, audio, vad_filter=vad_filter)
            logger.info("warm-up pass %d: %d ms, %r", i + 1, (time.monotonic() - t0) * 1000, r.text)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not token:
            raise RuntimeError("GLANCE_STT_TOKEN is not set")
        if load_in_background:
            threading.Thread(target=load, name="model-load", daemon=True).start()
        else:
            load()
        yield

    app = FastAPI(lifespan=lifespan)

    def check_auth(request: Request) -> None:
        got = request.headers.get("authorization", "")
        if not token or not hmac.compare_digest(got.encode(), f"Bearer {token}".encode()):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health(request: Request):
        check_auth(request)
        body = {"ready": state["ready"], **settings}
        if not state["ready"]:
            if state["error"]:
                body["error"] = state["error"]
            return JSONResponse(body, status_code=503)
        return body

    @app.post("/transcribe")
    async def transcribe(request: Request):
        arrived = time.monotonic()
        check_auth(request)
        data = await request.body()
        if len(data) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail=f"audio longer than {MAX_SECONDS} s")
        try:
            samples, rate, channels, width = read_wav(data)
        except (wave.Error, EOFError, ValueError):
            raise HTTPException(status_code=415, detail="expected WAV, 16 kHz, mono, PCM16")
        if rate != SAMPLE_RATE or channels != 1 or width != 2:
            raise HTTPException(status_code=415, detail="expected WAV, 16 kHz, mono, PCM16")
        duration = len(samples) / SAMPLE_RATE
        if duration > MAX_SECONDS:
            raise HTTPException(status_code=413, detail=f"audio longer than {MAX_SECONDS} s")
        if not state["ready"]:
            return JSONResponse({"ready": False, "model": model_name}, status_code=503)
        return await _in_thread(lambda: _locked_run(samples, duration, arrived))

    def _locked_run(samples: np.ndarray, duration: float, arrived: float):
        with lock:
            t0 = time.monotonic()
            if t0 - arrived > stale_s:
                logger.warning("dropping a request that waited %.1fs for the model", t0 - arrived)
                return JSONResponse({"error": "stale", "waited_s": round(t0 - arrived, 1)}, status_code=503)
            raw = run(state["model"], to_float(samples))
            infer_ms = int((time.monotonic() - t0) * 1000)
        return {"text": raw.text, "duration_s": round(duration, 3), "infer_ms": infer_ms,
                "no_speech_prob": round(raw.no_speech_prob, 4), "avg_logprob": round(raw.avg_logprob, 4),
                "model": model_name}

    return app


async def _in_thread(fn):
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(fn)


app = create_app()
