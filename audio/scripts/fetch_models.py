"""Fetch the model files the Pi (or a dev laptop) needs. Run from the repo root with the audio venv:

    python audio/scripts/fetch_models.py [--skip-wake] [--skip-whisper]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
VOICE = "en_US-lessac-medium"
VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name}")
        return
    print(f"downloading {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wake", action="store_true")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--whisper-model", default="tiny.en")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    # Piper voice (section 4)
    for name in (f"{VOICE}.onnx", f"{VOICE}.onnx.json"):
        fetch(f"{VOICE_BASE}/{name}", MODELS / name)

    # openWakeWord feature models plus the hey_jarvis stand-in (section 8)
    if not args.skip_wake:
        import openwakeword.utils

        openwakeword.utils.download_models(model_names=["hey_jarvis"])
        print("openWakeWord models ready")

    # Local fallback STT (section 10): cache it now so the venue needs no download
    if not args.skip_whisper:
        from faster_whisper import WhisperModel

        WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
        print(f"faster-whisper {args.whisper_model} cached")
    print(f"models ready in {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
