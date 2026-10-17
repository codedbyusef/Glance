"""Speech to text: audio clip in, text out. faster-whisper (small or distilled)."""

from __future__ import annotations


def transcribe(audio: bytes) -> str:
    raise NotImplementedError
