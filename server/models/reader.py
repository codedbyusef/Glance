"""Scene 3 backend: VLM or PaddleOCR + LLM, reading a close frame of printed text."""

from __future__ import annotations


def read_text(frame: bytes) -> str:
    raise NotImplementedError
