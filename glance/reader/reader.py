"""Scene 3: close frame -> OCR or vision-language read -> speech. No glove involved."""

from __future__ import annotations


def read_aloud(frame: bytes) -> str:
    raise NotImplementedError
