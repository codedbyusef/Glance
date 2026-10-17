"""Text-to-speech output. Hazard warnings interrupt whatever is currently speaking."""

from __future__ import annotations


class AudioOut:
    def speak(self, text: str, interrupt: bool = False) -> None:
        raise NotImplementedError
