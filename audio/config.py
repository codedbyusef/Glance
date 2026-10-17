"""Pi-side configuration (CLAUDE.md section 13). Defaults live here; audio/config.yaml overrides them."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

AUDIO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = AUDIO_DIR / "config.yaml"


@dataclass
class Config:
    # endpointing and wake word (section 7)
    vad_aggressiveness: int = 3  # decision 20
    speech_start_ms: int = 300
    end_silence_ms: int = 700
    preroll_ms: int = 300
    min_utterance_ms: int = 250
    max_utterance_s: float = 10
    no_speech_after_wake_s: float = 4
    wake_threshold: float = 0.5
    wake_vad_threshold: float = 0.5
    wake_cooldown_s: float = 1.5
    # After a detection, capture starts when the wake phrase ends: a 60 ms unvoiced gap, the score falling below
    # wake_end_threshold, or wake_end_max_s, whichever comes first (decision 19).
    wake_end_threshold: float = 0.3
    wake_end_max_s: float = 1.0
    wake_tone: bool = True  # short tone on a wake detection when nothing is playing (decision 24)

    # devices
    input_device: str | int | None = None  # None means the system default
    output_device: str | int | None = None
    sample_rate: int = 16000

    wake_model: str = "models/hey_glance.onnx"  # path, or a pretrained openWakeWord name such as hey_jarvis
    ptt: str = "none"  # none | key | gpio:<pin>
    assistant_name: str = "Glance"

    # STT client (section 10)
    stt_timeout_s: float = 2.0
    fallback_model: str = "tiny.en"
    fallback_preload: bool = True
    hallucination_blocklist: list[str] = field(
        default_factory=lambda: ["you", "thanks for watching", "thank you for watching"])

    # TTS (section 11)
    piper_voice: str = "models/en_US-lessac-medium.onnx"
    speaking_rate: float = 1.0

    save_clips: bool = False
    log_dir: str = "logs"

    def path(self, value: str) -> Path:
        """Resolve a config path relative to audio/."""
        p = Path(value)
        return p if p.is_absolute() else AUDIO_DIR / p


def load(path: str | os.PathLike | None = None) -> Config:
    """Defaults, overridden by the YAML file if it exists. Unknown keys are an error."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {', '.join(unknown)}")
    return Config(**data)


def stt_url() -> str | None:
    return os.environ.get("GLANCE_STT_URL") or None


def stt_token() -> str | None:
    return os.environ.get("GLANCE_STT_TOKEN") or None
