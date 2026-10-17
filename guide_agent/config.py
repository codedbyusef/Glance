"""Configuration (CLAUDE.md section 4). Defaults live here; an optional config.yaml overrides them."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

AGENT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    assistant_name: str = "Nova"
    glove_hand: str = "right"
    provider: str = "gemini"
    model: str = "gemini-3.8-flash"
    judge_model: str = ""  # empty means "same as model"
    thinking_level: str = "low"
    temperature: float = 0.2
    max_model_calls_per_turn: int = 4
    recent_turns_max: int = 10
    recent_turns_max_age_s: float = 600
    history_max_steps: int = 8
    notes_max_chars: int = 600
    step_max_words: int = 25
    answer_max_words: int = 40
    idle_timeout_done_s: float = 30
    idle_timeout_answer_s: float = 15
    max_idle_checkins: int = 2
    speech_start_timeout_s: float = 8
    task_inactivity_close_s: float = 600
    frame_max_long_edge_px: int = 1024
    log_dir: str = "logs"

    @property
    def judge(self) -> str:
        return self.judge_model or self.model


def load_config(path: str | Path | None = None, **overrides) -> Config:
    cfg = Config()
    path = Path(path) if path else AGENT_ROOT / "config.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        cfg = replace(cfg, **data)
    return replace(cfg, **overrides) if overrides else cfg


def load_dotenv(path: str | Path | None = None) -> None:
    """Copy KEY=value lines from guide_agent/.env into os.environ. Variables already set win."""
    path = Path(path) if path else AGENT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
