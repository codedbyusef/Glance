"""System prompt (CLAUDE.md sections 14 and 15), filled in from config."""
from __future__ import annotations

from pathlib import Path
from string import Template

from .config import Config

PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "system.md"


def build_system_prompt(cfg: Config) -> str:
    template = Template(PROMPT_FILE.read_text(encoding="utf-8"))
    return template.substitute(
        assistant_name=cfg.assistant_name,
        glove_hand=cfg.glove_hand,
        step_max_words=cfg.step_max_words,
        answer_max_words=cfg.answer_max_words,
        notes_max_chars=cfg.notes_max_chars,
    )
