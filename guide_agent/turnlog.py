"""JSONL turn log (CLAUDE.md section 16). One line per turn, plus event lines between turns.

Never pass image bytes or keys in here; frames are logged by id.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _default(value):
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return str(value)


class TurnLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    @classmethod
    def daily(cls, log_dir: str | Path) -> "TurnLog":
        return cls(Path(log_dir) / f"turns-{datetime.now():%Y%m%d}.jsonl")

    def write(self, record: dict) -> None:
        self.records.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=_default, ensure_ascii=False) + "\n")

    def turns(self) -> list[dict]:
        return [r for r in self.records if r.get("kind") == "turn"]
