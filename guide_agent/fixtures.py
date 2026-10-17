"""Fixture frames (CLAUDE.md section 17.5): manifest lookup and loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import AGENT_ROOT

FIXTURE_DIR = AGENT_ROOT / "fixture_frames"


@dataclass
class Fixture:
    id: str
    path: Path
    description: str
    expected_text: str | None
    marked_missing: bool

    @property
    def available(self) -> bool:
        return self.path.exists()

    def image(self) -> bytes:
        return self.path.read_bytes()


def load_manifest(image_dir: str | Path | None = None) -> dict[str, Fixture]:
    """Read fixture_frames/manifest.json. image_dir overrides where the photos are looked up."""
    data = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    base = Path(image_dir) if image_dir else FIXTURE_DIR
    return {
        f["id"]: Fixture(id=f["id"], path=base / f["file"], description=f["description"],
                         expected_text=f.get("expected_text"), marked_missing=bool(f.get("missing")))
        for f in data["fixtures"]
    }
