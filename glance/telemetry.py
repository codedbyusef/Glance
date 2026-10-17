"""One JSON line per frame and per event, so any trial can be replayed.

Named telemetry.py, not logging.py, because logging.py would shadow the stdlib module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Telemetry:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **fields}
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
