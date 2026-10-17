"""Aggregates eval/logs/*.jsonl into the write-up numbers: per-stage latency and
the glove-vs-audio-only trial comparison (3 testers x 5 trials x 2 conditions).
"""

from __future__ import annotations

from pathlib import Path


def analyze(logs_dir: Path) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
