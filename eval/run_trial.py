"""Runs one blindfolded trial (glove guidance or audio-only baseline) and writes
a JSONL log to eval/logs/ via shared.telemetry.Telemetry.
"""

from __future__ import annotations


def run_trial(condition: str, tester_id: str, trial_num: int) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
