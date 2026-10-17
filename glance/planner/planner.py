"""LLM call (transcript + one frame + state -> JSON plan), schema validation, one retry.

Vision-capable LLM API (Gemini preferred). Called at task start and on events, never per frame.
"""

from __future__ import annotations

from glance.schemas import Plan


def plan(transcript: str, frame: bytes, state: dict) -> Plan:
    raise NotImplementedError
