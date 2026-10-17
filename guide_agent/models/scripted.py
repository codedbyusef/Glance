"""Scripted fake model for deterministic controller tests. No API calls."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .base import Message, ModelResponse, ToolCall


@dataclass
class Request:
    system: str
    messages: list[Message]
    tools: list[dict]


class Gate:
    """Holds a scripted response until release(), to simulate a model call in flight."""

    def __init__(self, response: ModelResponse | Exception):
        self.response = response
        self.entered = asyncio.Event()
        self._released = asyncio.Event()

    def release(self) -> None:
        self._released.set()


class ScriptedModel:
    def __init__(self, *script: ModelResponse | Gate | Exception):
        self.script = list(script)
        self.requests: list[Request] = []

    def add(self, *script: ModelResponse | Gate | Exception) -> None:
        self.script.extend(script)

    async def generate(self, system, messages, tools, temperature) -> ModelResponse:
        self.requests.append(Request(system, list(messages), tools))
        if not self.script:
            raise AssertionError("scripted model has no response left")
        item = self.script.pop(0)
        if isinstance(item, Gate):
            item.entered.set()
            await item._released.wait()
            item = item.response
        if isinstance(item, Exception):
            raise item
        return item


def call(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args, id=f"call-{name}")


def respond(*calls: ToolCall, text: str = "") -> ModelResponse:
    return ModelResponse(text=text, tool_calls=list(calls))


def plain(text: str) -> ModelResponse:
    return ModelResponse(text=text)
