"""Provider-neutral model interface. The controller only sees these types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class UserInput:
    text: str
    image: bytes | None = None  # JPEG


@dataclass
class ModelTurn:
    """A model response sent back as history when the turn continues."""
    text: str
    tool_calls: list[ToolCall]
    raw: Any = None  # provider content, kept verbatim (it can carry thought signatures)


@dataclass
class ToolResult:
    call: ToolCall
    result: dict


@dataclass
class ToolResults:
    results: list[ToolResult]


Message = UserInput | ModelTurn | ToolResults


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider content for the next call in this turn
    raw_log: Any = None  # JSON-safe form for the turn log
    latency_ms: float = 0.0
    tokens: dict | None = None

    def as_turn(self) -> ModelTurn:
        return ModelTurn(self.text, list(self.tool_calls), self.raw)


class ModelAdapter(Protocol):
    async def generate(
        self, system: str, messages: list[Message], tools: list[dict], temperature: float
    ) -> ModelResponse: ...
