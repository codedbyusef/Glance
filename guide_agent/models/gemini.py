"""Gemini adapter (google-genai, generate_content). The only module that imports the Gemini SDK."""
from __future__ import annotations

import asyncio
import os
import random
import time

from google import genai
from google.genai import errors, types

from .base import Message, ModelResponse, ModelTurn, ToolCall, ToolResults, UserInput

RETRY_CODES = {408, 429, 500, 502, 503, 504}


class GeminiAdapter:
    def __init__(self, model: str, thinking_level: str = "low", api_key: str | None = None,
                 max_retries: int = 2, timeout_s: float = 30.0):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set; put it in guide_agent/.env or the environment")
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.model = model
        self.thinking_level = thinking_level
        self.max_retries = max_retries

    async def generate(self, system: str, messages: list[Message], tools: list[dict],
                       temperature: float) -> ModelResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"])
                for t in tools
            ])] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=self._thinking(),
        )
        contents = self._contents(messages)
        started = time.perf_counter()
        response = await self._call(contents, config)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return self._parse(response, latency_ms)

    async def generate_json(self, system: str, prompt: str, schema: dict, temperature: float = 0.0) -> dict | None:
        """One JSON-mode call with no tools (used by the eval judge)."""
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, response_mime_type="application/json",
            response_json_schema=schema, thinking_config=self._thinking(),
        )
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        response = await self._call(contents, config)
        return response.parsed if isinstance(response.parsed, dict) else None

    def _thinking(self) -> types.ThinkingConfig | None:
        if not self.thinking_level:
            return None
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel(self.thinking_level.upper()))

    async def _call(self, contents, config):
        for attempt in range(self.max_retries + 1):
            try:
                return await self.client.aio.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.APIError as exc:
                if getattr(exc, "code", None) not in RETRY_CODES or attempt == self.max_retries:
                    raise
            except (asyncio.TimeoutError, TimeoutError, OSError):
                if attempt == self.max_retries:
                    raise
            await asyncio.sleep(min(8.0, 2 ** attempt) + random.random() * 0.5)

    @staticmethod
    def _contents(messages: list[Message]) -> list[types.Content]:
        contents: list[types.Content] = []
        for message in messages:
            if isinstance(message, UserInput):
                parts = [types.Part.from_text(text=message.text)]
                if message.image:
                    parts.append(types.Part.from_bytes(data=message.image, mime_type="image/jpeg"))
                contents.append(types.Content(role="user", parts=parts))
            elif isinstance(message, ModelTurn):
                if message.raw is not None and message.raw.parts:
                    contents.append(message.raw)  # verbatim, so thought signatures go back unchanged
                elif message.text or message.tool_calls:
                    parts = [types.Part.from_text(text=message.text)] if message.text else []
                    parts += [types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args))
                              for c in message.tool_calls]
                    contents.append(types.Content(role="model", parts=parts))
            elif isinstance(message, ToolResults) and message.results:
                contents.append(types.Content(role="user", parts=[
                    types.Part(function_response=types.FunctionResponse(id=r.call.id, name=r.call.name,
                                                                        response=r.result))
                    for r in message.results
                ]))
        return contents

    @staticmethod
    def _parse(response, latency_ms: float) -> ModelResponse:
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate is not None else None
        texts: list[str] = []
        calls: list[ToolCall] = []
        logged_parts: list[dict] = []
        for part in (content.parts or []) if content is not None else []:
            if part.thought:
                logged_parts.append({"thought": True})
            elif part.function_call is not None:
                fc = part.function_call
                calls.append(ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id))
                logged_parts.append({"function_call": {"name": fc.name, "args": dict(fc.args or {})}})
            elif part.text:
                texts.append(part.text)
                logged_parts.append({"text": part.text})
        usage = response.usage_metadata
        tokens = None
        if usage is not None:
            tokens = {
                "prompt": usage.prompt_token_count,
                "output": usage.candidates_token_count,
                "thoughts": usage.thoughts_token_count,
                "total": usage.total_token_count,
            }
        raw_log = {
            "model_version": response.model_version,
            "finish_reason": str(candidate.finish_reason) if candidate is not None and candidate.finish_reason else None,
            "parts": logged_parts,
        }
        return ModelResponse(text="".join(texts), tool_calls=calls, raw=content, raw_log=raw_log,
                             latency_ms=latency_ms, tokens=tokens)
