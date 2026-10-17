"""Rubric grading with judge_model (CLAUDE.md section 17.3). The judge never sees the image."""
from __future__ import annotations

import json

SYSTEM = (
    "You grade one turn of a voice assistant for blind users against a rubric. The assistant cannot be seen, "
    "only heard. You get the input the assistant received (its image is not included), the tool calls it made, "
    "and everything that was spoken. Pass the turn only if the rubric is clearly met. Reply in JSON."
)
SCHEMA = {
    "type": "object",
    "properties": {"pass": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["pass", "reason"],
}


async def grade(adapter, rubric: str, model_input: str | None, tool_calls: list[dict], spoken: list[str]) -> tuple[bool, str]:
    calls = [{"name": c["name"], "args": c["args"], "result": c.get("result")} for c in tool_calls]
    prompt = (
        f"Rubric: {rubric}\n\n"
        f"Assistant input:\n{model_input or '(none)'}\n\n"
        f"Tool calls:\n{json.dumps(calls, ensure_ascii=False, indent=1)}\n\n"
        f"Spoken, in order:\n" + ("\n".join(f"- {s}" for s in spoken) if spoken else "(nothing)")
    )
    result = await adapter.generate_json(SYSTEM, prompt, SCHEMA)
    if not result or "pass" not in result:
        return False, "judge returned no verdict"
    return bool(result["pass"]), str(result.get("reason", ""))
