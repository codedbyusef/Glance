"""Tool schemas (CLAUDE.md section 9) and argument validation (section 6 error codes).

Names, fields and enums are a contract; only descriptions may be expanded.
"""
from __future__ import annotations

TOOL_SPECS = [
    {
        "name": "start_task",
        "description": (
            "Open a task for a request that needs the user to move or act. "
            "Send the first step (speak or reach) in the same response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Short goal, at most 12 words."},
                "replace_active": {
                    "type": "boolean",
                    "description": "True only when the user clearly wants to drop the active task for this one.",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "speak",
        "description": (
            "Give one instruction or ask one question, then wait for the user's reply. Ends your turn. "
            "Use expect done for an action the user confirms by saying done, expect answer for a question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "expect": {"type": "string", "enum": ["done", "answer"]},
                "notes": {"type": "string"},
            },
            "required": ["text", "expect", "notes"],
        },
    },
    {
        "name": "reach",
        "description": "Start haptic guidance of the gloved hand to a target within arm's reach. Ends your turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "1 to 4 word noun phrase for the object detector. No positions.",
                },
                "refer": {"type": "string", "description": "What tells this object apart from similar ones."},
                "say_first": {"type": "string", "description": "Spoken before guidance starts."},
                "notes": {"type": "string"},
            },
            "required": ["target", "refer", "say_first", "notes"],
        },
    },
    {
        "name": "finish_task",
        "description": "End the active task. Ends your turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["success", "failed", "cancelled"]},
                "summary": {"type": "string", "description": "One line for the log."},
                "say": {"type": "string", "description": "Optional closing words for the user."},
            },
            "required": ["outcome", "summary"],
        },
    },
]

SPECS_BY_NAME = {spec["name"]: spec for spec in TOOL_SPECS}
TERMINAL_TOOLS = ("speak", "reach", "finish_task")
TARGET_MAX_WORDS = 5  # decision 3: validation rejects over 5; the prompt asks for 1 to 4
NOT_VALIDATED = ("notes",)  # decision 1: missing notes is a warning, not invalid_args


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    def as_result(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


def word_count(text: str) -> int:
    return len(text.split())


def validate(name: str, args: dict | None) -> dict:
    """Return the checked arguments, or raise ToolError("invalid_args")."""
    spec = SPECS_BY_NAME.get(name)
    if spec is None:
        raise ToolError("invalid_args", f"unknown tool {name!r}; use start_task, speak, reach or finish_task")
    args = dict(args or {})
    props = spec["parameters"]["properties"]
    required = spec["parameters"]["required"]

    for key in required:
        if key in NOT_VALIDATED:
            continue
        if key not in args or args[key] is None:
            raise ToolError("invalid_args", f"missing required field {key!r}")

    for key, prop in props.items():
        if key in NOT_VALIDATED or key not in args or args[key] is None:
            continue
        value = args[key]
        if prop["type"] == "string":
            if not isinstance(value, str):
                raise ToolError("invalid_args", f"{key!r} must be a string")
            if not value.strip():
                if key in required:
                    raise ToolError("invalid_args", f"{key!r} must not be empty")
                args[key] = None  # an empty optional string counts as absent
                continue
            if "enum" in prop and value not in prop["enum"]:
                raise ToolError("invalid_args", f"{key!r} must be one of {prop['enum']}, got {value!r}")
        elif prop["type"] == "boolean" and not isinstance(value, bool):
            raise ToolError("invalid_args", f"{key!r} must be true or false")

    if name == "reach" and word_count(args["target"]) > TARGET_MAX_WORDS:
        raise ToolError("invalid_args", "target must be a 1 to 4 word noun phrase with no positions")
    if name == "start_task":
        args["replace_active"] = bool(args.get("replace_active") or False)
    return args
