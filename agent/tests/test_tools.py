"""Section 9 schemas (a contract) and section 6 argument validation."""
import pytest

from guide_agent.tools import TOOL_SPECS, ToolError, validate


def test_tool_schemas_match_the_spec():
    shape = {
        spec["name"]: (
            {k: (v["type"], tuple(v.get("enum", ()))) for k, v in spec["parameters"]["properties"].items()},
            tuple(spec["parameters"]["required"]),
        )
        for spec in TOOL_SPECS
    }
    assert shape == {
        "start_task": ({"goal": ("string", ()), "replace_active": ("boolean", ())}, ("goal",)),
        "speak": (
            {"text": ("string", ()), "expect": ("string", ("done", "answer")), "notes": ("string", ())},
            ("text", "expect", "notes"),
        ),
        "reach": (
            {"target": ("string", ()), "refer": ("string", ()), "say_first": ("string", ()), "notes": ("string", ())},
            ("target", "refer", "say_first", "notes"),
        ),
        "finish_task": (
            {"outcome": ("string", ("success", "failed", "cancelled")), "summary": ("string", ()),
             "say": ("string", ())},
            ("outcome", "summary"),
        ),
    }


def _code(name, args):
    with pytest.raises(ToolError) as err:
        validate(name, args)
    return err.value.code


def test_missing_required_field():
    assert _code("speak", {"text": "Hi", "notes": "n"}) == "invalid_args"
    assert _code("finish_task", {"outcome": "success"}) == "invalid_args"


def test_empty_string():
    assert _code("speak", {"text": "  ", "expect": "done", "notes": "n"}) == "invalid_args"
    assert _code("start_task", {"goal": ""}) == "invalid_args"


def test_enum_violation():
    assert _code("speak", {"text": "Hi", "expect": "maybe", "notes": "n"}) == "invalid_args"
    assert _code("finish_task", {"outcome": "superseded", "summary": "s"}) == "invalid_args"


def test_wrong_types():
    assert _code("speak", {"text": 3, "expect": "done", "notes": "n"}) == "invalid_args"
    assert _code("start_task", {"goal": "g", "replace_active": "yes"}) == "invalid_args"


def test_target_over_five_words_is_invalid_and_five_is_allowed():
    base = {"refer": "r", "say_first": "s", "notes": "n"}
    assert _code("reach", {**base, "target": "the big red soda can here"}) == "invalid_args"
    assert validate("reach", {**base, "target": "big red soda can here"})["target"] == "big red soda can here"


def test_unknown_tool_is_invalid_args():
    assert _code("look", {}) == "invalid_args"


def test_notes_are_not_validated():
    validate("speak", {"text": "Hi", "expect": "done"})
    validate("reach", {"target": "can", "refer": "r", "say_first": "s", "notes": ""})


def test_empty_optional_say_counts_as_absent_and_replace_active_defaults_false():
    assert validate("finish_task", {"outcome": "success", "summary": "s", "say": ""})["say"] is None
    assert validate("start_task", {"goal": "g"})["replace_active"] is False
