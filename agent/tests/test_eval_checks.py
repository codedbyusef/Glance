"""The eval's matchers and global checks (section 17.3)."""
from evals.checks import match, spoken_violations
from evals.run import _check_outputs


def test_matchers_are_case_insensitive_and_ignore_punctuation():
    assert match("Bring your Right Hand out.", {"contains_all": ["right hand"]}, "s") == []
    assert match("done", {"equals": "Done"}, "s") == []
    assert match(True, {"equals": True}, "s") == []
    assert match("speak", {"in": ["plain", "speak"]}, "s") == []
    assert match("a b c", {"max_words": 3, "regex": r"^a"}, "s") == []
    assert match("the cancan", {"contains_any": ["can"]}, "s")  # whole words only
    assert match("near the chair", {"not_contains_any": ["chair"]}, "s")
    assert match("x", {"bogus": 1}, "s")


def test_global_checks():
    assert spoken_violations("Take two steps forward, then say done.") == []
    assert spoken_violations("Look at the table, you can see it over there.")
    assert spoken_violations("It is within arm’s length.")
    assert spoken_violations("**Step 1**: walk")
    assert spoken_violations("- walk forward")
    assert spoken_violations("Great job \U0001F44D")


def test_output_order():
    names = ["stop_speech", "haptic.cancel", "cue", "model", "say"]
    assert _check_outputs(names, {"order": ["haptic.cancel", "model"], "includes": ["say"]}) == []
    assert _check_outputs(names, {"order": ["model", "haptic.cancel"]})
    assert _check_outputs(names, {"excludes": ["say"]})
