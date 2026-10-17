"""Section 6 (turns, plain answers, tool errors, protocol warnings, turn cap) and section 9 (tool effects)."""
from guide_agent.controller import KEEP_GOING, SAY_AGAIN
from guide_agent.models.base import ModelResponse, ToolResults, UserInput
from rig import NOTES, Rig, call, finish, plain, reach, respond, speak, start

# --- plain answers ------------------------------------------------------------


async def test_plain_answer_without_task_is_said_with_expect_none():
    rig = Rig(plain("A laptop and a mug."))
    await rig.utter("what am I looking at")
    assert rig.says() == [("A laptop and a mug.", "none")]
    assert rig.ctl.task is None
    assert rig.last_turn()["terminal"] == "plain"


async def test_plain_answer_with_pending_step_heard_is_said_with_the_steps_expect():
    rig = Rig(plain("It is red."))
    step = await rig.pending_speak(expect="done")
    m = rig.mark()
    await rig.utter("what color is the can")
    assert rig.says(m) == [("It is red.", "done")]
    assert rig.ctl.step is step and step.state == "awaiting_reply"
    assert rig.ctl.idle_deadline is not None


async def test_plain_answer_with_pending_step_cut_off_re_says_the_step():
    rig = Rig(plain("It is red."))
    step = await rig.pending_speak(text="Step forward, then say done.", heard=False)
    m = rig.mark()
    await rig.utter("what color is the can")
    assert rig.says(m) == [("It is red.", "none"), ("Step forward, then say done.", "done")]
    assert step.heard_fully


async def test_plain_answer_with_active_task_and_no_pending_step_becomes_speak_answer():
    rig = Rig(plain("Did you find it?"))
    rig.set_task(notes="keep me")
    await rig.utter("hmm")
    step = rig.ctl.step
    assert step.type == "speak" and step.expect == "answer" and step.text == "Did you find it?"
    assert rig.warnings() == ["no_pending_step"]
    assert rig.ctl.task.notes == "keep me"


# --- the loop ---------------------------------------------------------------------


async def test_start_task_alone_continues_the_loop():
    rig = Rig(respond(start()), respond(speak()))
    await rig.utter("get me the can")
    assert len(rig.model.requests) == 2
    assert rig.tool_results(1) == [{"task_id": "t-1"}]
    assert rig.ctl.task.goal == "get the red can"
    assert rig.ctl.task.request == "get me the can"
    assert rig.ctl.step.type == "speak"


async def test_start_task_and_step_in_one_response_execute_in_order():
    rig = Rig(respond(start(), speak("Turn right, then say done.")))
    await rig.utter("get me the can")
    assert len(rig.model.requests) == 1
    assert [t["name"] for t in rig.last_turn()["tool_calls"]] == ["start_task", "speak"]
    assert rig.says()[-1] == ("Turn right, then say done.", "done")
    assert rig.ctl.task.current_step.n == 1


async def test_each_turn_starts_a_fresh_conversation():
    rig = Rig(respond(start(), speak()), plain("It is red."))
    await rig.utter("get me the can")
    await rig.utter("what color is it")
    second = rig.model.requests[1]
    assert second.system == "SYSTEM"
    assert len(second.messages) == 1 and isinstance(second.messages[0], UserInput)


async def test_task_active_error_then_recovery():
    rig = Rig(respond(start("find the door")), respond(speak()))
    await rig.pending_speak()
    await rig.utter("now the door")
    assert rig.tool_results(1)[0]["error"]["code"] == "task_active"
    assert rig.ctl.task.id == "t-setup"
    assert rig.ctl.step.text == "Take two steps forward, then say done."


async def test_no_active_task_error_then_recovery():
    rig = Rig(respond(speak()), plain("I can help with that."))
    await rig.utter("hello")
    assert rig.tool_results(1)[0]["error"]["code"] == "no_active_task"
    assert rig.says() == [("I can help with that.", "none")]


async def test_invalid_args_error_then_recovery():
    rig = Rig(respond(reach(target="the big shiny red soda can")), respond(reach()))
    await rig.pending_speak()
    await rig.utter("done")
    assert rig.tool_results(1)[0]["error"]["code"] == "invalid_args"
    assert rig.ctl.step.type == "reach" and rig.ctl.step.target == "red soda can"


async def test_error_stops_the_response_and_later_calls_are_skipped():
    rig = Rig(respond(start("find the door"), speak("Turn around.")), respond(speak("Keep going, then say done.")))
    await rig.pending_speak()
    await rig.utter("find the door")
    results = rig.tool_results(1)
    assert results[0]["error"]["code"] == "task_active"
    assert results[1]["status"] == "skipped"
    assert ("Turn around.", "done") not in rig.says()
    assert rig.ctl.step.text == "Keep going, then say done."


async def test_model_exception_counts_toward_the_cap():
    rig = Rig(RuntimeError("boom"), plain("Here you go."))
    await rig.utter("hello")
    assert len(rig.last_turn()["model_calls"]) == 2
    assert rig.says() == [("Here you go.", "none")]


# --- protocol warnings ----------------------------------------------------------


async def test_extra_terminal_executes_the_first_and_drops_the_rest():
    rig = Rig(respond(speak("First, then say done."), speak("Second, then say done.")))
    await rig.pending_speak()
    await rig.utter("done")
    assert rig.ctl.step.text == "First, then say done."
    assert "extra_terminal" in rig.warnings()
    assert all(text != "Second, then say done." for text, _ in rig.says())


async def test_missing_notes_keeps_previous_notes():
    rig = Rig(respond(speak(notes=None)), respond(speak(notes="   ")))
    rig.set_task(notes="old notes")
    await rig.pending_speak()
    await rig.utter("done")
    await rig.utter("done")
    assert rig.ctl.task.notes == "old notes"
    assert rig.warnings() == ["missing_notes", "missing_notes"]


async def test_notes_truncated():
    rig = Rig(respond(speak(notes="x" * 700)))
    await rig.pending_speak()
    await rig.utter("done")
    assert rig.ctl.task.notes == "x" * 600
    assert rig.warnings() == ["notes_truncated"]


async def test_step_and_answer_too_long_are_still_spoken():
    long = " ".join(["word"] * 30)
    rig = Rig(respond(speak(long)), plain(" ".join(["word"] * 45)))
    await rig.pending_speak()
    await rig.utter("done")
    await rig.utter("tell me everything")
    assert rig.warnings() == ["step_too_long", "answer_too_long"]
    assert [t for t, _ in rig.says()][-2:] == [long, " ".join(["word"] * 45)]


async def test_text_next_to_a_tool_call_is_dropped_and_logged():
    rig = Rig(respond(speak("Go on, then say done."), text="Sure! Here is the next step."))
    await rig.pending_speak()
    await rig.utter("done")
    assert all("Sure!" not in text for text, _ in rig.says())
    assert rig.last_turn()["info"] == ["text_with_tool"]


async def test_empty_response_is_retried():
    rig = Rig(ModelResponse(), plain("Hello."))
    await rig.utter("hello")
    assert rig.says() == [("Hello.", "none")]
    assert rig.last_turn()["info"] == ["empty_response"]


# --- turn cap ---------------------------------------------------------------------


async def test_turn_cap_with_task_and_no_pending_step_records_keep_going_step():
    rig = Rig(*[ModelResponse()] * 4)
    await rig.running_reach()
    await rig.haptic_result("target_lost")
    assert len(rig.model.requests) == 4
    step = rig.ctl.step
    assert (step.type, step.text, step.expect) == ("speak", KEEP_GOING, "answer")
    assert "turn_cap" in rig.warnings()


async def test_turn_cap_without_task_says_please_repeat():
    rig = Rig(*[respond(speak())] * 4)
    await rig.utter("hello")
    assert rig.says() == [(SAY_AGAIN, "none")]
    assert rig.warnings() == ["turn_cap"]


async def test_turn_cap_with_pending_step_restores_it():
    rig = Rig(*[ModelResponse()] * 4)
    step = await rig.pending_speak(expect="answer")
    m = rig.mark()
    await rig.utter("uh")
    assert rig.says(m) == [(SAY_AGAIN, "answer")]
    assert rig.ctl.step is step


# --- tool effects (section 9) ------------------------------------------------------


async def test_speak_says_text_and_arms_idle_timer_only_after_playback():
    rig = Rig(respond(start(), speak("Turn right, then say done.")))
    await rig.utter("get me the can", settle=False)
    await rig.drain()
    assert rig.says()[-1] == ("Turn right, then say done.", "done")
    assert rig.ctl.step.state == "speaking" and rig.ctl.idle_deadline is None
    rig.audio.finish()
    assert rig.ctl.step.state == "awaiting_reply" and rig.ctl.step.heard_fully
    assert rig.ctl.idle_deadline == rig.clock.now() + 30


async def test_reach_says_first_then_starts_guidance_when_playback_completes():
    rig = Rig(respond(reach()))
    await rig.pending_speak()
    await rig.utter("done", settle=False)
    await rig.drain()
    step = rig.ctl.step
    assert step.state == "starting" and "haptic.start" not in rig.names()
    assert rig.says()[-1][1] == "none"
    rig.audio.finish()
    assert step.state == "running" and step.session_id == "s-1"
    start_call = rig.rec.of("haptic.start")[0]
    assert start_call.args == {"target": "red soda can", "refer": "the red can on the table"}


async def test_finish_task_closes_and_says_optional_words():
    rig = Rig(respond(finish("success", "got the can", "Great, you have the can.")), respond(finish("failed", "gave up")))
    await rig.pending_speak(expect="answer")
    await rig.utter("yes")
    assert rig.ctl.task is None and rig.ctl.closed_tasks[-1].outcome == "success"
    assert rig.says()[-1] == ("Great, you have the can.", "none")
    rig.set_task()
    await rig.pending_speak()
    m = rig.mark()
    await rig.utter("no")
    assert rig.ctl.task is None and rig.says(m) == []


async def test_finish_task_cancels_a_running_reach():
    rig = Rig(respond(finish("cancelled", "user stopped", "Okay.")))
    step = await rig.running_reach()
    await rig.utter("never mind the can, I'll do it later")
    assert "haptic.cancel" in rig.names()
    assert rig.ctl.closed_tasks[-1].history[-1] is step


async def test_start_task_replace_active_supersedes_and_cancels_reach():
    rig = Rig(respond(start("find the door", replace_active=True), speak("Turn left, then say done.")))
    await rig.running_reach()
    await rig.utter("forget the can, find the door")
    old = rig.ctl.closed_tasks[-1]
    assert old.outcome == "superseded"
    assert rig.haptic.sessions["s-1"] == "cancelled"
    assert rig.ctl.task.goal == "find the door"


async def test_thinking_cue_only_for_user_turns():
    rig = Rig(respond(speak()), respond(speak("Pull your hand back, then say done.")))
    await rig.pending_speak()
    m = rig.mark()
    await rig.utter("done")
    assert rig.rec.of("cue", m)[0].args == {"name": "thinking"}
    rig.ctl.task.current_step = None
    await rig.running_reach()
    m = rig.mark()
    await rig.haptic_result("target_lost")
    assert rig.rec.of("cue", m) == []


async def test_turn_log_records_the_turn():
    rig = Rig(respond(start(), speak()))
    await rig.utter("get me the can")
    t = rig.last_turn()
    assert t["trigger"]["type"] == "user_utterance" and t["trigger"]["text"] == "get me the can"
    assert t["input"].startswith("<task_state>none</task_state>")
    assert t["frame_id"] == "f-1"
    assert t["state_before"]["task"] is None and t["state_after"]["task"]["goal"] == "get the red can"
    assert [o["call"] for o in t["outputs"]] == ["cue", "say"]
    assert t["latency_ms"] == 0.0
    assert t["abandoned"] is False
