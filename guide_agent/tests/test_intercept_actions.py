"""Section 7: what each intercept does in the controller."""
from guide_agent.controller import CANCELLED
from guide_agent.models.scripted import Gate
from guide_agent.tests.rig import Rig, plain, respond, speak, start


async def test_halt_during_reach_cancels_before_the_model_turn():
    rig = Rig(respond(speak("Okay, I stopped. Do you want to keep going?", expect="answer")))
    step = await rig.running_reach()
    m = rig.mark()
    await rig.utter("stop")
    names = rig.names(m)
    assert names.index("haptic.cancel") < names.index("cue")
    assert step.result == "interrupted" and rig.ctl.task.history[-1] is step
    assert 'intercepted="halt">stop</trigger>' in rig.input_text()
    assert "current_step: none" in rig.input_text()
    assert rig.ctl.step.expect == "answer"
    assert rig.last_turn()["intercept"] == "halt"


async def test_halt_without_task_only_stops_speech():
    rig = Rig()
    await rig.utter("wait", speech_start=False)
    assert rig.names() == ["stop_speech"]
    assert rig.model.requests == []


async def test_cancel_closes_the_task_without_a_model_turn():
    rig = Rig()
    await rig.running_reach()
    m = rig.mark()
    await rig.utter("never mind")
    assert rig.names(m)[-3:] == ["stop_speech", "haptic.cancel", "say"]
    assert rig.says(m) == [(CANCELLED, "none")]
    assert rig.ctl.task is None and rig.ctl.closed_tasks[-1].outcome == "cancelled"
    assert rig.model.requests == []


async def test_cancel_without_task_says_nothing():
    rig = Rig()
    await rig.utter("cancel", speech_start=False)
    assert rig.names() == ["stop_speech"]


async def test_cancel_abandons_an_in_flight_turn():
    gate = Gate(respond(start(), speak("Walk forward, then say done.")))
    rig = Rig(gate)
    await rig.pending_speak()
    await rig.utter("go to the kitchen", settle=False)
    await gate.entered.wait()
    await rig.utter("cancel", settle=False)
    gate.release()
    await rig.settle()
    assert ("Walk forward, then say done.", "done") not in rig.says()
    assert rig.ctl.task is None
    abandoned = [t for t in rig.turns() if t["abandoned"]]
    assert len(abandoned) == 1 and abandoned[0]["tool_calls"] == []


async def test_repeat_says_the_last_output_with_its_expect_and_keeps_the_step():
    rig = Rig(plain("It is red."))
    step = await rig.pending_speak(expect="done")
    await rig.utter("what color")  # answer said with the step's expect
    m = rig.mark()
    await rig.utter("say that again")
    assert rig.says(m) == [("It is red.", "done")]
    assert rig.ctl.step is step
    assert rig.ctl.idle_deadline is not None
    assert len(rig.model.requests) == 1


async def test_repeat_of_cut_off_step_speech_counts_as_heard_and_is_said_once():
    rig = Rig()
    step = await rig.pending_speak(text="Turn right, then say done.", heard=False)
    assert not step.heard_fully
    m = rig.mark()
    await rig.utter("what?")
    assert rig.says(m) == [("Turn right, then say done.", "done")]
    assert step.heard_fully and step.state == "awaiting_reply"
    assert rig.model.requests == []


async def test_repeat_with_nothing_said_does_nothing():
    rig = Rig()
    await rig.utter("repeat", speech_start=False)
    assert rig.says() == [] and rig.model.requests == []


async def test_repeat_during_reach_resumes_after_the_repeat():
    rig = Rig()
    step = await rig.running_reach()
    await rig.utter("pardon", settle=False)
    assert step.state == "paused"
    rig.audio.finish()
    assert step.state == "running" and rig.names()[-1] == "haptic.resume"


async def test_non_intercept_goes_to_the_model():
    rig = Rig(plain("Okay."))
    await rig.utter("stop, I got it")
    assert 'intercepted="none">stop, I got it</trigger>' in rig.input_text()
