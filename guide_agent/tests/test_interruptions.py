"""Section 13: speech_start, restore, abandonment, staleness, timers, haptic.start failure."""
import asyncio

from guide_agent.models.scripted import Gate
from guide_agent.schemas import HapticResult
from guide_agent.tests.rig import Rig, plain, reach, respond, speak, start

# --- speech_start ----------------------------------------------------------------


async def test_speech_start_stops_speech_and_the_idle_timer():
    rig = Rig()
    step = await rig.pending_speak()
    assert rig.ctl.idle_deadline is not None
    rig.ctl.on_speech_start()
    assert rig.names()[-1] == "stop_speech" and rig.ctl.idle_deadline is None
    assert step.heard_fully  # playback had finished, so nothing was cut off


async def test_speech_start_during_step_speech_marks_it_cut_off():
    rig = Rig()
    step = await rig.pending_speak(heard=False)
    assert not step.heard_fully and step.state == "awaiting_reply"
    assert rig.ctl.transcript.entries[-1].cut_off


async def test_speech_start_pauses_a_running_reach():
    rig = Rig()
    step = await rig.running_reach()
    rig.ctl.on_speech_start()
    assert step.state == "paused" and rig.names()[-1] == "haptic.pause"


async def test_speech_start_timeout_restores_the_step():
    rig = Rig()
    step = await rig.pending_speak(text="Turn right, then say done.", heard=False)
    rig.ctl.on_speech_start()
    m = rig.mark()
    await rig.advance(7.9)
    assert rig.says(m) == []
    await rig.advance(0.1)
    assert rig.says(m) == [("Turn right, then say done.", "done")]
    assert step.heard_fully


async def test_empty_utterance_restores_the_step():
    rig = Rig()
    step = await rig.running_reach()
    rig.ctl.on_speech_start()
    assert step.state == "paused"
    rig.ctl.on_utterance("  ")
    await rig.settle()
    assert step.state == "running" and rig.names()[-1] == "haptic.resume"
    assert rig.model.requests == []


# --- restore matrix ------------------------------------------------------------------


async def test_restore_speak_heard_says_nothing_and_rearms_idle():
    rig = Rig()
    await rig.pending_speak()
    rig.ctl.on_speech_start()
    m = rig.mark()
    rig.ctl.on_utterance("")
    assert rig.says(m) == [] and rig.ctl.idle_deadline is not None


async def test_restore_reach_cut_off_before_start_re_says_then_starts():
    rig = Rig(respond(reach()), plain("It is red."))
    await rig.pending_speak()
    await rig.utter("done", settle=False)
    await rig.drain()
    step = rig.ctl.step
    rig.ctl.on_speech_start()  # cuts off say_first, guidance never started
    assert not step.heard_fully and step.session_id is None
    m = rig.mark()
    rig.ctl.on_utterance("what color is it", frame=rig.frame())
    await rig.drain()
    assert [t for t, _ in rig.says(m)] == ["It is red.", step.say_first]
    assert "haptic.start" not in rig.names(m)
    rig.audio.finish()
    assert step.state == "running" and rig.names()[-1] == "haptic.start"


async def test_side_question_during_reach_resumes_after_the_answer():
    rig = Rig(plain("It is red."))
    step = await rig.running_reach()
    await rig.utter("what color is it", settle=False)
    await rig.drain()
    assert step.state == "paused" and rig.says()[-1] == ("It is red.", "none")
    rig.audio.finish()
    assert step.state == "running" and rig.names()[-1] == "haptic.resume"
    assert step.n == rig.ctl.step.n  # same step


# --- abandonment ----------------------------------------------------------------------


async def test_utterance_abandons_an_in_flight_turn():
    gate = Gate(respond(speak("Old step, then say done.")))
    rig = Rig(gate, respond(speak("New step, then say done.")))
    await rig.pending_speak()
    await rig.utter("done", settle=False)
    await gate.entered.wait()
    await rig.utter("actually wait for the blue one", settle=False)
    gate.release()
    await rig.settle()
    texts = [t for t, _ in rig.says()]
    assert "Old step, then say done." not in texts and rig.ctl.step.text == "New step, then say done."
    abandoned = [t for t in rig.turns() if t["abandoned"]]
    assert len(abandoned) == 1 and abandoned[0]["trigger"]["text"] == "done"
    assert abandoned[0]["model_calls"] and abandoned[0]["outputs"] == [{"call": "cue", "name": "thinking",
                                                                        "at": rig.clock.now()}]


async def test_effects_of_an_abandoned_turn_stay_and_the_next_turn_repairs():
    gate = Gate(respond(speak("Walk forward, then say done.")))
    rig = Rig(respond(start()), gate, plain("Sure, which can?"))
    await rig.utter("get me the can", settle=False)
    await gate.entered.wait()
    assert rig.ctl.task is not None  # start_task already executed
    await rig.utter("the red one", settle=False)
    gate.release()
    await rig.settle()
    assert rig.ctl.task.id == "t-1" and rig.ctl.step.text == "Sure, which can?"
    assert "no_pending_step" in rig.warnings()


async def test_speech_start_holds_an_in_flight_turn_until_the_utterance():
    gate = Gate(respond(speak("Old step, then say done.")))
    rig = Rig(gate, plain("Okay."))
    await rig.pending_speak()
    await rig.utter("done", settle=False)
    await gate.entered.wait()
    rig.ctl.on_speech_start()
    gate.release()
    for _ in range(10):
        await asyncio.sleep(0)  # the held turn gets its response but must not act on it
    assert rig.ctl.turn is not None and rig.ctl.turn.held
    assert all(t != "Old step, then say done." for t, _ in rig.says())
    rig.ctl.on_utterance("never mind that, what is this", frame=rig.frame())
    await rig.settle()
    assert all(t != "Old step, then say done." for t, _ in rig.says())


async def test_empty_utterance_releases_a_held_turn():
    gate = Gate(respond(speak("Next step, then say done.")))
    rig = Rig(gate)
    await rig.pending_speak()
    await rig.utter("done", settle=False)
    await gate.entered.wait()
    rig.ctl.on_speech_start()
    gate.release()
    rig.ctl.on_utterance("")
    await rig.settle()
    assert rig.ctl.step.text == "Next step, then say done."


# --- staleness ------------------------------------------------------------------------


async def test_stale_and_cancelled_haptic_results_are_dropped():
    rig = Rig()
    await rig.running_reach()
    await rig.haptic_result("target_lost", session_id="s-old")
    rig.ctl.on_haptic_result(HapticResult(session_id="s-1", status="cancelled"))
    await rig.settle()
    assert rig.model.requests == [] and rig.ctl.step.state == "running"


async def test_haptic_result_during_a_turn_waits_for_it_and_is_checked_when_dequeued():
    gate = Gate(respond(speak("Move closer, then say done.")))
    rig = Rig(gate)
    await rig.running_reach()
    await rig.utter("is it far", settle=False)
    await gate.entered.wait()
    await rig.haptic_result("timeout", settle=False, session_id="s-1")
    gate.release()
    await rig.settle()
    # the turn replaced the reach step, so the queued result was stale
    assert len(rig.model.requests) == 1
    assert rig.ctl.step.text == "Move closer, then say done."


async def test_haptic_start_failure_becomes_stale_link():
    rig = Rig(respond(reach()), respond(speak("Guidance stopped because of a connection problem.", expect="answer")))
    await rig.pending_speak()
    rig.haptic.fail_next_start = True
    await rig.utter("done")
    assert '"status": "stale_link"' in rig.input_text()
    assert rig.ctl.task.history[-1].result == "stale_link"


# --- idle timer and inactivity ---------------------------------------------------------------


async def test_idle_timeout_uses_the_steps_expect_and_counts_check_ins():
    rig = Rig(respond(speak("Say yes or no.", expect="answer")), respond(speak("Yes or no?", expect="answer")))
    await rig.pending_speak(expect="answer")
    await rig.advance(14)
    assert rig.model.requests == []
    await rig.advance(1)
    await rig.settle()
    assert 'type="idle_timeout" checkin="1"' in rig.input_text()
    await rig.advance(15)
    await rig.settle()
    assert 'checkin="2"' in rig.input_text()
    assert rig.ctl.idle_deadline is None  # cap reached
    await rig.advance(100)
    assert len(rig.model.requests) == 2 and rig.ctl.step.pending


async def test_idle_count_resets_on_utterance():
    rig = Rig(respond(speak("Still there? Say done.")), respond(speak("Next, then say done.")))
    await rig.pending_speak()
    await rig.advance(30)
    await rig.settle()
    assert rig.ctl.idle_checkins == 1
    await rig.utter("done")
    assert rig.ctl.idle_checkins == 0


async def test_idle_timer_is_frozen_during_turns():
    gate = Gate(plain("It is red."))
    rig = Rig(gate)
    await rig.pending_speak()
    rig.ctl.on_utterance("what color is it", frame=rig.frame())  # no speech_start
    await gate.entered.wait()
    rig.clock.advance(31)
    rig.ctl.tick()  # the turn is still in flight, so no idle trigger
    assert len(rig.model.requests) == 1 and not rig.ctl.queue
    gate.release()
    await rig.settle()
    assert rig.ctl.idle_deadline == rig.clock.now() + 30


async def test_inactivity_closes_the_task_silently():
    rig = Rig()
    await rig.pending_speak()
    rig.ctl.idle_checkins = 2  # check-ins used up
    m = rig.mark()
    await rig.advance(599)
    assert rig.ctl.task is not None
    await rig.advance(1)
    assert rig.ctl.task is None and rig.ctl.closed_tasks[-1].outcome == "cancelled"
    assert rig.says(m) == []
