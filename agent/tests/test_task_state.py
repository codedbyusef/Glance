"""Section 8 (task and step lifecycle) and section 12 (notes)."""
from rig import Rig, finish, plain, reach, respond, speak, start


async def test_step_results_record_what_ended_them():
    rig = Rig(
        respond(speak("A, then say done.")),  # ends the setup step: "user: done"
        respond(speak("B, then say done.")),  # idle turn ends A: "no reply"
        respond(reach()),  # ends B: "user: done"
        respond(speak("Pull back, then say done.")),  # after target_lost
    )
    await rig.pending_speak()
    await rig.utter("done")
    await rig.advance(30)
    await rig.settle()
    await rig.utter("done")
    await rig.haptic_result("target_lost")
    results = [s.result for s in rig.ctl.task.history]
    assert results == ["user: done", "no reply", "user: done", "target_lost"]
    assert [s.n for s in rig.ctl.task.history] == [1, 2, 3, 4]
    assert rig.ctl.step.n == 5


async def test_halt_result_is_interrupted_and_inferred_success_is_marked():
    rig = Rig(respond(speak("Stopped. Keep going?", expect="answer")), respond(reach()),
              respond(speak("Close your fingers around it and say yes if you have it.", expect="answer")))
    await rig.running_reach()
    await rig.utter("hold on")
    assert rig.ctl.task.history[-1].result == "interrupted"
    await rig.utter("yes, keep going")
    await rig.haptic_result("success", inferred=True)
    assert rig.ctl.task.history[-1].result == "success (inferred)"


async def test_one_task_at_a_time_and_new_task_records_the_request():
    rig = Rig(respond(start("get the can"), speak()))
    await rig.utter("get me the can please")
    assert rig.ctl.task.request == "get me the can please"
    assert rig.ctl.task.id == "t-1"


async def test_every_close_path_cancels_the_reach():
    # finish_task
    rig = Rig(respond(finish("failed", "x")))
    await rig.running_reach()
    await rig.utter("forget about it, it fell")
    assert rig.haptic.sessions["s-1"] == "cancelled"
    # cancel intercept
    rig = Rig()
    await rig.running_reach()
    await rig.utter("forget it")
    assert rig.haptic.sessions["s-1"] == "cancelled"
    # replace_active
    rig = Rig(respond(start("find the door", replace_active=True), speak()))
    await rig.running_reach()
    await rig.utter("forget the can, find the door")
    assert rig.haptic.sessions["s-1"] == "cancelled"
    # inactivity (the reach is paused by speech_start and never resumed)
    rig = Rig()
    await rig.running_reach()
    rig.ctl.on_speech_start()
    await rig.advance(600)
    assert rig.ctl.task is None and rig.haptic.sessions["s-1"] == "cancelled"


async def test_new_step_cancels_a_paused_reach_first():
    rig = Rig(respond(speak("Move your hand to the left, then say done.")))
    step = await rig.running_reach()
    await rig.utter("no, the blue one")
    assert rig.haptic.sessions["s-1"] == "cancelled"
    assert step.result == "user: no, the blue one"


# --- notes (section 12) -----------------------------------------------------------


async def test_notes_are_stored_verbatim_and_shown_next_turn():
    notes = "Step 1. Table about 4 steps ahead, slightly right.  Remaining: walk, reach."
    rig = Rig(respond(start(), speak(notes=notes)), plain("It is red."))
    await rig.utter("get me the can")
    assert rig.ctl.task.notes == notes
    await rig.utter("what color is it")
    assert f"notes: {notes}" in rig.input_text()


async def test_notes_are_rewritten_wholesale_on_each_step():
    rig = Rig(respond(start(), speak(notes="first")), respond(speak(notes="second")))
    await rig.utter("get me the can")
    await rig.utter("done")
    assert rig.ctl.task.notes == "second"


async def test_plain_answers_finish_and_intercepts_leave_notes_unchanged():
    rig = Rig(plain("It is red."), respond(finish("success", "ok")))
    rig.set_task(notes="keep")
    await rig.pending_speak()
    await rig.utter("what color is it")
    await rig.utter("repeat")
    assert rig.ctl.task.notes == "keep"
    await rig.utter("yes")
    assert rig.ctl.closed_tasks[-1].notes == "keep"
