"""Section 11: the per-turn model input, and section 15: the system prompt is filled in."""
from guide_agent.compose import Transcript, compose_input
from guide_agent.config import Config
from guide_agent.prompt import build_system_prompt
from guide_agent.schemas import Frame, HapticResult, HapticTrigger, IdleTrigger, LastSeen, UtteranceTrigger
from guide_agent.state import Step, Task
from rig import Rig, plain, respond, speak

T0 = 1_800_000_000.0
STEP1 = ("The table is about four steps ahead, slightly to your right. Turn slightly right and take two steps "
         "forward, then say done.")
STEP2 = ("Hold your hands out at waist height and step forward slowly until you touch the table, about two "
         "steps, then say done.")
NOTES = ("Step 2. Table about 2 steps ahead, can on its right half at waist height. Remaining: walk to the "
         "table, reach for the can, grasp.")


def spec_example():
    now = T0 + 41
    task = Task(id="t-1", goal="get the red can", request="get me the red can on the table", started_at=T0,
                last_user_activity_at=T0 + 11, notes=NOTES)
    task.history.append(Step(n=1, type="speak", issued_at=T0 + 2, text=STEP1, expect="done", result="user: done",
                             heard_fully=True))
    task.current_step = Step(n=2, type="speak", issued_at=now - 12, text=STEP2, expect="done",
                             state="awaiting_reply", heard_fully=True)
    transcript = Transcript()
    transcript.add(T0, "user", "get me the red can on the table")
    transcript.add(T0 + 2, "assistant", STEP1)
    transcript.add(T0 + 11, "user", "done")
    transcript.add(T0 + 13, "assistant", STEP2)
    return task, transcript, now


def test_matches_the_spec_example():
    task, transcript, now = spec_example()
    frame = Frame(id="f-203", image=b"x", captured_at=now - 0.38)
    text = compose_input(task=task, transcript=transcript, frame=frame, now=now, cfg=Config(),
                         trigger=UtteranceTrigger(text="done", frame=frame, arrived_at=now))
    assert text == f"""<task_state>
goal: get the red can
request: "get me the red can on the table"
current_step: n=2 type=speak state=awaiting_reply expect=done heard_fully=true age_s=12
  text: "{STEP2}"
history:
  1 speak "{STEP1}" -> user: done
notes: {NOTES}
</task_state>
<recent_turns>
[41s] user: get me the red can on the table
[39s] assistant: {STEP1}
[30s] user: done
[28s] assistant: {STEP2}
</recent_turns>
<trigger type="user_utterance" intercepted="none">done</trigger>
<frame id="f-203" age_ms="380"/>"""


def test_variants_no_task_haptic_idle_no_image():
    cfg = Config()
    event = HapticResult(session_id="s-17", status="target_lost", elapsed_s=6.1, target="red soda can",
                         last_seen=LastSeen("right-middle", "near", 2.3), closest_approach="near",
                         frame=Frame(id="f-311", image=b"x", captured_at=T0))
    text = compose_input(task=None, transcript=Transcript(), trigger=HapticTrigger(event, T0), frame=None,
                         now=T0, cfg=cfg)
    assert text.startswith("<task_state>none</task_state>\n<recent_turns>none</recent_turns>")
    assert ('<trigger type="haptic_result">{"event": "haptic_result", "status": "target_lost", "inferred": false, '
            '"elapsed_s": 6.1, "target": "red soda can", "last_seen": {"region": "right-middle", "depth": "near", '
            '"age_s": 2.3}, "closest_approach": "near"}</trigger>') in text
    assert "s-17" not in text and "f-311" not in text
    assert text.endswith("<frame>none</frame>")
    idle = compose_input(task=None, transcript=Transcript(), trigger=IdleTrigger(1, T0), frame=None, now=T0, cfg=cfg)
    assert '<trigger type="idle_timeout" checkin="1"/>' in idle


def test_reach_step_cut_off_speech_and_history_limits():
    cfg = Config(history_max_steps=2, recent_turns_max=2)
    task = Task(id="t-1", goal="g", request="r", started_at=T0, last_user_activity_at=T0)
    for n in range(1, 4):
        task.history.append(Step(n=n, type="reach", issued_at=T0, target="red soda can", refer="the red can",
                                 result="target_lost"))
    task.current_step = Step(n=4, type="reach", issued_at=T0, target="red soda can", refer="the red can",
                             say_first="Bring your right hand out.", state="paused", heard_fully=True)
    transcript = Transcript()
    transcript.add(T0 - 700, "user", "too old")
    transcript.add(T0 - 5, "user", "hello")
    transcript.add(T0 - 4, "assistant", "Hi there.").cut_off = True
    transcript.add(T0 - 3, "event", "reach started: red soda can")
    text = compose_input(task=task, transcript=transcript, trigger=IdleTrigger(1, T0), frame=None, now=T0, cfg=cfg)
    assert "current_step: n=4 type=reach state=paused heard_fully=true age_s=0" in text
    assert '  target: "red soda can"' in text and '  say_first: "Bring your right hand out."' in text
    assert "  (1 earlier steps not shown)" in text and "  1 reach" not in text
    assert '  3 reach target="red soda can" refer="the red can" -> target_lost' in text
    assert "[4s] assistant (cut off): Hi there." in text and "[3s] [reach started: red soda can]" in text
    assert "hello" not in text and "too old" not in text


async def test_turn_never_sees_its_own_trigger_in_recent_turns():
    rig = Rig(plain("A mug."), plain("It is white."))
    await rig.utter("what is this")
    assert "what is this" not in rig.input_text(0).split("<trigger")[0]
    await rig.utter("what color")
    recent = rig.input_text(1).split("<recent_turns>")[1]
    assert "user: what is this" in recent and "assistant: A mug." in recent
    assert "what color" not in recent.split("</recent_turns>")[0]


async def test_images_attached_per_trigger():
    rig = Rig(respond(speak("Close your fingers around it and say yes if you have it.", expect="answer")),
              respond(speak("Still there? Say yes if you have it.", expect="answer")))
    await rig.running_reach()
    await rig.haptic_result("success")
    assert rig.model.requests[0].messages[0].image is None  # no image on success
    assert rig.input_text(0).endswith("<frame>none</frame>")
    rig.camera.frame = Frame(id="f-latest", image=b"jpeg", captured_at=rig.clock.now())
    await rig.advance(15)
    await rig.settle()
    assert rig.model.requests[1].messages[0].image == b"jpeg"  # idle turns use the latest frame
    assert '<frame id="f-latest"' in rig.input_text(1)


def test_system_prompt_is_filled_in():
    prompt = build_system_prompt(Config(assistant_name="Nova", glove_hand="right"))
    assert "$" not in prompt
    assert "You are Nova" in prompt and "right hand" in prompt
