"""Integration: the real Controller, Player and Mic, wired by runtime.voice_agent.App on a real event loop thread
(dispatch = loop.call_soon_threadsafe). Only the devices, STT and model are fakes: FakeSynth/FakeOutput,
scripted mic blocks with FakeVad/FakeDetector, a fake transcriber, a ScriptedModel and a fake camera.

The test thread plays the mic worker: it feeds blocks straight into Mic.handle, so every sink call still goes
through dispatch onto the loop thread.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from audio.capture import FakeCapture
from audio.config import Config as AudioConfig
from audio.listen import Mic
from audio.log import null_log
from audio.player import FakeOutput, FakeSynth, Player
from audio.stt_client import SttResult
from audio.vad import FakeVad
from audio.wake import FakeDetector
from guide_agent.config import Config as AgentConfig
from guide_agent.models.scripted import ScriptedModel, call, plain, respond
from guide_agent.schemas import Frame
from runtime.voice_agent import App

LOUD = np.full(1280, 3000, np.int16)
QUIET = np.zeros(1280, np.int16)
B = 0.08
WARM_TIMEOUT_S = 60.0   # one-time cost: the Player's synth thread imports scipy and caches the fixed phrases
WAIT_TIMEOUT_S = 20.0   # behaviour waits; they return as soon as the condition holds, so headroom is free
SPEECH_SAMPLES = 8000  # more than the thinking cue writes: the speech itself is playing
LONG = "Take two small steps forward and keep your hands low in front of you. " * 4  # ~3 s with FakeSynth


def blocks(*parts):
    out = []
    for kind, n in parts:
        out += [LOUD if kind == "l" else QUIET] * n
    return out


class FakeTranscriber:
    def __init__(self, *texts):
        self.texts = list(texts)

    def transcribe(self, wav):
        return SttResult(self.texts.pop(0) if self.texts else "", "vultr", 5)


class FakeCamera:
    def __init__(self):
        self.n = 0
        self.fps = 0.0

    def latest_frame(self):
        self.n += 1
        return Frame(id=f"cam-{self.n}", image=b"\xff\xd8 not a real jpeg", captured_at=time.time())

    def newest_bgr(self):
        return None


def start_warm(player, app) -> None:
    """Start the Player and wait for its warm-up before the App takes input, as runtime.voice_agent does, so
    the one-time scipy import and fixed-phrase cache never eat into a behaviour wait."""
    player.start()
    assert player.warm.wait(WARM_TIMEOUT_S), "Player warm-up did not finish"
    app.start()


class Rig:
    def __init__(self, model, texts, scores=None, ptt_only=False):
        import asyncio

        loop = asyncio.new_event_loop()
        self.output = FakeOutput(realtime=True)
        self.player = Player(FakeSynth(), self.output, dispatch=loop.call_soon_threadsafe, log=null_log())
        self.camera = FakeCamera()
        self.model = model
        self.transcriber = FakeTranscriber(*texts)
        acfg = AudioConfig(ptt_only=ptt_only, wake_tone=False)

        def mic_factory(sink, dispatch, task_active):
            return Mic(acfg, sink, self.transcriber, FakeCapture([]), FakeVad(), detector=FakeDetector(scores),
                       player=self.player, dispatch=dispatch, frame_source=self.camera.latest_frame,
                       task_active=task_active, log=null_log())

        self.app = App(AgentConfig(assistant_name="Glance"), model, self.player, self.camera,
                       mic_factory=mic_factory, tick_s=0.02, loop=loop)
        self.mic = self.app.mic
        self.t = time.time()
        start_warm(self.player, self.app)

    def push(self, bs, ptt=None):
        """Feed 80 ms blocks to the Mic; ptt maps a block index to a press (True) or release (False)."""
        for i, b in enumerate(bs):
            if ptt and i in ptt:
                self.mic.handle(("ptt", ptt[i], self.t))
            self.mic.handle(("audio", b, self.t))
            self.t += B

    def ctl(self, fn):
        return self.app.on_loop(lambda: fn(self.app.controller))

    def step(self):
        return self.ctl(lambda c: c.step)

    def wait(self, pred, timeout=WAIT_TIMEOUT_S, what=""):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {what or pred}")

    def close(self):
        self.app.stop()


@pytest.fixture
def rigs():
    made = []
    yield made
    for r in made:
        r.close()


def model_inputs(model):
    return [str(r.messages) for r in model.requests]


def test_wake_utterance_say_and_reply_window(rigs):
    model = ScriptedModel(
        respond(call("start_task", goal="get the can"),
                call("speak", text="Turn right, then say done.", expect="done", notes="Step 1.")),
        respond(call("finish_task", outcome="success", summary="done", say="Great.")),
    )
    r = Rig(model, ["get me the can", "done"], scores=[0.1] * 5 + [0.9])
    rigs.append(r)
    assert r.mic.wake_needed()  # idle: no reply window, nothing playing, no task

    r.push(blocks(("l", 6), ("q", 2), ("l", 10), ("q", 12)))  # "hey glance", then the request
    r.wait(lambda: (s := r.step()) is not None and s.state == "awaiting_reply", what="speak step played")
    assert len(model.requests) == 1 and "get me the can" in model_inputs(model)[0]
    assert r.output.samples_written > 0
    assert r.player.window.is_open and not r.mic.wake_needed()  # expect=done opens the reply window

    r.push(blocks(("q", 5), ("l", 8), ("q", 12)))  # "done" with no wake word
    r.wait(lambda: r.ctl(lambda c: c.task) is None, what="task finished")
    assert r.ctl(lambda c: c.closed_tasks[-1].outcome) == "success"
    assert "done" in model_inputs(model)[1]


def test_barge_in_stops_speech_and_halt_runs_a_turn(rigs):
    model = ScriptedModel(
        respond(call("start_task", goal="get the can"), call("speak", text=LONG, expect="done", notes="Step 1.")),
        respond(call("speak", text="Okay, I stopped. Keep going?", expect="answer", notes="Halted.")),
    )
    r = Rig(model, ["get me the can", "stop"], scores=[0.1] * 5 + [0.9])
    rigs.append(r)
    r.push(blocks(("l", 6), ("q", 2), ("l", 10), ("q", 12)))
    r.wait(lambda: r.player.busy and r.output.samples_written > SPEECH_SAMPLES, what="long speech playing")
    assert r.output.aborts == 0

    r.push(blocks(("q", 2), ("l", 8), ("q", 12)))  # the user talks over the speech: open state, VAD start
    r.wait(lambda: (s := r.step()) is not None and s.expect == "answer", what="halt turn")
    r.wait(lambda: r.output.aborts >= 1, what="playback cut")  # the writer aborts at its next 50 ms chunk
    assert 'intercepted="halt"' in model_inputs(model)[1]


def test_ptt_only_ignores_voice_and_wake_word_and_ptt_barges_in(rigs):
    model = ScriptedModel(
        respond(call("start_task", goal="get the can"), call("speak", text=LONG, expect="done", notes="Step 1.")),
        respond(call("speak", text="Okay, I stopped. Keep going?", expect="answer", notes="Halted.")),
    )
    r = Rig(model, ["get me the can", "stop"], scores=[0.9] * 400, ptt_only=True)
    rigs.append(r)
    r.push(blocks(("l", 20), ("q", 15)))  # talking, and "hey glance" scores, without the key: nothing
    time.sleep(0.2)
    assert model.requests == []

    r.push(blocks(("q", 3), ("l", 10), ("q", 3)), ptt={3: True, 15: False})
    r.wait(lambda: r.player.busy and r.output.samples_written > SPEECH_SAMPLES, what="long speech playing")
    r.push(blocks(("l", 15), ("q", 15)))  # the speaker's own output (or talking) during playback: no barge-in
    time.sleep(0.2)
    assert r.output.aborts == 0 and len(model.requests) == 1

    r.push(blocks(("l", 8), ("q", 2)), ptt={0: True, 9: False})  # Enter while it speaks = barge-in, then "stop"
    r.wait(lambda: (s := r.step()) is not None and s.expect == "answer", what="halt turn")
    r.wait(lambda: r.output.aborts >= 1, what="playback cut")
    assert 'intercepted="halt"' in model_inputs(model)[1]


def test_real_reach_path_start_progress_result_through_the_controller(rigs):
    """--reach real wiring: say_first plays, then ReachHaptic starts the ReachAPI reach; progress drives the
    glove; the success result reaches the controller through dispatch and the model gets its turn."""
    from runtime.guidance import Guidance
    from runtime.reach_haptic import ReachHaptic
    from runtime.tests.test_reach_haptic import FakeReachAPI, progress, result

    holder = {}

    def haptic_factory(dispatch, on_result):
        holder["h"] = ReachHaptic(FakeReachAPI, dispatch, on_result, Guidance("table"), holder["glove"])
        return holder["h"]

    model = ScriptedModel(
        respond(call("start_task", goal="get the can"),
                call("reach", target="red soda can", refer="the red can on the table",
                     say_first="Bring your right hand out.", notes="Reach 1.")),
        respond(call("speak", text="Your hand should be on the can. Say yes if you have it.", expect="answer",
                     notes="Touch check.")),
    )
    from runtime.glove_out import FakeGlove
    holder["glove"] = FakeGlove()

    class RealReachRig(Rig):
        def __init__(self):
            import asyncio
            loop = asyncio.new_event_loop()
            self.output = FakeOutput(realtime=True)
            self.player = Player(FakeSynth(), self.output, dispatch=loop.call_soon_threadsafe, log=null_log())
            self.camera = FakeCamera()
            self.model = model
            self.app = App(AgentConfig(assistant_name="Glance"), model, self.player, self.camera,
                           glove=holder["glove"], tick_s=0.02, loop=loop, haptic_factory=haptic_factory)
            self.mic = None
            start_warm(self.player, self.app)

    r = RealReachRig()
    rigs.append(r)
    r.app.type_utterance("get me the can")
    h = holder["h"]
    r.wait(lambda: h.session_id is not None, what="reach started after say_first")
    assert h.api.calls[0] == ("start", "red soda can", "the red can on the table")
    assert (s := r.step()).type == "reach" and s.state == "running" and s.session_id == h.session_id
    sid = h.session_id
    h.api.on_progress(progress(sid))
    assert holder["glove"].command.pulse_hz  # guidance drives the glove (pulsing direction motors)
    h.api.on_result(result(sid, "success"))
    assert holder["glove"].command.pulse_hz == {} and holder["glove"].command.on == frozenset()
    r.wait(lambda: (s := r.step()) is not None and s.type == "speak" and s.expect == "answer", what="touch check")
    assert '"status": "success"' in model_inputs(model)[1]


def test_overlay_renders_without_a_camera_image(rigs):
    from runtime.voice_agent import Overlay

    r = Rig(ScriptedModel(), [])
    rigs.append(r)
    r.app.status.set(transcript="hello", sent_frame=None)
    canvas = Overlay(r.app, 640, "ptt-only").render()
    assert canvas.shape[1] == 640 and canvas.any()


def test_typed_utterance_gets_the_camera_frame_and_is_spoken(rigs):
    model = ScriptedModel(plain("A red can on a desk."))
    r = Rig(model, [])
    rigs.append(r)
    r.app.type_utterance("what is in front of me")
    r.wait(lambda: r.ctl(lambda c: c.last_spoken) is not None and r.output.samples_written > 0, what="answer spoken")
    assert 'frame id="cam-1"' in model_inputs(model)[0]
    st = r.app.status.get()
    assert st["transcript"] == "what is in front of me" and st["sent_frame"].id == "cam-1"
    r.wait(lambda: r.app.status.get()["spoken"] == "A red can on a desk.", what="status updated by the ticker")
