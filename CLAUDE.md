# Glance

A voice-controlled assistive prototype for blind and low-vision users, built for SwampHacks XII (Oct 16-18, 2026).
A head-worn camera sees the room. The user speaks a request ("guide me to the paper"). Speech handles coarse
navigation and reading; a vibrating glove (8 motors) handles the last arm's length of reaching.

This is a hackathon prototype. It supplements a cane or guide dog and makes no safety claims. Never add
language to the code, README or demo that says it keeps a user safe.

Targets: First Timers track, Best Use of Vultr (MLH). Also eligible: Hardware, AI, Gemini (if the planner uses Gemini).
Hacking ends Sun Oct 18, 8:00 AM, which is also the Devpost deadline.

## Priorities (in order)

1. A first working reach (hand guided to an object) by Saturday ~2 PM. Everything else waits for this.
2. Per-stage latency numbers (the Vultr story).
3. A small blindfolded trial: glove guidance vs audio-only baseline (3 testers x 5 trials x 2 conditions).
4. Scene 3 (read a page aloud) and everything stretch.

Freeze scope after the first working reach. Do not add features before then.

## The demo (three voice scenes, one table, one sheet of paper)

1. Identify: "what is on the desk?" -> detector over the scene -> speech ("a piece of paper, about 3 steps ahead and slightly left").
2. Guide: "guide me to the paper" -> audio step (walk), then haptic reach step until the hand touches the paper, then a confirm pulse.
3. Read: "read this paper" -> close frame -> OCR or vision-language read -> speech. No glove.

## Architecture

```
 WORN BY USER            LAPTOP                         CLOUD
 camera ───────────────► orchestrator (state machine) ─► Vultr GPU (one instance, all per-frame models)
 mic ──► wake word ─────►   │  ▲ events                    whisper, detector, tracker, hand, depth, reader
 headphones ◄── speech ◄────┘  │                         LLM planner API (only call off the Vultr path;
 ESP32 + drivers ◄── BLE/WiFi ◄ vision loop + controller     called at task start and on events, never per frame)
   └─► 8 coin motors
```

- The laptop owns the session and the glove link. Camera, headphones and mic connect to the laptop, never to the glove.
- One Vultr GPU instance serves every per-frame model behind a single streaming (WebSocket) service.
- The LLM is NOT in the per-frame loop. It runs before a task (plan) and after events (replan).
- Network path is the main risk. Fallbacks: phone hotspot, smaller frames, run hand pose locally.

## Components and interfaces

| Component | Job | Tech | Runs on |
|---|---|---|---|
| Wake word | Hear the phrase, then record to end of speech | openWakeWord ("hey jarvis" pretrained), VAD, push-to-talk backup | Laptop |
| Speech to text | Audio clip in, text out | faster-whisper (small or distilled) | Vultr |
| LLM planner | Transcript + one frame + state -> JSON plan; replan on events | Vision-capable LLM API (Gemini preferred) | API |
| Perception server | Find target, disambiguate, track it, locate hand, estimate depth | Grounding DINO / OWLv2 / YOLO-World (benchmark), VLM disambiguator, SAM 2 video tracker, MediaPipe Hands, Depth Anything V2 | Vultr (hand pose may run on the laptop) |
| Guidance controller | Hand-to-target error -> 8 motor values | Plain Python, no model | Laptop |
| Orchestrator | Session state machine; runs plan steps; routes events to planner | Explicit enum state machine | Laptop |
| Audio out and reader | Speak audio steps; read printed text | Piper or API TTS; VLM or PaddleOCR + LLM | Mixed |

### Planner output (validated JSON, retry once, else speak "I did not understand, try again")

```json
{"steps": [
  {"type": "audio", "text": "Walk forward about 4 steps toward the desk."},
  {"type": "haptic", "action": "reach", "target": "sheet of paper", "refer": "the sheet of paper on the table"},
  {"type": "read", "region": "held_object"}
]}
```

- `reach(target, refer)` is the only haptic action. Grasp, lift and turn become audio steps.
- `target` is a 1-4 word noun phrase, no positions ("red soda can", "door handle") -- it goes straight into the open-vocabulary detector. `refer` is a fuller description used only to disambiguate between similar candidates ("the red can on the table"); position words in it may be stale if the user moved.
- Distances are spoken as "about", with a "say done when you arrive" confirmation.
- Ambiguity (three cups) becomes a spoken question.
- The planner must list hazards (stairs, glass, open drawer, person in path) before any walking instruction.
- The planner refuses haptic targets that are hot or sharp (stove burner, knife) and says so aloud.

### Object detection and tracking (settled)

When the user is within arm's reach of the target, the orchestrator calls `reach(target, refer)` on the
vision side, which finds that object in the glasses-camera frame and keeps it located while the glove
guides the hand. Frames are not mirrored: image left = user's left.

**Pipeline** (re-run from step 1 on re-acquire):
1. Open-vocab detector on `target`: Grounding DINO, OWLv2, or YOLO-World. All take a text query and need
   no training; a fixed-class YOLO can't cover arbitrary objects. Tradeoffs to verify: YOLO-World fastest,
   Grounding DINO best on descriptive phrases, OWLv2 good on short nouns but weak on relations. Benchmark
   on the demo objects and check licenses before committing.
2. Several candidates: draw numbered boxes and have a VLM (Qwen-VL class or Florence-2) pick one using
   `refer`, or have it ground `refer` directly. Runs once per reach, so a few hundred ms is fine.
3. Seed SAM 2's video tracker with the chosen box. SAM 2 runs every frame; the detector only runs at
   session start and on re-acquire.
4. Depth Anything V2 on small patches around the target and the hand, every few frames. Absolute values
   are noisy -- only ordering matters. It drives forward/back and proximity and is the weakest signal, so
   test it early.
5. No fine-tuning unless the pretrained models fail on our objects.

Per frame, the server reports: target box, tracking confidence, depth at target and hand, frame id, timestamps.

**Session contract**: `start(target, refer)` returns a session id, plus `pause`, `resume`, `cancel`.
`pause` zeroes the motors and freezes the loss timers. Resuming after more than ~3 s re-runs the detector.
Motors go to zero on every exit.

**End statuses** (starting thresholds, tune them):

| Status | Condition |
|---|---|
| `success` | a fingertip inside the box and hand depth near target depth for 5-10 frames in a row |
| `success`, `inferred: true` | the target vanished, but the hand covered over half its box in the last few frames (the hand hides the object at the end of a good reach) |
| `target_not_found` | no acquisition within ~3 s |
| `target_lost` | gone over ~2 s without that hand overlap |
| `hand_lost` | no hand for over ~1.5 s |
| `stalled` | the error hasn't shrunk in ~6-8 s |
| `timeout` | 25-30 s cap |
| `stale_link` | no results for 500 ms |

The report to the orchestrator also carries `elapsed_s`, `last_seen` (region in image thirds like
"right-middle", depth near/mid/far, `age_s`), `closest_approach` (touching/near/far), and the last frame
(omitted on success). Keep these status names identical everywhere (planner prompt, orchestrator, vision
loop, logs) -- they replace the earlier `step_done`/`user_stop` event sketch.

### Perception server (WebSocket)

- One Vultr GPU instance, one service, every model loaded once (detector, SAM 2, depth, Whisper), Docker on
  a CUDA base with weights baked in and a warm-up pass, behind HTTPS with a token check. fp16 small/base
  variants should fit a 24 GB GPU -- check which GPUs the MLH credits actually cover.
- One WebSocket per session. Client sends JPEG frames ~320-480 px wide with `frame_id` and a client timestamp.
- Server processes only the NEWEST frame and drops stale ones. Never build a queue; a queue makes the glove lag the hand.
- Network time = laptop round trip minus server time (only the laptop clock is used).
- Budget ~30-60 ms for tracking inside the ~200 ms capture-to-vibration target; log timestamps at every stage.
- Also run the pipeline laptop-only, to compare latency, frame rate, and detection quality for the Vultr prize write-up.
- Detector runs at session start and on re-acquire. Tracker (SAM 2) runs every frame. Hand pose every frame.
  Depth every frame or every 2nd-3rd frame on small patches.

Order of work: webcam + detector printing boxes, then SAM 2 + depth, then move to Vultr and measure latency.

### Guidance controller (pure rules, per frame)

- Errors, normalised: `dx = (target_cx - hand_x) / W`, `dy = (target_cy - hand_y) / H`, `dz` = normalised target depth minus hand depth.
- Depth Anything V2 gives relative depth (larger = closer), so normalise per frame (patch medians divided by the frame's 5th-95th percentile range). Verify the forward/back sign by hand-waving at the camera.
- Positive dx -> right channel; negative -> left. Positive dy (target lower in image) -> down; negative -> up. Target farther than hand -> forward.
- Wrist channel = proximity, stronger as total distance shrinks.
- Deadband per axis, hysteresis, EMA smoothing, and at most TWO direction channels active at once.
- Success itself is decided server-side (see "Object detection and tracking (settled)" above): the controller
  just drives motors from dx/dy/dz every frame; once the orchestrator gets a `success` `ReachReport`, it fires
  the confirm pulse and stops.
- Starting values (tune, not measured): deadband 0.05, hysteresis 0.03, EMA alpha 0.3. Keep them in `config/controller.yaml`.
- No hand in frame: zero all motors. Tracker mask empty for several frames: re-detect once; the session's
  own `hand_lost`/`target_lost` timeouts (config/settings.yaml: `reach_session`) are what actually end the reach.
- No fine-tuned or learned policy. Rules only.
- No hand-rotation compensation. The mapping is fixed in the camera's image frame; the user learns it.

## Haptics

### Channel order (the 8-value array, ints 0-255, ~20 Hz, BLE or WiFi: undecided)

`[thumb, index, middle, ring, pinky, top_of_hand, bottom_of_hand, wrist]` (indices 0-7)

### Cue-to-channel mapping is PROVISIONAL. Keep it in `config/mapping.yaml`, never hardcode it.

| Cue | Mapping A (proposed for first test) | Mapping B (earlier planning chat) |
|---|---|---|
| left | thumb | index |
| right | pinky | pinky |
| forward | index | middle |
| back | ring | ring |
| up | top of hand | top of hand |
| down | bottom of hand (palm) | bottom of hand (palm) |
| proximity | wrist | wrist |
| confirm | middle, short double pulse | thumb, short double pulse |

The design doc also lists a "teammate version" with no forward/back channels; it cannot guide depth, so it is not used.
Decide between A and B after the first blindfolded tester. Right hand only for the demo. The glove's physical wiring does not change between mappings.

### Safety and comfort (firmware + software)

- Failsafe: no packet for 300 ms -> all motors off. Do not resend stale values when vision stalls.
- "stop" (voice) cancels the step and zeroes all motors. All motors pulsing together is the haptic stop signal.
- Cap motor intensity and continuous duty cycle; check with testers that sustained vibration does not numb the hand.
- Firmware test mode: pulse one channel at a time. Software ships a fake stream (scripted sweep) so hardware never waits on vision.

### Hardware summary

- 8x 10 mm x 2.7 mm 3 V coin ERM motors, right-hand glove. Motors only spin one way.
- ESP32 DevKit (classic) on a forearm strap with a USB power bank (5V). Glove connects through a 10-pin plug at the wrist (8 signals + 2 shared returns).
- Two 16-pin half-H drivers (SN754410 and/or L293D, same pinout, both fed 5V). U1 drives channels 0-3, U2 drives channels 4-7.
- ESP32 GPIO assumed: thumb 13, index 27, middle 16, ring 17, pinky 18, top 19, bottom 21, wrist 22 (avoid strapping pins 0, 2, 5, 12, 15 and GPIO14). Change if the board is not a classic ESP32.
- 10k pulldown on every driver input line so motors stay off during boot. 470 uF across 5V/GND near the drivers.
- PWM ~20 kHz, 8-bit. Motors do not spin below ~2.3 V, so firmware maps 0 -> off and 1-255 -> roughly 60%-85% duty, with a per-channel max duty so the two driver chips feel matched. Flash the ESP32 with the main switch off.

## Latency

The headline number is hand movement -> felt change in vibration. Log timestamps at: camera capture, network up, detection/tracking, depth, controller, network down, motor update.
~200 ms capture-to-vibration (with ~30-60 ms of that budgeted for tracking) is the current proposal, not a
measured fact. Set the real target after the first end-to-end measurement and record it in the eval log.

## Proposed file structure

```
glance/
├── CLAUDE.md
├── README.md                    # needed for Devpost and the AI track
├── requirements.txt
├── config/
│   ├── mapping.yaml             # channel order, cue-to-channel (A and B)
│   ├── controller.yaml          # deadband, hysteresis, EMA, N frames, max channels
│   └── settings.yaml            # server URL, rates, thresholds, camera source
├── glance/                      # laptop-side Python package
│   ├── schemas.py               # dataclasses: Plan, Step, ReachCommand, FrameResult, ReachReport
│   ├── telemetry.py             # one JSON line per frame and per event
│   ├── orchestrator/
│   │   ├── state_machine.py     # idle, listening, planning, audio step, haptic step, replanning
│   │   ├── wake_word.py         # openWakeWord + VAD + push-to-talk
│   │   └── audio_out.py         # TTS, hazard warnings interrupt speech
│   ├── planner/
│   │   ├── planner.py           # LLM call, schema validation, one retry
│   │   └── prompts/
│   ├── vision/
│   │   ├── camera.py            # frame source (webcam, phone stream)
│   │   ├── client.py            # WebSocket client to the Vultr server
│   │   ├── loop.py              # sends frames, drives haptics per FrameResult, surfaces the ReachReport
│   │   ├── controller.py        # error -> 8 values
│   │   └── fake.py              # webcam-only / canned results, no server needed
│   ├── haptics/
│   │   ├── link.py              # sends the 8-value array at ~20 Hz
│   │   └── fake_stream.py       # scripted sweep for the hardware team
│   └── reader/
│       └── reader.py            # scene 3
├── server/                      # runs on the Vultr GPU instance
│   ├── app.py                   # FastAPI + WebSocket, /health endpoint
│   ├── config.py                # tiny shared YAML loader (resolves config/ from repo root)
│   ├── session.py               # ReachSession state machine: ACQUIRING -> TRACKING -> DONE
│   ├── timing.py                # per-stage timers returned to the client
│   └── models/
│       ├── stt.py               # faster-whisper
│       ├── detector.py          # Grounding DINO / OWLv2 / YOLO-World (benchmark)
│       ├── tracker.py           # SAM 2 video tracker, seeded by the detector box
│       ├── hand.py              # MediaPipe Hands (can also run on the laptop)
│       ├── depth.py             # Depth Anything V2 small
│       └── reader.py
├── firmware/glove/              # ESP32 sketch: BLE/WiFi in, PWM out, 300 ms failsafe, test mode
├── eval/
│   ├── run_trial.py
│   ├── analyze.py
│   └── logs/                    # JSONL per trial, every number in the write-up traces here
├── scripts/
│   └── benchmark_latency.py
└── tests/
```

Do not name a package `logging` (it shadows the standard library); telemetry lives in `telemetry.py`.

## Working conventions

- Python 3. Dataclasses in `schemas.py` are the single source of truth for messages; do not pass loose dicts between modules.
- Each model sits behind a small interface (input -> output) so it can be swapped or stubbed. Every model has a fake or fallback.
- Build and test on a laptop webcam first (print dx, dy, depth) before moving any model to Vultr.
- Move models to Vultr one at a time; record GPU memory and per-call latency after each.
- Hardcode the plan (e.g. `reach("sheet of paper", "the sheet of paper on the table")`) until the loop works, then swap in the real planner.
- Config over constants: mapping, thresholds and URLs live in `config/`.
- Log one JSON line per frame and per event (transcript, plan, step transitions, intensities sent) so any trial can be replayed.
- Keep it simple and demo-reliable. This is a 36-hour build; prefer boring, explicit code over abstractions.
- Verify model IDs and library APIs on Hugging Face or the project docs before relying on them (candidates: `google/owlv2-base-patch16-ensemble`, `IDEA-Research/grounding-dino-tiny`, `depth-anything/Depth-Anything-V2-Small-hf`; SAM 2 ships as Meta's `sam2` package). Note OWLv2 boxes are relative to a padded square image, so convert box coordinates in one tested helper.

## Team and ownership

- Computer vision (detector, tracker, hand, depth, controller, vision loop): one teammate.
- LLM planner: one teammate.
- Glove, motors, firmware, haptic link: the hardware teammates.
- Orchestrator (state machine, step runner, event routing) and the reader: NOT ASSIGNED YET. Assign at the start.

## Open decisions

- Detector: Grounding DINO vs OWLv2 vs YOLO-World -- benchmark on the demo objects and check licenses before committing.
- Wake word: pretrained "hey jarvis" for the weekend, or a custom name (stretch). The planner prompt and demo script must use the same phrase.
- Finger mapping A vs B (decide after the first blindfolded test).
- Camera path from head-worn camera to laptop: phone camera, USB, or WiFi stream.
- Haptic transport: BLE or WiFi.
- Planner model: Gemini (eligible for the MLH Gemini prize) or another vision-capable LLM.
- Hand pose: local or on Vultr (decided by the first latency measurement).
- Demo objects and table layout (pick matte objects like paper and a matte can; shiny or thin objects detect poorly).
- Whether the team meets the First Timers requirement (at least 50% first-time hackers).

## Risks to keep in mind

Network latency in the loop; noisy monocular depth (fallback: audio for depth, glove for left/right/up/down); the hand occluding the target at the final reach (rely on tracker memory and hand-box overlap); MediaPipe struggling with a gloved hand from a head-worn view (fallback: bright marker on the glove tracked by color); bad planner JSON (schema check, retry, hardcoded plan); venue noise breaking the wake word (push-to-talk).

## Out of scope for the weekend

3D mapping, hand-rotation compensation, grasping or turning actions, outdoor use, left-hand glove, any safety-critical claim.
