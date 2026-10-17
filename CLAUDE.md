# Glance

A voice-controlled assistive prototype for blind and low-vision users, built for SwampHacks XII (Oct 16-18, 2026).
A head-worn camera sees the room. The user speaks a request ("guide me to the paper"). Speech handles coarse
navigation and answers; a vibrating glove (8 motors) handles the last arm's length of reaching.

This is a hackathon prototype. It supplements a cane or guide dog and makes no safety claims. Never add
language to the code, README or demo that says it keeps a user safe.

Targets: First Timers track, Best Use of Vultr (MLH). Also eligible: Hardware, AI, Gemini (if the agent uses Gemini).
Hacking ends Sun Oct 18, 8:00 AM, which is also the Devpost deadline.

## Priorities (in order)

1. A first working reach (hand guided to an object) by Saturday ~2 PM. Everything else waits for this.
2. Per-stage latency numbers (the Vultr story).
3. A small blindfolded trial: glove guidance vs audio-only baseline (3 testers x 5 trials x 2 conditions).
4. Everything stretch.

Freeze scope after the first working reach. Do not add features before then.

## The demo (three voice scenes, one table, one sheet of paper)

1. Identify: "what is on the desk?" -> the agent answers from the camera frame.
2. Guide: "guide me to the paper" -> spoken steps to get within arm's reach, then a haptic reach until the hand
   touches the paper, then a touch check.
3. Read: "read this paper" -> the agent reads the text from a close frame. No glove.

Scenes 1 and 3 are plain questions to the agent; there is no separate detector-over-the-scene or OCR path.

## Architecture

```
 WORN BY USER            LAPTOP (runtime/)                    CLOUD
 camera ───────────────► agent (controller + LLM turns) ────► Vultr GPU (server/, one instance, all per-frame models)
 mic ──► wake word ─────►   │  ▲ haptic_result                  whisper, detection, depth, hand (or local)
 headphones ◄── speech ◄────┘  │                             LLM API (Gemini; one call per turn, never per frame)
 ESP32 + drivers ◄── BLE/WiFi ◄ vision loop + guidance controller
   └─► 8 coin motors
```

- The laptop owns the session and the glove link. Camera, headphones and mic connect to the laptop, never to the glove.
- One Vultr GPU instance serves every per-frame model behind a single streaming (WebSocket) service.
- The LLM is NOT in the per-frame loop. It runs once per turn (user utterance, haptic result, idle timeout).
- Network path is the main risk. Fallbacks: phone hotspot, smaller frames, run hand pose locally.

## Repo layout and ownership

| Dir | What | Owner | Read |
|---|---|---|---|
| `agent/` | LLM brain: controller, tools, system prompt, model adapter, evals | LLM agent | `agent/CLAUDE.md` |
| `detection/` | Open-vocab detector, tracker, depth. Pure functions, frame in, results out | object detection | `detection/CLAUDE.md` |
| `server/` | Vultr FastAPI/WebSocket service, `ReachSession`, STT | object detection | `detection/CLAUDE.md` |
| `hand/` | Hand localization (MediaPipe, marker fallback) | hand detection | `hand/CLAUDE.md` |
| `haptics/` | Guidance controller (error -> 8 motor values), glove link, fake stream | hardware | `haptics/CLAUDE.md` |
| `firmware/glove/` | ESP32 sketch | hardware | `haptics/CLAUDE.md` |
| `runtime/` | Laptop glue: camera, wake word, speech out, perception client, vision loop. Implements the agent's interfaces | unassigned | this file |
| `shared/` | Dataclasses (`schemas.py`), config loader, telemetry. Change with care, everyone imports it | everyone | this file |
| `config/` | YAML tunables, never hardcode these | everyone | this file |
| `eval/`, `scripts/` | Blindfolded trial runner and analysis, latency benchmark | unassigned | this file |

Who may import whom:
- `agent/` imports nothing else in this repo. It talks to the rest only through the interfaces in `agent/CLAUDE.md`
  section 5, and `runtime/` provides the adapters for them.
- `detection/` and `hand/` are pure functions (frame bytes in, results out). No FastAPI, no imports from `server/`,
  `runtime/` or `haptics/`.
- `server/` composes `detection/` and `hand/`.
- `runtime/` uses `haptics/`, `shared/` and the perception server's WebSocket protocol.
- `shared/` imports nothing else in this repo.

Where the contracts live:
- Agent <-> rest of the system (speech, camera, haptic start/pause/resume/cancel, `haptic_result`): `agent/CLAUDE.md`
  sections 5 and 10.
- Reach session (WebSocket protocol, end statuses, `ReachReport`): `detection/CLAUDE.md`, types in `shared/schemas.py`.
- Motor array and cue-to-channel mapping: `haptics/CLAUDE.md` and `config/mapping.yaml`.

Run everything from the repo root (for example `uvicorn server.app:app`). Each component has its own
`requirements.txt`; install only the ones you need.

## Latency

The headline number is hand movement -> felt change in vibration. Log timestamps at: camera capture, network up, detection/tracking, depth, controller, network down, motor update.
~200 ms capture-to-vibration (with ~30-60 ms of that budgeted for tracking) is the current proposal, not a
measured fact. Set the real target after the first end-to-end measurement and record it in the eval log.

## Working conventions

- Python 3. Dataclasses in `shared/schemas.py` are the single source of truth for messages between components; do not pass loose dicts between them. The agent keeps its own types behind its interfaces.
- Each model sits behind a small interface (input -> output) so it can be swapped or stubbed. Every model has a fake or fallback.
- Build and test on a laptop webcam first (print dx, dy, depth) before moving any model to Vultr.
- Move models to Vultr one at a time; record GPU memory and per-call latency after each.
- Hardcode the reach (for example `reach("sheet of paper", "the sheet of paper on the table")`) until the loop works, then let the agent drive it.
- Config over constants: mapping, thresholds and URLs live in `config/`.
- Log one JSON line per frame and per event (transcript, steps, intensities sent) so any trial can be replayed.
- Keep it simple and demo-reliable. This is a 36-hour build; prefer boring, explicit code over abstractions.
- Do not name a package `logging` (it shadows the standard library); telemetry lives in `shared/telemetry.py`.
- Component docs describe their own component only. If a change affects another component's contract, update that doc in the same commit and tell its owner.

## Open decisions

- Detector: Grounding DINO vs OWLv2 vs YOLO-World -- benchmark on the demo objects and check licenses before committing.
- Wake word: pretrained "hey jarvis" for the weekend, or a custom name (stretch). The agent's `assistant_name` and the demo script must use the same phrase.
- Finger mapping A vs B (decide after the first blindfolded test). `config/mapping.yaml` has `active_mapping` set to `a`.
- Camera path from head-worn camera to laptop: phone camera, USB, or WiFi stream.
- Haptic transport: BLE or WiFi.
- Agent model provider: Gemini is the default (`agent/CLAUDE.md` section 4; eligible for the MLH Gemini prize).
- Hand pose: local or on Vultr (decided by the first latency measurement).
- Demo objects and table layout (pick matte objects like paper and a matte can; shiny or thin objects detect poorly).
- Whether the team meets the First Timers requirement (at least 50% first-time hackers).

## Risks to keep in mind

Network latency in the loop; noisy monocular depth (fallback: audio for depth, glove for left/right/up/down); the hand occluding the target at the final reach (rely on tracker memory and hand-box overlap); MediaPipe struggling with a gloved hand from a head-worn view (fallback: bright marker on the glove tracked by color); venue noise breaking the wake word (push-to-talk).

## Out of scope for the weekend

3D mapping, hand-rotation compensation, grasping or turning actions, outdoor use, left-hand glove, any
safety-critical claim.
