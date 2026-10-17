# Glance

A voice-controlled assistive prototype for blind and low-vision users, built for **SwampHacks XII** (Oct 16-18, 2026).

A head-worn camera sees the room. The user speaks a request ("guide me to the paper"). Speech handles coarse
navigation and reading; a vibrating glove (8 motors) handles the last arm's length of reaching.

This is a hackathon prototype. It supplements a cane or guide dog and makes no safety claims.

## The demo

Three voice scenes, one table, one sheet of paper:

1. **Identify** — "what is on the desk?" -> detector over the scene -> speech ("a piece of paper, about 3 steps ahead and slightly left").
2. **Guide** — "guide me to the paper" -> audio step (walk), then haptic reach step until the hand touches the paper, then a confirm pulse.
3. **Read** — "read this paper" -> close frame -> OCR or vision-language read -> speech. No glove.

## Architecture

```
 WORN BY USER            LAPTOP                         CLOUD
 camera ───────────────► orchestrator (state machine) ─► Vultr GPU (one instance, all per-frame models)
 mic ──► wake word ─────►   │  ▲ events                    whisper, detector, tracker, hand, depth, reader
 headphones ◄── speech ◄────┘  │                         LLM planner API (only call off the Vultr path;
 ESP32 + drivers ◄── BLE/WiFi ◄ vision loop + controller     called at task start and on events, never per frame)
   └─► 8 coin motors
```

See [CLAUDE.md](CLAUDE.md) for the full design doc: components, message schemas, the guidance controller,
haptic channel mapping, latency plan, file structure, and team ownership.

## Getting started

```bash
pip install -r requirements.txt
```

Build and test on a laptop webcam first (`glance/vision/fake.py`) before moving any model to Vultr. The
perception server (`server/`) runs on the Vultr GPU instance, reached over an SSH tunnel.

## Layout

| Path | What's there |
|---|---|
| `glance/` | Laptop-side package: orchestrator, planner client, vision loop + controller, haptic link, reader |
| `server/` | FastAPI/WebSocket perception server + per-frame models, runs on the Vultr GPU instance |
| `firmware/glove/` | ESP32 sketch: BLE/WiFi in, PWM out, 300 ms failsafe, test mode |
| `config/` | `mapping.yaml`, `controller.yaml`, `settings.yaml` — tunables, never hardcoded |
| `eval/` | Trial runner + analysis for the blindfolded glove-vs-audio comparison |
| `scripts/` | Latency benchmarking |
| `tests/` | Unit tests |

## Team

- Computer vision (detector, tracker, hand, depth, controller, vision loop)
- LLM planner
- Glove, motors, firmware, haptic link
- Orchestrator + reader (not assigned yet)

## Out of scope for the weekend

3D mapping, hand-rotation compensation, grasping or turning actions, outdoor use, left-hand glove, any
safety-critical claim.
