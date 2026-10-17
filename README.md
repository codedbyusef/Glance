# Glance

A voice-controlled assistive prototype for blind and low-vision users, built for **SwampHacks XII** (Oct 16-18, 2026).

A head-worn camera sees the room. The user speaks a request ("guide me to the paper"). Speech handles coarse
navigation and answers; a vibrating glove (8 motors) handles the last arm's length of reaching.

This is a hackathon prototype. It supplements a cane or guide dog and makes no safety claims.

## The demo

Three voice scenes, one table, one sheet of paper:

1. **Identify** — "what is on the desk?" -> the agent answers from the camera frame.
2. **Guide** — "guide me to the paper" -> spoken steps, then a haptic reach until the hand touches the paper, then a touch check.
3. **Read** — "read this paper" -> the agent reads the text from a close frame. No glove.

## Architecture

```
 WORN BY USER            LAPTOP (runtime/)                    CLOUD
 camera ───────────────► agent (controller + LLM turns) ────► Vultr GPU (server/, one instance, all per-frame models)
 mic ──► wake word ─────►   │  ▲ haptic_result                  whisper, detection, depth, hand (or local)
 headphones ◄── speech ◄────┘  │                             LLM API (Gemini; one call per turn, never per frame)
 ESP32 + drivers ◄── BLE/WiFi ◄ vision loop + guidance controller
   └─► 8 coin motors
```

See [CLAUDE.md](CLAUDE.md) for priorities, ownership and import rules. Each component has its own doc.

## Layout

| Path | What's there | Doc |
|---|---|---|
| `agent/` | LLM brain: controller, tools, prompt, model adapter, evals | `agent/CLAUDE.md` |
| `detection/` | Open-vocab detector, tracker, depth (pure functions) | `detection/CLAUDE.md` |
| `server/` | FastAPI/WebSocket perception server, `ReachSession`, STT (runs on Vultr) | `detection/CLAUDE.md` |
| `hand/` | Hand localization | `hand/CLAUDE.md` |
| `haptics/` | Guidance controller, glove link, fake stream | `haptics/CLAUDE.md` |
| `firmware/glove/` | ESP32 sketch | `haptics/CLAUDE.md` |
| `runtime/` | Laptop glue: camera, wake word, audio out, perception client, vision loop | `CLAUDE.md` |
| `shared/` | Dataclasses, config loader, telemetry | `CLAUDE.md` |
| `config/` | `mapping.yaml`, `controller.yaml`, `settings.yaml` — tunables, never hardcoded | |
| `eval/`, `scripts/` | Blindfolded trial runner and analysis, latency benchmark | |
| `tests/` | Unit tests | |

## Getting started

Run everything from the repo root. Install only what your component needs:

```bash
pip install -r detection/requirements.txt   # detector, tracker, depth
pip install -r hand/requirements.txt        # hand localization
pip install -r server/requirements.txt      # perception server (includes the two above)
pip install -r runtime/requirements.txt     # laptop side
pip install -r eval/requirements.txt        # trial analysis
# agent/ has its own setup, see agent/CLAUDE.md
```

Build and test on a laptop webcam first (`runtime/fake_perception.py`) before moving any model to Vultr.

```bash
uvicorn server.app:app   # perception server, on the Vultr GPU instance
```

## Out of scope for the weekend

3D mapping, hand-rotation compensation, grasping or turning actions, outdoor use, left-hand glove, any
safety-critical claim.
