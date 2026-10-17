# Glance

A voice-controlled assistant for blind users, built for **SwampHacks XII**. A head-worn camera sees what the user
faces, they talk to it through a mic and headphones, and a glove with vibration motors guides their hand
during the last arm's length of a reach ("hey glance, get me the can").

This is a hackathon prototype. It supplements a cane or guide dog and makes no safety claims.

[CLAUDE.md](CLAUDE.md) is the integration spec (how the components are wired together, step by step).
Each package keeps its own spec or README.

## Architecture

```
 worn by the user          client: laptop now, Raspberry Pi at the demo           GPU host: PC (RTX 3070) or Vultr
 camera ──────────────► camera thread ──► agent (guide_agent) ──► Gemini API
 mic ─────────────────► audio (wake word, VAD) ──HTTP──────────────────────────► STT server (audio.server, :8001)
 headphones ◄────────── audio (Piper TTS) ◄── agent
 glove (ESP32) ◄─ USB ─ haptics ◄── guidance ◄── ReachAPI ──WebSocket──────────► vision server (object_detection, :8000)
                                                                                  detect + track + hand band + depth
```

Clients reach the servers by URL only (`GLANCE_STT_URL`, `GLANCE_VISION_URL`), so moving from the PC to Vultr
changes env vars, nothing else.

## Layout

Every component is a Python package importable from the repo root. Each one has its own `config.yaml` (found
relative to its own file), its tests in `<package>/tests/`, and its runnable tools in `<package>/scripts/`.

| Package | What | Doc |
|---|---|---|
| `guide_agent/` | LLM controller: triggers, intercepts, task state, tools, Gemini adapter, harness, evals | `guide_agent/CLAUDE.md`, `README.md` |
| `audio/` | Wake word, VAD, push-to-talk, STT client + server, Piper TTS player | `audio/CLAUDE.md`, `README.md` |
| `object_detection/` | Grounding DINO + SAM 2.1, reach session, WebSocket vision server, client `ReachAPI` | `object_detection/README.md` |
| `hand_detection/` | Green tape band tracker (the hand point) | `hand_detection/README.md` |
| `depth_estimation/` | Depth Anything V2 Small + `DepthTracker` | `depth_estimation/README.md` |
| `haptics/` | `GuidanceController` (pixels -> motor commands), glove link, wire protocol | `haptics/CLAUDE.md` |
| `runtime/` | Integration glue (from Step 1). For now: the client requirements file | `CLAUDE.md` |
| `firmware/glove/` | ESP32 sketch | `haptics/CLAUDE.md` |

## Requirements by machine role

`pyproject.toml` declares the packages only, no dependencies. Each machine installs the file for its role, then
`pip install -e .`:

| Role | Machine | Venv | Requirements |
|---|---|---|---|
| Client (agent, audio, camera, glove) | laptop, Pi (no torch) | `.venv` | `runtime/requirements.txt`, then `pip install --no-deps openwakeword==0.6.0` |
| Unit tests | any | `.venv` | `requirements-dev.txt` (client + pytest, fastapi, uvicorn) |
| Vision server | GPU PC / Vultr | `.venv-vision` | torch CUDA build first, then `object_detection/requirements-server.txt` |
| STT server | GPU PC / Vultr | `audio\.venv` on the PC | `audio/requirements-server-windows.txt` (Windows) or `audio/requirements-server.txt` |

Keep one OpenCV wheel per venv: `opencv-python` on the client, `opencv-contrib-python` on the vision server.
The STT and vision servers run as separate processes in separate venvs (faster-whisper and PyTorch ship
conflicting CUDA DLLs on Windows).

## Laptop: setup and unit tests (PowerShell, repo root)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\pip install --no-deps openwakeword==0.6.0
.venv\Scripts\pip install -e .
.venv\Scripts\python audio\scripts\fetch_models.py      # Piper voice, wake models, tiny.en
.venv\Scripts\python -m pytest                          # every package; no hardware, GPU, network or keys
.venv\Scripts\python -m pytest object_detection         # one package
```

Secrets come from env vars only: `GEMINI_API_KEY` (or `guide_agent/.env`), `GLANCE_STT_TOKEN`, `GLANCE_TOKEN`.

## GPU host: servers (PowerShell, repo root)

STT server (port 8001):

```powershell
$env:GLANCE_STT_TOKEN = "<token>"; $env:GLANCE_STT_MODEL = "small.en"; $env:GLANCE_STT_DEVICE = "cuda"
audio\.venv\Scripts\python -m uvicorn audio.server.stt_service:app --host 0.0.0.0 --port 8001
```

Vision server (port 8000), in its own venv:

```powershell
python -m venv .venv-vision
.venv-vision\Scripts\python -m pip install --upgrade pip
.venv-vision\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv-vision\Scripts\pip install -r object_detection\requirements-server.txt
.venv-vision\Scripts\pip install -e .
$env:GLANCE_TOKEN = "<token>"
.venv-vision\Scripts\python -m uvicorn object_detection.server:app --host 0.0.0.0 --port 8000 --ws-ping-interval 5
```

Add `$env:GLANCE_FAKE_MODELS = "1"` for canned models (wiring checks, no GPU). On Vultr the vision server runs
from `object_detection/Dockerfile` via `object_detection/deploy/`.

## Running the parts (laptop, repo root)

```powershell
$env:GLANCE_STT_URL = "http://100.112.104.87:8001"; $env:GLANCE_STT_TOKEN = "<token>"
$env:GLANCE_VISION_URL = "ws://100.112.104.87:8000/ws"; $env:GLANCE_TOKEN = "<token>"

.venv\Scripts\python -m guide_agent.harness                     # agent, typed, stubbed I/O
.venv\Scripts\python -m guide_agent.scripts.webcam_chat         # agent, typed, live camera
.venv\Scripts\python -m guide_agent.evals.run --case halt --runs 1
.venv\Scripts\python -m audio.cli selftest                      # mic, speaker, STT server
.venv\Scripts\python -m audio.cli echo                          # voice loop without the agent
.venv\Scripts\python -m object_detection.client --health        # vision server health
.venv\Scripts\python -m object_detection.reach_api --target "sheet of paper"
.venv\Scripts\python -m hand_detection.scripts.tune_marker      # pick the band's color range
.venv\Scripts\python -m hand_detection.scripts.test_hand_live
.venv\Scripts\python -m haptics.fake_stream --port COM5 test    # glove, no vision
```

The depth live test and benchmark (`python -m depth_estimation.scripts.test_depth_live`, `...benchmark_depth`)
need torch for the real model, so run them in `.venv-vision` (or with `GLANCE_FAKE_MODELS=1`).

## Out of scope for the weekend

3D mapping, hand-rotation compensation, grasping or turning actions, outdoor use, left-hand glove, any
safety-critical claim.
