# Glance

A voice-controlled assistant for blind users, built for **SwampHacks XII**. A head-worn camera sees what the user
faces, they talk to it through a mic and headphones, and a glove with vibration motors guides their hand
during the last arm's length of a reach ("hey glance, get me the can").

This is a hackathon prototype. It supplements a cane or guide dog and makes no safety claims.

[CLAUDE.md](CLAUDE.md) is the integration spec (how the components are wired together, step by step).
Each package keeps its own spec or README.

[LIVE_TESTS.md](LIVE_TESTS.md) is the ordered end-to-end live test list (servers, audio, hand band, depth
calibration, guidance, voice agent, real reaches, hardware glove, Pi).

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
| `runtime/` | Integration glue: camera hub, live view, glove output, `voice_agent.py`, `live_guidance.py`, `reach_haptic.py`, `glove_check.py`, `selftest.py`, Pi install | `CLAUDE.md` |
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

.venv\Scripts\python -m runtime.voice_agent --ptt-only         # Step 1: voice + agent + camera, keyboard reach
.venv\Scripts\python -m runtime.voice_agent                    # same with the wake word (needs headphones)
.venv\Scripts\python -m runtime.voice_agent --typed            # type instead of talking; real TTS and camera
.venv\Scripts\python -m runtime.latency                        # end of speech -> first audio, last session
.venv\Scripts\python -m runtime.live_guidance --target "red can"  # Step 2: vision + depth + glove commands, no agent
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

## The full system (agent + voice + real reaches)

1. **GPU host:** start the STT server (port 8001) and the vision server (port 8000) as above, each in its own
   venv, with the same tokens the laptop will use.
2. **Laptop, repo root:**

   ```powershell
   $env:GLANCE_STT_URL = "http://100.112.104.87:8001"; $env:GLANCE_STT_TOKEN = "<token>"
   $env:GLANCE_VISION_URL = "ws://100.112.104.87:8000/ws"; $env:GLANCE_TOKEN = "<token>"
   .venv\Scripts\python -m audio.cli stt-check                     # STT reachable
   .venv\Scripts\python -m object_detection.client --health        # vision reachable, models loaded, depth on
   .venv\Scripts\python -m runtime.voice_agent --reach real --ptt-only    # speakers: Enter to talk
   .venv\Scripts\python -m runtime.voice_agent --reach real              # headphones: "hey glance"
   ```

   The view shows the camera, what was heard and said, the task and step, the reach (target box, hand point,
   arrow, command words, dx/dy/dz, distance, touching/frozen) and the fake glove's motors. `m` toggles
   table/wall guidance, `q` quits. Per-frame reach timings go to `runtime/logs/voice-reach-*.jsonl` and a
   summary prints on exit; `python -m runtime.latency` gives end of speech to first reply.
3. **Without the agent** (tune vision, depth and guidance): `python -m runtime.live_guidance --target "red can"`.
4. **Without a GPU** (wiring only): run the vision server locally with `$env:GLANCE_FAKE_MODELS = "1"` and
   point `GLANCE_VISION_URL` at `ws://127.0.0.1:8000/ws`.

## Raspberry Pi (worn by the user)

```bash
git clone <repo> ~/Glance && cd ~/Glance && git checkout integration
bash runtime/scripts/install_pi.sh            # apt deps, .venv (no torch), openwakeword --no-deps, models
. .venv/bin/activate
nano runtime/glance.env                       # server URLs, tokens, GEMINI_API_KEY (made by the install)
python -m runtime.selftest                    # env, camera, STT, vision, glove, audio: PASS/WARN/FAIL
bash runtime/scripts/run_glance.sh --ptt-only # everything; the live view is at http://<pi>:8090/ (web is
                                              # the default without a display); keys: a letter + Enter
```

Per-machine settings (camera index or `picam` for the camera module, glove port) go in
`runtime/config.local.yaml` (gitignored); the install script creates one with `camera.source: 0`.

## Hardware glove

The glove is the Feather ESP32-S3 running `firmware/glove/code.py` (CircuitPython, 6 motors: thumb, index,
ring, pinky, back of hand, palm), on USB serial (`haptics/link.py`, protocol in `haptics/protocol.py`). Motors
are on or off, one per direction; the active direction motors pulse faster as the hand gets closer (1 Hz far,
6 Hz near; CLAUDE.md decision 24). See `firmware/glove/README.md` for flashing. The port is found automatically (`port: auto`).

```bash
python -m runtime.glove ports                       # which serial port is the glove
python -m runtime.glove test                        # firmware test: each motor once
python -m runtime.glove buzz LEFT                   # one direction through the runtime's mapping
python -m runtime.glove sweep                       # a simulated reach: guidance -> glove, no camera or server
python -m runtime.glove_check --glove hardware      # each motor in turn, with the live view
python -m runtime.voice_agent --reach real --glove hardware
python -m runtime.live_guidance --target "red can" --glove hardware
python3 firmware/glove/send_test.py /dev/ttyACM0 test   # the teammate's raw-frame bench test, unchanged
```

Or set `glove: {kind: hardware}` in `runtime/config.local.yaml`. The live view still draws the motors,
mirroring what the glove was told, with the ones this glove lacks crossed out.

## Run the whole thing

```bash
python -m runtime.run --ptt-only        # or: bash runtime/scripts/run_glance.sh --ptt-only   (Pi)
```

Loads `runtime/glance.env` (copy `runtime/glance.env.example`: server URLs, tokens, `GEMINI_API_KEY`), uses
the hardware glove if one is plugged in (fake otherwise), runs `runtime.selftest` and stops on any FAIL
(`--force` to go on), then starts `runtime.voice_agent --reach real`. Without `--ptt-only` it listens for
"hey glance" (headphones).

## Out of scope for the weekend

3D mapping, hand-rotation compensation, grasping or turning actions, outdoor use, left-hand glove, any
safety-critical claim.
