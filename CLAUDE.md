# Glance: integration spec (repo root)

## 1. What this file is for

Glance is a voice-controlled assistant for blind users:
- a head-worn camera, a mic and headphones
- a glove with vibration motors that guides the hand during the last arm's length of a reach

Built for SwampHacks XII. Hacking ends **Sun Oct 18, 8:00 AM**.

The components were built and tested in isolation by different people. This file covers **wiring them together, one step at a time** (section 7). Each component keeps its own spec, and those specs win for anything inside their folders.

| Folder | State | Entry points used by integration | Its spec |
|---|---|---|---|
| `guide_agent/` (was `agent/`) | 95 controller tests, Gemini adapter, harness, eval, `scripts/webcam_chat.py` | `guide_agent.controller.Controller`; `interfaces.py` (`AudioOut`, `HapticLoop`, `Camera`); `schemas.py` (`Frame`, `HapticResult`, `LastSeen`); `stubs.py` | `guide_agent/CLAUDE.md` (sections 5, 10, 13, 19) |
| `audio/` | 176 tests; tested live on the laptop with the GPU STT server | `audio.player.Player` (implements `AudioOut`); `audio.listen.Mic` (sink = the controller); `audio.cli` | `audio/CLAUDE.md` (sections 5, 6, 15, 17) |
| `object_detection/` (was `object-detection/`) | Grounding DINO + SAM 2.1 tracking, the reach session with every end status, the WebSocket server (`/ws`, `/health`) and the client-side `ReachAPI`. Tested with fake models and on a 3060 Ti. The hand tracker is already called inside the session. **Depth isn't wired.** | `reach_api.ReachAPI(on_result, frame_source, on_progress)`; `server.py`; `session.ReachSession(hand=, depth=)` | `object_detection/README.md` ("Agent integration"), `object_detection/CLAUDE.md` |
| `hand_detection/` (was `hand-detection/`) | Green tape band tracker (no MediaPipe) | `hand_marker.detect`, `marker_mask`, `reset` | `hand_detection/README.md` |
| `depth_estimation/` (was `depth-estimation/`) | Depth Anything V2 Small + `DepthTracker`. Unit-tested on synthetic maps; the real model hasn't been run yet | `depth_model.DepthModel`, `DepthCache`; `depth_tracker.DepthTracker.update(...)` | `depth_estimation/README.md` ("Interface") |
| `haptics/` | `GuidanceController` (pixels -> 8 motor levels, table or wall mode), `GloveLink`, protocol | `haptics.controller`, `haptics.protocol` | `haptics/CLAUDE.md` |
| `firmware/glove/` | ESP32 sketch | Untouched until step 4 | |

## 2. Working rules

- **One step at a time, in order** (section 7):
  1. Before coding a step, propose a short plan and wait for my approval.
  2. Finish the step and run the unit tests.
  3. Give me the live-test commands and **stop for my review**.

  Don't start the next step until I say so.
- **New integration code goes in a new `runtime/` package** at the repo root.
  - **Reuse existing component code. Don't rebuild it.** For example, the vision server, reach session and `ReachAPI` already exist in `object-detection/`.
  - Component folders may be reorganized for consistency and cohesion (Step 0), and changed in later steps when that is cleaner than an adapter. Keep every component's tests passing, keep the contracts below, and list every change outside `runtime/` in the stop report.
- **Contracts are fixed. Ask before changing any of these:**
  - the agent's tool names, the `HapticResult` schema, and the `HapticLoop` and `AudioOut` interfaces
  - the audio `AudioEvents` sink and its `dispatch` rule
  - the `ReachAPI` interface
  - the depth and hand APIs
- **Imports.** After Step 0, every component is a real Python package importable from the repo root, with package-qualified imports (no `sys.path` hacks, no flat imports of generic names like `client`, `session`, `common`, `fake`, `server`, `config`).
- **Every live test script shows a live camera feed** with its overlays (section 6).
- **Unit tests run without hardware or credentials:** no camera, mic, speaker, GPU, network, model weights or API key. Use fakes for everything, as each component already does (`GLANCE_FAKE_MODELS=1` for object detection and depth).
- **Secrets come from env vars:** `GEMINI_API_KEY` (in `agent/.env`), `GLANCE_STT_TOKEN`, `GLANCE_TOKEN`. Never log them, or image bytes.
- **Simplest thing that works.** The demo is tomorrow.
- **Commits.** Commit at the end of each step on the branch `integration`, with a clear message. Push when I confirm the step.
- **Ask instead of guessing** when the specs are silent or contradict each other. Record every decision in section 9.

## 3. Machines and environments

| Machine | Role now | Role at the demo |
|---|---|---|
| Windows laptop (Python 3.12) | Runs `runtime/` (agent, audio, camera, guidance, glove output). Camera: Logitech Brio, usually index 1 (index 0 is the built-in webcam) | Fallback runner if the Pi fails |
| Windows PC, RTX 3070 (Tailscale `100.112.104.87`) | GPU stand-in for Vultr: STT server on port **8001** (running), vision server (`object-detection/server.py`) on port **8000** | Same, unless Vultr is up |
| Raspberry Pi 5 | Not in use yet | Runs `runtime/` worn by the user |
| Vultr GPU | Not set up | Same servers as the PC; only the URLs change |

- **Servers run as separate processes with separate venvs:**
  - STT: `audio/.venv`.
  - Vision: its own venv, with the PyTorch CUDA build.

  faster-whisper (ctranslate2) and PyTorch each bring their own CUDA and cuDNN DLLs, and mixing them in one Windows process is a known source of load errors.
- **Clients reach servers by URL only:**
  - `GLANCE_STT_URL=http://100.112.104.87:8001`
  - `GLANCE_VISION_URL=ws://100.112.104.87:8000/ws` (object-detection's existing variable)

  Moving to Vultr changes env vars, nothing else.
- **No torch on the laptop or Pi side.** The runtime venv is `.venv` at the repo root, from `runtime/requirements.txt`. Write that file from what is actually imported. Don't `-r` component files that pull in torch.
- **One OpenCV wheel per venv.** `object-detection` lists `opencv-contrib-python`, while `hand-detection` and `depth-estimation` list `opencv-python`. Installing both breaks `cv2`. On the vision server, install only `opencv-contrib-python`.

## 4. Data flow (end state)

```
 camera hub ──► Frame (JPEG, <=1024 px) ──► agent (on utterance / haptic result)
      │
      └──► BGR ──► ReachAPI ──WebSocket──► vision server (GPU): detect + track + hand band + depth
                       ▲                         │ progress per frame: box, hand point, depth reading
                       │                         │ end: status, last_seen, closest_approach
 mic ──► audio.Mic ──► agent.Controller ──► audio.Player
                       │  HapticLoop.start/pause/resume/cancel
                       └──► runtime.reach_haptic ──► progress ──► runtime.guidance ──► GloveOutput (fake view | real glove)
                                                 └──► ReachResult -> HapticResult ──► controller.on_haptic_result
```

## 5. Glove design change: on/off motors, proximity as pulse rate

The motors are **on or off only**. Intensity carries no information.
- **Direction channels** (left, right, forward, back, up, down) are steadily on or off.
- **Proximity** is the **wrist motor's pulse rate**: slow when far, fast when close. Starting values are 1 Hz far and 6 Hz near; coin motors need about 40 ms to spin up, so faster blurs.
- **Channel order and meaning** come from `haptics/protocol.py`, the source of truth:
  - thumb = left, index = forward, middle = reserved, ring = back, pinky = right
  - back of hand = up, palm = down, wrist = proximity

  Tell me if anything else in the repo disagrees.
- **Keep `haptics/GuidanceController` unchanged where possible.** Map its output: a channel is on if its level > 0, and the wrist pulse rate is `cmd.wrist_hz`. Run it with `success_frames=None`, because the vision server's reach session owns `success`.
- **Output goes through one interface,** `runtime/glove_out.py`:

```python
@dataclass
class GloveCommand:
    on: frozenset[int]          # channel indices from haptics/protocol.py that are steadily on
    pulse_hz: dict[int, float]  # channel -> pulse rate (today only WRIST); 0 or missing = no pulsing

class GloveOutput(Protocol):
    def apply(self, cmd: GloveCommand) -> None: ...   # called each vision frame; must not block
    def all_off(self) -> None: ...                    # called on every exit path
```

- **Implementations:**
  - `FakeGlove`: draws the 8 motors on the live view, blinking at their pulse rates.
  - `HardwareGlove` (later): wraps the teammates' per-motor on/off functions. If their interface has no pulse function, it pulses in software on a 50 Hz thread (as `haptics/link.py` `frame_at` does). In steps 1 to 3 it's a stub that raises `NotImplementedError` with a clear message.
- **All-off** happens on: session end, pause, cancel, any exception, and process exit.

## 6. Live views

Every live script uses one shared helper, `runtime/view.py`, with `--view window|web|none`:
- **`window`:** an OpenCV window. Run the UI loop on the **main thread**, and the asyncio loop (agent, controller ticks) on a background thread.
- **`web`:** an MJPEG stream at `http://<host>:8090/`, so a headless Pi can be watched from the laptop's browser over Tailscale. Use the standard multipart/x-mixed-replace approach and stdlib or FastAPI, nothing exotic.
- **`none`:** logs only.

Overlays always show the camera frame, FPS, and the latest per-stage timings. Each step adds its own overlays. Keys work in `window` mode. In `web` mode, mirror the important keys on the terminal (type a letter, then Enter).

Camera: one `runtime/camera.py` capture thread, newest frame only, thread-safe. It offers:
- `newest_bgr()` for `ReachAPI`'s `frame_source`
- `latest_frame()` for the agent and the Mic: the agent's `Frame` (JPEG, long edge at most `frame_max_long_edge_px`, not mirrored)

Reuse the capture logic in `agent/scripts/webcam_chat.py`: the warm-up frames and `CAP_DSHOW` on Windows. The camera source and index come from `runtime/config.yaml`.

## 7. Steps

Folder names below are the current ones (`object-detection/` and so on). After Step 0, read them as the new package paths.

### Step 0: Repo cleanup (packages and consistency)

Goal: one coherent codebase before any wiring. The folders were written by different people with different conventions:
- hyphenated folder names that can't be imported
- flat imports of generic module names
- tests sometimes in `tests/`, sometimes next to the code
- different requirements file names
- a root README describing an old layout

Do this:
1. **Propose a target layout and wait for approval.** For example, make the repo installable with a root `pyproject.toml` (`pip install -e .`) exposing:
   - `guide_agent` (from `agent/`)
   - `audio`, `haptics`, `runtime`
   - `object_detection`, `hand_detection`, `depth_estimation` (renamed from the hyphenated folders)

   Keep each package's own config file, README or CLAUDE.md, and tests, in one consistent place.
2. **Move files with `git mv`**, so history follows, and convert imports to package-qualified ones.
3. **Update every reference to the moved paths:**
   - docs and READMEs
   - run commands
   - `object-detection`'s Dockerfile and `deploy/` scripts
   - the config path lookups (each component finds its config relative to its own file)
   - the agent's `AGENT_ROOT` / `.env` lookup
   - the audio and agent CLI commands
4. **Unify small conventions where it's cheap:**
   - the fake-model switch (`GLANCE_FAKE_MODELS=1` everywhere)
   - requirements file naming per machine role: client (laptop/Pi, no torch), vision server, STT server
5. **No behavior changes in this step.** Every component's test suite passes before and after, and the commands in each README still work under their new paths.
6. **Rewrite the root README** to describe the real layout and how to run each part.
7. In the stop report, include **a short "what moved" table** for teammates whose branches will need rebasing, and the exact new commands for the servers on the GPU PC.

### Step 1: Audio + agent (voice in, speech out, live camera, simulated reach)

Goal: talk to the real agent with the real mic, wake word, STT and TTS, using live camera frames. The reach is simulated.

Build:
- **`runtime/voice_agent.py`.** Wire everything per `audio/CLAUDE.md` section 15:
  - `dispatch = loop.call_soon_threadsafe`
  - `Player` is the controller's `audio`, and the controller is the `Mic`'s sink
  - `task_active = lambda: controller.task is not None`
  - `frame_source = camera.latest_frame`
  - `controller.tick()` every 100 ms
- **Agent config.** Add `agent/config.yaml` with `assistant_name: Glance` and `speech_start_timeout_s: 20` (audio decision 1). The name change alters the prompt, so run the agent's controller tests, plus the eval cases whose fixtures exist, and report the results.
- **`KeyboardReach`**, implementing the agent's `HapticLoop`.
  - When the agent calls `start`, the view shows "REACH: <target> (<refer>)".
  - Keys report results: `s` success, `l` target_lost, `h` hand_lost, `n` target_not_found, `t` timeout, `k` stale_link.
  - `pause`, `resume` and `cancel` show on screen.
  - The `FakeGlove` panel is shown, idle.
- **No-headphones mode.** I have no earbuds right now, and with speakers the open-state VAD hears the TTS and loops. Add `--ptt-only`: no wake word and no open-state VAD; Enter starts and stops each utterance. Implement it in `audio/` behind a config flag if that's the right place, keeping the `audio/CLAUDE.md` section 5 contracts. Also add `--typed`: typed utterances, with real TTS and camera.
- **Overlays:**
  - listening state (wake, open, capture)
  - the last transcript and the last spoken text
  - the agent's task goal, current step and expect
  - reach state
  - the frame sent with the last utterance, as a small inset
- **Tests:** a scripted-model integration test that drives the controller through the real `Player` and `Mic` code paths, with fake audio devices. Cover wake, utterance, say, reply window, barge-in and the ptt-only mode.
- **Live tests I'll run:**
  - The agent's reference flow (`agent/CLAUDE.md` 17.6) with a real object.
  - "stop" during a speak step.
  - A side question during a reach.
  - Report the latency from end of speech to first audio (p50 and max) from the logs.

### Step 2: Vision + depth + guidance, standalone (no agent)

Goal: point the camera at the glove and an object, and see the right commands, dx, dy, dz and a distance estimate live.

**Server side** (extend `object-detection/`; don't build a second server):
- Wire `depth-estimation` into the reach session.
  - Load `DepthModel` and `DepthCache` once per server process, warmed up at startup. Keep one `DepthTracker` per reach, reset per reach.
  - Feed the tracker the band mask (`depth_regions.band_mask_from_marker(hand_marker.marker_mask(bgr) > 0, marker.box)`), the target box, the SAM 2 mask if the tracker can expose it cheaply, the hand point and the band area.
  - `session.py`'s current `DepthFn` (points in, depths out) is too narrow, so widen the hook. Keep `object-detection`'s tests passing.
- **Progress messages** gain the depth reading: `hand`, `target`, `dz`, `gap`, `band`, `touching`, `units`, `target_frozen`, `inconsistent`, and the depth timing. Also add the hand box and band area if they aren't there.
- **End reports:** `last_seen.depth` comes from `DepthTracker.last_seen_depth()`, and `closest_approach` from `DepthTracker.closest_approach()`, when depth is available.
- **Success:** the session uses depth `touching` when depth is available, and keeps the 2D rule otherwise.
- **Frame size.** Object detection streams 480x360, while the hand and depth configs assume 640x480. `hand_marker.min_area_px` is in pixels, so it depends on the size. Measure both sizes on the 3070 and pick one; record it as a decision.
- Respect `depth.every_n_frames`, and measure the per-frame server time with every model loaded on the 3070.
- Give me exact PowerShell commands to set up the vision server venv on the GPU PC and run it on port 8000 with `GLANCE_TOKEN`.

**Client side:**
- **`runtime/guidance.py`:** per progress message, call `haptics.GuidanceController.update(hand, box, frame_size, depth_error, touching)` in the configured axis mode (default `table`; a demo key toggles `wall`), then produce a `GloveCommand` per section 5. For display, also compute:
  - `dx`, `dy`: normalized image error, from the controller
  - `dz`: the depth `gap`, in depth units
  - the active commands as words (LEFT, RIGHT, FORWARD, BACK, UP, DOWN) and the wrist pulse rate
  - a distance estimate: cm when depth is metric (with lateral cm from `focal_px` if it's calibrated in `depth-estimation/config.yaml`), otherwise normalized values labeled as such
- **`runtime/live_guidance.py --target "red can" [--refer ...] [--view ...]`** uses `CameraHub` and `ReachAPI` against `GLANCE_VISION_URL`. It also accepts a local `GLANCE_FAKE_MODELS=1` server, for wiring checks.
  - Overlays: target box, hand point, an arrow from hand to target, the command words, dx/dy/dz, the distance, touching and frozen state, and the `FakeGlove` motor diagram blinking at the pulse rates.
  - Keys: `r` restart the reach, `space` pause/resume, `m` toggle table/wall, `q` quit.
  - It logs one JSONL line per frame with per-stage timings (capture, encode, network, detect/track, hand, depth, guidance) and prints a latency and FPS summary on exit.
- **Tests:** guidance mapping from synthetic progress messages (each direction, the deadband and hysteresis, the pulse rate rising as distance falls, all-off on lost, stop and end), plus the widened depth hook with fake models.
- **Live tests I'll run:**
  - Move the gloved hand around a can and a sheet of paper on the desk. Check that the commands point the right way, dz falls as the hand descends, and touching fires at contact.
  - Report FPS and the timing breakdown, with the server on the 3070.

### Step 3: Agent + full reach loop

Goal: the agent drives real reaches; only the motors are simulated.

- **`runtime/reach_haptic.py`:** implements the agent's `HapticLoop` on top of `ReachAPI`.
  - `ReachStartError` becomes `HapticStartError`.
  - `ReachResult` becomes `HapticResult`, and `ReachFrame` becomes `Frame`. The field names already match; see `reach_api.py`'s docstring.
  - Results go to `controller.on_haptic_result` through `dispatch`. `cancelled` results are dropped (the controller ignores them anyway).
  - `on_progress` feeds `runtime/guidance.py` and then the `GloveOutput`. On pause, cancel or end: all-off.
- **`voice_agent.py`** gains `--reach real` (this) versus `--reach keys` (step 1). The overlays combine steps 1 and 2 in one view.
- **Root README:** add how to run the full system (servers, runtime, live scripts).
- **Tests:** `reach_haptic` with a fake `ReachAPI` for every status. Check pause, resume and cancel; all-off on every exit; result conversion; and that results arrive through `dispatch`.
- **Live tests I'll run:**
  - The full reference flow by voice: "hey glance, get me the can" through to touching it, with `FakeGlove` on screen.
  - `target_lost` (I'll cover the can).
  - `hand_lost` (I'll pull my hand out of view).
  - "stop" mid-reach.

### Step 4: Hardware glove (later; I'll tell you when)

Build:
- `HardwareGlove` on the teammates' per-motor interface
- `runtime/glove_check.py`, which steps through each motor and pulse rate
- `--glove fake|hardware` on both live scripts

### Step 5: Move to the Pi (later)

- Do this after step 3 passes on the laptop and before step 4, since the glove plugs into whatever runs `runtime/`.
- A Pi smoke test can happen earlier, in parallel:
  - OS, Tailscale, `audio/scripts/install_pi.sh` and `audio.cli selftest` with a USB headset
  - camera capture
  - `pip install -r runtime/requirements.txt`
  - `voice_agent.py --view web --reach keys`
- Build: `runtime/scripts/install_pi.sh` (extending `audio/scripts/install_pi.sh`), Pi camera or USB camera selection, `--view web` as the default on the Pi, and `runtime/selftest.py` covering camera, audio, STT server, vision server and glove.

## 8. Definition of done per step

- All unit tests in every touched folder pass, with fake models: `agent`, `audio`, `object-detection`, `depth-estimation`, `hand-detection`, `haptics`, `runtime`.
- The live script runs on the Windows laptop, with servers on the 3070 PC where applicable.
- The stop report includes:
  - what was built
  - every change outside `runtime/`
  - exact PowerShell commands for the laptop and the GPU PC (venv, env vars, run)
  - what I should see for each live test
  - open questions

## 9. Decisions

1. Components are packages importable from the repo root (Step 0). Integration lives in `runtime/`. The laptop stands in for the Pi, and the GPU PC (over Tailscale) for Vultr. Switching hosts changes only URLs.
2. The existing `object-detection` server and `ReachAPI` are the vision path. Depth and hand detection run there, on the GPU host.
3. The vision server's reach session owns `success`. The client-side `GuidanceController` runs with `success_frames=None`.
4. The default axis mode is `table` (paper or can on a desk, camera looking down).
5. Motors are on or off. Proximity is the wrist's pulse rate.
6. STT and vision servers run as separate processes with separate venvs.
7. Step 0 layout: each top-level folder is its package (`guide_agent`, `audio`, `haptics`, `object_detection`, `hand_detection`, `depth_estimation`, `runtime`). `agent/` was flattened into `guide_agent/`, and its photo folder `fixtures/` is now `fixture_frames/`. Tests go in `<pkg>/tests/`, runnable tools in `<pkg>/scripts/` (`python -m <pkg>.scripts.<name>`). Everything runs from the repo root. `pyproject.toml` declares packages only.
8. Requirements by machine role: `runtime/requirements.txt` (client: laptop and Pi, aarch64, no torch), `requirements-dev.txt` (unit tests), `object_detection/requirements-server.txt` (vision server, installed after the torch CUDA build, `opencv-contrib-python` only), and `audio/requirements-server[-windows].txt` (STT server, unchanged). Venvs: `.venv` (client), `.venv-vision` (vision server), `audio\.venv` (STT server on the GPU PC). `audio/scripts/install_pi.sh` keeps the openWakeWord `--no-deps` step.
9. Hand-band tests draw the band in the middle of the configured HSV range, so they follow retuning. `hand_detection/config.yaml` keeps its tuned values.
10. Depth convention: larger = farther (`depth_estimation`). `object_detection/CLAUDE.md` is a partly stale design doc; its README is current.
11. Step 1, no headphones: `audio` config key `ptt_only` (audio decision 26). There's no wake word and no open-state VAD, and Enter starts and stops each utterance; Enter during playback is a barge-in. `runtime.voice_agent --ptt-only` turns it on. `build_mic` takes an optional `ptt` source, so the runtime owns stdin.
12. Keys: one stdin reader per process (`runtime/keys.py`). An empty line toggles push-to-talk, a line that is exactly one known key letter is that key, and any other line is a typed utterance in `--typed` mode (otherwise ignored). In `--view window` the same keys and Enter also work in the window.
13. `KeyboardReach` results: success gives `frame=None`, `last_seen=None`, `closest_approach=touching`. Every other status gives `frame=camera.latest_frame()` and `closest_approach=near`; `last_seen` is center-middle/near, or None for `target_not_found`. `elapsed_s` is real time. Keys work only while a reach is running (not paused).
14. The voice agent waits for the Piper warm-up (the fixed phrases are cached first, several seconds on the laptop) before it starts the mic or takes input, so the first reply isn't queued behind it. The live view keeps rendering meanwhile.
15. Step 2 frame size: `object_detection` now streams **640x480** (it was 480x360), which matches `hand_detection` and `depth_estimation`, whose `min_area_px` and `focal_px` are in pixels at that size. This is provisional until `object_detection/scripts/bench_server.py` has been run on the 3070 at both sizes. Keep 640x480 unless the per-frame server time there is clearly too slow (then go back to 480x360 and scale `hand_marker.min_area_px` by 0.56).
16. Step 2 depth hook: `ReachSession` takes a per-reach `DepthProbe` (`object_detection/depth_hook.py`), replacing the old points-in/depths-out `DepthFn`. On a frame with a depth gap, success is depth's `touching`. On a frame without one, the 2D rule applies. Once any frame has had a depth gap, the end report's `last_seen.depth` and `closest_approach` come from the `DepthTracker`. The contact offset is chosen by an object name that appears in the target phrase ("red soda can" → can), falling back to `active_object`. `depth.enabled` in `object_detection/config.yaml` turns depth off.
17. Step 2 client: `runtime/guidance.py` passes `depth_error` to the controller in both modes (the controller uses it only in wall mode) and `touching` from depth. `ReachAPI` adds a `client` dict (encode, round trip, network ms) to each progress message; this adds fields to the dict and leaves the interface unchanged. Terminal mirror of the space key in `live_guidance`: `p` + Enter.
18. Step 3: `runtime/reach_haptic.py` `ReachHaptic` is the `HapticLoop` on `ReachAPI`. The interface is synchronous and the controller calls it on its loop, so `start()` blocks that loop while the WebSocket connects (milliseconds normally, at most `reach_api.start_timeout_s` + 1 s on failure, which becomes `HapticStartError` → stale_link). `cancelled` results are dropped. `ambiguous` (off by default on the server) maps to `target_not_found`, the nearest status the agent knows. Guidance's table/wall mode survives across reaches; `m` toggles it.
19. Step 5 (built; the Pi run is still to do): `runtime/scripts/install_pi.sh` sets up the root `.venv` on the Pi with `--system-site-packages` (gpiozero, picamera2), installs `runtime/requirements.txt` and `openwakeword --no-deps`, fetches the models, and adds the user to `dialout`. Per-machine settings go in the gitignored `runtime/config.local.yaml`. `--view` defaults to `web` on Linux without a display. Camera source `picam` uses Picamera2; a USB webcam is the default. `runtime/selftest.py` checks env, camera, STT, vision (health plus a `/ws` start and cancel), glove and audio.
20. Step 4 (built; untested on hardware): `HardwareGlove` drives the glove through `haptics.link.GloveLink`, over the USB serial protocol the repo's firmware implements. A steady channel is sent at `glove.on_level`. GloveLink pulses the wrist in software at 50 Hz, and the firmware turns everything off after 300 ms without a frame. Section 5 says to wrap "the teammates' per-motor on/off functions"; no such interface is in the repo. If the teammates have one, it replaces GloveLink inside `HardwareGlove` and nothing else changes. `--glove fake|hardware` and `--glove-port` work on `voice_agent`, `live_guidance` and `glove_check`, and the hardware glove is mirrored on the fake panel.
21. The glove in use is the 6-motor ESP32-S3 (`firmware/glove/code.py`, CircuitPython): thumb, index, ring, pinky, back, palm, in that wire order. It has no middle-finger motor and **no wrist motor**, which contradicts section 5's proximity-as-wrist-pulse. `haptics/protocol.py` keeps the 8 logical channels (the GuidanceController and GloveCommand use them) and adds `S3_CHANNELS`. `GloveLink(channels=...)` sends only those 6 (6-level frames); the default still sends 8 for `glove.ino`. Runtime `glove.firmware: s3` (default) or `ino8`. On the S3, proximity is shown on the live view only. **Open question:** convey proximity some other way (for example, pulse the active direction motors at the proximity rate), or drop it.
22. `python -m runtime.run` (and `runtime/scripts/run_glance.sh` on the Pi) runs the whole thing. It loads `runtime/glance.env` (gitignored; template `runtime/glance.env.example`), uses the hardware glove if one is plugged in (`port: auto` finds the Adafruit/Espressif USB vendor id), runs the selftest (stops on FAIL unless `--force`), then `voice_agent --reach real`. `python -m runtime.glove ports|test|buzz|sweep|off` drives the glove through the runtime's own path; `firmware/glove/send_test.py` stays as the raw-frame bench test.
23. Unit tests ignore a developer's `object_detection/.env` and shell `GLANCE_*` variables (`object_detection/tests/conftest.py`). A local `.env` with a token made the server tests fail with 403.
