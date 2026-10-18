# Live tests: end to end, every component

Run these in order. Each block builds on the one before, so a failure points at one component. Commands are
PowerShell from the repo root unless marked Pi (bash). Everything below is on branch `integration`.
Record the numbers marked **Record** and fill in CLAUDE.md section 9 where a decision depends on them.

Tokens: pick one value each for `GLANCE_STT_TOKEN` and `GLANCE_TOKEN` and use the same on every machine.

## 0. Setup (once)

**GPU PC** (`git pull`, `git checkout integration`):

```powershell
# STT server: the existing audio\.venv, unchanged
$env:GLANCE_STT_TOKEN = "<stt token>"; $env:GLANCE_STT_MODEL = "small.en"; $env:GLANCE_STT_DEVICE = "cuda"
audio\.venv\Scripts\python -m uvicorn audio.server.stt_service:app --host 0.0.0.0 --port 8001

# Vision server venv (second terminal, once)
python -m venv .venv-vision
.venv-vision\Scripts\python -m pip install --upgrade pip
.venv-vision\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv-vision\Scripts\pip install -r object_detection\requirements-server.txt
.venv-vision\Scripts\pip install -e .
.venv-vision\Scripts\python -c "import torch, cv2; print(torch.cuda.is_available(), cv2.__version__)"   # True
.venv-vision\Scripts\python -m pytest object_detection hand_detection depth_estimation   # fake models, all pass

# Firewall, elevated PowerShell, once
New-NetFirewallRule -DisplayName "Glance vision (Tailscale only)" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any
```

**Laptop** (`git pull`, `git checkout integration`):

```powershell
.venv\Scripts\pip install -r requirements-dev.txt; .venv\Scripts\pip install -e .
.venv\Scripts\python -m pytest                      # 465 pass
$env:GLANCE_STT_URL = "http://100.112.104.87:8001"; $env:GLANCE_STT_TOKEN = "<stt token>"
$env:GLANCE_VISION_URL = "ws://100.112.104.87:8000/ws"; $env:GLANCE_TOKEN = "<vision token>"
```

Plug in the Brio (camera index 1). If you're using the built-in webcam, put `camera: {source: 0}` in
`runtime/config.local.yaml`.

## 1. Vision server with every model (GPU PC)

```powershell
$env:GLANCE_TOKEN = "<vision token>"
.venv-vision\Scripts\python -m uvicorn object_detection.server:app --host 0.0.0.0 --port 8000 --ws-ping-interval 5
```

- First start downloads Grounding DINO, SAM 2.1 and Depth Anything V2 (a few minutes), then warms up.
- Laptop: `.venv\Scripts\python -m object_detection.client --health` should show `models_loaded: true`, the
  GPU name and `depth: {kind: relative, ...}`.
- Laptop: `.venv\Scripts\python -m runtime.selftest --skip-audio`: `env`, `camera`, `stt`, `vision` (health
  and `/ws` start+cancel) and `glove` should all PASS. A `vision` FAIL on `/ws` usually means a token mismatch.

## 2. Audio (laptop)

```powershell
.venv\Scripts\python -m audio.cli stt-check     # transcribes "Get me the can." through the 3070
.venv\Scripts\python -m audio.cli selftest      # speaker, tone, 3 s recording (say something), both STT backends
```

**Record:** the stt-check round trip.

## 3. Hand band under the demo lighting (laptop)

```powershell
.venv\Scripts\python -m hand_detection.scripts.tune_marker       # click the band, tune, p, paste into hand_detection/config.yaml
.venv\Scripts\python -m hand_detection.scripts.test_hand_live    # 1/2/3/4 poses, r records clips, q summary
```

- Palm down should be found close to 100% of the time; the point must not jump to other green things.
- While this script runs, record the depth clips in step 4: press `r` to start, `r` to stop, and rename
  `hand_test.mp4` after each clip. You need `paper_down.mp4`, `can_reach.mp4` and `poses.mp4`
  (`depth_estimation/README.md` step 2).
- If the HSV range changes, the unit tests follow it automatically. Rerun `pytest hand_detection` once to confirm.

## 4. Depth calibration (GPU PC, on the recorded clips)

Copy the three clips to the GPU PC. Then follow `depth_estimation/README.md` steps 3 to 7 with
`.venv-vision\Scripts\python -m depth_estimation.scripts.test_depth_live --video <clip> ...`:

- **Sign check:** the hand value must get **smaller** as the hand comes toward the camera.
- **Choose relative or metric,** and set `model.kind` in `depth_estimation/config.yaml`.
- **Contact offsets:** key `c`, once for paper and once for the can. Paste them under `contact_offset`.
- **Tune thresholds:** touching must not fire while hovering, and must fire at real contact.
- **`focal_px`** (key `f` at 40 cm). This enables lateral cm in the guidance view.

Commit the config changes, then restart the vision server.

## 5. Frame size and server time (GPU PC)

```powershell
.venv-vision\Scripts\python -m object_detection.scripts.bench_server --video can_reach.mp4 --target "red can" --json bench_3070.json
```

- **Record:** server p50 and p95 at 480x360 and at 640x480, the per-stage times (detect, track, hand,
  depth_model, depth_sample), the hand-found rate, and GPU memory.
- **Choose a size:** keep 640x480 (`object_detection/config.yaml` `frame`) unless it's clearly too slow. Record
  the choice as CLAUDE.md decision 15. If you go back to 480x360, scale `hand_marker.min_area_px` by 0.56.

## 6. Step 2: vision + depth + guidance, no agent (laptop)

```powershell
.venv\Scripts\python -m runtime.live_guidance --target "red can"
.venv\Scripts\python -m runtime.live_guidance --target "sheet of paper"
```

The camera looks down at the desk, with the gloved hand in view. Check:

- **Directions (table mode):** hand left of the can → RIGHT, hand right → LEFT, hand nearer to you than the
  can → FORWARD, past it → BACK. Over the can → DOWN.
- **Wrist:** pulses faster as the hand closes in, from 1 Hz far to 6 Hz near.
- **dz:** falls as the hand comes down onto the paper. FROZEN shows when the hand covers the target.
  TOUCHING fires at real contact and not while hovering, then END: success.
- **Wall mode** (`m`, can on a shelf, camera looking ahead): UP and DOWN come from the image, and FORWARD and
  BACK come from depth.
- **Pause:** `space` (all motors off), `space` again resumes. `r` restarts.
- **Record:** the summary printed on `q`: FPS, and p50/p95 for capture, encode, network, server stages,
  guidance and round trip. The JSONL is in `runtime/logs/`.

## 7. Step 1: voice + agent, simulated reach (laptop; speakers → `--ptt-only`)

```powershell
.venv\Scripts\python -m runtime.voice_agent --ptt-only --reach keys
```

Enter to talk, Enter to stop.

1. **Reference flow** (`guide_agent/CLAUDE.md` 17.6), with a real can:
   - "get me the red can on the table" → walking steps → "done" (and "done" again)
   - REACH appears after the "bring your right hand out" line, then press `s` → touch check → "yes" → done
2. **"stop" during a speak step:** press Enter while it's talking, and the speech cuts. Say "stop"; it asks
   whether to keep going.
3. **Side question during a reach:** press Enter and the reach shows PAUSED. Ask "what color is it"; you get
   an answer and the reach goes back to RUNNING.
4. **The other reach results:** `l`, `h`, `n`, `t`, `k` each give a sensible recovery step (14.7). Press `k`
   twice in one task and it ends as failed.
5. **Record:** `.venv\Scripts\python -m runtime.latency`, the end → first speech p50 and max.

## 8. Step 3: the agent drives real reaches (laptop, servers on the 3070)

```powershell
.venv\Scripts\python -m runtime.voice_agent --ptt-only --reach real
```

1. **Full flow by voice:** "get me the can" through to touching it. The view shows the target box, hand
   point, arrow and command words, and the fake glove blinks. The success leads to the touch check.
2. **target_lost:** cover the can with a sheet mid-reach. After about 2 s the agent asks you to pull the hand
   back, then reaches again.
3. **hand_lost:** pull the gloved hand out of view. After about 1.5 s the agent tells you where to hold it.
4. **"stop" mid-reach:** the reach is cancelled, the glove goes all-off, and the agent asks whether to continue.
5. **Record:** the reach summary printed on `q` (`runtime/logs/voice-reach-*.jsonl`), and `runtime.latency`.

## 9. Wake word (when headphones are available)

```powershell
.venv\Scripts\python -m runtime.voice_agent --reach real
```

- "hey glance" plus a request, with no Enter.
- Replies need no wake word while the reply window is open.
- Talking over the speech cuts it (barge-in).

## 10. Step 4: hardware glove (ESP32-S3, 6 motors)

Copy `firmware/glove/boot.py` and `code.py` to CIRCUITPY, press RESET, and plug the Feather into the laptop or
the Pi over USB. The port is found automatically.

```bash
python3 firmware/glove/send_test.py /dev/ttyACM0 test    # the teammate's raw bench test (COMx on Windows)
python -m runtime.glove ports                            # marks the glove's port
python -m runtime.glove test                             # firmware test through the runtime link: TEST 0 thumb ... TEST DONE
python -m runtime.glove buzz LEFT                        # thumb; then RIGHT pinky, FORWARD index, BACK ring, UP back, DOWN palm
python -m runtime.glove sweep                            # simulated reach: RIGHT+FORWARD, then fewer, then DOWN
python -m runtime.glove_check --glove hardware --firmware-test
```

- Each direction word buzzes the motor printed next to it. If one is wrong, the wiring or code.py's `PINS` order
  disagrees with `haptics/protocol.py` `S3_CHANNELS`.
- There is no wrist motor, so proximity only shows on the screen (decision 21).
- Then repeat step 6 and step 8 with `--glove hardware` added.
- Pulling the USB cable mid-reach must stop every motor (the firmware's 300 ms failsafe).

## 11. Step 5: Raspberry Pi (bash on the Pi)

```bash
git clone <repo> ~/Glance && cd ~/Glance && git checkout integration
bash runtime/scripts/install_pi.sh                # then log out and in once (dialout group, for the glove)
nano runtime/glance.env                           # server URLs, tokens, GEMINI_API_KEY
. .venv/bin/activate
python -m runtime.selftest                        # everything PASS, with the USB headset and the glove plugged in
python -m runtime.glove test                      # the glove from the Pi
python -m runtime.voice_agent --reach keys        # watch http://<pi>:8090/ from the laptop; keys = letter + Enter
bash runtime/scripts/run_glance.sh --ptt-only     # the whole thing: selftest, agent, real reaches, glove
```

- The Pi camera module needs `camera: {source: picam}` in `runtime/config.local.yaml`.
- `config.local.yaml` from the install script already says `glove: {kind: hardware}` (port auto-detected).
- **Record:** FPS and timings on the Pi, against the laptop numbers from steps 6 and 8.
- Audio acceptance on the Pi: `audio/README.md` "Acceptance run on the Pi".

## 12. Agent eval with real photos (any machine with the Gemini key)

- Take the fixture photos listed in `guide_agent/fixture_frames/manifest.json`, and set `missing: false`.
- Run `.venv\Scripts\python -m guide_agent.evals.run`. Target: at least 90% of cases pass, protocol warnings
  in under 5% of turns, and 0 global-check violations.

## 13. Vultr (only if it gets set up; untested since Step 0)

- Follow `object_detection/README.md` "Vultr" with `deploy.sh --fake` first, then real.
- Run the STT server there as in `audio/README.md`.
- Then repeat steps 1, 6 and 8 with the Vultr URLs.
