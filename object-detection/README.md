# object-detection

Finds the target the agent names ("sheet of paper") in the head-camera frame and keeps it located
while the glove guides the hand. Grounding DINO finds the target from the phrase. SAM 2 then follows
it from frame to frame. Runs as a WebSocket server on a Vultr GPU instance; the laptop streams frames
to it. Design: [CLAUDE.md](CLAUDE.md).

**Status: runs locally, not yet on Vultr or a live webcam.** Tested on a laptop RTX 3060 Ti (fp16) with
still photos and a synthetic panning clip: Grounding DINO ~130 ms per call, SAM 2.1 tiny 40-60 ms per
frame, ~65 ms round trip through the local server, ~1.2 GB GPU for both. Still to check: the real demo
objects under venue light, a real hand covering the target, and Vultr latency. The Vultr IPv4 is still
unknown, so it lives in `.env` as a placeholder (see below).

## How a reach works

1. `start(target, refer)` -> Grounding DINO runs once on the newest frame with `target`.
2. Top box at or above `detector.box_threshold` -> seeds SAM 2 on that frame. Nothing passes for
   3 s -> `target_not_found`.
3. Every later frame SAM 2 updates the mask -> box + confidence -> sent to the controller with the
   hand point (from `hand-detection/`).
4. Mask gone for `tracker.lost_frames` frames -> Grounding DINO runs once to re-acquire. If that fails:
   `success` with `inferred=true` when the hand was over the box, otherwise `target_lost` after 2 s.
5. Other endings: `hand_lost`, `stalled`, `timeout`, `stale_link`, `cancelled`
   (and `ambiguous`, off by default because the agent doesn't handle it yet).

`refer` is accepted and ignored for now. Depth (Depth Anything V2) is not added yet: `session.py`
takes a depth function when it exists; until then success = fingertip in the box for 7 frames.

## Files

| File | What it does |
|---|---|
| `detector.py` | Grounding DINO (`transformers`), `detect(image, phrase) -> [Detection]`, best first. |
| `tracker.py` | SAM 2.1 video, streaming, one frame at a time; mask -> box + score. `CsrtTracker` fallback (CPU, no mask). |
| `follow.py` | Detect once, then track, with one detector retry per loss. Model-free. |
| `session.py` | Reach session state machine, timers, end statuses, report for the agent. |
| `server.py` | FastAPI `/ws` + `/health`; newest frame wins; optional `GLANCE_TOKEN`. |
| `client.py` | Laptop client library + `--health` / live webcam check. |
| `reach_api.py` | Synchronous `start/pause/resume/cancel` + result callback for the agent (see "Agent integration"). |
| `boxes.py` | All box conversion and geometry, in one tested place. |
| `fake.py` | Canned detector/tracker for tests and wiring checks (`GLANCE_FAKE_MODELS=1`). |
| `config.yaml` | Model ids, thresholds, timers, frame size, Vultr ports. |
| `Dockerfile` | CUDA image with weights baked in. Build from the repo root. |
| `deploy/` | `setup_instance.sh`, `deploy.sh`, `tunnel.sh`. They read `.env`. |
| `.env.example` | Template for the Vultr IPv4 (and optional token / SSH key). |
| `test_*.py` | Box helpers, follower, every end status, server protocol, client URLs. No GPU needed. |

The layout differs from the one sketched in CLAUDE.md (`server/app.py`, `glance/vision/...`):
everything is kept inside this folder, like `hand-detection/`.

## Agent integration

The agent's LLM never sees frames. It calls `reach(target, refer, say_first)`; the agent's controller
then calls `haptic.start(target, refer)` and waits for one `haptic_result` (agent/CLAUDE.md sections 5
and 10). `reach_api.ReachAPI` is that interface on this side:

```python
api = ReachAPI(on_result=handle, frame_source=camera.newest_bgr, on_progress=glove_controller)
session_id = api.start("sheet of paper", "the white sheet on the desk")   # raises ReachStartError
api.pause(session_id); api.resume(session_id); api.cancel(session_id)
```

- It streams the newest camera frame to the server at `reach_api.stream_fps` and returns one
  `ReachResult` per reach, with the agent's `haptic_result` fields (status, inferred, elapsed_s,
  target, last_seen, closest_approach, frame). `frame` is the JPEG the server ended on (None on success).
- Link dead or server silent for `reach_api.no_reply_s` -> `stale_link`, made on this side.
- `cancel` -> a `cancelled` result (the agent drops those). `pause` stops frames and freezes timers.
- `on_result` and `on_progress` run on the API's own thread; hop to your loop with
  `loop.call_soon_threadsafe`.
- `on_progress` gets each per-frame `progress` message (target box, hand point, timings): the input
  for the glove controller, which does not exist yet.

Connecting it to the agent later is a small adapter in the agent (not written yet): `ReachStartError`
-> `HapticStartError`, `ReachResult` -> `guide_agent.schemas.HapticResult` (same field names;
`LastSeen` and `Frame` map one to one), and swap it in for `StubHaptic`. Share one camera thread
between the agent's utterance frames and `frame_source`.

Standalone check with the webcam (server running): `python object-detection/reach_api.py --target "sheet of paper"`.

## Vultr: what to do once the IPv4 is known

1. `cp object-detection/.env.example object-detection/.env` and set `GLANCE_VULTR_HOST=<ipv4>`
   (and `GLANCE_SSH_KEY` if the key isn't your default). `.env` is gitignored.
2. One-time instance setup (skip if `nvidia-smi` and Docker with GPU already work there):
   `bash object-detection/deploy/setup_instance.sh`
3. Wiring check with canned models first: `bash object-detection/deploy/deploy.sh --fake`
4. Real models: `bash object-detection/deploy/deploy.sh`
   (first build downloads Grounding DINO tiny + SAM 2.1 tiny into the image; a few minutes).
5. Leave the tunnel running in its own terminal: `bash object-detection/deploy/tunnel.sh`
6. From the laptop:
   - `python object-detection/client.py --health`: shows the GPU and memory used
   - `python object-detection/client.py --target "sheet of paper"`: prints box, confidence, round trip,
     server time and network time for each frame

The container listens on the instance's `127.0.0.1:8000` only, so the SSH tunnel is the way in.
To expose it directly instead, remove `127.0.0.1:` in `deploy.sh`, open port 8000 in the Vultr
firewall, set `GLANCE_TOKEN` in `.env`, and set `vultr.use_tunnel: false` in `config.yaml`.

## Laptop-only run (for the latency comparison)

```
pip install -r object-detection/requirements.txt
cd object-detection && uvicorn server:app --port 8000
python object-detection/client.py --target "sheet of paper"
```

`server.device: auto` picks CUDA, then Apple MPS, then CPU. Switch `tracker.backend` to `csrt` if
SAM 2 is too slow without a GPU.

## Tests

`pytest object-detection -v` (fake models only; `test_reach_api.py` starts the server on a free local port).

## Things to check on the first real run

- `transformers>=4.56` for `Sam2VideoModel` streaming. The Docker base is PyTorch 2.5.1 / CUDA 12.4;
  bump it if the installed `transformers` asks for a newer torch.
- SAM 2 streaming keeps the seed frame plus the last 20 frames (`tracker._KEEP_FRAMES`) to bound GPU
  memory. Watch `/health` memory over a 30 s reach.
- fp16 is used for both models on CUDA. If boxes look wrong, set `server.fp16: false`.
- Try phrase variants per demo object and tune `box_threshold`, `min_score`, `lost_frames`.
