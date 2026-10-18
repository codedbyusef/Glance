# object_detection

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
   hand point (from `hand_detection/`).
4. Mask gone for `tracker.lost_frames` frames -> Grounding DINO runs once to re-acquire. If that fails:
   `success` with `inferred=true` when the hand was over the box, otherwise `target_lost` after 2 s.
5. Other endings: `hand_lost`, `stalled`, `timeout`, `stale_link`, `cancelled`
   (and `ambiguous`, off by default because the agent doesn't handle it yet).

`refer` is accepted and ignored for now.

**Depth** (integration Step 2): the server loads Depth Anything V2 once (`depth_hook.load_depth_cache`,
settings in `depth_estimation/config.yaml`, `depth.enabled` here) and gives each reach its own
`depth_estimation` `DepthTracker` (`depth_hook.ReachDepth`). The hand is sampled over the band's own pixels,
the target over SAM 2's mask (or the box for CSRT). On frames with a depth gap, success is depth's `touching`
(hand point in the box and the gap closed, held for `touching.hold_frames`); otherwise the 2D rule above
(`success_hold_frames`). With depth, the end report's `last_seen.depth` and `closest_approach` come from
the depth tracker.

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
| `depth_hook.py` | Depth for the reach session: the shared Depth Anything V2 cache and a `DepthTracker` per reach. |
| `scripts/bench_server.py` | Per-frame server time with every model loaded, and hand detection, per frame size (run on the GPU host). |
| `fake.py` | Canned detector/tracker for tests and wiring checks (`GLANCE_FAKE_MODELS=1`). |
| `config.yaml` | Model ids, thresholds, timers, frame size, Vultr ports. |
| `Dockerfile` | CUDA image with weights baked in. Build from the repo root. |
| `requirements-server.txt` | Vision server deps (GPU host). The laptop/Pi client deps are in `runtime/requirements.txt`. |
| `deploy/` | `setup_instance.sh`, `deploy.sh`, `tunnel.sh`. They read `.env`. |
| `.env.example` | Template for the Vultr IPv4 (and optional token / SSH key). |
| `tests/` | Box helpers, follower, every end status, server protocol, client URLs. No GPU needed. |

The layout differs from the one sketched in CLAUDE.md (`server/app.py`, `glance/vision/...`):
everything is kept inside this package, like `hand_detection/`. It imports `hand_detection` package-qualified,
so run every command from the repo root.

## Agent integration

The agent's LLM never sees frames. It calls `reach(target, refer, say_first)`; the agent's controller
then calls `haptic.start(target, refer)` and waits for one `haptic_result` (guide_agent/CLAUDE.md sections 5
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
- `on_progress` gets each per-frame `progress` message: `target_box`, `hand_point`, `hand_box`,
  `band_area_px`, `frame_size`, `depth` (`hand`, `target`, `dz`, `gap`, `depth_error`, `band`, `touching`,
  `units`, `target_frozen`, `inconsistent`, `map_age`, or null), `timings_ms` (detect/track, hand, depth,
  depth_model, band_mask, depth_sample), `server_ms`, and a `client` dict added on this side (`encode_ms`,
  `round_trip_ms`, `network_ms`). `runtime/guidance.py` turns it into glove commands.

Connecting it to the agent later is a small adapter in the agent (not written yet): `ReachStartError`
-> `HapticStartError`, `ReachResult` -> `guide_agent.schemas.HapticResult` (same field names;
`LastSeen` and `Frame` map one to one), and swap it in for `StubHaptic`. Share one camera thread
between the agent's utterance frames and `frame_source`.

Standalone check with the webcam (server running): `python -m object_detection.reach_api --target "sheet of paper"`.

## Vultr: what to do once the IPv4 is known

1. `cp object_detection/.env.example object_detection/.env` and set `GLANCE_VULTR_HOST=<ipv4>`
   (and `GLANCE_SSH_KEY` if the key isn't your default). `.env` is gitignored.
2. One-time instance setup (skip if `nvidia-smi` and Docker with GPU already work there):
   `bash object_detection/deploy/setup_instance.sh`
3. Wiring check with canned models first: `bash object_detection/deploy/deploy.sh --fake`
4. Real models: `bash object_detection/deploy/deploy.sh`
   (first build downloads Grounding DINO tiny + SAM 2.1 tiny into the image; a few minutes).
5. Leave the tunnel running in its own terminal: `bash object_detection/deploy/tunnel.sh`
6. From the laptop:
   - `python -m object_detection.client --health`: shows the GPU and memory used
   - `python -m object_detection.client --target "sheet of paper"`: prints box, confidence, round trip,
     server time and network time for each frame

The container listens on the instance's `127.0.0.1:8000` only, so the SSH tunnel is the way in.
To expose it directly instead, remove `127.0.0.1:` in `deploy.sh`, open port 8000 in the Vultr
firewall, set `GLANCE_TOKEN` in `.env`, and set `vultr.use_tunnel: false` in `config.yaml`.

## Laptop-only run (for the latency comparison)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # CUDA build first
pip install -r object_detection/requirements-server.txt
pip install -e .
uvicorn object_detection.server:app --port 8000          # from the repo root
python -m object_detection.client --target "sheet of paper"
```

`server.device: auto` picks CUDA, then Apple MPS, then CPU. Switch `tracker.backend` to `csrt` if
SAM 2 is too slow without a GPU.

## Tests

`python -m pytest object_detection -v` from the repo root (fake models only; `test_reach_api.py` starts the server on a free local port).

## Things to check on the first real run

- `transformers>=4.56` for `Sam2VideoModel` streaming. The Docker base is PyTorch 2.5.1 / CUDA 12.4;
  bump it if the installed `transformers` asks for a newer torch.
- SAM 2 streaming keeps the seed frame plus the last 20 frames (`tracker._KEEP_FRAMES`) to bound GPU
  memory. Watch `/health` memory over a 30 s reach.
- fp16 is used for both models on CUDA. If boxes look wrong, set `server.fp16: false`.
- Try phrase variants per demo object and tune `box_threshold`, `min_score`, `lost_frames`.
