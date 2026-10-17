# depth-estimation

Depth Anything V2 Small for the reach: how far the hand and the target are from the head camera,
read from **one depth map per frame**, so the two can be compared directly. Self-contained like
`hand-detection/`: everything it needs is in this folder.

**Status: logic built and unit-tested on synthetic depth maps; the real model has not been run yet.**
Nothing here is wired into object-detection, haptics or the agent. The tests and the fake model run
with no GPU, torch or weights.

## What depth is for

The haptics controller (`haptics/controller.py`) defaults to **table mode**: the camera looks down,
image left/right -> thumb/pinky, image up/down -> index/ring (forward/back), and once the hand point is
inside the target box the palm cues "down". In 2D, a hand hovering 15 cm above the paper looks the same
as a hand on it. Depth tells them apart. Its jobs, in order:

1. **Table mode:** the remaining gap once the hand is over the target -> when "down" is done, and
   whether it is really touching.
2. **Agent labels:** `last_seen.depth` (near/mid/far) and `closest_approach` (touching/near/far) for
   the `haptic_result` event in `agent/CLAUDE.md`.
3. **Wall mode:** `depth_error` for the controller's forward/back, in its units and sign
   (positive = target farther than hand).

## How it works

```
frame ──► depth model (once) ──► depth map, larger = farther
                                    │
        band pixels (hand-detection) ├─► hand median   ─┐
        target mask or box           └─► target median ─┴─► dz = target - hand
                                                              gap = dz - contact offset
                                                              touching, labels, depth_error
```

- **Hand:** median over the band's own pixels (`hand_marker.marker_mask()` limited to the tracked
  band's box), eroded a few pixels because depth blurs at edges.
- **Target:** median over the tracker's mask when one is passed in, else the box shrunk by 20%. The
  hand region is always cut out of it.
- **Same frame only.** Head motion shifts both together, so `dz` stays valid. `DepthCache` runs the
  model at most once per frame id, and only every Nth frame if `every_n_frames` > 1.
- **Contact offset.** The band is on the knuckles. At real contact the knuckles are still a few cm
  above flat paper, or several cm back from an upright can, so `dz` at contact is not 0. The offset is
  measured per object and subtracted: `gap` ~ 0 at contact.
- **Smoothing:** median of the last 5 samples, then an EMA, separately for hand and target. A sample
  that jumps more than `thresholds.jump` is dropped; after 3 in a row the new level is accepted.
  (The controller smooths only the hand point, not depth, so this is not double smoothing.)
- **Occlusion:** once the hand region (band box x2) covers 30% of the target, or the visible target
  shrinks below half, the target depth freezes at its last good value. It unfreezes when the hand
  leaves or the target box moves.
- **Touching:** hand point inside the target box AND `|gap|` under `contact` for `hold_frames` in a row.
- **Cross-checks:** the band's area grows as the hand nears the camera; if area and depth move in
  opposite directions the frame is flagged `inconsistent`. The pinhole estimate from the tape width
  is logged per pose so you can see whether it's usable.

### Units

`model.kind` picks the checkpoint and the threshold set:

| kind | model | values | thresholds |
|---|---|---|---|
| `relative` | `Depth-Anything-V2-Small-hf` | fraction of the frame's 5th-95th percentile range | `thresholds.relative` |
| `metric` | `Depth-Anything-V2-Metric-Indoor-Small-hf` | meters | `thresholds.metric` |

Both are flipped/kept so **larger = farther**. Metric was trained on synthetic room-scale scenes; whether
it holds at 30-70 cm with a hand in front is what the tape test (step 3) decides. Small is Apache-2.0;
confirm the metric checkpoint's license on its model card before committing to it.

## Files

| File | What it does |
|---|---|
| `depth_model.py` | Loads the model (CUDA, then MPS, then CPU; fp16 on CUDA), `infer(bgr)` -> depth map. `DepthCache` = once per frame, every Nth frame. Fake model with `GLANCE_FAKE_MODELS=1`. |
| `depth_tracker.py` | `DepthTracker.update(...)` -> `DepthReading` per frame; `last_seen_depth()`, `closest_approach()` per reach. The hand-target relation lives here. |
| `depth_regions.py` | Masks, boxes, medians, overlap, band width and pinhole math. |
| `depth_config.py` | Loads `config.yaml` over built-in defaults. |
| `config.yaml` | Model choice, input size, sampling, smoothing, occlusion, thresholds, contact offsets, focal length. |
| `test_depth_live.py` | Live/replay test with the band tracker, calibration keys and JSONL logs. |
| `benchmark_depth.py` | Headless latency + GPU memory per input size (laptop and Vultr). |
| `test_depth.py` | Unit tests on synthetic depth maps. No camera, GPU or weights. |

## Interface (for whoever wires it in later)

```python
import sys; sys.path.insert(0, "depth-estimation")
from depth_model import DepthModel, DepthCache
from depth_tracker import DepthTracker

cache = DepthCache(DepthModel())          # load once per process
tracker = DepthTracker(object_name="paper")   # one per reach; tracker.reset() between reaches

dmap, age = cache.get(bgr, frame_id)
r = tracker.update(dmap, frame_id, band_mask=band_mask,
                   target_box=box, target_mask=sam2_mask_or_None,
                   hand_point=marker.point, band_area_px=marker.area_px, map_age=age)
r.gap, r.touching, r.depth_error, r.target_frozen, r.timings_ms
tracker.last_seen_depth(), tracker.closest_approach()
```

`band_mask` comes from hand-detection:
`depth_regions.band_mask_from_marker(hand_marker.marker_mask(bgr) > 0, marker.box)`.
Boxes are pixel `(x1, y1, x2, y2)`, masks are bool arrays the size of the frame.

Open questions for integration (not decided here): the depth function signature
`object-detection/session.py` expects; whether the server or the controller owns the final success
decision; and how depth weights get into object-detection's Docker image.

## Walkthrough

Run everything from the repo root.

### 1. Install and run the tests (5 min)

```bash
pip install -r depth-estimation/requirements.txt
pytest depth-estimation -v          # 33 tests, no GPU needed
```

If pip's default torch doesn't match your GPU, install torch from pytorch.org first. Check that
`GLANCE_FAKE_MODELS=1 python depth-estimation/benchmark_depth.py --runs 10` prints a table: that
proves the wiring before any weights download.

### 2. Record test clips (15 min)

You need the glove with the green band, the head camera (or webcam) at head height looking down at a
table, a sheet of paper, a matte can, and a tape measure.

```bash
python hand-detection/test_hand_live.py
```

Press `r` to record, `r` to stop (saves `hand_test.mp4`; rename it after each clip). Record:

- **paper_down.mp4:** hand starts 30 cm above the paper, comes down slowly, rests flat on it 2 s, lifts.
- **can_reach.mp4:** hand approaches the can from the side, fingers touch it, hold 2 s.
- **poses.mp4:** hand at ~40 cm, hold palm down, sideways, palm up, ~3 s each.

Write down, for each clip, the camera-to-paper distance and the knuckle height when the hand rests on
the paper. Use the demo lighting if you can.

### 3. First real run and the sign check (15 min)

```bash
python depth-estimation/test_depth_live.py --model relative
```

First run downloads the weights. Press `t`, drag a box over the paper, Enter. Move your hand toward
the camera and away. **hand** must get smaller as the hand comes closer (larger = farther everywhere).
If it moves the wrong way, stop: the sign is wrong and nothing downstream will work.

Copy the `--target-box ...` line it prints; you'll reuse it for replays.

### 4. Pick relative or metric (20 min)

Replay the same clip with both models and the same box:

```bash
python depth-estimation/test_depth_live.py --video paper_down.mp4 --model metric   --target-box X1,Y1,X2,Y2
python depth-estimation/test_depth_live.py --video paper_down.mp4 --model relative --target-box X1,Y1,X2,Y2
```

With **metric**, pause (space) and click the paper, the table near the camera and the far table edge:
the printed values should be within a few cm of your tape measurements. Then watch `dz` as the hand
comes down: it should shrink smoothly and stop at a small positive value when the hand rests.

Keep **metric** if it's within ~3-5 cm at arm's length and `dz` is smooth. Otherwise use **relative**.
Set `model.kind` in `config.yaml`.

### 5. Measure the contact offsets (10 min)

Live or on the replays: get the hand into real contact (resting flat on the paper), hold still with
the band point inside the target box for a second, press `c`. Paste the printed line into
`config.yaml` under `contact_offset.<kind>`. Repeat with `--object can` for the can. Do it 2-3 times
per object; if the values spread a lot, use the median.

After this, `gap` should read ~0 at contact and **TOUCHING** should appear after 7 frames.

### 6. Tune thresholds on the replays (20 min)

Replay `paper_down.mp4` and `can_reach.mp4` and watch the overlay:

- **TOUCHING appears while hovering:** lower `thresholds.<kind>.contact`.
- **TOUCHING never appears at real contact:** raise `contact`, or re-measure the offset.
- **FROZEN never shows as the hand covers the target:** lower `occlusion.freeze_overlap`, or raise
  `sampling.hand_region_scale` if the fingers stick out past the scaled band box.
- **Target depth drifts toward the hand at the end:** same as above; it should freeze first.
- **Hand value lags:** raise `smoothing.ema_alpha`. **Jittery:** lower it.
- **Real moves get rejected as jumps:** raise `thresholds.<kind>.jump`.

For the relative model, set `controller.relative_scale` so a ~30 cm miss gives `depth_error` ~0.35
(the controller's full scale). Metric meters need no scaling.

### 7. Band cross-checks (10 min)

```bash
python depth-estimation/test_depth_live.py --calib-distance 0.40
```

Hold the band (palm down) exactly 40 cm from the camera, press `f`, paste `focal_px` into
`config.yaml`. Replay `poses.mp4`, labeling poses with `1` `2` `3`. The summary prints how far the
pinhole estimate is from the model per pose. If sideways/palm-up are way off (likely), treat the
pinhole number as a palm-down-only sanity check and don't use it for scale. The `area-vs-depth flags`
line should be near 0% on a clean reach; a high rate means either depth or the band is unreliable.

### 8. Latency, laptop then Vultr (15 min)

```bash
python depth-estimation/benchmark_depth.py --video paper_down.mp4 --json benchmark_laptop.json
```

On the Vultr instance (once it exists), clone this branch, install the requirements, copy a clip
over, and run the same command with `--json benchmark_vultr.json`. Pick the smallest `input_size`
whose `dz` still looks clean in step 6, and set `every_n_frames: 2` if depth doesn't fit the budget
next to SAM 2's 40-60 ms. Once object-detection runs on the same GPU, run the benchmark while it's
serving to measure the combined load. Both JSON files are the depth numbers for the Vultr write-up.

### 9. Hand off

Tell the object-detection and haptics owners which `model.kind` won, the measured latency, and the
calibrated offsets. Wiring it in (the `session.py` depth hook, the controller's `depth_error`, the
agent's labels) is their call; the interface above is all they need.

If a depth camera (OAK-D, RealSense) turns up at the hardware desk, it would beat all of this on
reliability.
