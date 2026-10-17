# Glance: object detection and tracking (vision side)

> **Status note (integration Step 0):** this is the original design doc and is partly out of date.
> [README.md](README.md) describes the current code and wins where they differ. In particular: the hand point
> is the glove's green tape band (`hand_detection/`), not MediaPipe; depth follows `depth_estimation/`, where
> **larger = farther** (not "larger = closer" as below); and the "Code and interfaces" file layout and
> `config/vision.yaml` are out of date (the code lives in `object_detection/`, settings in `object_detection/config.yaml`).

Scope of this file: finding the named object in the glasses-camera frame, keeping it located while the glove
guides the hand, estimating depth, and reporting the outcome. The overall project, the guidance controller,
haptics and the planner are described in the root CLAUDE.md. Read that first for context.

## Job

When the user is within arm's reach of the target, the agent starts a reach session. Our side:
1. finds the target in the frame,
2. tracks it every frame while the glove guides the hand,
3. gives the controller a per-frame result,
4. ends the session with a clear status and zeroes the motors on every exit.

## Inputs from the agent

- `target`: 1-4 word noun phrase, no positions ("red soda can", "door handle"). This is the detector text query.
- `refer`: a fuller description for choosing between similar objects ("the red can on the table"). Position words in it may be stale if the user moved.
  - The demo uses one of each object on the table, so `refer` is accepted but IGNORED for now. Keep the field in the contract so nothing needs rewiring later.
- Frames are not mirrored: image left = the user's left. Verify the camera app does not flip the preview.

## Pipeline

1. **Detect** with an open-vocabulary detector on `target` (no training, any phrase). Candidates to benchmark: Grounding DINO (tiny/base), OWLv2, YOLO-World, and SAM 3 (text-prompted detect + segment + track; weights are gated on Hugging Face, so request access early). A fixed-class YOLO cannot cover arbitrary objects.
   - Hypotheses to verify on our demo objects, not facts: YOLO-World is fastest; Grounding DINO is better on descriptive phrases; OWLv2 is good on short nouns and weak on relations.
   - Take all candidates above a low score threshold (start ~0.1-0.2), then use the top box.
   - Always return the candidate list and the score gap between the top two. Only the top box is used for now, but it is logged and keeps the door open for a disambiguation step.
   - If the top two scores are close, report `ambiguous` instead of guessing (proposed status, agree it with the planner side).
   - Wording matters: try variants ("door handle" not "door knob", "white sheet of paper" not "paper") and keep the best.
2. **Track** by seeding SAM 2's video tracker with the chosen box (or use SAM 3 if it replaces steps 1-2; OpenCV CSRT is the fallback if streaming SAM 2 is a time sink). The tracker runs every frame; the detector runs only at start and on re-acquire.
3. **Depth** with Depth Anything V2 (small) on small patches at the target and at the hand, every 1-3 frames.
   - Values are relative (larger = closer) and noisy in absolute terms. Only ordering matters. It drives forward/back and proximity and is the weakest signal, so test it early.
   - Normalise per frame (patch medians divided by the frame's 5th-95th percentile range). Verify the sign by waving a hand toward the camera.
4. **Hand point** (MediaPipe Hands, fingertip landmark) is part of the per-frame result. Whether it runs on the laptop or the server is decided by the first latency measurement.
5. **No fine-tuning** unless the pretrained models fail on our demo objects. If it comes to that, 100-200 photos of our own objects in venue lighting.

Out of scope for the demo: VLM disambiguation (designed, not built: numbered candidate boxes + a VLM that picks one using `refer`, with a spoken question as the fallback), clutter handling, 3D mapping.

## Per-frame result (to the guidance controller)

target box (pixel xyxy), tracking confidence, hand point (fingertip, pixels), depth at target, depth at hand, frame id, timestamps for each stage.

## Session contract

- `start(target, refer)` returns a session id. Also `pause`, `resume`, `cancel`.
- `pause` zeroes the motors and freezes the loss timers. Resuming after more than ~3 s re-runs the detector.
- Motors go to zero on EVERY exit path.
- One WebSocket per session.

## End statuses (all thresholds are starting values; keep them in `config/vision.yaml`, then tune)

| Status | Meaning |
|---|---|
| `success` | A fingertip is inside the box and hand depth is near target depth for 5-10 frames in a row |
| `success` with `inferred=true` | The target vanished, but the hand covered over half its box in the last few frames (the hand hides the object at the end of a good reach) |
| `target_not_found` | No acquisition within ~3 s |
| `target_lost` | Gone for over ~2 s without that hand overlap |
| `hand_lost` | No hand for over ~1.5 s |
| `stalled` | The error has not shrunk in ~6-8 s |
| `timeout` | 25-30 s cap |
| `stale_link` | No results for 500 ms |
| `ambiguous` | (proposed) top candidates score too close to choose |
| `cancelled` | (proposed) the agent called `cancel` |

Every report to the agent also carries:
- `elapsed_s`
- `last_seen`: region in image thirds (e.g. right-middle), depth near/mid/far, `age_s`
- `closest_approach`: touching / near / far
- the last frame (none on success)
- candidate count and top scores (for the log)

Link timeouts are two different layers: `stale_link` (500 ms, no results from the vision service) is a session status; the glove firmware separately zeroes motors after 300 ms with no packet. Do not merge them.

## Serving (Vultr)

- One GPU instance, one service, models loaded once (detector, tracker, depth, plus Whisper). Use fp16 small/base variants. Check which GPUs the MLH credits cover, and record GPU memory after each model is added. If memory runs short, drop optional pieces first.
- Frames: JPEG, ~320-480 px wide. Test that width on our small demo objects; widen if recall suffers.
- Latest frame wins. Drop stale frames, never queue; a queue makes the glove lag the hand.
- Docker on a CUDA base, weights baked in, a warm-up pass at startup.
- Access: default to an SSH tunnel for the weekend (fastest to set up). Add HTTPS + token only if the service must be public.
- Latency target: about 200 ms capture-to-vibration, with ~30-60 ms for tracking. This is a target, NOT a measured number. Log timestamps at each stage (capture, upload, detect/track, depth, controller, download, motor update); network time = laptop round trip minus server-reported time.
- Also run the whole pipeline laptop-only to compare latency, frame rate and detection quality against Vultr (evidence for the Vultr prize).

## Code and interfaces

Keep every model behind a small interface so it can be swapped or faked:
- `Detector.detect(image, phrase) -> list[Detection]` where Detection = box (pixel xyxy), score, label
- `Tracker.init(frame, box)` and `Tracker.update(frame) -> (box | None, confidence)`
- `Depth.at(frame, points) -> {point: value}`
- Convert box formats in ONE tested helper. OWLv2 boxes are relative to a padded square image; Grounding DINO returns normalized center/width/height.

Files for this part (matching the root layout, plus two additions):
```
server/app.py              # FastAPI + WebSocket, /health
server/session.py          # reach session state machine, statuses, timers   (new)
server/models/detector.py  tracker.py  depth.py  hand.py
glance/vision/client.py  loop.py  fake.py     # fake.py = webcam-only / canned results
config/vision.yaml         # thresholds, timers, frame size, score cutoffs
scripts/bench_detectors.py                     # same frames through each detector   (new)
tests/                     # box-conversion helper, session timers, status logic
```

## Build order

1. Webcam + detector printing boxes and scores for each demo phrase (`bench_detectors.py`: same ~10 frames through each candidate, log box, score, ms).
2. Add the tracker and depth.
3. Move to Vultr one model at a time and measure latency.
4. Wrap in the session contract and statuses.

## Licenses (check the model cards; do not rely on this list)

Ultralytics YOLO-World is AGPL-3.0 as far as I know (fine for a public hackathon repo). Depth Anything V2 Small is Apache-2.0 but larger sizes are non-commercial. SAM 3 weights are gated with Meta's license.

## Risks

- Shiny cans and thin door handles detect poorly. Pick matte demo objects (paper, a matte can) and test them under venue lighting.
- White paper on a white desk has almost no contrast. Use a dark surface or colored sheet.
- The hand occludes the target at the end of the reach. Hold the last box for a few frames and use the `inferred` success rule.
- Venue wifi: use a phone hotspot and measure early.

## Reconcile before coding

This file uses `start(target, refer)`, `pause/resume/cancel`, and the statuses above. The root CLAUDE.md still shows `start_reach {target, success}`, `step_done` and `user_stop`. Pick one set of names and update the root file, the planner JSON (add `refer`) and the planner prompt to match.
