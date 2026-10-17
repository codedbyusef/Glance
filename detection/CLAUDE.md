# Detection and perception server

Covers `detection/` (open-vocab detector, tracker, depth), `server/` (FastAPI/WebSocket service, `ReachSession`, STT)
and the reach session contract the agent uses. Hand localization is in `hand/` (see `hand/CLAUDE.md`). Read the
root `CLAUDE.md` first for priorities, ownership and conventions.

Code map: `detection/detector.py`, `detection/tracker.py`, `detection/depth.py` are pure functions (frame bytes in,
results out, no FastAPI). `server/session.py` composes them with `hand/locate.py` and decides when a reach ends.
Thresholds live in `config/settings.yaml` (`reach_session`, `detector`).

## Object detection and tracking (settled)

When the user is within arm's reach of the target, the agent calls `reach(target, refer)`. The runtime's vision
loop turns that into a session on the perception server, which finds the object in the glasses-camera frame and
keeps it located while the glove guides the hand. Frames are not mirrored: image left = user's left.

**Pipeline** (re-run from step 1 on re-acquire):
1. Open-vocab detector on `target`: Grounding DINO, OWLv2, or YOLO-World. All take a text query and need
   no training; a fixed-class YOLO can't cover arbitrary objects. Tradeoffs to verify: YOLO-World fastest,
   Grounding DINO best on descriptive phrases, OWLv2 good on short nouns but weak on relations. Benchmark
   on the demo objects and check licenses before committing.
2. Several candidates: draw numbered boxes and have a VLM (Qwen-VL class or Florence-2) pick one using
   `refer`, or have it ground `refer` directly. Runs once per reach, so a few hundred ms is fine.
3. Seed SAM 2's video tracker with the chosen box. SAM 2 runs every frame; the detector only runs at
   session start and on re-acquire.
4. Depth Anything V2 on small patches around the target and the hand, every few frames. Absolute values
   are noisy -- only ordering matters. It drives forward/back and proximity and is the weakest signal, so
   test it early.
5. No fine-tuning unless the pretrained models fail on our objects.

Per frame, the server reports: target box, tracking confidence, depth at target and hand, frame id, timestamps.

**Session contract**: `start(target, refer)` returns a session id, plus `pause`, `resume`, `cancel`.
`pause` zeroes the motors and freezes the loss timers. Resuming after more than ~3 s re-runs the detector.
Motors go to zero on every exit.

**End statuses** (starting thresholds, tune them):

| Status | Condition |
|---|---|
| `success` | a fingertip inside the box and hand depth near target depth for 5-10 frames in a row |
| `success`, `inferred: true` | the target vanished, but the hand covered over half its box in the last few frames (the hand hides the object at the end of a good reach) |
| `target_not_found` | no acquisition within ~3 s |
| `target_lost` | gone over ~2 s without that hand overlap |
| `hand_lost` | no hand for over ~1.5 s |
| `stalled` | the error hasn't shrunk in ~6-8 s |
| `timeout` | 25-30 s cap |
| `stale_link` | no results for 500 ms |

The report to the agent also carries `elapsed_s`, `last_seen` (region in image thirds like
"right-middle", depth near/mid/far, `age_s`), `closest_approach` (touching/near/far), and the last frame
(omitted on success). Keep these status names identical everywhere (agent, runtime vision loop, server, logs).
The agent's own spec of this event is in `agent/CLAUDE.md` section 10; the runtime maps `ReachReport` onto it.

## Perception server (WebSocket)

- One Vultr GPU instance, one service, every model loaded once (detector, SAM 2, depth, Whisper), Docker on
  a CUDA base with weights baked in and a warm-up pass, behind HTTPS with a token check. fp16 small/base
  variants should fit a 24 GB GPU -- check which GPUs the MLH credits actually cover.
- One WebSocket per session. Client sends JPEG frames ~320-480 px wide with `frame_id` and a client timestamp.
- Server processes only the NEWEST frame and drops stale ones. Never build a queue; a queue makes the glove lag the hand.
- Network time = laptop round trip minus server time (only the laptop clock is used).
- Budget ~30-60 ms for tracking inside the ~200 ms capture-to-vibration target; log timestamps at every stage.
- Also run the pipeline laptop-only, to compare latency, frame rate, and detection quality for the Vultr prize write-up.
- Detector runs at session start and on re-acquire. Tracker (SAM 2) runs every frame. Hand pose every frame.
  Depth every frame or every 2nd-3rd frame on small patches.

Order of work: webcam + detector printing boxes, then SAM 2 + depth, then move to Vultr and measure latency.

## Notes

- Verify model IDs and library APIs on Hugging Face or the project docs before relying on them (candidates: `google/owlv2-base-patch16-ensemble`, `IDEA-Research/grounding-dino-tiny`, `depth-anything/Depth-Anything-V2-Small-hf`; SAM 2 ships as Meta's `sam2` package). Note OWLv2 boxes are relative to a padded square image, so convert box coordinates in one tested helper.
