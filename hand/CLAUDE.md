# Hand detection

Covers `hand/` (hand localization). Read the root `CLAUDE.md` first for priorities, ownership and conventions.

The glove guides the hand by comparing the gloved hand's position with the target's position in the glasses-camera
image, so we need the hand's position every frame. `server/session.py` calls `hand/locate.py` on each frame. It
runs in the server process by default and can move to the laptop (open decision, settled by the first latency
measurement), so keep it a pure function with no server imports. If it moves, the client sends the hand point with
each frame so the server can sample depth there.

## Decisions

- Use MediaPipe Hands, not the object detector with "hand" as the target. It gives 21 landmarks (fingertips, palm
  center) and a clean "no hand" signal. An open-vocab detector gives only a box, runs slower, and has no fingertips.
- Track only the gloved hand (right by default) and ignore other hands in view. MediaPipe labels handedness assuming
  a mirrored selfie image and our frames are not mirrored, so flip the label (confirm with a quick test).
- No hand orientation. The finger mapping is fixed in the image frame (see `haptics/CLAUDE.md`).

## Output

Per frame: fingertips and palm center in image coordinates, confidence, hand present or not, frame id, timestamp.

`locate_hand` currently returns a single point, but the success rule needs fingertips and a hand region (`server/session.py`
has a note about this). Extend its return type and `FrameResult` in `shared/schemas.py` together, and tell the server
owner.

## Where the output is used

- `success`: any fingertip inside the target box with hand depth near target depth for 5-10 frames in a row.
- `hand_lost`: no hand for over ~1.5 s. Keep that grace period; a head camera often loses the hand when it drops low
  or to the side.
- At the end of a reach the hand covers the object. If the hand covered over half the target box in the last few
  frames, the session counts it as `success` (`inferred`) even after the tracker loses the target.

## Risk

The glove's motors and wires may break MediaPipe, especially from a head-mounted view. Test with the real glove early.
Fallback: a bright colored marker or an AprilTag on the glove, tracked with OpenCV (a second module in `hand/` behind
the same `locate_hand` interface).

## Context from the agent

The agent tells the user to hold the gloved hand out in front of them, keep their head still, and tilt their head
down when the target is low in the frame. Tilting the glasses camera slightly down would keep the hand in view more
often (raise with hardware).

## Order of work

Webcam + MediaPipe printing fingertip coordinates, then the same with the glove on, then print dx/dy against the
object box.
