# hand_detection

Finds the user's hand in head-camera frames by tracking a neon tape band around the
glove's knuckles. Used instead of MediaPipe Hands, which the glove's motors and wires
break. Self-contained: everything it needs is in this package. Deps: `runtime/requirements.txt` (client)
or `object_detection/requirements-server.txt` (vision server).

## The glove

One matte fluorescent-green tape band, 2-3 cm wide, wrapped all the way around the
knuckles (back, both edges, palm). No other marker. Some of the band faces the camera
whether the hand is palm down, sideways or palm up. Keep wires off it, avoid glossy tape,
and keep other green things out of view.

## How it works

Each frame: HSV color threshold -> clean up the mask -> find band pieces -> merge pieces
that are close together -> return their area-weighted center as the hand point.
Hand orientation is never estimated: thumb = left in the camera's view, always.

```python
from hand_detection import hand_marker
point = hand_marker.locate_hand(jpeg_bytes)   # (x, y) in pixels, or None
info  = hand_marker.detect(bgr_image)         # point, box, area_px, pieces
hand_marker.reset()                           # call at the start of each reach
```

## Files

| File | What it does |
|---|---|
| `hand_marker.py` | The tracker. |
| `config.yaml` | Band color range and thresholds, frame size, camera source. |
| `scripts/tune_marker.py` | Pick the band's color range under real lighting. |
| `scripts/test_hand_live.py` | Live test on a camera or recorded video: detection rate per pose, time per frame. |
| `tests/test_hand_marker.py` | Unit tests on drawn frames, no camera needed. |

## Testing (run from the repo root)

1. **Unit tests**: `python -m pytest hand_detection -v`
2. **Tune the color** with the glove on: `python -m hand_detection.scripts.tune_marker`.
   Click the band, adjust sliders until only the band is white in the mask window,
   press `p`, paste the printed block into `config.yaml`.
3. **Live test**: `python -m hand_detection.scripts.test_hand_live`.
   Keys: `1` palm down, `2` sideways, `3` palm up, `4` moving fast, `r` record,
   `q` quit and print the summary. Replay a recording with `--video hand_test.mp4`.
4. Repeat 2-3 with the head camera under the demo lighting.

Good result: palm down near 100% found; sideways/palm up can be lower (short gaps are
expected); the point doesn't jump to other green objects; area grows as the hand nears
the camera.
