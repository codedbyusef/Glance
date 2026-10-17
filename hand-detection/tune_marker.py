"""Pick the marker's HSV range under real lighting, then paste it into hand-detection/config.yaml.

    python hand-detection/tune_marker.py            # webcam index from config.yaml
    python hand-detection/tune_marker.py --cam 1

Click on the marker in the "camera" window to auto-fill a range around that
pixel, then nudge the sliders until the "mask" window shows ONLY the marker
(white blob) while you move your hand around the table. Press p to print the
YAML block for config.yaml, q to quit. Shows the tracked point and the per-frame time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hand_marker as hand  # noqa: E402

SETTINGS = yaml.safe_load(hand.CONFIG_PATH.read_text())

WIN, MASK = "camera", "mask"
SLIDERS = [("H lo", 179), ("H hi", 179), ("S lo", 255), ("S hi", 255), ("V lo", 255), ("V hi", 255)]


def set_range(lo, hi):
    for (name, _), v in zip(SLIDERS, [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]):
        cv2.setTrackbarPos(name, WIN, int(v))


def get_range():
    v = [cv2.getTrackbarPos(n, WIN) for n, _ in SLIDERS]
    return [v[0], v[2], v[4]], [v[1], v[3], v[5]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default=None)
    args = ap.parse_args()
    src = args.cam if args.cam is not None else SETTINGS["camera"]["source"]
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)

    cfg = hand._load_cfg()
    cv2.namedWindow(WIN)
    cv2.namedWindow(MASK)
    for name, mx in SLIDERS:
        cv2.createTrackbar(name, WIN, 0, mx, lambda _: None)
    set_range(*cfg["hsv_ranges"][0])

    frame_holder = {}

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and "f" in frame_holder:
            hsv = cv2.cvtColor(frame_holder["f"], cv2.COLOR_BGR2HSV)
            patch = hsv[max(0, y - 3):y + 4, max(0, x - 3):x + 4].reshape(-1, 3)
            h, s, v = np.median(patch, axis=0)
            set_range([max(0, h - 10), max(0, s - 70), max(0, v - 70)],
                      [min(179, h + 10), 255, 255])

    cv2.setMouseCallback(WIN, on_click)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 480))
        frame_holder["f"] = frame
        lo, hi = get_range()
        cfg["hsv_ranges"] = [[lo, hi]]

        t0 = time.perf_counter()
        pt = hand.locate_in_image(frame)
        ms = (time.perf_counter() - t0) * 1000

        view = frame.copy()
        if pt:
            cv2.circle(view, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), 2)
        cv2.putText(view, f"{ms:.1f} ms  {'FOUND' if pt else 'none'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(WIN, view)
        cv2.imshow(MASK, hand.marker_mask(frame))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print(f"hand_marker:\n  hsv_ranges:\n    - [{lo}, {hi}]\n"
                  f"  min_area_px: {cfg['min_area_px']}\n  max_area_frac: {cfg['max_area_frac']}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
