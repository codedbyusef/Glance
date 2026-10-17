"""Live test of the hand band tracker on its own: no server, no glove motors.

    python -m hand_detection.scripts.test_hand_live              # webcam from config.yaml
    python -m hand_detection.scripts.test_hand_live --cam 1      # another camera
    python -m hand_detection.scripts.test_hand_live --video clip.mp4

Each frame goes through the same path as the server: resize to the server
frame size, JPEG-encode at the server quality, then hand.locate_hand(bytes).

On screen: the hand point (red), the box around the band (yellow), how many
pieces were merged, the band area, and the time per frame.

Keys while it runs (label what you're doing so the summary splits by pose):
    1 palm down    2 sideways    3 palm up    4 moving fast    0 no label
    r start/stop recording to hand_test.mp4 (replay later with --video)
    q quit and print the summary
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

import yaml

from hand_detection import hand_marker as hand

SETTINGS = yaml.safe_load(hand.CONFIG_PATH.read_text())

LABELS = {ord("0"): "unlabeled", ord("1"): "palm down", ord("2"): "sideways",
          ord("3"): "palm up", ord("4"): "moving fast"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    settings = SETTINGS
    W, H = settings["frame"]["frame_width"], settings["frame"]["frame_height"]
    quality = settings["frame"]["jpeg_quality"]
    src = args.video or (args.cam if args.cam is not None else settings["camera"]["source"])
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"Could not open camera/video: {src}")

    hand.reset()
    label = "unlabeled"
    stats = defaultdict(lambda: {"frames": 0, "found": 0})
    times_ms: list[float] = []
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H))
        if writer is not None:
            writer.write(frame)

        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        t0 = time.perf_counter()
        point = hand.locate_hand(jpg)                     # exactly what the server calls
        times_ms.append((time.perf_counter() - t0) * 1000)

        stats[label]["frames"] += 1
        view = frame.copy()
        if point is not None:
            stats[label]["found"] += 1
            r = hand.detect(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
            if r:
                cv2.rectangle(view, r.box[:2], r.box[2:], (0, 255, 255), 2)
                info = f"pieces {r.pieces}  area {r.area_px:.0f}px"
                cv2.putText(view, info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(view, (int(point[0]), int(point[1])), 8, (0, 0, 255), -1)
            cv2.putText(view, f"({point[0]:.0f}, {point[1]:.0f})", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        s = stats[label]
        rate = 100 * s["found"] / s["frames"]
        status = "FOUND" if point else "LOST"
        cv2.putText(view, f"{status}  {times_ms[-1]:.1f} ms  [{label}] {rate:.0f}% found",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if point else (0, 0, 200), 2)
        if writer is not None:
            cv2.putText(view, "REC", (W - 60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("hand band test", view)
        cv2.imshow("mask", hand.marker_mask(frame))

        key = cv2.waitKey(1 if not args.video else 30) & 0xFF
        if key == ord("q"):
            break
        if key in LABELS:
            label = LABELS[key]
        if key == ord("r"):
            if writer is None:
                writer = cv2.VideoWriter("hand_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
            else:
                writer.release()
                writer = None
                print("saved hand_test.mp4")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    print("\n=== hand band test summary ===")
    for name, s in stats.items():
        if s["frames"]:
            print(f"{name:12s} {s['found']:5d}/{s['frames']:<5d} found  ({100 * s['found'] / s['frames']:.0f}%)")
    if times_ms:
        t = np.array(times_ms)
        print(f"time per frame: p50 {np.percentile(t, 50):.1f} ms, p95 {np.percentile(t, 95):.1f} ms")


if __name__ == "__main__":
    main()
