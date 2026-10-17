"""Live test of depth on its own: band tracker + depth model, no server, no glove.

    python depth-estimation/test_depth_live.py                       # webcam from config.yaml
    python depth-estimation/test_depth_live.py --video hand_test.mp4 # replay a recording
    python depth-estimation/test_depth_live.py --video clip.mp4 --model metric --target-box 260,200,380,300

Each frame goes through the same path as the server: resize to the frame size,
JPEG-encode at the server quality, decode. The hand comes from
hand-detection/hand_marker.py (band pixels only). The target is a box you draw
(key t) until object-detection's tracker is wired in.

Windows: "depth live" (frame + boxes + numbers) and "depth map" (closer = brighter).

Keys:
    t  draw the target box (drag, then Enter). Prints a --target-box value for replays.
    click   probe the depth at a point (for the tape-measure check)
    1 palm down   2 sideways   3 palm up   4 moving fast   0 no label
    c  capture the contact offset: hold the hand in real contact with the target first
    f  calibrate the focal length: hold the band at --calib-distance meters first
    r  start/stop recording raw frames to depth_test.mp4
    space  pause/resume      q  quit and print the summary

Every frame and event is logged as one JSON line (default: depth-estimation/logs/).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "hand-detection"))
import depth_config  # noqa: E402
import depth_regions as regions  # noqa: E402
from depth_model import DepthCache, DepthModel  # noqa: E402
from depth_tracker import DepthTracker  # noqa: E402
import hand_marker as hand  # noqa: E402

LABELS = {ord("0"): "unlabeled", ord("1"): "palm down", ord("2"): "sideways",
          ord("3"): "palm up", ord("4"): "moving fast"}
CAPTURE_FRAMES = 15


def parse_box(text):
    if not text:
        return None
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--target-box needs x1,y1,x2,y2")
    return tuple(parts)


def fmt(v, units, digits=3):
    if v is None:
        return "--"
    return f"{v * 100:.1f} cm" if units == "metric" else f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--model", choices=["relative", "metric"], default=None, help="override model.kind")
    ap.add_argument("--object", default=None, help="contact_offset object (paper, can, ...)")
    ap.add_argument("--every-n", type=int, default=None, help="override model.every_n_frames")
    ap.add_argument("--target-box", type=parse_box, default=None, help="x1,y1,x2,y2 (from key t)")
    ap.add_argument("--calib-distance", type=float, default=None, help="meters, for key f")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    overrides = {}
    if args.model:
        overrides["model"] = {"kind": args.model}
    if args.every_n:
        overrides.setdefault("model", {})["every_n_frames"] = args.every_n
    cfg = depth_config.load(**overrides)
    units = depth_config.units(cfg)
    W, H = cfg["frame"]["frame_width"], cfg["frame"]["frame_height"]
    quality = cfg["frame"]["jpeg_quality"]

    src = args.video or (args.cam if args.cam is not None else cfg["camera"]["source"])
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"Could not open camera/video: {src}")

    print(f"loading {units} model ({cfg['model']['ids'][units]}) ...")
    model = DepthModel(cfg)
    model.warmup()
    print(f"device: {model.device}{'  (FAKE MODEL)' if model.fake else ''}")
    cache = DepthCache(model)
    tracker = DepthTracker(cfg, object_name=args.object)
    hand.reset()

    log_path = Path(args.log) if args.log else HERE / "logs" / f"live_{units}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")

    def emit(kind, **fields):
        log.write(json.dumps({"ts": time.time(), "kind": kind, **fields}, default=float) + "\n")

    emit("start", units=units, model_id=cfg["model"]["ids"][units], device=model.device,
         fake=model.fake, object=tracker.object_name, offset=tracker.offset, source=str(src))

    target_box = args.target_box
    label = "unlabeled"
    stats = defaultdict(lambda: defaultdict(int))
    band_vs_model = defaultdict(list)
    times = defaultdict(list)
    recent = deque(maxlen=CAPTURE_FRAMES)
    probe = {"point": None}
    writer, paused, frame_id = None, False, 0
    frame = view = None
    dmap = None

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            probe["point"] = (x, y)

    cv2.namedWindow("depth live")
    cv2.setMouseCallback("depth live", on_mouse)

    while True:
        if not paused:
            ok, raw = cap.read()
            if not ok:
                break
            raw = cv2.resize(raw, (W, H))
            if writer is not None:
                writer.write(raw)
            jpg = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            frame_id += 1

            t0 = time.perf_counter()
            marker = hand.detect(frame)
            band_mask = None
            if marker is not None:
                band_mask = regions.band_mask_from_marker(hand.marker_mask(frame) > 0, marker.box)
            t1 = time.perf_counter()
            dmap, age = cache.get(frame, frame_id)
            t2 = time.perf_counter()
            reading = tracker.update(
                dmap, frame_id, band_mask=band_mask, target_box=target_box,
                hand_point=marker.point if marker else None,
                band_area_px=marker.area_px if marker else None, map_age=age)
            t3 = time.perf_counter()

            timings = {"hand": (t1 - t0) * 1000, "depth_map": (t2 - t1) * 1000, **reading.timings_ms,
                       "total": (t3 - t0) * 1000}
            if age == 0:
                timings.update(model.last_timings)
            for k, v in timings.items():
                if k == "depth_map" and age > 0:
                    continue                      # reused map: don't count as an inference
                times[k].append(v)
            recent.append(reading)

            s = stats[label]
            s["frames"] += 1
            s["hand"] += reading.hand_valid
            s["target"] += reading.target_valid
            s["inconsistent"] += reading.inconsistent
            s["frozen"] += reading.target_frozen
            if reading.band_distance_m is not None and reading.hand is not None and units == "metric":
                band_vs_model[label].append(abs(reading.band_distance_m - reading.hand))

            emit("frame", pose=label, target_box=target_box,
                 hand_point=marker.point if marker else None,
                 band_area_px=marker.area_px if marker else None,
                 reading=reading.to_dict(), timings_ms=timings)

            view = draw(frame, marker, target_box, reading, units, timings, label, writer is not None)

        if probe["point"] is not None and dmap is not None:
            x, y = probe["point"]
            patch = dmap[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
            value = float(np.median(patch))
            print(f"probe ({x}, {y}): {fmt(value, units)}  -- compare with the tape measure")
            emit("probe", point=[x, y], value=value, units=units, frame_id=frame_id)
            if view is not None:
                cv2.circle(view, (x, y), 6, (255, 0, 255), 2)
                cv2.putText(view, fmt(value, units), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            probe["point"] = None

        if view is not None:
            cv2.imshow("depth live", view)
        if dmap is not None:
            cv2.imshow("depth map", colorize(dmap))

        key = cv2.waitKey(1 if not args.video else 30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key in LABELS:
            label = LABELS[key]
        if key == ord("t") and frame is not None:
            x, y, w, h = cv2.selectROI("depth live", frame, showCrosshair=True)
            if w > 0 and h > 0:
                target_box = (float(x), float(y), float(x + w), float(y + h))
                tracker.reset()
                print(f"target box set. For replays: --target-box {x},{y},{x + w},{y + h}")
                emit("target_box", box=target_box)
        if key == ord("c"):
            capture_offset(recent, tracker, units, emit)
        if key == ord("f"):
            capture_focal(recent, cfg, args.calib_distance, emit)
        if key == ord("r"):
            if writer is None:
                writer = cv2.VideoWriter("depth_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
            else:
                writer.release()
                writer = None
                print("saved depth_test.mp4")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    summary(stats, band_vs_model, times, units, model)
    emit("end", frames=frame_id, gpu_memory_mb=model.gpu_memory_mb())
    log.close()
    print(f"log: {log_path}")


def capture_offset(recent, tracker, units, emit):
    dz = [r.dz for r in recent if r.dz is not None and r.hand_in_box]
    if len(dz) < CAPTURE_FRAMES // 2:
        print("contact offset: not enough frames with the hand inside the target box and both depths valid")
        return
    value = float(np.median(dz))
    print(f"\ncontact offset for '{tracker.object_name}' ({units}): {value:.4f}"
          f"  (spread {np.std(dz):.4f} over {len(dz)} frames)")
    print(f"paste into config.yaml under contact_offset.{units}:\n    {tracker.object_name}: {value:.4f}\n")
    emit("contact_offset", object=tracker.object_name, units=units, value=value, spread=float(np.std(dz)), n=len(dz))


def capture_focal(recent, cfg, distance, emit):
    if distance is None:
        print("focal calibration: start the tool with --calib-distance <meters>")
        return
    widths = [r.band_width_px for r in recent if r.band_width_px]
    if len(widths) < CAPTURE_FRAMES // 2:
        print("focal calibration: band not visible in enough recent frames")
        return
    width_px = float(np.median(widths))
    f = regions.focal_from_band(width_px, distance, cfg["band"]["width_m"])
    print(f"\nband width {width_px:.1f} px at {distance} m -> focal_px {f:.1f}")
    print(f"paste into config.yaml under band:\n    focal_px: {f:.1f}\n")
    emit("focal", width_px=width_px, distance_m=distance, focal_px=f)


def colorize(dmap):
    lo, hi = np.percentile(dmap, [2, 98])
    norm = np.clip((hi - dmap) / max(hi - lo, 1e-6), 0, 1)        # closer = brighter
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def draw(frame, marker, target_box, r, units, timings, label, recording):
    view = frame.copy()
    if target_box is not None:
        color = (0, 255, 0) if r.touching else (255, 200, 0)
        x1, y1, x2, y2 = (int(v) for v in target_box)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        if r.target_frozen:
            cv2.putText(view, "FROZEN", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if marker is not None:
        cv2.rectangle(view, marker.box[:2], marker.box[2:], (0, 255, 255), 2)
        cv2.circle(view, (int(marker.point[0]), int(marker.point[1])), 6, (0, 0, 255), -1)
    lines = [
        f"[{label}] {units}  map age {r.map_age}  {timings['total']:.0f} ms total",
        f"hand {fmt(r.hand, units)}   target {fmt(r.target, units)}",
        f"dz {fmt(r.dz, units)}   gap {fmt(r.gap, units)}   band {r.band or '--'}",
        f"in box {r.hand_in_box}  streak {r.touch_streak}  {'TOUCHING' if r.touching else ''}"
        f"{'  CHECK: area vs depth' if r.inconsistent else ''}",
    ]
    if r.band_distance_m is not None:
        lines.append(f"band pinhole {r.band_distance_m * 100:.1f} cm  (width {r.band_width_px:.1f} px)")
    for i, text in enumerate(lines):
        cv2.putText(view, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if recording:
        cv2.putText(view, "REC", (view.shape[1] - 60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return view


def summary(stats, band_vs_model, times, units, model):
    print(f"\n=== depth test summary ({units}, {model.device}{', FAKE' if model.fake else ''}) ===")
    for name, s in stats.items():
        n = s["frames"]
        if n:
            print(f"{name:12s} {n:5d} frames  hand {100 * s['hand'] / n:3.0f}%  target {100 * s['target'] / n:3.0f}%"
                  f"  frozen {100 * s['frozen'] / n:3.0f}%  area-vs-depth flags {100 * s['inconsistent'] / n:3.0f}%")
    for name, diffs in band_vs_model.items():
        if diffs:
            print(f"band pinhole vs model hand depth [{name}]: median |diff| {np.median(diffs) * 100:.1f} cm"
                  f" over {len(diffs)} frames")
    for k in ("depth_map", "depth_pre", "depth_infer", "depth_post", "depth_sample", "hand", "total"):
        if times.get(k):
            t = np.array(times[k])
            print(f"{k:13s} p50 {np.percentile(t, 50):6.1f} ms   p95 {np.percentile(t, 95):6.1f} ms")
    mem = model.gpu_memory_mb()
    if mem is not None:
        print(f"peak GPU memory: {mem:.0f} MB")


if __name__ == "__main__":
    main()
