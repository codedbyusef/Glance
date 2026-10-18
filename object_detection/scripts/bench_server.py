"""Per-frame server time with every model loaded, and hand detection, at each frame size (CLAUDE.md Step 2).

Runs the same frames through the server's own reach session (detect + track + hand band + depth) in-process,
at each size, and prints p50/p95 per stage, the band detection rate and its area. Run it on the GPU host in
the vision venv; record a clip first with the glove and the object in view.

    .venv-vision\\Scripts\\python -m object_detection.scripts.bench_server --video can_reach.mp4 --target "red can"
    .venv-vision\\Scripts\\python -m object_detection.scripts.bench_server --camera 0 --target "red can" --frames 150
    ... --sizes 480x360 640x480 --json bench_3070.json

GLANCE_FAKE_MODELS=1 runs it without a GPU (wiring check only; the numbers mean nothing).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import cv2
import numpy as np

from object_detection import server


def read_frames(video: str | None, camera: int | None, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video if video else camera)
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            if video and frames:  # loop a short clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit("no frames read")
    return frames


def pct(values: list[float], q: float) -> float:
    vs = sorted(values)
    return vs[min(len(vs) - 1, int(round(q * (len(vs) - 1))))]


def run_size(frames: list[np.ndarray], size: tuple[int, int], target: str, jpeg_quality: int) -> dict:
    stages: dict[str, list[float]] = {}
    total: list[float] = []
    hands, areas, sessions = 0, [], 0
    session = None
    for i, frame in enumerate(frames, 1):
        if session is None or session.done:
            session = server.new_session(target, "")
            sessions += 1
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        # what the wire does: JPEG on the laptop, decode on the server
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        t0 = time.perf_counter()
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        result = session.handle_frame(decoded, i)
        total.append((time.perf_counter() - t0) * 1000)
        if result is None:
            continue
        for k, v in result.timings_ms.items():
            stages.setdefault(k, []).append(v)
        if result.hand_point is not None:
            hands += 1
            if result.band_area_px:
                areas.append(result.band_area_px)
    out = {"size": f"{size[0]}x{size[1]}", "frames": len(frames), "sessions": sessions,
           "server_ms": {"p50": statistics.median(total), "p95": pct(total, 0.95)},
           "stages": {k: {"p50": statistics.median(v), "p95": pct(v, 0.95), "n": len(v)} for k, v in stages.items()},
           "hand_found": hands / len(frames), "band_area_px_median": statistics.median(areas) if areas else None,
           "jpeg_kb": round(len(buf) / 1024, 1)}
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu_mem_max_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    except ImportError:
        pass
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--camera", type=int)
    parser.add_argument("--target", required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--sizes", nargs="+", default=["480x360", "640x480"])
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args(argv)

    cfg = server.load_config()
    server._models.update(server.load_models())
    frames = read_frames(args.video, args.camera, args.frames)
    print(f"{len(frames)} frames, target {args.target!r}, depth "
          f"{'on' if server._models.get('depth') is not None else 'off'}, fake models: "
          f"{server._models['detector'].__class__.__name__.startswith('Fake')}")
    results = []
    for spec in args.sizes:
        w, h = (int(v) for v in spec.lower().split("x"))
        r = run_size(frames, (w, h), args.target, int(cfg["frame"]["jpeg_quality"]))
        results.append(r)
        print(f"\n== {r['size']}: server p50 {r['server_ms']['p50']:.1f} ms, p95 {r['server_ms']['p95']:.1f} ms; "
              f"hand found {r['hand_found']:.0%}, band area median {r['band_area_px_median']}; "
              f"JPEG {r['jpeg_kb']} KB; reach sessions {r['sessions']}"
              + (f"; GPU max {r['gpu_mem_max_mb']} MB" if "gpu_mem_max_mb" in r else ""))
        for k, v in r["stages"].items():
            print(f"   {k:14s} p50 {v['p50']:7.1f}  p95 {v['p95']:7.1f}  n={v['n']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
