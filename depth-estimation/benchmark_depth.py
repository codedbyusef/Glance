"""Depth latency per input size. Headless, so it runs over SSH on the Vultr instance.

    python depth-estimation/benchmark_depth.py                          # synthetic frames
    python depth-estimation/benchmark_depth.py --video hand_test.mp4    # real frames
    python depth-estimation/benchmark_depth.py --model metric --sizes 252 364 518 --runs 100
    python depth-estimation/benchmark_depth.py --json results.json      # for the Vultr write-up

Reports p50/p95 per stage (preprocess, inference, postprocess, sampling) and
peak GPU memory. Run it on the laptop and on Vultr with the same arguments so
the numbers compare directly. When object-detection is running on the same GPU,
run this at the same time to see the combined load.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import depth_config  # noqa: E402
import depth_regions as regions  # noqa: E402
from depth_model import DepthModel  # noqa: E402
from depth_tracker import DepthTracker  # noqa: E402


def load_frames(video, count, w, h, quality):
    frames = []
    if video:
        cap = cv2.VideoCapture(video)
        while len(frames) < count:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            sys.exit(f"no frames read from {video}")
    else:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (h, w, 3), np.uint8) for _ in range(min(count, 20))]
    out = []
    for f in frames:                            # same path as the server: resize + JPEG round trip
        f = cv2.resize(f, (w, h))
        jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        out.append(cv2.imdecode(jpg, cv2.IMREAD_COLOR))
    return out


def pct(values):
    a = np.array(values)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--model", choices=["relative", "metric"], default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[252, 308, 364, 518])
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base = depth_config.load(**({"model": {"kind": args.model}} if args.model else {}))
    w, h = base["frame"]["frame_width"], base["frame"]["frame_height"]
    frames = load_frames(args.video, args.runs, w, h, base["frame"]["jpeg_quality"])

    # Fixed sampling regions so the sampling cost is measured too.
    band = regions.box_to_mask((w * 0.25, h * 0.7, w * 0.35, h * 0.73), (h, w))
    target = (w * 0.4, h * 0.4, w * 0.6, h * 0.6)

    results = {"host": platform.node(), "frame": [w, h], "video": args.video, "sizes": {}}
    model = None
    for size in args.sizes:
        cfg = depth_config.load(**depth_config._merge(
            {"model": {"kind": args.model}} if args.model else {}, {"model": {"input_size": size}}))
        if model is None:
            model = DepthModel(cfg)
            results.update(units=model.kind, device=model.device, fake=model.fake,
                           model_id=cfg["model"]["ids"][model.kind], fp16=cfg["model"]["fp16"])
        else:
            model.cfg = cfg                      # same weights, new input size
        tracker = DepthTracker(cfg)
        for i in range(args.warmup):
            model.infer(frames[i % len(frames)])

        stages = {"depth_pre": [], "depth_infer": [], "depth_post": [], "depth_sample": []}
        for i in range(args.runs):
            dmap = model.infer(frames[i % len(frames)])
            for k, v in model.last_timings.items():
                stages.setdefault(k, []).append(v)
            r = tracker.update(dmap, i, band_mask=band, target_box=target)
            stages["depth_sample"].append(r.timings_ms["depth_sample"])
        stages = {k: v for k, v in stages.items() if v}
        total = [sum(vals) for vals in zip(*stages.values())]
        results["sizes"][size] = {**{k: pct(v) for k, v in stages.items()}, "total": pct(total)}

    results["gpu_memory_mb"] = model.gpu_memory_mb()

    print(f"\ndepth benchmark: {results['units']} on {results['device']}"
          f"{' (FAKE MODEL)' if results['fake'] else ''}, {w}x{h} frames, {args.runs} runs")
    for size, s in results["sizes"].items():
        parts = "  ".join(f"{k.replace('depth_', '')} {v['p50']:.1f}/{v['p95']:.1f}" for k, v in s.items())
        print(f"input {size:4d}: {parts}   (ms, p50/p95)")
    if results["gpu_memory_mb"] is not None:
        print(f"peak GPU memory: {results['gpu_memory_mb']:.0f} MB")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
