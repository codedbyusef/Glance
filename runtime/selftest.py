"""Client selftest (Step 5): run it first after each boot, on the Pi or the laptop.

    python -m runtime.selftest                  # everything, including audio (plays a sentence, records 3 s)
    python -m runtime.selftest --skip-audio     # camera, servers and glove only
    python -m runtime.selftest --glove hardware --glove-port /dev/ttyUSB0

Checks, each PASS / WARN / FAIL (exit 1 if any FAIL):
    env       GEMINI_API_KEY, GLANCE_STT_URL/TOKEN, GLANCE_VISION_URL/TOKEN set (values are never printed)
    camera    opens, warms up, frame size and FPS
    stt       the STT server's /health (with the token)
    vision    the vision server's /health (models loaded, depth on), then a reach start + cancel over /ws
    glove     fake: nothing to check; hardware: the serial port opens and takes an all-off
    audio     audio's own selftest: speaker, thinking tone, mic level, both STT backends
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request

from . import config as runtime_config

RESULTS: list[tuple[str, str, str]] = []


def report(check: str, status: str, detail: str = "") -> None:
    RESULTS.append((check, status, detail))
    print(f"{status:4s}  {check:8s} {detail}", flush=True)


def check_env() -> None:
    from guide_agent.config import load_dotenv

    load_dotenv()
    for name, needed in (("GEMINI_API_KEY", True), ("GLANCE_STT_URL", True), ("GLANCE_STT_TOKEN", False),
                         ("GLANCE_VISION_URL", True), ("GLANCE_TOKEN", False)):
        is_set = bool(os.environ.get(name, "").strip())
        report("env", "PASS" if is_set else ("FAIL" if needed else "WARN"), f"{name} {'set' if is_set else 'NOT set'}")


def check_camera(rt: dict, source) -> None:
    from .camera import CameraHub

    try:
        hub = CameraHub.from_config(rt["camera"], 1024, source=source).start()
    except SystemExit as e:
        report("camera", "FAIL", str(e))
        return
    try:
        if not hub.ready.wait(15):
            report("camera", "FAIL", "opened but no frames within 15 s")
            return
        time.sleep(1.0)
        image = hub.newest_bgr()
        report("camera", "PASS", f"source {source if source is not None else rt['camera']['source']}: "
                                 f"{image.shape[1]}x{image.shape[0]} at {hub.fps:.0f} FPS")
    finally:
        hub.stop()


def check_stt() -> None:
    url, token = os.environ.get("GLANCE_STT_URL", "").rstrip("/"), os.environ.get("GLANCE_STT_TOKEN", "")
    if not url:
        report("stt", "FAIL", "GLANCE_STT_URL not set (the local tiny.en fallback still works)")
        return
    req = urllib.request.Request(url + "/health", headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            info = json.loads(r.read().decode())
        report("stt", "PASS" if info.get("ready") else "WARN", f"{url}: {info}")
    except Exception as e:
        report("stt", "FAIL", f"{url}/health: {e}")


def check_vision() -> None:
    import numpy as np

    from object_detection.client import server_url
    from object_detection.reach_api import ReachAPI

    try:
        url = server_url("/health", scheme="http")
        with urllib.request.urlopen(url, timeout=5) as r:
            info = json.loads(r.read().decode())
    except Exception as e:
        report("vision", "FAIL", f"/health: {e}")
        return
    ok = info.get("models_loaded")
    report("vision", "PASS" if ok else "FAIL", f"{url}: models_loaded={ok} fake={info.get('fake')} "
                                               f"gpu={info.get('gpu')} depth={info.get('depth')}")
    if info.get("fake"):
        report("vision", "WARN", "the server runs FAKE models (GLANCE_FAKE_MODELS=1)")
    if not info.get("depth"):
        report("vision", "WARN", "no depth model on the server: success uses the 2D rule")

    got = threading.Event()
    results = []
    frame = np.zeros((480, 640, 3), np.uint8)
    api = ReachAPI(on_result=lambda r: (results.append(r), got.set()), frame_source=lambda: frame)
    try:
        t0 = time.monotonic()
        sid = api.start("sheet of paper")
        started_ms = (time.monotonic() - t0) * 1000
        time.sleep(0.3)
        api.cancel(sid)
        if got.wait(5) and results[0].status == "cancelled":
            report("vision", "PASS", f"/ws reach start + cancel ok (start {started_ms:.0f} ms)")
        else:
            report("vision", "FAIL", f"/ws: no cancelled result ({[r.status for r in results]})")
    except Exception as e:
        report("vision", "FAIL", f"/ws reach start: {e} (wrong GLANCE_TOKEN?)")
    finally:
        api.close()


def check_glove(kind: str, port, on_level: int, firmware: str = "s3") -> None:
    from .glove_out import OFF, make_glove

    if kind == "fake":
        report("glove", "PASS", "fake glove (drawn on the live view)")
        return
    try:
        glove = make_glove(kind, port, on_level, firmware)
        hw = glove.outputs[0]
        glove.apply(OFF)
        glove.all_off()
        said = hw.read_lines()
        glove.close()
        report("glove", "PASS", f"hardware ({firmware}) on {hw.port}: port opened, all-off sent"
                                f"{'; firmware said ' + repr(said[-1]) if said else ''} "
                                f"(feel each motor with python -m runtime.glove test)")
    except Exception as e:
        report("glove", "FAIL", f"hardware on {port}: {e!r}")


def check_audio() -> None:
    from audio import cli

    print("---- audio selftest (plays a sentence, then records 3 s: say something) ----", flush=True)
    try:
        rc = cli.main(["selftest"])
    except SystemExit as e:
        rc = e.code or 0
    except Exception as e:
        report("audio", "FAIL", repr(e))
        return
    report("audio", "PASS" if rc == 0 else "FAIL", "audio selftest (details above)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Client selftest: env, camera, STT, vision, glove, audio.")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--camera", help="camera index or video path (default: runtime/config)")
    parser.add_argument("--glove", choices=("fake", "hardware"), help="default: glove.kind in runtime/config")
    parser.add_argument("--glove-port")
    args = parser.parse_args(argv)
    runtime_config.load_env()  # runtime/glance.env, if there is one
    from object_detection.common import load_env_file

    load_env_file()  # object_detection/.env too (the vision client reads it), so the env check agrees with it
    rt = runtime_config.load()

    check_env()
    check_camera(rt, args.camera)
    check_stt()
    check_vision()
    check_glove(args.glove or rt["glove"]["kind"], args.glove_port or rt["glove"]["port"], rt["glove"]["on_level"],
                rt["glove"]["firmware"])
    if not args.skip_audio:
        check_audio()
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS)} checks: {len(fails)} FAIL, {sum(r[1] == 'WARN' for r in RESULTS)} WARN", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
