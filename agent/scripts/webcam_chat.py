"""Temporary: talk to the agent by typing, with live webcam frames and no glove.

    .venv\\Scripts\\python -m scripts.webcam_chat [--camera 1] [--no-window]

Defaults to camera index 1, the Logitech Brio 101 (index 0 is the laptop's built-in webcam). Indexes can shift
when cameras are plugged or unplugged; if the wrong one opens, pass --camera.

A background thread reads the camera continuously, so the frame sent when you press Enter is the newest one.
Two windows open: "live" shows the feed, "sent to model" shows the last frame the agent received, with its id and
the time it was captured. The last frame sent is also saved to logs/last_frame.jpg.

There is no glove: when the agent starts a reach, it is reported as success right away, so you go straight to the
touch check. The harness commands (/speech_start, /idle, /state, /haptic ..., /help) still work; /frame is ignored.

Point the webcam at the scene, the way the glasses camera would see it. Frames are not mirrored.
Needs opencv-python (pip install opencv-python); it is not in requirements.txt.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from datetime import datetime

import cv2

from guide_agent.config import AGENT_ROOT, load_config, load_dotenv
from guide_agent.harness import Harness
from guide_agent.models.gemini import GeminiAdapter
from guide_agent.schemas import Frame, HapticResult

LAST_FRAME = AGENT_ROOT / "logs" / "last_frame.jpg"
WARMUP_FRAMES = 45  # the first frames come out dark while auto-exposure settles
PREVIEW_WIDTH = 400  # preview windows only; the model still gets the full frame


class CameraThread(threading.Thread):
    """Reads the camera nonstop and keeps the newest frame. Also owns the preview windows (all GUI calls stay here)."""

    def __init__(self, index: int, show: bool):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open webcam {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.show = show
        self.lock = threading.Lock()
        self.latest = None
        self.latest_at = 0.0
        self.count = 0
        self.ready = threading.Event()
        self.running = True
        self._sent = None  # annotated copy of the last frame sent, waiting to be displayed

    def run(self) -> None:
        while self.running:
            ok, image = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest, self.latest_at = image, time.time()
                self.count += 1
                sent, self._sent = self._sent, None
            if self.count >= WARMUP_FRAMES:
                self.ready.set()
            if self.show:
                cv2.imshow("live", _small(image))
                if sent is not None:
                    cv2.imshow("sent to model", _small(sent))
                cv2.waitKey(1)
        self.cap.release()
        if self.show:
            cv2.destroyAllWindows()

    def snapshot(self, label: str):
        """Return (image, capture time) of the newest frame, and queue an annotated copy for the 'sent' window."""
        with self.lock:
            image, at = self.latest, self.latest_at
        if image is None:
            return None, 0.0
        shown = image.copy()
        text = f"{label}  captured {datetime.fromtimestamp(at):%H:%M:%S.%f}"[:-3]
        cv2.putText(shown, text, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7)
        cv2.putText(shown, text, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        with self.lock:
            self._sent = shown
        return image, at

    def stop(self) -> None:
        self.running = False
        self.join(timeout=2)


def _small(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)


class WebcamHarness(Harness):
    def __init__(self, cfg, model, camera: CameraThread):
        super().__init__(cfg, model)
        self.camera_thread = camera

    def current_frame(self) -> Frame | None:
        frame_id = f"cam-{self._frames + 1}"
        image, at = self.camera_thread.snapshot(frame_id)
        if image is None:
            self.out("  !! no webcam frame yet; sending no frame")
            return None
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        LAST_FRAME.parent.mkdir(exist_ok=True)
        LAST_FRAME.write_bytes(jpeg.tobytes())
        self._frames += 1
        self.out(f"  frame {frame_id}: captured {(time.time() - at) * 1000:.0f} ms before sending")
        return Frame(id=frame_id, image=jpeg.tobytes(), captured_at=at)

    async def run_command(self, line: str) -> bool:
        keep = await super().run_command(line)
        # No glove: let say_first finish, then report the reach as a success.
        self.audio.finish()
        await self.ctl.drain()
        step = self.ctl.step
        if step is not None and step.type == "reach" and step.state == "running":
            self.out("  (no glove: reporting the reach as success)")
            self.ctl.on_haptic_result(HapticResult(session_id=step.session_id, status="success",
                                                   target=step.target))
            await self.ctl.drain()
            self._print_new_turns()
            self.print_state()
        return keep


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the agent using webcam frames.")
    parser.add_argument("--camera", type=int, default=1, help="webcam index (default 1: the Brio 101)")
    parser.add_argument("--no-window", action="store_true", help="don't open the preview windows")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    load_dotenv()
    cfg = load_config()
    camera = CameraThread(args.camera, show=not args.no_window)
    camera.start()
    if not camera.ready.wait(timeout=10):
        print("webcam is slow to start; frames may be dark at first")
    harness = WebcamHarness(cfg, GeminiAdapter(cfg.model, thinking_level=cfg.thinking_level), camera)
    print(f"model {cfg.model}, webcam {args.camera}. Type what the user says; /quit to exit.")
    try:
        while True:
            try:
                line = await asyncio.to_thread(input, "\n> ")
            except EOFError:
                break
            if not await harness.run_command(line):
                break
    finally:
        camera.stop()


if __name__ == "__main__":
    asyncio.run(main())
