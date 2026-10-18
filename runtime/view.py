"""The shared live view for every live script (CLAUDE.md section 6): --view window | web | none.

    window  an OpenCV window. show() must be called on the main thread; it returns the keys pressed.
    web     MJPEG at http://<host>:<port>/, so a headless Pi can be watched from a browser over Tailscale.
            Keys come from the terminal instead (runtime/keys.py).
    none    nothing is drawn; logs only.

Also the overlay helpers every script uses: text panels, an inset image, FPS and per-stage timings.
"""
from __future__ import annotations

import collections
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

MODES = ("window", "web", "none")
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Timings:
    """Latest per-stage timings in ms, written from any thread, drawn by the UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ms: dict[str, float] = {}

    def set(self, stage: str, ms: float) -> None:
        with self._lock:
            self._ms[stage] = ms

    def items(self) -> list[tuple[str, float]]:
        with self._lock:
            return list(self._ms.items())


class FpsCounter:
    def __init__(self, window: int = 30):
        self._stamps: collections.deque[float] = collections.deque(maxlen=window)

    def tick(self) -> float:
        self._stamps.append(time.monotonic())
        if len(self._stamps) < 2 or self._stamps[-1] == self._stamps[0]:
            return 0.0
        return (len(self._stamps) - 1) / (self._stamps[-1] - self._stamps[0])


# ------------------------------------------------------------------ drawing helpers

def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image.copy()
    return cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def text_panel(img: np.ndarray, lines: list[str | tuple[str, tuple[int, int, int]]], origin: tuple[int, int],
               scale: float = 0.5, width: int | None = None) -> int:
    """Lines of text on a dark translucent box. A line is a str or (str, BGR color). Returns the box's bottom y."""
    if not lines:
        return origin[1]
    x, y = origin
    step = int(22 * scale / 0.5)
    items = [(ln, (235, 235, 235)) if isinstance(ln, str) else ln for ln in lines]
    w = width or max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t, _ in items) + 12
    h = step * len(items) + 8
    y2, x2 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
    if y2 > y and x2 > x:
        roi = img[y:y2, x:x2]
        roi[:] = (roi * 0.35).astype(np.uint8)
    for i, (t, color) in enumerate(items):
        cv2.putText(img, t, (x + 6, y + step * (i + 1)), FONT, scale, color, 1, cv2.LINE_AA)
    return y + h


def inset(img: np.ndarray, small: np.ndarray, corner: str = "br", width: int = 240, label: str = "") -> None:
    """Paste a scaled copy of small into a corner of img (br, bl, tr, tl), with an optional label."""
    small = fit_width(small, width)
    h, w = small.shape[:2]
    H, W = img.shape[:2]
    if h + 24 > H or w + 10 > W:
        return
    x = W - w - 8 if corner[1] == "r" else 8
    y = H - h - 8 if corner[0] == "b" else 8
    img[y:y + h, x:x + w] = small
    cv2.rectangle(img, (x - 1, y - 1), (x + w, y + h), (255, 255, 255), 1)
    if label:
        text_panel(img, [label], (x, max(0, y - 22)), scale=0.42)


def blank(width: int = 960, height: int = 540, message: str = "waiting for the camera...") -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    cv2.putText(img, message, (20, height // 2), FONT, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ outputs

class _MjpegServer:
    """multipart/x-mixed-replace stream of the newest frame. / is a page with the stream, /stream the stream."""

    def __init__(self, port: int):
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (b"<!doctype html><title>Glance live view</title>"
                            b"<body style='margin:0;background:#111'><img src='/stream' style='max-width:100%'></body>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with server._cond:
                            server._cond.wait_for(lambda: server._seq != seen, timeout=5)
                            jpeg, seen = server._jpeg, server._seq
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="mjpeg", daemon=True)
        self._thread.start()

    def push(self, image: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._cond:
                self._jpeg, self._seq = buf.tobytes(), self._seq + 1
                self._cond.notify_all()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class View:
    def __init__(self, mode: str, title: str = "glance", port: int = 8090, max_fps: float = 30.0):
        if mode not in MODES:
            raise ValueError(f"--view must be one of {MODES}")
        self.mode = mode
        self.title = title
        self.port = port
        self._period = 1.0 / max_fps
        self._last = 0.0
        self._server = _MjpegServer(port) if mode == "web" else None
        if mode == "window":  # resizable (drag the corner), keeping the aspect ratio
            cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        if mode == "web":
            print(f"live view: http://<this host>:{port}/  (keys: type a letter, then Enter)", flush=True)

    def show(self, image: np.ndarray | None) -> list[str]:
        """Display one frame. Returns the keys pressed since the last call (window mode only), as
        one-character strings; Enter is returned as "\\n". Paces web/none to max_fps."""
        keys: list[str] = []
        if self.mode == "window":
            if image is not None:
                cv2.imshow(self.title, image)
            k = cv2.waitKey(1)
            if k != -1:
                k &= 0xFF
                keys.append("\n" if k in (10, 13) else chr(k).lower())
            return keys
        if self.mode == "web" and image is not None:
            self._server.push(image)
        wait = self._period - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        return keys

    def close(self) -> None:
        if self.mode == "window":
            cv2.destroyAllWindows()
        if self._server is not None:
            self._server.close()
