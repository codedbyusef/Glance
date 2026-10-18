"""Step 4: step through each glove motor and the wrist pulse rates, with the live view showing which motor
should be buzzing right now. Check each one by touch.

    python -m runtime.glove_check                                   # fake glove: wiring of the view only
    python -m runtime.glove_check --glove hardware --glove-port COM5
    python -m runtime.glove_check --glove hardware --firmware-test  # also the firmware's own test (each channel once)

Keys (window, or the letter + Enter in the terminal): n next step, space / p pause, q quit. Every exit is all-off.
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

from haptics import protocol

from . import config as runtime_config
from .glove_out import FIRMWARE_CHANNELS, OFF, GloveCommand, glove_from_config
from .keys import StdinRouter
from .view import View, blank, fit_width, text_panel

MEANING = {protocol.THUMB: "LEFT", protocol.INDEX: "FORWARD", protocol.MIDDLE: "reserved (touch sensor)",
           protocol.RING: "BACK", protocol.PINKY: "RIGHT", protocol.BACK: "UP", protocol.PALM: "DOWN",
           protocol.WRIST: "proximity"}
PULSE_RATES = (1.0, 2.0, 4.0, 6.0)


@dataclass
class Step:
    label: str
    cmd: GloveCommand
    seconds: float


def check_steps(hold_s: float = 1.5, gap_s: float = 0.5, pulse_s: float = 3.0, channels=None) -> list[Step]:
    """Every channel the glove has steadily on, one at a time (with an off gap), in its wire order; then the
    wrist at each pulse rate, if it has a wrist motor. channels: logical channels (default all 8)."""
    channels = tuple(range(8)) if channels is None else tuple(channels)
    steps: list[Step] = []
    for wire, ch in enumerate(channels):
        name = protocol.CHANNELS[ch]
        steps.append(Step(f"{name} (wire channel {wire}) = {MEANING[ch]}: steady", GloveCommand(on=frozenset({ch})),
                          hold_s))
        steps.append(Step("off", OFF, gap_s))
    if protocol.WRIST not in channels:
        return steps
    for hz in PULSE_RATES:
        steps.append(Step(f"wrist pulsing at {hz:g} Hz (faster = closer)",
                          GloveCommand(pulse_hz={protocol.WRIST: hz}), pulse_s))
        steps.append(Step("off", OFF, gap_s))
    return steps


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step through each glove motor and pulse rate.")
    parser.add_argument("--glove", choices=("fake", "hardware"), help="default: glove.kind in runtime/config")
    parser.add_argument("--glove-port", help="serial port for --glove hardware (default: glove.port, auto = detect)")
    parser.add_argument("--hold", type=float, default=1.5, help="seconds each motor stays on")
    parser.add_argument("--firmware-test", action="store_true", help="first run the firmware's test mode")
    parser.add_argument("--view", choices=("window", "web", "none"), default=runtime_config.default_view())
    parser.add_argument("--camera", help="camera index or video path (default: runtime/config.yaml)")
    args = parser.parse_args(argv)

    from .camera import CameraHub

    rt = runtime_config.load()
    glove = glove_from_config(rt, args.glove, args.glove_port)
    camera = CameraHub.from_config(rt["camera"], 1024, source=args.camera).start()
    view = View(args.view, title="glance glove check", port=rt["view"]["web_port"])
    hw = getattr(glove, "outputs", [None])[0]
    steps = check_steps(args.hold, channels=getattr(hw, "channels", None)
                        or FIRMWARE_CHANNELS[rt["glove"]["firmware"]])
    quit_flag, skip, paused = threading.Event(), threading.Event(), threading.Event()

    def on_key(k: str) -> None:
        if k == "q":
            quit_flag.set()
        elif k == "n":
            skip.set()
        elif k in ("p", " "):
            paused.clear() if paused.is_set() else paused.set()

    StdinRouter({"q", "n", "p"}, on_key=on_key).start()
    try:
        if args.firmware_test and hw is not None:
            print("firmware test mode: each channel pulses once", flush=True)
            hw.link.test()
            end = time.monotonic() + 6.0
            while time.monotonic() < end:
                for line in hw.read_lines():
                    print(f"  glove: {line}", flush=True)
                time.sleep(0.1)
        i = 0
        while not quit_flag.is_set():
            step = steps[i % len(steps)]
            if step.label != "off":
                print(f"[{i % len(steps) // 2 + 1}/{len(steps) // 2}] {step.label}", flush=True)
            glove.apply(step.cmd)
            end = time.monotonic() + step.seconds
            while not quit_flag.is_set() and not skip.is_set() and (time.monotonic() < end or paused.is_set()):
                canvas = None
                if args.view != "none":
                    image = camera.newest_bgr()
                    canvas = fit_width(image, rt["view"]["width"]) if image is not None else blank(rt["view"]["width"])
                    text_panel(canvas, [(f"NOW: {step.label}", (0, 220, 255)),
                                        "PAUSED" if paused.is_set() else f"next in {max(0, end - time.monotonic()):.1f}s",
                                        "keys: n next | space/p pause | q quit"], (8, 8), scale=0.6)
                    h = canvas.shape[0]
                    if h > 230 and hasattr(glove, "draw"):
                        glove.draw(canvas, (8, h - 230))
                for k in view.show(canvas):
                    on_key(k)
            skip.clear()
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        glove.all_off()
        if hasattr(glove, "close"):
            glove.close()
        camera.stop()
        view.close()
        print("all off.", flush=True)


if __name__ == "__main__":
    main()
