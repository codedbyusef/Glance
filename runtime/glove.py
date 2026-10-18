"""Glove commands over the runtime's real path: GloveCommand -> HardwareGlove -> GloveLink -> USB serial ->
firmware/glove/code.py on the ESP32-S3. (firmware/glove/send_test.py writes raw frames instead; both work.)

    python -m runtime.glove ports                    # serial ports; marks the one the glove auto-detect picks
    python -m runtime.glove test                     # firmware test mode: each motor once, prints TEST lines
    python -m runtime.glove buzz LEFT                # a direction word (LEFT RIGHT FORWARD BACK UP DOWN),
    python -m runtime.glove buzz thumb --seconds 3   #   a motor name (thumb index ring pinky back palm), or all
    python -m runtime.glove sweep                    # a simulated reach through runtime.guidance: the commands
    python -m runtime.glove sweep --mode wall        #   change as a fake hand closes in on a fake target
    python -m runtime.glove off

Options: --port (default glove.port in runtime/config, "auto" detects), --firmware s3|ino8, --level.
Ctrl+C at any time: all off.
"""
from __future__ import annotations

import argparse
import sys
import time

from haptics import protocol

from . import config as runtime_config
from .glove_out import GLOVE_USB_VIDS, OFF, GloveCommand, HardwareGlove, find_glove_port
from .guidance import Guidance

WORD_CHANNEL = {"LEFT": protocol.LEFT, "RIGHT": protocol.RIGHT, "FORWARD": protocol.FORWARD,
                "BACK": protocol.BACKWARD, "UP": protocol.UP, "DOWN": protocol.DOWN}


def channel_for(name: str) -> int | None:
    """A direction word or a motor name -> logical channel; None for 'all'."""
    if name.lower() == "all":
        return None
    if name.upper() in WORD_CHANNEL:
        return WORD_CHANNEL[name.upper()]
    if name.lower() in protocol.CHANNELS:
        return protocol.CHANNELS.index(name.lower())
    raise SystemExit(f"unknown motor {name!r}: use {', '.join(WORD_CHANNEL)}, {', '.join(protocol.CHANNELS)} or all")


def sweep_path(seconds: float, hz: float = 20.0, size=(640, 480), box=(280.0, 200.0, 360.0, 280.0)):
    """Progress messages for a hand moving from the bottom-left corner into the box, then resting in it."""
    n = max(2, int(seconds * hz))
    start, target = (40.0, 450.0), ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    for i in range(n):
        f = min(1.0, i / (0.75 * n))  # the last quarter rests on the target
        hand = (start[0] + f * (target[0] - start[0]), start[1] + f * (target[1] - start[1]))
        yield {"session_id": "sweep", "frame_id": i, "frame_size": list(size), "target_box": list(box),
               "hand_point": list(hand), "depth": None, "timings_ms": {}}


def run_sweep(glove, mode: str, seconds: float, hz: float = 20.0, out=print) -> list[list[str]]:
    """Drive the glove through guidance along sweep_path; returns the words at each step (tests)."""
    g = Guidance(mode)
    seen, last = [], None
    for msg in sweep_path(seconds, hz):
        st = g.update(msg)
        glove.apply(st.cmd)
        seen.append(st.words)
        line = f"{' '.join(st.words) or '-':16s} wrist {st.wrist_hz:.1f} Hz (proximity; no wrist motor on the S3)"
        if line != last:
            out(f"  {line}")
            last = line
        time.sleep(1.0 / hz)
    glove.all_off()
    return seen


def cmd_ports(args, rt) -> int:
    from serial.tools import list_ports

    ports = list_ports.comports()
    pick = find_glove_port(ports)
    if not ports:
        print("no serial ports. Is the glove plugged in (USB data cable, boot.py on CIRCUITPY)?")
    for p in ports:
        vid = f"{p.vid:04x}:{p.pid:04x}" if p.vid is not None else "----:----"
        mark = "  <- glove" if p.device == pick else ""
        print(f"{p.device:16s} {vid}  {p.description}{mark}")
    if ports and pick is None:
        print(f"none looks like the glove (USB vendor {', '.join(f'{v:04x}' for v in GLOVE_USB_VIDS)}); "
              f"pass --port")
    return 0 if pick else 1


def open_glove(args, rt) -> HardwareGlove:
    g = rt["glove"]
    glove = HardwareGlove(args.port or g["port"], args.level or g["on_level"], args.firmware or g["firmware"])
    print(f"glove: {glove.port} ({glove.firmware}: {', '.join(protocol.CHANNELS[c] for c in glove.channels)})",
          flush=True)
    return glove


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m runtime.glove", description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port (default glove.port; auto = detect)")
    parser.add_argument("--firmware", choices=("s3", "ino8"), help="default glove.firmware")
    parser.add_argument("--level", type=int, help="on level 1-255 (default glove.on_level)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ports", help="list serial ports and the auto-detected glove")
    sub.add_parser("test", help="firmware test mode: each motor once")
    b = sub.add_parser("buzz", help="one motor (or all) on, through the runtime's mapping")
    b.add_argument("motor", help="LEFT RIGHT FORWARD BACK UP DOWN, thumb index ring pinky back palm, or all")
    b.add_argument("--seconds", type=float, default=2.0)
    s = sub.add_parser("sweep", help="a simulated reach through runtime.guidance")
    s.add_argument("--mode", choices=("table", "wall"), default="table")
    s.add_argument("--seconds", type=float, default=8.0)
    sub.add_parser("off", help="all motors off")
    args = parser.parse_args(argv)
    rt = runtime_config.load()

    if args.cmd == "ports":
        return cmd_ports(args, rt)
    glove = open_glove(args, rt)
    try:
        for line in glove.read_lines():
            print(f"  glove says: {line}")
        if args.cmd == "test":
            glove.link.test()
            end = time.monotonic() + 6.0
            while time.monotonic() < end:
                lines = glove.read_lines()
                for line in lines:
                    print(f"  glove says: {line}", flush=True)
                if "TEST DONE" in lines:
                    break
                time.sleep(0.05)
        elif args.cmd == "buzz":
            ch = channel_for(args.motor)
            on = frozenset(glove.channels) if ch is None else frozenset({ch})
            if ch is not None and ch not in glove.channels:
                print(f"  {protocol.CHANNELS[ch]} has no motor on the {glove.firmware} glove: nothing will buzz")
            names = ", ".join(protocol.CHANNELS[c] for c in sorted(on))
            print(f"  on: {names} for {args.seconds:g} s", flush=True)
            glove.apply(GloveCommand(on=on))
            time.sleep(args.seconds)
        elif args.cmd == "sweep":
            print(f"  simulated reach ({args.mode} mode), {args.seconds:g} s:", flush=True)
            run_sweep(glove, args.mode, args.seconds)
        elif args.cmd == "off":
            glove.apply(OFF)
    except KeyboardInterrupt:
        pass
    finally:
        glove.all_off()
        glove.close()
        print("all off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
