"""Drive the glove without any vision, so hardware can be built and tested on its own.

Run from the repo root:
    python -m haptics.fake_stream --list                     # find the ESP32's port
    python -m haptics.fake_stream --port PORT test           # firmware pulses each channel once
    python -m haptics.fake_stream --port PORT sweep          # ramps each channel 0 -> 255 in turn
    python -m haptics.fake_stream --port PORT one 3          # buzz one channel (here: ring) for 2 s
    python -m haptics.fake_stream --port PORT reach          # simulated hand moving to a target,
                                                             #   through the real controller
Leave out --port to print frames instead of sending them.
"""
import argparse
import time

from haptics.controller import GuidanceController
from haptics.link import GloveLink
from haptics.protocol import CHANNELS


def list_ports():
    from serial.tools import list_ports as lp
    for p in lp.comports():
        print(p.device, "-", p.description)


def run_sweep(link):
    for ch, name in enumerate(CHANNELS):
        print(f"channel {ch}: {name}")
        for level in range(0, 256, 15):
            levels = [0] * 8
            levels[ch] = level
            link.set(levels)
            time.sleep(0.05)
        link.set([0] * 8)
        time.sleep(0.4)


def run_one(link, ch, seconds=2.0):
    print(f"channel {ch}: {CHANNELS[ch]}")
    levels = [0] * 8
    levels[ch] = 255
    link.set(levels)
    time.sleep(seconds)
    link.set([0] * 8)


def run_reach(link, fps=15):
    """A hand starts bottom-left and moves to a box near the top-right of a 640x480 frame."""
    ctrl = GuidanceController()
    box = (420, 120, 500, 200)
    start, goal = (80.0, 420.0), (460.0, 160.0)
    steps = 60
    for i in range(steps + 25):
        f = min(i / steps, 1.0)
        hand = (start[0] + f * (goal[0] - start[0]), start[1] + f * (goal[1] - start[1]))
        cmd = ctrl.update(hand, box, (640, 480))
        link.set(cmd.levels, cmd.wrist_hz)
        print(f"{cmd.status:8s} levels={cmd.levels} wrist={cmd.wrist_hz:.1f}Hz")
        if cmd.status == "success":
            break
        time.sleep(1 / fps)
    link.set([0] * 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("mode", nargs="?", choices=["test", "sweep", "one", "reach"], default="sweep")
    ap.add_argument("channel", nargs="?", type=int, default=0, help="channel for 'one' (0-7)")
    args = ap.parse_args()

    if args.list:
        list_ports()
        return

    link = GloveLink(args.port, rate_hz=50 if args.port else 2)
    if args.mode != "test":
        link.start()  # test mode is run by the firmware alone, so no frames are streamed
    try:
        if args.mode == "test":
            link.test()
            time.sleep(6)
        elif args.mode == "sweep":
            run_sweep(link)
        elif args.mode == "one":
            run_one(link, args.channel)
        else:
            run_reach(link)
    finally:
        link.stop()


if __name__ == "__main__":
    main()
