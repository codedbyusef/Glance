"""Run the whole thing: selftest, then the voice agent with real reaches and the glove.

    python -m runtime.run                  # wake word ("hey glance"); needs headphones
    python -m runtime.run --ptt-only       # speakers: Enter starts and stops each utterance
    python -m runtime.run --typed          # type instead of talking
    bash runtime/scripts/run_glance.sh     # the same on the Pi (activates .venv first)

Steps:
1. Load runtime/glance.env (server URLs, tokens, GEMINI_API_KEY; copy runtime/glance.env.example).
2. Pick the glove: --glove, else glove.kind in runtime/config if it says hardware, else the hardware glove if
   one is plugged in (auto-detected), else the fake one (with a warning).
3. runtime.selftest (no audio unless --full-selftest). Any FAIL stops here unless --force.
4. runtime.voice_agent --reach real with that glove.
"""
from __future__ import annotations

import argparse
import sys

from . import config as runtime_config


def pick_glove(choice: str | None, port: str | None, rt: dict) -> tuple[str, str | None]:
    """(kind, port) for the run: an explicit choice wins, then the config, then whatever is plugged in."""
    from .glove_out import find_glove_port

    want_port = port or rt["glove"]["port"]
    if choice == "fake":
        return "fake", None
    explicit = choice == "hardware" or rt["glove"]["kind"] == "hardware"
    found = want_port if want_port and want_port != "auto" else find_glove_port()
    if found:
        return "hardware", found
    if explicit:
        raise SystemExit("hardware glove requested but no glove port found: plug it in, or pass --glove-port. "
                         "python -m runtime.glove ports lists the ports.")
    print("[run] no glove plugged in: using the fake glove (drawn on the live view)", flush=True)
    return "fake", None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m runtime.run", description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ptt-only", action="store_true", help="no wake word: Enter starts and stops each utterance")
    mode.add_argument("--typed", action="store_true", help="type utterances in the terminal")
    parser.add_argument("--glove", choices=("fake", "hardware"), help="default: hardware if plugged in, else fake")
    parser.add_argument("--glove-port", help="default: glove.port in runtime/config (auto = detect)")
    parser.add_argument("--view", choices=("window", "web", "none"), help="default: web without a display")
    parser.add_argument("--camera", help="camera index or video path (default: runtime/config)")
    parser.add_argument("--mode", choices=("table", "wall"), default="table", help="guidance axis mode")
    parser.add_argument("--full-selftest", action="store_true", help="include the audio selftest (speaks, records 3 s)")
    parser.add_argument("--skip-selftest", action="store_true")
    parser.add_argument("--force", action="store_true", help="start even if the selftest has a FAIL")
    args = parser.parse_args(argv)

    names = runtime_config.load_env()
    print(f"[run] runtime/glance.env: {', '.join(names) if names else 'nothing new (missing, or all already set)'}",
          flush=True)
    rt = runtime_config.load()
    kind, port = pick_glove(args.glove, args.glove_port, rt)
    glove_args = ["--glove", kind] + (["--glove-port", port] if port else [])
    camera_args = ["--camera", args.camera] if args.camera else []

    if not args.skip_selftest:
        from . import selftest

        rc = selftest.main(glove_args + camera_args + ([] if args.full_selftest else ["--skip-audio"]))
        if rc != 0 and not args.force:
            print("[run] selftest failed: fix the FAIL lines above, or rerun with --force", flush=True)
            return rc

    from . import voice_agent

    va = ["--reach", "real", "--mode", args.mode] + glove_args + camera_args
    va += ["--ptt-only"] if args.ptt_only else ["--typed"] if args.typed else []
    va += ["--view", args.view] if args.view else []
    print(f"[run] python -m runtime.voice_agent {' '.join(va)}", flush=True)
    voice_agent.main(va)
    return 0


if __name__ == "__main__":
    sys.exit(main())
