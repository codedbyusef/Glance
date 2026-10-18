"""Progress message -> glove command (CLAUDE.md section 5), plus what the live view shows.

haptics.GuidanceController is used unchanged, with success_frames=None (the vision server's reach session
owns success). Its output is mapped to on/off motors (CLAUDE.md section 5 and decision 24): a direction
channel with level > 0 is active, and every active direction motor pulses at the proximity rate
(cmd.wrist_hz: 1 Hz far, 6 Hz near). There is no wrist motor; the controller's PROXIMITY level is unused.

    g = Guidance("table")
    state = g.update(progress)        # one per progress message
    glove.apply(state.cmd)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from haptics import protocol
from haptics.controller import ControllerConfig, GuidanceController

from .glove_out import GloveCommand

MODES = ("table", "wall")
WORDS = {protocol.LEFT: "LEFT", protocol.RIGHT: "RIGHT", protocol.FORWARD: "FORWARD",
         protocol.BACKWARD: "BACK", protocol.UP: "UP", protocol.DOWN: "DOWN"}
_WORD_ORDER = (protocol.LEFT, protocol.RIGHT, protocol.FORWARD, protocol.BACKWARD, protocol.UP, protocol.DOWN)


@dataclass
class GuidanceState:
    cmd: GloveCommand = field(default_factory=GloveCommand)
    words: list[str] = field(default_factory=list)   # active direction commands, e.g. ["LEFT", "FORWARD"]
    pulse_hz: float = 0.0                            # proximity: the rate the active direction motors pulse at
    status: str = "idle"                             # the controller's: guiding | aligned | success | hand_lost | ...
    mode: str = "table"
    dx: Optional[float] = None                       # normalized image error (fraction of width), + = target right
    dy: Optional[float] = None                       # + = target lower in the image
    dz: Optional[float] = None                       # depth gap (dz - contact offset), depth units; + = target farther
    units: Optional[str] = None                      # metric | relative | None (no depth)
    touching: bool = False
    frozen: bool = False
    distance: str = "no depth"                       # human-readable distance estimate
    ms: float = 0.0                                  # time spent in update()


DIRECTIONS = frozenset(WORDS)


def command_from(levels: list[int], proximity_hz: float) -> GloveCommand:
    """Decision 24: each active direction motor (level > 0) pulses at the proximity rate; nothing is steady.
    Without a rate (0) the active motors are steadily on. The wrist/PROXIMITY level is ignored."""
    active = [ch for ch, level in enumerate(levels) if level > 0 and ch in DIRECTIONS]
    if proximity_hz > 0:
        return GloveCommand(pulse_hz={ch: proximity_hz for ch in active})
    return GloveCommand(on=frozenset(active))


def distance_text(depth: Optional[dict], dx: Optional[float], dy: Optional[float],
                  frame_size: tuple[int, int], focal_px: Optional[float]) -> str:
    """cm when depth is metric (lateral cm too if focal_px is calibrated), else normalized values, labeled."""
    if not depth or depth.get("gap") is None:
        lateral = f"dx {dx:+.2f} dy {dy:+.2f} (frame fractions)" if dx is not None else ""
        return f"no depth gap; {lateral}".rstrip("; ")
    gap, hand = depth["gap"], depth.get("hand")
    if depth.get("units") == "metric":
        text = f"gap {gap * 100:+.0f} cm"
        if hand is not None:
            text += f", hand {hand:.2f} m from camera"
        if focal_px and hand is not None and dx is not None:
            w, h = frame_size
            lx = dx * w / focal_px * hand * 100
            ly = dy * h / focal_px * hand * 100
            text += f", lateral {lx:+.0f} / {ly:+.0f} cm"
        elif dx is not None:
            text += f", lateral dx {dx:+.2f} dy {dy:+.2f} (frame fractions; focal_px not calibrated)"
        return text
    return f"gap {gap:+.2f} (relative depth units), dx {dx:+.2f} dy {dy:+.2f} (frame fractions)" \
        if dx is not None else f"gap {gap:+.2f} (relative depth units)"


class Guidance:
    def __init__(self, mode: str = "table", controller_cfg: Optional[ControllerConfig] = None,
                 focal_px: Optional[float] = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        base = controller_cfg or ControllerConfig()
        self._base = base
        self.focal_px = focal_px
        self.mode = mode
        self.ctrl = self._make(mode)
        self.state = GuidanceState(mode=mode)

    @classmethod
    def from_config(cls, mode: str = "table") -> "Guidance":
        from depth_estimation import depth_config

        return cls(mode, focal_px=depth_config.load()["band"].get("focal_px"))

    def _make(self, mode: str) -> GuidanceController:
        cfg = ControllerConfig(**{**vars(self._base), "axis_mode": mode, "success_frames": None})
        return GuidanceController(cfg)

    def toggle_mode(self) -> str:
        self.mode = "wall" if self.mode == "table" else "table"
        self.ctrl = self._make(self.mode)
        self.state = GuidanceState(mode=self.mode)
        return self.mode

    def reset(self) -> None:
        self.ctrl.reset()
        self.state = GuidanceState(mode=self.mode)

    def update(self, progress: dict, frame_size: Optional[tuple[int, int]] = None) -> GuidanceState:
        t0 = time.perf_counter()
        size = tuple(progress.get("frame_size") or ()) or frame_size or (640, 480)
        if not size[0]:
            size = frame_size or (640, 480)
        depth = progress.get("depth") or None
        hand = progress.get("hand_point")
        box = progress.get("target_box")
        touching = bool(depth and depth.get("touching"))
        cmd = self.ctrl.update(tuple(hand) if hand else None, tuple(box) if box else None, size,
                               depth_error=depth.get("depth_error") if depth else None, touching=touching)
        glove = command_from(cmd.levels, cmd.wrist_hz)
        has_error = box is not None and hand is not None
        dx, dy = (cmd.error if has_error else (None, None))
        active = glove.on | set(glove.pulse_hz)
        st = GuidanceState(
            cmd=glove,
            words=[WORDS[ch] for ch in _WORD_ORDER if ch in active],
            pulse_hz=max(glove.pulse_hz.values(), default=0.0),
            status=cmd.status,
            mode=self.mode,
            dx=dx, dy=dy,
            dz=depth.get("gap") if depth else None,
            units=depth.get("units") if depth else None,
            touching=touching,
            frozen=bool(depth and depth.get("target_frozen")),
            distance=distance_text(depth, dx, dy, size, self.focal_px),
        )
        st.ms = (time.perf_counter() - t0) * 1000
        self.state = st
        return st
