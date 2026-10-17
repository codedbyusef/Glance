"""Guidance controller: turns the hand-to-target error into glove motor levels.

Pure Python, no models. Call update() once per vision frame with the hand point and the
target box (both in pixels of the same frame); it returns a HapticCommand for GloveLink.

Axis modes
  "table" (default): head camera looks down at a desk. Image left/right -> thumb/pinky,
      image up (target farther across the table) -> index "forward", image down -> ring "back".
      Once the hand is over the target, the palm motor cues "down" until success.
  "wall": camera looks straight ahead. Image up/down -> back-of-hand "up" / palm "down";
      forward/back only fire if a depth error is passed in.

Image coordinates: x grows to the right, y grows downward, origin top-left.
"""
from dataclasses import dataclass, field
import math

from haptics.protocol import LEFT, RIGHT, FORWARD, BACKWARD, UP, DOWN, PROXIMITY


@dataclass
class ControllerConfig:
    axis_mode: str = "table"
    deadband: float = 0.04        # normalized error below which an axis is "aligned"
    on_threshold: float = 0.06    # hysteresis: an off channel turns on above this
    full_scale: float = 0.35      # normalized error that gives full intensity
    min_level: int = 110          # coin motors stall below roughly this level
    max_level: int = 255
    smoothing: float = 0.5        # EMA weight on the newest hand point (1.0 = no smoothing)
    max_direction_channels: int = 2
    down_level: int = 160         # palm "down" cue once over the target (table mode)
    wrist_level: int = 200
    wrist_hz_far: float = 1.0     # wrist pulse rate when far away
    wrist_hz_near: float = 6.0    # faster blurs: coin motors take ~40 ms to spin up
    wrist_far_distance: float = 0.5
    success_frames: int | None = 15  # None = success is decided elsewhere (e.g. the server)


@dataclass
class HapticCommand:
    levels: list = field(default_factory=lambda: [0] * 8)
    wrist_hz: float = 0.0
    status: str = "idle"          # guiding | aligned | success | hand_lost | target_lost | idle
    error: tuple = (0.0, 0.0)     # normalized (ex, ey) after smoothing, for logging


class GuidanceController:
    def __init__(self, config: ControllerConfig | None = None):
        self.cfg = config or ControllerConfig()
        self.reset()

    def reset(self):
        self._hand = None
        self._active = set()      # direction channels currently on (for hysteresis)
        self._inside_count = 0

    def update(self, hand, target_box, frame_size, depth_error=None, touching=False):
        """hand: (x, y) or None. target_box: (x1, y1, x2, y2) or None. frame_size: (w, h).
        depth_error: optional, positive = target farther than hand (wall mode only).
        touching: optional contact-sensor reading; ends the reach immediately when True."""
        cfg = self.cfg
        if target_box is None:
            self._soft_reset()
            return HapticCommand(status="target_lost")
        if hand is None:
            self._soft_reset()
            self._hand = None  # don't smooth a reappearing hand toward its old position
            return HapticCommand(status="hand_lost")

        w, h = frame_size
        if self._hand is None:
            self._hand = (float(hand[0]), float(hand[1]))
        else:
            a = cfg.smoothing
            self._hand = (a * hand[0] + (1 - a) * self._hand[0], a * hand[1] + (1 - a) * self._hand[1])
        hx, hy = self._hand

        x1, y1, x2, y2 = target_box
        inside = x1 <= hx <= x2 and y1 <= hy <= y2
        # Error to the nearest point of the box, so "inside" means zero error on both axes.
        ex = (_clamp(hx, x1, x2) - hx) / w
        ey = (_clamp(hy, y1, y2) - hy) / h

        cmd = HapticCommand(error=(ex, ey))

        if touching:
            self._soft_reset()
            cmd.status = "success"
            return cmd

        # Candidate direction channels: (channel, magnitude)
        candidates = []
        if ex < 0:
            candidates.append((LEFT, -ex))
        elif ex > 0:
            candidates.append((RIGHT, ex))
        if cfg.axis_mode == "table":
            if ey < 0:
                candidates.append((FORWARD, -ey))
            elif ey > 0:
                candidates.append((BACKWARD, ey))
        else:
            if ey < 0:
                candidates.append((UP, -ey))
            elif ey > 0:
                candidates.append((DOWN, ey))
            if depth_error is not None:
                if depth_error > 0:
                    candidates.append((FORWARD, depth_error))
                elif depth_error < 0:
                    candidates.append((BACKWARD, -depth_error))

        # Hysteresis: a channel that is off needs on_threshold to start, one that is on stays
        # on until it drops under the deadband. This stops flicker at the edge.
        active = []
        for ch, mag in candidates:
            threshold = cfg.deadband if ch in self._active else cfg.on_threshold
            if mag > threshold:
                active.append((ch, mag))
        active.sort(key=lambda cm: cm[1], reverse=True)
        active = active[: cfg.max_direction_channels]
        self._active = {ch for ch, _ in active}

        for ch, mag in active:
            frac = min(mag / cfg.full_scale, 1.0)
            cmd.levels[ch] = int(cfg.min_level + frac * (cfg.max_level - cfg.min_level))

        # Wrist: pulse faster as the hand gets closer.
        dist = math.hypot(ex, ey)
        near = 1.0 - min(dist / cfg.wrist_far_distance, 1.0)
        cmd.wrist_hz = cfg.wrist_hz_far + near * (cfg.wrist_hz_near - cfg.wrist_hz_far)
        cmd.levels[PROXIMITY] = cfg.wrist_level

        if inside:
            self._inside_count += 1
            cmd.status = "aligned"
            if cfg.axis_mode == "table":
                cmd.levels[DOWN] = cfg.down_level
            if cfg.success_frames is not None and self._inside_count >= cfg.success_frames:
                self._soft_reset()
                return HapticCommand(status="success", error=(ex, ey))
        else:
            self._inside_count = 0
            cmd.status = "guiding"
        return cmd

    def _soft_reset(self):
        self._active = set()
        self._inside_count = 0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
