"""Hand-to-target error -> 8 motor values. Plain rules, no model, no learned policy.

Deadband per axis, hysteresis, EMA smoothing, at most two direction channels active
at once. Tuning lives in config/controller.yaml, not here.
"""

from __future__ import annotations


def compute_motor_values(dx: float, dy: float, dz: float) -> list[int]:
    raise NotImplementedError
