import pytest

from haptics.controller import ControllerConfig, GuidanceController
from haptics.link import GloveLink
from haptics.protocol import (BACKWARD, DOWN, FORWARD, LEFT, PROXIMITY, RIGHT, UP, WRIST,
                              encode_command, encode_frame)

FRAME = (640, 480)
BOX = (300, 200, 340, 240)  # 40 px target in the middle


def ctrl(**kw):
    kw.setdefault("smoothing", 1.0)  # no smoothing, so each test sees exactly its input
    return GuidanceController(ControllerConfig(**kw))


# ---------- protocol ----------

def test_frame_layout_and_checksum():
    f = encode_frame([1, 2, 3, 4, 5, 6, 7, 8])
    assert f[0] == 0xAA and list(f[1:9]) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert f[9] == 1 ^ 2 ^ 3 ^ 4 ^ 5 ^ 6 ^ 7 ^ 8


def test_frame_rejects_bad_input():
    with pytest.raises(ValueError):
        encode_frame([0] * 7)
    with pytest.raises(ValueError):
        encode_frame([0] * 7 + [256])


def test_s3_frames_carry_six_levels_in_wire_order():
    from haptics.protocol import BACK, INDEX, PALM, PINKY, RING, S3_CHANNELS, THUMB, encode_levels, wire_levels

    levels = [10, 11, 12, 13, 14, 15, 16, 17]       # thumb index middle ring pinky back palm wrist
    wire = wire_levels(levels, S3_CHANNELS)
    assert wire == [levels[ch] for ch in (THUMB, INDEX, RING, PINKY, BACK, PALM)] == [10, 11, 13, 14, 15, 16]
    f = encode_levels(wire)
    assert len(f) == 8 and f[0] == 0xAA and f[-1] == 10 ^ 11 ^ 13 ^ 14 ^ 15 ^ 16
    assert wire_levels(levels) == levels


def test_link_sends_only_the_firmware_channels():
    from haptics.protocol import PALM, S3_CHANNELS, THUMB

    link = GloveLink(dry_run=True, channels=S3_CHANNELS)
    levels = [0] * 8
    levels[THUMB], levels[PALM], levels[WRIST] = 255, 200, 255
    link.set(levels, wrist_hz=2.0)
    f = link.wire_frame_at(0.1)
    assert list(f[1:7]) == [255, 0, 0, 0, 0, 200]      # no middle, no wrist on the S3 glove
    assert len(GloveLink(dry_run=True).wire_frame_at(0.0)) == 10  # default: all 8 (glove.ino)


def test_command():
    assert encode_command(1) == bytes([0xA5, 0x01])


# ---------- directions (table mode) ----------

def test_target_to_the_right_buzzes_pinky():
    cmd = ctrl().update((100, 220), BOX, FRAME)
    assert cmd.levels[RIGHT] > 0 and cmd.levels[LEFT] == 0
    assert cmd.status == "guiding"


def test_target_to_the_left_buzzes_thumb():
    cmd = ctrl().update((600, 220), BOX, FRAME)
    assert cmd.levels[LEFT] > 0 and cmd.levels[RIGHT] == 0


def test_target_higher_in_image_means_forward():
    cmd = ctrl().update((320, 450), BOX, FRAME)
    assert cmd.levels[FORWARD] > 0 and cmd.levels[BACKWARD] == 0


def test_target_lower_in_image_means_back():
    cmd = ctrl().update((320, 20), BOX, FRAME)
    assert cmd.levels[BACKWARD] > 0 and cmd.levels[FORWARD] == 0


def test_farther_means_stronger():
    near = ctrl().update((390, 220), BOX, FRAME).levels[LEFT]
    far = ctrl().update((600, 220), BOX, FRAME).levels[LEFT]
    assert far > near > 0


def test_inside_box_cues_down_and_no_directions():
    cmd = ctrl(success_frames=None).update((320, 220), BOX, FRAME)
    assert cmd.status == "aligned"
    assert cmd.levels[DOWN] > 0
    assert all(cmd.levels[ch] == 0 for ch in (LEFT, RIGHT, FORWARD, BACKWARD))


def test_wall_mode_uses_up_down():
    c = ctrl(axis_mode="wall")
    assert c.update((320, 450), BOX, FRAME).levels[UP] > 0
    c.reset()
    assert c.update((320, 20), BOX, FRAME).levels[DOWN] > 0


def test_wall_mode_depth_error_drives_forward():
    cmd = ctrl(axis_mode="wall").update((320, 220), BOX, FRAME, depth_error=0.2)
    assert cmd.levels[FORWARD] > 0


# ---------- deadband, hysteresis, cap ----------

def test_small_error_is_ignored():
    # 10 px left of the box edge = 0.016 normalized, under the 0.06 on-threshold
    cmd = ctrl().update((290, 220), BOX, FRAME)
    assert cmd.levels[RIGHT] == 0


def test_hysteresis_keeps_channel_on_near_the_edge():
    c = ctrl()
    assert c.update((100, 220), BOX, FRAME).levels[RIGHT] > 0          # far: on
    # 0.05 error: below on-threshold (0.06) but above deadband (0.04) -> stays on
    assert c.update((300 - 0.05 * 640, 220), BOX, FRAME).levels[RIGHT] > 0
    c.reset()
    # same 0.05 error from a cold start -> stays off
    assert c.update((300 - 0.05 * 640, 220), BOX, FRAME).levels[RIGHT] == 0


def test_at_most_two_direction_channels():
    cmd = ctrl(axis_mode="wall").update((20, 450), BOX, FRAME, depth_error=0.3)
    directions = [cmd.levels[ch] for ch in (LEFT, RIGHT, FORWARD, BACKWARD, UP, DOWN)]
    assert sum(1 for v in directions if v > 0) == 2


def test_levels_never_exceed_255():
    cmd = ctrl().update((0, 479), BOX, FRAME)
    assert max(cmd.levels) <= 255


# ---------- wrist, lost, success ----------

def test_wrist_pulses_faster_when_closer():
    far = ctrl().update((20, 460), BOX, FRAME).wrist_hz
    near = ctrl().update((280, 220), BOX, FRAME).wrist_hz
    assert near > far > 0


def test_hand_lost_zeros_everything():
    cmd = ctrl().update(None, BOX, FRAME)
    assert cmd.status == "hand_lost" and cmd.levels == [0] * 8


def test_target_lost_zeros_everything():
    cmd = ctrl().update((100, 100), None, FRAME)
    assert cmd.status == "target_lost" and cmd.levels == [0] * 8


def test_success_after_n_frames_inside():
    c = ctrl(success_frames=3)
    assert c.update((320, 220), BOX, FRAME).status == "aligned"
    assert c.update((320, 220), BOX, FRAME).status == "aligned"
    cmd = c.update((320, 220), BOX, FRAME)
    assert cmd.status == "success" and cmd.levels == [0] * 8


def test_leaving_the_box_resets_success_count():
    c = ctrl(success_frames=3)
    c.update((320, 220), BOX, FRAME)
    c.update((320, 220), BOX, FRAME)
    c.update((100, 220), BOX, FRAME)  # drifted out
    assert c.update((320, 220), BOX, FRAME).status == "aligned"


def test_touch_sensor_ends_reach():
    cmd = ctrl().update((100, 220), BOX, FRAME, touching=True)
    assert cmd.status == "success" and cmd.levels == [0] * 8


def test_smoothing_damps_a_jump():
    c = GuidanceController(ControllerConfig(smoothing=0.5))
    c.update((100, 220), BOX, FRAME)
    e_smoothed = c.update((600, 220), BOX, FRAME).error[0]
    e_raw = ctrl().update((600, 220), BOX, FRAME).error[0]
    assert abs(e_smoothed) < abs(e_raw)


# ---------- link ----------

def test_link_gates_wrist_at_its_pulse_rate():
    link = GloveLink(dry_run=True)
    levels = [0] * 8
    levels[PROXIMITY] = 200
    link.set(levels, wrist_hz=2.0)       # period 0.5 s: on for 0.25 s, off for 0.25 s, from the first frame
    assert link.frame_at(0.10)[WRIST] == 200
    assert link.frame_at(0.40)[WRIST] == 0
    assert link.frame_at(0.62)[WRIST] == 200


def test_link_pulses_any_channel_and_rate_changes_stay_smooth():
    link = GloveLink(dry_run=True)
    levels = [0] * 8
    levels[0] = 255
    link.set(levels, pulse_hz={0: 2.0})
    assert link.frame_at(1000.00)[0] == 255     # a pulse starts with its "on" half
    assert link.frame_at(1000.30)[0] == 0       # 0.3 s at 2 Hz: second half
    link.set(levels, pulse_hz={0: 2.1})         # a small rate change does not jump the phase
    assert link.frame_at(1000.32)[0] == 0
    assert link.frame_at(1000.52)[0] == 255     # next period begins
    link.set(levels)                            # no pulse: steady on
    assert link.frame_at(1000.80)[0] == 255


def test_link_clamps_levels():
    link = GloveLink(dry_run=True)
    link.set([300, -5, 0, 0, 0, 0, 0, 0])
    assert link.frame_at(0.0)[:2] == [255, 0]
