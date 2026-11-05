"""Planner + FakeHardware tests, no API and no photos.

    python -m pytest tests/test_planner.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import geometry as g  # noqa: E402
import hardware as hw_mod  # noqa: E402
import planner as pl  # noqa: E402
from hardware import FakeHardware, PiHardware  # noqa: E402

W, H, D, L = 1296, 972, 12.0, 25.0


def calib(**robot) -> g.Calib:
    r = {"placeholder": False, "camera_height_cm": 25.0, "ridge_height_cm": 8.0,
         "camera_to_nozzle_cm": D, "camera_side_offset_cm": 0.0, "nozzle_height_cm": 15.0,
         "cm_per_tick": 1.0, "camera_hfov_deg": 66.0, "image_width_px": W,
         "image_height_px": H, "arm_length_cm": L, "spray_radius_cm": 1.5,
         "protect_margin_cm": 0.5, "pump_flow_ml_per_s": 2.0}
    r.update(robot)
    return g.Calib(robot=r, servo={}, homographies={}, quiet=True)


class NoVision:
    def detect(self, image, tick):
        return []


class OneWeed:
    """Reports a weed at a fixed world position, in the pixels of whatever
    frame the planner just took (tick * cm_per_tick is where the planner
    thinks the car is)."""

    def __init__(self, c: g.Calib, x: float, y: float):
        self.c, self.x, self.y = c, x, y

    def detect(self, image, tick):
        p = tick * float(self.c.robot["cm_per_tick"])
        u, v = g.ground_to_pixel(self.x - p, self.y, "ridge", (W, H), self.c)
        if not (0 <= u < W and 0 <= v < H):
            return []
        return [{"label": "weed", "action": "spray", "box": [u - 5, v - 5, u + 5, v + 5],
                 "spray_px": (u, v), "polygon_px": []}]


def make(c: g.Calib, vision, tmp_path, **cfg) -> tuple[FakeHardware, pl.Planner]:
    park = g.arm_to_servo(float(c.servo["arm"]["park_deg"]), c)
    hw = FakeHardware(lambda cam_x: np.zeros((4, 4, 3), np.uint8), camera_to_nozzle_cm=D,
                      cm_per_tick=float(c.robot["cm_per_tick"]), park_deg=park)
    return hw, pl.Planner(hw, vision, c, None, pl.Config(out_dir=tmp_path, **cfg))


def landing(hw: FakeHardware, c: g.Calib, i: int = 0) -> tuple[float, float]:
    """Where pump i really put the nozzle (true car position, real servo)."""
    p = hw.pumps[i]
    t1 = p["t0"] + p["ms"] / 1000.0
    mid = [r for r in hw.trace if p["t0"] < r[0] <= t1]
    t, pos, servo, *_ = mid[len(mid) // 2]
    phi = math.radians(g.servo_to_arm(servo, c))
    return pos + L * math.cos(phi), L * math.sin(phi)


# (a) the stop tick
@pytest.mark.parametrize("y", [0.0, 5.0, 10.0, -12.0, 20.0])
def test_stop_tick_is_weed_tick_minus_arm_forward_offset(y, tmp_path):
    c = calib()
    _hw, p = make(c, NoVision(), tmp_path)
    stop_tick, phi = p.plan_stop(100.0, y)
    assert phi == pytest.approx(math.degrees(math.asin(y / L)), abs=1e-9)
    assert stop_tick == 100 - round(L * math.cos(math.radians(phi)))


def test_arm_out_of_reach_is_refused(tmp_path):
    c = calib()
    _hw, p = make(c, NoVision(), tmp_path)
    with pytest.raises(g.OutOfReach):
        p.plan_stop(100.0, L)          # farther than the arm swings (max 80 deg)


# (b) a weed inside a protected zone is skipped
def test_weed_inside_protected_zone_is_skipped(tmp_path):
    c = calib()
    hw, p = make(c, NoVision(), tmp_path)
    p.weeds = [{"id": 0, "x": 60.0, "y": 6.0, "plane": "ridge", "sightings": 1}]
    p.protected = [{"label": "crop", "polygon": [(55, 1), (65, 1), (65, 11), (55, 11)]}]
    p.spray_pass()
    assert hw.pumps == []
    assert p.weeds[0]["status"] == "skipped: near crop"


def test_weed_just_outside_the_safety_distance_is_sprayed(tmp_path):
    c = calib()
    hw, p = make(c, NoVision(), tmp_path)
    p.weeds = [{"id": 0, "x": 60.0, "y": 6.0, "plane": "ridge", "sightings": 1}]
    far = 6.0 + p.safe_distance() + 1.0
    p.protected = [{"label": "crop", "polygon": [(55, far), (65, far), (65, far + 5), (55, far + 5)]}]
    p.spray_pass()
    assert len(hw.pumps) == 1 and p.weeds[0]["status"] == "sprayed"


# (c) zero weeds -> zero sprays
@pytest.mark.parametrize("continuous", [False, True])
def test_no_weeds_no_sprays(continuous, tmp_path):
    c = calib()
    hw, p = make(c, NoVision(), tmp_path, continuous=continuous, vision_s=0.5)
    p.run(80.0)
    assert hw.pumps == [] and p.weeds == []
    assert (tmp_path / "weed_map.json").is_file()


# (d) the encoder matches the distance driven
def test_ticks_match_distance_driven():
    hw = FakeHardware(cm_per_tick=1.0)
    hw.drive(4.0)
    hw.sleep(25.0)
    hw.stop()
    hw.sleep(1.0)                       # coasts a little: still counted
    assert hw.pos == pytest.approx(100.0 + 4.0 * hw_mod.COAST_S, abs=0.05)
    assert abs(hw.ticks() - hw.pos) <= 1
    hw.drive(-4.0)
    hw.sleep(10.0)
    assert hw.pos == pytest.approx(100.4 - 40.0, abs=0.05)
    assert abs(hw.ticks() - hw.pos) <= 1


def test_ticks_follow_cm_per_tick():
    hw = FakeHardware(cm_per_tick=0.5)
    hw.drive(5.0)
    hw.sleep(10.0)
    assert hw.ticks() == 100


def test_servo_takes_time_and_events_are_logged():
    hw = FakeHardware(park_deg=160.0)
    hw.servo_set(100.0)                 # 60 degrees = 0.1 s
    hw.sleep(0.05)
    assert 100.0 < hw.servo_deg() < 160.0
    hw.sleep(0.06)
    assert hw.servo_deg() == pytest.approx(100.0)
    hw.pump(150)
    kinds = [e["type"] for e in hw.events]
    assert kinds == ["start", "servo", "pump"]
    assert hw.now() == pytest.approx(0.11 + 0.15)
    assert all({"t", "type", "value", "tick", "arm"} <= set(e) for e in hw.events)


def test_pi_hardware_implements_the_interface():
    # Was a stub until the robot was built; tests/conftest.py makes it a dry run.
    hw = PiHardware()
    try:
        assert hw.dry_run
        hw.drive(1.0)
        hw.stop()
        assert [e["type"] for e in hw.events][:1] == ["start"]
    finally:
        hw.close()


# end to end on the fake robot: the spray lands on the weed
@pytest.mark.parametrize("continuous,speed", [(False, 3.0), (False, 5.0), (True, 3.0)])
@pytest.mark.parametrize("wy", [0.0, 8.0, -11.0])
def test_spray_lands_on_the_weed(continuous, speed, wy, tmp_path):
    c = calib()
    hw, p = make(c, OneWeed(c, 70.0, wy), tmp_path, continuous=continuous, speed=speed, vision_s=2.0)
    p.run(110.0)
    assert len(hw.pumps) == 1
    x, y = landing(hw, c)
    # a tick is 1 cm and the spray wet radius 1.5 cm: within 1.5 cm is a hit
    assert math.dist((x, y), (70.0, wy)) < 1.5, (x, y)


def test_two_sightings_of_one_weed_are_merged(tmp_path):
    c = calib()
    # frames at ticks 0, 19, 38: the ground from x=37.8 to 43.2 is in two of them
    hw, p = make(c, OneWeed(c, 40.0, 4.0), tmp_path, vision_s=1.0)
    p.run(90.0)
    assert len(p.frames) >= 3
    assert sum(1 for f in p.frames if f["dets"]) >= 2
    assert len(p.weeds) == 1 and p.weeds[0]["sightings"] >= 2
    assert len(hw.pumps) == 1


def test_pass_two_waits_for_vision(tmp_path):
    c = calib()
    hw, p = make(c, OneWeed(c, 30.0, 4.0), tmp_path, vision_s=40.0, vision_workers=1)
    p.run(60.0)
    assert any(d["type"] == "wait_vision" for d in p.log)
    assert hw.pumps and hw.pumps[0]["t0"] >= p.ready_at


# ---- one pass, real time ----
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("wy", [0.0, 9.0, -13.0])
def test_realtime_one_pass_sprays_without_returning(continuous, wy, tmp_path):
    c = calib(camera_to_nozzle_cm=40.0)          # camera sees a weed before the arm can reach it
    park = g.arm_to_servo(float(c.servo["arm"]["park_deg"]), c)
    hw = FakeHardware(lambda x: np.zeros((4, 4, 3), np.uint8), camera_to_nozzle_cm=40.0, park_deg=park)
    p = pl.Planner(hw, OneWeed(c, 90.0, wy), c, None,
                   pl.Config(out_dir=tmp_path, vision_s=0.1, continuous=continuous, image_size=(W, H)))
    p.run_realtime(140.0)
    assert len(hw.pumps) == 1
    assert min(r[1] for r in hw.trace) >= -0.5       # never drove back
    x, y = landing(hw, c)
    assert math.dist((x, y), (90.0, wy)) < 1.5, (x, y)


def test_realtime_with_the_camera_too_close_misses_central_weeds(tmp_path):
    # camera 12 cm ahead, arm 25 cm long: the weed is seen after the arm passed it
    c = calib()
    hw, p = make(c, OneWeed(c, 90.0, 0.0), tmp_path, vision_s=0.1)
    p.run_realtime(140.0)
    assert hw.pumps == [] and p.weeds[0]["status"] == "skipped: passed"


# ---- re-aiming a weed that sits next to a crop ----
def test_reaim_moves_to_the_far_side_of_a_big_weed(tmp_path):
    c = calib()
    hw, p = make(c, NoVision(), tmp_path)
    weed = [(58, 2), (64, 2), (64, 12), (58, 12)]           # a big weed, y 2..12
    p.weeds = [{"id": 0, "x": 61.0, "y": 3.0, "plane": "ridge", "sightings": 1,
                "poly": weed, "conf": 0.97}]
    p.protected = [{"label": "crop", "polygon": [(55, -4), (67, -4), (67, 0.5), (55, 0.5)]}]
    p.spray_pass()
    assert len(hw.pumps) == 1 and p.weeds[0]["status"] == "sprayed"
    assert p.weeds[0]["y"] > 3.0
    assert p.dist_to_protected(p.weeds[0]["x"], p.weeds[0]["y"]) >= p.safe_distance()


def test_reaim_needs_a_very_sure_judge(tmp_path):
    c = calib()
    hw, p = make(c, NoVision(), tmp_path)
    weed = [(58, 2), (64, 2), (64, 12), (58, 12)]
    p.weeds = [{"id": 0, "x": 61.0, "y": 3.0, "plane": "ridge", "sightings": 1,
                "poly": weed, "conf": 0.90}]
    p.protected = [{"label": "crop", "polygon": [(55, -4), (67, -4), (67, 0.5), (55, 0.5)]}]
    p.spray_pass()
    assert hw.pumps == [] and p.weeds[0]["status"] == "skipped: near crop"
